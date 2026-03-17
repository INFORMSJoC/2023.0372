import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class UserNetwork(nn.Module):
    def __init__(self, input_dim, time_period_dim, hidden_dim, latent_dim, num_user, character_dim, dropout, character_vector):
        super().__init__()
        self.input_dim = input_dim
        self.time_period_dim = time_period_dim
        self.latent_dim = latent_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim + time_period_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim + time_period_dim, latent_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.beta = nn.Parameter(torch.randn((latent_dim, input_dim)), requires_grad=True)
        self.u = nn.Parameter(torch.randn((num_user, latent_dim, character_dim)), requires_grad=True)
        self.register_buffer('character_vector', character_vector)

    def reparameterize(self, mean, log_var):
        epsilon = torch.randn_like(log_var)
        std = torch.exp(0.5 * log_var)
        return torch.softmax(mean + std * epsilon, dim=-1)

    def forward(self, bag_of_item):
        if bag_of_item.size(0) == 0:
            # 处理空输入，返回空结果（根据实际情况可能需要调整）
            batch_size, num_time = 0, bag_of_item.size(1)
            device = bag_of_item.device
            mean = torch.empty((0, num_time, self.latent_dim), device=device)
            log_var = torch.empty((0, num_time, self.latent_dim), device=device)
            word_dist = torch.empty((0, num_time, self.input_dim), device=device)
            return mean, log_var, word_dist

        item_part = bag_of_item[..., :self.input_dim]
        time_part = bag_of_item[..., self.input_dim:]

        batch_size, num_time = item_part.shape[0], item_part.shape[1]

        H1 = self.fc(item_part)
        H1_flat = H1.view(batch_size * num_time, -1)
        H1_norm = self.bn(H1_flat).view(batch_size, num_time, -1)
        H = self.dropout(torch.relu(H1_norm))

        H_combined = torch.cat([H, time_part], dim=-1)

        mean = self.fc_mean(H_combined)
        log_var = self.fc_log_var(H_combined)
        theta = self.reparameterize(mean, log_var)

        word_distribution = torch.einsum('ijk, kt->ijt', theta, torch.softmax(self.beta, dim=-1))
        return mean, log_var, word_distribution

    def kl_loss(self, mean, log_var):
        var = torch.exp(log_var)
        term1 = log_var
        term2 = -mean.pow(2)
        term3 = -var
        return -0.5 * (1 + term1 + term2 + term3).sum()

    def log_likelihood_loss(self, bag_of_item, wd):
        item_part = bag_of_item[..., :self.input_dim]
        loss = item_part * torch.log(wd)
        return -loss.sum()

    def rating_loss(self, mean, rating, purchase, user_idx, valid_len):
        user_motivation_proportion = torch.softmax(mean, dim=-1)
        u = self.u[user_idx]
        select_vector = self.character_vector[purchase]
        inferred_rating_all_motivation = torch.einsum('ijkp, itp->ijkt', select_vector, u)
        weighted_rating = torch.einsum('ijkp, ijp->ijk', inferred_rating_all_motivation, user_motivation_proportion)
        whole_loss = torch.square(weighted_rating - rating)

        mask = torch.arange(rating.shape[-1], device=valid_len.device)[None, None, :] <= valid_len[:, :, None] - 1
        mask = mask.float()
        loss = whole_loss * mask
        return loss.sum()

    def time_loss(self, mean):
        time_part1 = mean[:, :-1]
        time_part2 = mean[:, 1:]
        loss = torch.square(time_part2 - time_part1)
        return loss.sum()

    def u_loss(self, user_idx):
        u = self.u[user_idx]
        diff = u[:, :, None, :] - u[:, None, :, :]
        distance = torch.norm(diff, dim=-1)
        return -distance.sum()

class Model:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data', 'movie')
        self.intermediate_dir = os.path.join(self.base_dir, 'intermediate', 'movie')
        self.figure_dir = os.path.join(self.base_dir, 'results', 'movie')
        self.heat_dir = os.path.join(self.figure_dir, 'heat')
        os.makedirs(self.intermediate_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.heat_dir, exist_ok=True)

        self._load_dimensions_from_data()

        char_vec_path = os.path.join(self.intermediate_dir, 'character_vector.pt')
        if not os.path.exists(char_vec_path):
            raise FileNotFoundError(f"character_vector.pt not found in {self.intermediate_dir}. Please run FOP model first.")
        self.character_vector = torch.load(char_vec_path, map_location='cpu')


    def _load_dimensions_from_data(self):
        user_tensor_path = os.path.join(self.intermediate_dir, 'user_tensor.pt')
        if not os.path.exists(user_tensor_path):
            raise FileNotFoundError(f"user_tensor.pt not found in {self.intermediate_dir}. Please run data preparation first.")
        user_tensor = torch.load(user_tensor_path, map_location='cpu')
        self.num_user, self.num_time, total_dim = user_tensor.shape

        char_vec_path = os.path.join(self.intermediate_dir, 'character_vector.pt')
        if os.path.exists(char_vec_path):
            char_vec = torch.load(char_vec_path, map_location='cpu')
            self.character_dim = char_vec.shape[-1]
        else:
            self.character_dim = 13  # 默认值，可根据实际调整

        self.input_dim = 7643  # 根据预处理固定，但也可以从 total_dim 减去时间维度得到
        self.time_period_dim = self.num_time
        self.hidden_dim = 256
        self.latent_dim = 5
        self.dropout = 0.2
        self.lr = 0.005
        self.num_epochs = 400
        self.batch_size = 512

    def load_data(self):
        user_tensor = torch.load(os.path.join(self.intermediate_dir, 'user_tensor.pt'))
        user_purchase = torch.load(os.path.join(self.intermediate_dir, 'user_purchase_tensor.pt')).to(torch.long)
        rating = torch.load(os.path.join(self.intermediate_dir, 'rating_tensor.pt'))
        valid_len = torch.load(os.path.join(self.intermediate_dir, 'valid_len_tensor.pt'))
        return user_tensor, user_purchase, rating, valid_len

    def train(self):
        start_time = time.time()
        print(datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'))

        user_tensor, user_purchase, rating, valid_len = self.load_data()
        expected_dim = self.input_dim + self.time_period_dim
        if user_tensor.shape[-1] != expected_dim:
            print(f"Warning: user_tensor last dimension {user_tensor.shape[-1]} != expected {expected_dim}")

        network = UserNetwork(
            self.input_dim, self.time_period_dim, self.hidden_dim, self.latent_dim,
            self.num_user, self.character_dim, self.dropout, self.character_vector
        ).to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)

        user_idx = torch.arange(self.num_user)
        dataset = TensorDataset(user_idx, user_tensor, user_purchase, rating, valid_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        losses = []
        network.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in dataloader:
                idx_b, u_b, pur_b, rat_b, len_b = [x.to(self.device) for x in batch]
                mean, log_var, wd = network(u_b)
                loss = network.log_likelihood_loss(u_b, wd) + network.kl_loss(mean, log_var) + network.time_loss(mean)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(network.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss)
            if epoch % 10 == 0:
                print(f"epoch {epoch}, loss = {total_loss:.2f}")

        model_path = os.path.join(self.intermediate_dir, 'user_model.pt')
        torch.save(network.state_dict(), model_path)

        end_time = time.time()
        print(datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'))
        print(f"Training time: {end_time - start_time:.2f}s")
        return network

    def derive_top_item_for_motivation(self):
        network = self._load_network()
        beta = torch.softmax(network.beta, dim=-1)
        top_values, top_indices = beta.topk(10, dim=1)
        print("Top indices:", top_indices)

    def time_insights(self):
        user_tensor, _, _, _ = self.load_data()
        network = self._load_network()
        # 使用所有用户
        all_indices = torch.arange(self.num_user)
        user_batch = user_tensor[all_indices].to(self.device)
        mean, _, _ = network(user_batch)
        period_motivation = torch.softmax(mean, dim=-1)

        max_user_limit = 50

        for i in range(max_user_limit):
            user_idx = i
            data = period_motivation[i].T.cpu().detach()
            vmin, vmax = data.min(), data.max()
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            fig, ax = plt.subplots(figsize=(7.8, 4))
            im = ax.imshow(data, cmap='YlOrBr', interpolation='nearest', norm=norm)

            ax.set_xticks(range(self.time_period_dim))
            ax.set_xticklabels(range(1, self.time_period_dim + 1))
            ax.set_xlabel("Time period", fontsize=15)
            ax.tick_params(axis='x', labelsize=12)

            ax.set_yticks(range(self.latent_dim))
            ax.set_yticklabels(range(1, self.latent_dim + 1))
            ax.set_ylabel("Motivation", fontsize=15)
            ax.tick_params(axis='y', labelsize=12)

            fig.colorbar(im, ax=ax, shrink=1.1)
            plt.subplots_adjust(left=0.08, right=1.05, bottom=0.15, top=0.9)
            save_path = os.path.join(self.heat_dir, f'{user_idx}.png')
            plt.savefig(save_path)
            plt.close(fig)
        print(f"Heatmaps saved for all {self.num_user} users to {self.heat_dir}")


    def save_proportion_motivation(self):
        network = self._load_network()
        beta = torch.softmax(network.beta, dim=-1)
        torch.save(beta, os.path.join(self.intermediate_dir, 'beta.pt'))

        user_tensor, _, _, _ = self.load_data()
        dataloader = DataLoader(user_tensor, batch_size=self.batch_size, shuffle=False)
        user_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Saving user motivation'):
                batch = batch.to(self.device)
                mean, _, _ = network(batch)
                period_motivation = torch.softmax(mean, dim=-1)
                user_list.append(period_motivation.cpu())
        user_motivation = torch.cat(user_list, dim=0)
        torch.save(user_motivation, os.path.join(self.intermediate_dir, 'user_motivation.pt'))
        print('beta.pt and user_motivation.pt saved.')


    def _load_network(self):
        user_tensor, _, _, _ = self.load_data()
        network = UserNetwork(
            self.input_dim, self.time_period_dim, self.hidden_dim, self.latent_dim,
            self.num_user, self.character_dim, self.dropout, self.character_vector
        ).to(self.device)
        model_path = os.path.join(self.intermediate_dir, 'user_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"user_model.pt not found in {self.intermediate_dir}. Please train first.")
        network.load_state_dict(torch.load(model_path, map_location=self.device))
        network.eval()
        return network

    def main(self):
        self.train()
        self.save_proportion_motivation()
        self.derive_top_item_for_motivation()
        self.time_insights()


if __name__ == '__main__':
    M = Model(device='cuda:0')
    M.main()