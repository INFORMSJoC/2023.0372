import os
import json
import time
from datetime import datetime
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from bokeh.plotting import gmap
from bokeh.models import GMapOptions, ColumnDataSource
from bokeh.io import output_file, show

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = []
        self.token_to_idx = {}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        tokens_to_add = reserved_tokens + ['<unk>']
        for token in tokens_to_add:
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def count_corpus(self, tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    @property
    def unk(self):
        return self.token_to_idx['<unk>']

    @property
    def token_freqs(self):
        return self._token_freqs

class UserNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_user, character_dim, dropout, character_vector):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.beta = nn.Parameter(torch.randn((latent_dim, input_dim)), requires_grad=True)
        self.u = nn.Parameter(torch.randn((num_user, latent_dim, character_dim)), requires_grad=True)
        self.register_buffer('character_vector', character_vector)

    def reparameterize(self, mean, log_var):
        epsilon = torch.randn_like(log_var)
        std = torch.exp(0.5 * log_var)
        return torch.softmax(mean + std * epsilon, dim=-1)

    def forward(self, bag_of_item):
        H1 = self.fc(bag_of_item).view(bag_of_item.shape[0] * bag_of_item.shape[1], -1)
        H2 = self.bn(H1).view(bag_of_item.shape[0], bag_of_item.shape[1], -1)
        H = self.dropout(torch.relu(H2))
        mean = self.fc_mean(H)
        log_var = self.fc_log_var(H)
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
        loss = bag_of_item * torch.log(wd)
        return -loss.sum()

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
        self.data_dir = os.path.join(self.base_dir, 'data', 'google_map')
        self.intermediate_dir = os.path.join(self.base_dir, 'intermediate', 'google_map')
        self.model_dir = self.intermediate_dir
        self.figure_dir = os.path.join(self.base_dir, 'results', 'google_map')
        self.heat_dir = os.path.join(self.figure_dir, 'heat')
        for d in [self.intermediate_dir, self.figure_dir, self.heat_dir]:
            os.makedirs(d, exist_ok=True)

        self.hidden_dim = 256
        self.latent_dim = 4
        self.num_user = 4936
        self.character_dim = 9
        self.dropout = 0.2
        self.lr = 0.005
        self.num_epochs = 400
        self.batch_size = 512

        # Load character vector (item proportion derived from FOP model)
        char_vec_path = os.path.join(self.intermediate_dir, 'character_vector.pt')
        if os.path.exists(char_vec_path):
            self.character_vector = torch.load(char_vec_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"character_vector.pt not found in {self.intermediate_dir}")

        # Load meta data for visualization (if exists)
        meta_path = os.path.join(self.data_dir, 'location.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.location = json.load(f)
        else:
            self.location = None
            print("Warning: location.json not found, map visualization will be skipped.")

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
        input_dim = user_tensor.shape[2]

        network = UserNetwork(
            input_dim, self.hidden_dim, self.latent_dim,
            self.num_user, self.character_dim, self.dropout,
            self.character_vector
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
                loss = network.log_likelihood_loss(u_b, wd) + network.kl_loss(mean, log_var) + 1e-4 * network.time_loss(mean)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(network.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss)
            if epoch % 10 == 0:
                print(f"epoch {epoch}, loss = {total_loss:.2f}")

        model_path = os.path.join(self.model_dir, 'user_model.pt')
        torch.save(network.state_dict(), model_path)
        plt.plot(range(self.num_epochs), losses)
        plt.savefig(os.path.join(self.figure_dir, 'motivation_loss.png'))
        plt.close()

        end_time = time.time()
        print(datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'))
        print(f"Training time: {end_time - start_time:.2f}s")
        return network

    def derive_top_item_for_motivation(self):
        network = self._load_network()
        beta = torch.softmax(network.beta, dim=-1)
        beta = torch.softmax(network.beta, dim=-1)
        top_values, top_indices = beta.topk(10, dim=1)
        print("Top indices:", top_indices)


    def visualize_motivation_through_map(self, api_key=""):
        if self.location is None:
            print("Meta data not available, skipping map visualization.")
            return
        network = self._load_network()
        beta = torch.softmax(network.beta, dim=-1)
        top_indices = beta.topk(10, dim=1)[1]

        lat_center = 20.6
        lng_center = -156.5
        map_options = GMapOptions(lat=lat_center, lng=lng_center, map_type="satellite", zoom=7)
        google_map = gmap(api_key, map_options, title="Hawaii")

        colors = ['red', 'green', 'blue', 'orange']
        for k in range(self.latent_dim):
            lat_list = []
            lon_list = []
            for idx in top_indices[k].tolist():
                item = self.location.get(str(idx), {})
                lat_list.append(item.get('latitude', 0))
                lon_list.append(item.get('longitude', 0))
            source = ColumnDataSource(data=dict(lat=lat_list, lng=lon_list))
            google_map.scatter(x="lng", y="lat", size=5, fill_color=colors[k % len(colors)],
                              line_color="white", fill_alpha=1, source=source, marker='circle')

        out_path = os.path.join(self.figure_dir, 'map.html')
        output_file(out_path)
        show(google_map)
        print(f"Map saved to {out_path}")

    def time_insights(self):
        user_tensor, _, _, _ = self.load_data()
        period_count = torch.sum(user_tensor, dim=-1)
        non_zero_mask = torch.all(period_count != 0, dim=1)
        non_zero_indices = torch.nonzero(non_zero_mask).squeeze()

        network = self._load_network()
        user_batch = user_tensor[non_zero_indices].to(self.device)
        mean, _, _ = network(user_batch)
        period_motivation = torch.softmax(mean, dim=-1)

        for i in range(len(period_motivation)):
            user_idx = non_zero_indices[i].item()
            data = period_motivation[i].T.cpu().detach()
            vmin, vmax = data.min(), data.max()
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            fig, ax = plt.subplots(figsize=(7.8, 4))
            im = ax.imshow(data, cmap='YlOrBr', interpolation='nearest', norm=norm)

            ax.set_xticks(range(6))
            ax.set_xticklabels(range(1, 7))
            ax.set_xlabel("Time period", fontsize=15)
            ax.tick_params(axis='x', labelsize=12)

            ax.set_yticks(range(self.latent_dim))
            ax.set_yticklabels(range(1, self.latent_dim+1))
            ax.set_ylabel("Motivation", fontsize=15)
            ax.tick_params(axis='y', labelsize=12)

            fig.colorbar(im, ax=ax, shrink=1.1)
            plt.subplots_adjust(left=0.08, right=1.05, bottom=0.15, top=0.9)
            save_path = os.path.join(self.heat_dir, f'{user_idx}.png')
            plt.savefig(save_path)
            plt.close(fig)
        print(f"Heatmaps saved to {self.heat_dir}")


    def save_proportion_motivation(self):
        network = self._load_network()
        beta = torch.softmax(network.beta, dim=-1)
        torch.save(beta, os.path.join(self.intermediate_dir, 'beta.pt'))

        user_tensor, _, _, _ = self.load_data()
        dataloader = DataLoader(user_tensor, batch_size=512, shuffle=False)
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
        input_dim = user_tensor.shape[2]
        network = UserNetwork(
            input_dim, self.hidden_dim, self.latent_dim,
            self.num_user, self.character_dim, self.dropout,
            self.character_vector
        ).to(self.device)
        model_path = os.path.join(self.model_dir, 'user_model.pt')
        network.load_state_dict(torch.load(model_path, map_location=self.device))
        network.eval()
        return network

    def main(self):
        # self.train()
        self.save_proportion_motivation()
        self.derive_top_item_for_motivation()
        self.visualize_motivation_through_map(api_key="YOUR_API_KEY")
        self.time_insights()


if __name__ == '__main__':
    M = Model(device='cuda:0')
    M.main()