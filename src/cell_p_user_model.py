import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class UserNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_user, character_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.beta = nn.Parameter(torch.randn((latent_dim, input_dim)), requires_grad=True)

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

class UserModelTrainer:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data', 'cell_p')
        self.intermediate_dir = os.path.join(self.base_dir, 'intermediate', 'cell_p')
        self.heat_dir = os.path.join(self.base_dir, 'results', 'cell_p', 'heat')
        for d in [self.intermediate_dir, self.heat_dir]:
            os.makedirs(d, exist_ok=True)

        self.input_dim = 13089
        self.hidden_dim = 512
        self.latent_dim = 4
        self.num_user = 8629
        self.character_dim = 8
        self.dropout = 0.2
        self.lr = 0.005
        self.num_epochs = 200
        self.batch_size = 1024

    def load_user_tensor(self):
        path = os.path.join(self.intermediate_dir, 'user_tensor.pt')
        return torch.load(path)

    def train(self):
        start_time = time.time()
        print(datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'))

        network = UserNetwork(
            self.input_dim, self.hidden_dim, self.latent_dim,
            self.num_user, self.character_dim, self.dropout
        ).to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)

        user_tensor = self.load_user_tensor()
        dataloader = DataLoader(user_tensor, batch_size=self.batch_size, shuffle=False)

        network.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                mean, log_var, wd = network(batch)
                loss = (
                    network.kl_loss(mean, log_var) +
                    network.log_likelihood_loss(batch, wd) +
                    1e-4 * network.time_loss(mean)
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(network.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
            if epoch % 1 == 0:
                print(f"epoch {epoch}, loss = {total_loss:.2f}")

        model_path = os.path.join(self.intermediate_dir, 'user_model.pt')
        torch.save(network.state_dict(), model_path)
        end_time = time.time()
        print(datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'))
        print(f"Training time: {end_time - start_time:.2f}s")
        return network

    def motivation_insight(self):
        network = UserNetwork(
            self.input_dim, self.hidden_dim, self.latent_dim,
            self.num_user, self.character_dim, self.dropout
        ).to(self.device)
        model_path = os.path.join(self.intermediate_dir, 'user_model.pt')
        network.load_state_dict(torch.load(model_path, map_location=self.device))
        network.eval()

        beta = torch.softmax(network.beta, dim=-1)
        top_values, top_indices = beta.topk(10, dim=1)

        print("Top indices:", top_indices)

    def time_insights(self):
        user_tensor = self.load_user_tensor()
        period_count = torch.sum(user_tensor, dim=-1)
        non_zero_mask = torch.all(period_count != 0, dim=1)
        non_zero_indices = torch.nonzero(non_zero_mask).squeeze()

        network = UserNetwork(
            self.input_dim, self.hidden_dim, self.latent_dim,
            self.num_user, self.character_dim, self.dropout
        ).to(self.device)
        model_path = os.path.join(self.intermediate_dir, 'user_model.pt')
        network.load_state_dict(torch.load(model_path, map_location=self.device))
        network.eval()

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

            ax.set_xticks(range(10))
            ax.set_xticklabels(range(1, 11))
            ax.set_xlabel("Time period", fontsize=15)
            ax.tick_params(axis='x', labelsize=12)

            ax.set_yticks(range(self.latent_dim))
            ax.set_yticklabels(range(1, self.latent_dim+1))
            ax.set_ylabel("Motivation", fontsize=15)
            ax.tick_params(axis='y', labelsize=12)

            cbar = fig.colorbar(im, ax=ax, shrink=1.1)
            plt.subplots_adjust(left=0.08, right=1.05, bottom=0.15, top=0.9)
            save_path = os.path.join(self.heat_dir, f'{user_idx}.png')
            plt.savefig(save_path)
            plt.close(fig)
        print(f"Heatmaps saved to {self.heat_dir}")

    def save_proportion_motivation(self):
        network = UserNetwork(
            self.input_dim, self.hidden_dim, self.latent_dim,
            self.num_user, self.character_dim, self.dropout
        ).to(self.device)
        model_path = os.path.join(self.intermediate_dir, 'user_model.pt')
        network.load_state_dict(torch.load(model_path, map_location=self.device))
        network.eval()

        beta = torch.softmax(network.beta, dim=-1)
        torch.save(beta, os.path.join(self.intermediate_dir, 'beta.pt'))

        user_tensor = self.load_user_tensor()
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

    def run(self):
        self.train()
        self.save_proportion_motivation()
        self.motivation_insight()
        self.time_insights()

if __name__ == '__main__':
    trainer = UserModelTrainer(device='cuda:0')
    trainer.run()