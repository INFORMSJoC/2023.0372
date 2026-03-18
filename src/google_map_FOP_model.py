import os
import json
import time
from datetime import datetime
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

class FOPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)
        self.fc_fop = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.phi = nn.Parameter(torch.randn((latent_dim, input_dim)), requires_grad=True)

    def reparameterize(self, mean, log_var):
        epsilon = torch.randn_like(log_var)
        std = torch.exp(0.5 * log_var)
        return torch.softmax(mean + std * epsilon, dim=-1)

    def forward(self, bag_of_FOP):
        H = self.dropout(torch.relu(self.bn(self.fc_fop(bag_of_FOP))))
        mean = self.fc_mean(H)
        log_var = self.fc_log_var(H)
        gamma = self.reparameterize(mean, log_var)
        word_distribution = torch.mm(gamma, torch.softmax(self.phi, dim=-1))
        return mean, log_var, word_distribution

    def evaluate(self, bag_of_FOP):
        H = self.dropout(torch.relu(self.bn(self.fc_fop(bag_of_FOP))))
        proportion = torch.softmax(self.fc_mean(H), dim=-1)
        word_distribution = torch.mm(proportion, torch.softmax(self.phi, dim=-1))
        return proportion, word_distribution

    def kl_loss(self, mean, log_var):
        var = torch.exp(log_var)
        term1 = log_var
        term2 = -mean.pow(2)
        term3 = -var
        return -0.5 * (1 + term1 + term2 + term3).sum()

    def log_likelihood_loss(self, X, wd):
        loss = X * torch.log(wd)
        return -loss.sum()

class Model:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data', 'google_map')
        self.intermediate_dir = os.path.join(self.base_dir, 'intermediate', 'google_map')
        self.model_dir = self.intermediate_dir
        self.figure_dir = os.path.join(self.base_dir, 'results', 'google_map', 'wordcloud')
        os.makedirs(self.intermediate_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)

        self.hidden_dim = 512
        self.pos_latent_dim = 6
        self.neg_latent_dim = 3
        self.dropout = 0.2
        self.lr = 0.005
        self.num_epochs = 400
        self.batch_size = 512

    def load_data_pos(self):
        path = os.path.join(self.intermediate_dir, 'pos_tensor.pt')
        return torch.load(path)

    def load_data_neg(self):
        path = os.path.join(self.intermediate_dir, 'neg_tensor.pt')
        return torch.load(path)

    def train_one(self, tensor, latent_dim, model_name):
        print(f"Training {model_name}...")
        start_time = time.time()
        print(datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'))

        input_dim = tensor.shape[1]
        network = FOPNetwork(input_dim, self.hidden_dim, latent_dim, self.dropout).to(self.device)
        dataloader = DataLoader(tensor, batch_size=self.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)

        losses = []
        network.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                mean, log_var, wd = network(batch)
                loss = network.log_likelihood_loss(batch, wd) + network.kl_loss(mean, log_var)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(network.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss)
            if epoch % 10 == 0:
                print(f"epoch {epoch}, loss = {total_loss:.2f}")

        model_path = os.path.join(self.model_dir, f'{model_name}.pt')
        torch.save(network.state_dict(), model_path)

        end_time = time.time()
        print(datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'))
        print(f"Training time: {end_time - start_time:.2f}s")
        return network

    def train_pos(self):
        tensor = self.load_data_pos()
        self.pos_network = self.train_one(tensor, self.pos_latent_dim, 'pos_model')

    def train_neg(self):
        tensor = self.load_data_neg()
        self.neg_network = self.train_one(tensor, self.neg_latent_dim, 'neg_model')

    def visualize_topic(self, polarity, network, latent_dim):
        with open(os.path.join(self.data_dir, f'{polarity}_FOP_text.json'), 'r') as f:
            data = json.load(f)
        token_lists = [data[str(i)] for i in range(len(data))]
        vocab = Vocab(token_lists, min_freq=3)

        topic_word_dist = torch.softmax(network.phi, dim=-1)
        top_k = 20
        top_indices = topic_word_dist.topk(top_k, dim=1)[1]

        for i in range(latent_dim):
            tokens = [vocab.idx_to_token[idx] for idx in top_indices[i].tolist() if vocab.idx_to_token[idx] != "<unk>"]
            print(f"Topic {i}: {' | '.join(tokens)}")
            freq = {vocab.idx_to_token[idx]: topic_word_dist[i, idx].item() for idx in range(len(vocab)) if vocab.idx_to_token[idx] != "<unk>"}
            wc = WordCloud(width=800, height=400, background_color='white' if polarity=='pos' else 'black')
            wc.generate_from_frequencies(freq)
            plt.figure(figsize=(10,5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            save_path = os.path.join(self.figure_dir, f'{polarity}_{i}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved {save_path}")

    def visualize_pos_topic(self):
        self.visualize_topic('pos', self.pos_network, self.pos_latent_dim)

    def visualize_neg_topic(self):
        self.visualize_topic('neg', self.neg_network, self.neg_latent_dim)

    def derive_item_proportion(self):
        pos_tensor = self.load_data_pos().to(self.device)
        neg_tensor = self.load_data_neg().to(self.device)

        self.pos_network.eval()
        self.neg_network.eval()

        pos_proportions = []
        dataloader = DataLoader(pos_tensor, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                prop, _ = self.pos_network.evaluate(batch)
                pos_proportions.append(prop.cpu())
        pos_pro = torch.cat(pos_proportions, dim=0)

        neg_proportions = []
        dataloader = DataLoader(neg_tensor, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                prop, _ = self.neg_network.evaluate(batch)
                neg_proportions.append(prop.cpu())
        neg_pro = torch.cat(neg_proportions, dim=0)

        item_proportion = torch.cat([pos_pro, neg_pro], dim=1)
        torch.save(item_proportion, os.path.join(self.intermediate_dir, 'item_proportion.pt'))

        char_vec = torch.zeros_like(item_proportion)
        char_vec[:, :self.pos_latent_dim] = (self.pos_latent_dim / (self.pos_latent_dim + self.neg_latent_dim)) * \
                                            item_proportion[:, :self.pos_latent_dim]
        char_vec[:, self.pos_latent_dim:] = (self.neg_latent_dim / (self.pos_latent_dim + self.neg_latent_dim) / (
                    self.neg_latent_dim - 1)) * (1 - item_proportion[:, self.pos_latent_dim:])
        torch.save(char_vec, os.path.join(self.intermediate_dir, 'character_vector.pt'))
        print('item_proportion.pt and character_vector.pt saved.')

    def main(self):
        self.train_pos()
        self.train_neg()
        self.visualize_pos_topic()
        self.visualize_neg_topic()
        self.derive_item_proportion()

if __name__ == '__main__':
    M = Model(device='cuda:0')
    M.main()