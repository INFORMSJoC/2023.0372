import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.join('..', 'intermediate', 'google_map', 'evaluation', 'user')
INPUT_DIR = os.path.join(BASE_DIR, 'input')

class UserNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout):
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
        no_biased_wd = torch.einsum('ijk, kt->ijt', torch.softmax(mean, dim=-1), torch.softmax(self.beta, dim=-1))
        return mean, log_var, word_distribution, no_biased_wd

    def kl_loss(self, mean, log_var):
        var = torch.exp(log_var)
        term1 = log_var
        term2 = -mean.pow(2)
        term3 = -var
        return -0.5 * (1 + term1 + term2 + term3).sum()

    def log_likelihood_loss(self, bag_of_item, wd):
        loss = bag_of_item * torch.log(wd)
        return -loss.sum()

class pLSAUserNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.beta = nn.Parameter(torch.randn((latent_dim, input_dim)), requires_grad=True)

    def forward(self, bag_of_item):
        H = torch.relu(self.fc(bag_of_item))
        theta = torch.softmax(self.fc_latent(H), dim=-1)
        word_distribution = torch.einsum('ijk, kt->ijt', theta, torch.softmax(self.beta, dim=-1))
        return word_distribution

    def log_likelihood_loss(self, bag_of_item, wd):
        loss = bag_of_item * torch.log(wd)
        return -loss.sum()

class DMMNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.phi = nn.Parameter(torch.randn((latent_dim, input_dim)), requires_grad=True)

    def forward(self, Z):
        word_distribution = torch.einsum('ijk, kt->ijt', Z, torch.softmax(self.phi, dim=-1))
        return word_distribution

    def log_likelihood_loss(self, X, wd):
        loss = X * torch.log(wd)
        return -loss.sum()

class UserEvaluator:
    def __init__(self):
        self.user_train = torch.load(os.path.join(INPUT_DIR, 'user_train.pt'))
        self.user_test = torch.load(os.path.join(INPUT_DIR, 'user_test.pt'))

        self.input_dim = self.user_train.shape[2]
        self.hidden_dim = 512
        self.latent_dim = 4
        self.dropout = 0.2
        self.batch_size = 512

    def print_metrics(self, name, perplexity):
        print(f"\n========== {name} ==========")
        print(f"  Test Perplexity: {perplexity:.4f}")

    def _run_our_model_single_with_net(self, verbose=False):
        net = UserNetwork(self.input_dim, self.hidden_dim, self.latent_dim, self.dropout)
        train_loader = DataLoader(self.user_train, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.user_test, batch_size=self.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        for epoch in range(400):
            net.train()
            up_train, down_train = 0.0, 0.0
            for batch in train_loader:
                mean, log_var, wd, nwd = net(batch)
                loss = 0.1 * net.kl_loss(mean, log_var) + net.log_likelihood_loss(batch, wd)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 10.0)
                optimizer.step()
                up_train += net.log_likelihood_loss(batch, nwd).item()
                down_train += batch.sum().item()
            if verbose and epoch % 20 == 0:
                train_ppl = np.exp(up_train / down_train)
                print(f"  epoch {epoch}, train ppl = {train_ppl:.2f}")

        net.eval()
        up_test, down_test = 0.0, 0.0
        with torch.no_grad():
            for batch in test_loader:
                _, _, _, nwd = net(batch)
                up_test += net.log_likelihood_loss(batch, nwd).item()
                down_test += batch.sum().item()
        test_ppl = np.exp(up_test / down_test)
        return test_ppl, net

    def _run_plsa_single(self, verbose=False):
        net = pLSAUserNetwork(self.input_dim, self.hidden_dim, self.latent_dim)
        train_loader = DataLoader(self.user_train, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.user_test, batch_size=self.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        for epoch in range(150):
            net.train()
            up_train, down_train = 0.0, 0.0
            for batch in train_loader:
                wd = net(batch)
                loss = net.log_likelihood_loss(batch, wd)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 10.0)
                optimizer.step()
                up_train += loss.item()
                down_train += batch.sum().item()
            if verbose and epoch % 20 == 0:
                train_ppl = np.exp(up_train / down_train)
                print(f"  epoch {epoch}, train ppl = {train_ppl:.2f}")

        net.eval()
        up_test, down_test = 0.0, 0.0
        with torch.no_grad():
            for batch in test_loader:
                wd = net(batch)
                up_test += net.log_likelihood_loss(batch, wd).item()
                down_test += batch.sum().item()
        test_ppl = np.exp(up_test / down_test)
        return test_ppl

    def _run_dmm_single(self, our_net, verbose=False):
        def generate_Z(net, data):
            loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
            Z_list = []
            net.eval()
            with torch.no_grad():
                for batch in loader:
                    prop, _, _, _ = net(batch)
                    max_idx = torch.argmax(prop, dim=-1)
                    Z = nn.functional.one_hot(max_idx, num_classes=self.latent_dim).float()
                    Z_list.append(Z)
            return torch.cat(Z_list, dim=0)

        Z_train = generate_Z(our_net, self.user_train)
        Z_test = generate_Z(our_net, self.user_test)

        dmm_net = DMMNetwork(self.input_dim, self.latent_dim)
        train_dataset = TensorDataset(self.user_train, Z_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataset = TensorDataset(self.user_test, Z_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(dmm_net.parameters(), lr=0.005)

        for epoch in range(200):
            dmm_net.train()
            up_train, down_train = 0.0, 0.0
            for X, Z in train_loader:
                wd = dmm_net(Z)
                loss = dmm_net.log_likelihood_loss(X, wd)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(dmm_net.parameters(), 10.0)
                optimizer.step()
                up_train += loss.item()
                down_train += X.sum().item()
            if verbose and epoch % 20 == 0:
                train_ppl = np.exp(up_train / down_train)
                print(f"  epoch {epoch}, train ppl = {train_ppl:.2f}")

        dmm_net.eval()
        up_test, down_test = 0.0, 0.0
        with torch.no_grad():
            for X, Z in test_loader:
                wd = dmm_net(Z)
                up_test += dmm_net.log_likelihood_loss(X, wd).item()
                down_test += X.sum().item()
        test_ppl = np.exp(up_test / down_test)
        return test_ppl

    def run_all(self, n_runs=20):
        start_time = time.time()
        print("Current time:", datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'))
        print(f"Running each model {n_runs} times and averaging results...\n")

        our_ppls = []
        plsa_ppls = []
        dmm_ppls = []

        for run in range(n_runs):
            if run % 5 == 0:
                print(f"Run {run+1}/{n_runs}")

            our_ppl, our_net = self._run_our_model_single_with_net(verbose=False)
            our_ppls.append(our_ppl)

            plsa_ppl = self._run_plsa_single(verbose=False)
            plsa_ppls.append(plsa_ppl)

            dmm_ppl = self._run_dmm_single(our_net, verbose=False)
            dmm_ppls.append(dmm_ppl)

        avg_our = np.mean(our_ppls)
        avg_plsa = np.mean(plsa_ppls)
        avg_dmm = np.mean(dmm_ppls)

        self.print_metrics("Our Model (average)", avg_our)
        self.print_metrics("pLSA (average)", avg_plsa)
        self.print_metrics("DMM (average)", avg_dmm)

        print("\nAll models finished.")

if __name__ == '__main__':
    evaluator = UserEvaluator()
    evaluator.run_all(n_runs=20)