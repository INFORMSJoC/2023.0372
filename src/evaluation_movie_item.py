import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.join('..', 'intermediate', 'movie', 'evaluation', 'item')
INPUT_DIR = os.path.join(BASE_DIR, 'input')

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

class pLSANetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc_fop = nn.Linear(input_dim, hidden_dim)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.phi = nn.Parameter(torch.randn((latent_dim, input_dim)), requires_grad=True)

    def forward(self, bag_of_FOP):
        H = torch.relu(self.fc_fop(bag_of_FOP))
        gamma = torch.softmax(self.fc_latent(H), dim=-1)
        word_distribution = torch.mm(gamma, torch.softmax(self.phi, dim=-1))
        return word_distribution

    def log_likelihood_loss(self, X, wd):
        loss = X * torch.log(wd)
        return -loss.sum()

class DMMNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.phi = nn.Parameter(torch.randn((latent_dim, input_dim)), requires_grad=True)

    def forward(self, Z):
        word_distribution = torch.mm(Z, torch.softmax(self.phi, dim=-1))
        return word_distribution

    def log_likelihood_loss(self, X, wd):
        loss = X * torch.log(wd)
        return -loss.sum()

class ItemEvaluator:
    def __init__(self):
        self.pos_train = torch.load(os.path.join(INPUT_DIR, 'pos_train.pt'))
        self.pos_test = torch.load(os.path.join(INPUT_DIR, 'pos_test.pt'))
        self.neg_train = torch.load(os.path.join(INPUT_DIR, 'neg_train.pt'))
        self.neg_test = torch.load(os.path.join(INPUT_DIR, 'neg_test.pt'))

        self.pos_input_dim = self.pos_train.shape[1]
        self.neg_input_dim = self.neg_train.shape[1]

        self.hidden_dim = 512
        self.pos_latent_dim = 8
        self.neg_latent_dim = 5
        self.dropout = 0.2

        self.batch_size = 512

    def print_metrics(self, name, perplexity):
        print(f"\n========== {name} ==========")
        print(f"  Test Perplexity: {perplexity:.4f}")

    def _run_our_model_single_with_net(self, verbose=False):
        pos_net = FOPNetwork(self.pos_input_dim, self.hidden_dim, self.pos_latent_dim, self.dropout)
        neg_net = FOPNetwork(self.neg_input_dim, self.hidden_dim, self.neg_latent_dim, self.dropout)

        train_dataset = TensorDataset(self.pos_train, self.neg_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataset = TensorDataset(self.pos_test, self.neg_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        optim_pos = torch.optim.Adam(pos_net.parameters(), lr=0.001)
        optim_neg = torch.optim.Adam(neg_net.parameters(), lr=0.001)

        for epoch in range(700):
            pos_net.train()
            neg_net.train()
            for pos_batch, neg_batch in train_loader:
                mean_p, log_var_p, wd_p = pos_net(pos_batch)
                loss_p = pos_net.log_likelihood_loss(pos_batch, wd_p) + pos_net.kl_loss(mean_p, log_var_p)

                mean_n, log_var_n, wd_n = neg_net(neg_batch)
                loss_n = neg_net.log_likelihood_loss(neg_batch, wd_n) + neg_net.kl_loss(mean_n, log_var_n)

                optim_pos.zero_grad()
                loss_p.backward()
                nn.utils.clip_grad_value_(pos_net.parameters(), 10.0)
                optim_pos.step()

                optim_neg.zero_grad()
                loss_n.backward()
                nn.utils.clip_grad_value_(neg_net.parameters(), 10.0)
                optim_neg.step()

            if verbose and epoch % 100 == 0:
                pos_net.eval()
                neg_net.eval()
                up, down = 0.0, 0.0
                with torch.no_grad():
                    for pos_batch, neg_batch in train_loader:
                        _, _, wd_p = pos_net(pos_batch)
                        _, _, wd_n = neg_net(neg_batch)
                        up += (pos_net.log_likelihood_loss(pos_batch, wd_p) + neg_net.log_likelihood_loss(neg_batch, wd_n)).item()
                        down += (pos_batch.sum() + neg_batch.sum()).item()
                train_ppl = np.exp(up / down)
                print(f"  epoch {epoch}, train ppl = {train_ppl:.2f}")
                pos_net.train()
                neg_net.train()

        pos_net.eval()
        neg_net.eval()
        up_total, down_total = 0.0, 0.0
        with torch.no_grad():
            for pos_batch, neg_batch in test_loader:
                _, _, wd_p = pos_net(pos_batch)
                _, _, wd_n = neg_net(neg_batch)
                up_total += (pos_net.log_likelihood_loss(pos_batch, wd_p) + neg_net.log_likelihood_loss(neg_batch, wd_n)).item()
                down_total += (pos_batch.sum() + neg_batch.sum()).item()
        test_ppl = np.exp(up_total / down_total)
        return test_ppl, pos_net, neg_net

    def _run_plsa_single(self, verbose=False):
        pos_net = pLSANetwork(self.pos_input_dim, self.hidden_dim, self.pos_latent_dim)
        neg_net = pLSANetwork(self.neg_input_dim, self.hidden_dim, self.neg_latent_dim)

        train_dataset = TensorDataset(self.pos_train, self.neg_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataset = TensorDataset(self.pos_test, self.neg_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        optim_pos = torch.optim.Adam(pos_net.parameters(), lr=0.001)
        optim_neg = torch.optim.Adam(neg_net.parameters(), lr=0.001)

        for epoch in range(600):
            pos_net.train()
            neg_net.train()
            for pos_batch, neg_batch in train_loader:
                wd_p = pos_net(pos_batch)
                loss_p = pos_net.log_likelihood_loss(pos_batch, wd_p)

                wd_n = neg_net(neg_batch)
                loss_n = neg_net.log_likelihood_loss(neg_batch, wd_n)

                optim_pos.zero_grad()
                loss_p.backward()
                nn.utils.clip_grad_value_(pos_net.parameters(), 10.0)
                optim_pos.step()

                optim_neg.zero_grad()
                loss_n.backward()
                nn.utils.clip_grad_value_(neg_net.parameters(), 10.0)
                optim_neg.step()

            if verbose and epoch % 100 == 0:
                pos_net.eval()
                neg_net.eval()
                up, down = 0.0, 0.0
                with torch.no_grad():
                    for pos_batch, neg_batch in train_loader:
                        wd_p = pos_net(pos_batch)
                        wd_n = neg_net(neg_batch)
                        up += (pos_net.log_likelihood_loss(pos_batch, wd_p) + neg_net.log_likelihood_loss(neg_batch, wd_n)).item()
                        down += (pos_batch.sum() + neg_batch.sum()).item()
                train_ppl = np.exp(up / down)
                print(f"  epoch {epoch}, train ppl = {train_ppl:.2f}")
                pos_net.train()
                neg_net.train()

        pos_net.eval()
        neg_net.eval()
        up_total, down_total = 0.0, 0.0
        with torch.no_grad():
            for pos_batch, neg_batch in test_loader:
                wd_p = pos_net(pos_batch)
                wd_n = neg_net(neg_batch)
                up_total += (pos_net.log_likelihood_loss(pos_batch, wd_p) + neg_net.log_likelihood_loss(neg_batch, wd_n)).item()
                down_total += (pos_batch.sum() + neg_batch.sum()).item()
        test_ppl = np.exp(up_total / down_total)
        return test_ppl

    def _run_dmm_single(self, our_pos_net, our_neg_net, verbose=False):
        def generate_Z(net, data, latent_dim):
            loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
            Z_list = []
            net.eval()
            with torch.no_grad():
                for batch in loader:
                    prop, _ = net.evaluate(batch)
                    max_idx = torch.argmax(prop, dim=1)
                    Z = nn.functional.one_hot(max_idx, num_classes=latent_dim).float()
                    Z_list.append(Z)
            return torch.cat(Z_list, dim=0)

        Z_pos_train = generate_Z(our_pos_net, self.pos_train, self.pos_latent_dim)
        Z_pos_test = generate_Z(our_pos_net, self.pos_test, self.pos_latent_dim)
        Z_neg_train = generate_Z(our_neg_net, self.neg_train, self.neg_latent_dim)
        Z_neg_test = generate_Z(our_neg_net, self.neg_test, self.neg_latent_dim)

        pos_net = DMMNetwork(self.pos_input_dim, self.pos_latent_dim)
        neg_net = DMMNetwork(self.neg_input_dim, self.neg_latent_dim)

        train_dataset = TensorDataset(self.pos_train, self.neg_train, Z_pos_train, Z_neg_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataset = TensorDataset(self.pos_test, self.neg_test, Z_pos_test, Z_neg_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        optim_pos = torch.optim.Adam(pos_net.parameters(), lr=0.003)
        optim_neg = torch.optim.Adam(neg_net.parameters(), lr=0.003)

        for epoch in range(180):
            pos_net.train()
            neg_net.train()
            for Xp, Xn, Zp, Zn in train_loader:
                wd_p = pos_net(Zp)
                loss_p = pos_net.log_likelihood_loss(Xp, wd_p)

                wd_n = neg_net(Zn)
                loss_n = neg_net.log_likelihood_loss(Xn, wd_n)

                optim_pos.zero_grad()
                loss_p.backward()
                nn.utils.clip_grad_value_(pos_net.parameters(), 10.0)
                optim_pos.step()

                optim_neg.zero_grad()
                loss_n.backward()
                nn.utils.clip_grad_value_(neg_net.parameters(), 10.0)
                optim_neg.step()

            if verbose and epoch % 50 == 0:
                pos_net.eval()
                neg_net.eval()
                up, down = 0.0, 0.0
                with torch.no_grad():
                    for Xp, Xn, Zp, Zn in train_loader:
                        wd_p = pos_net(Zp)
                        wd_n = neg_net(Zn)
                        up += (pos_net.log_likelihood_loss(Xp, wd_p) + neg_net.log_likelihood_loss(Xn, wd_n)).item()
                        down += (Xp.sum() + Xn.sum()).item()
                train_ppl = np.exp(up / down)
                print(f"  epoch {epoch}, train ppl = {train_ppl:.2f}")
                pos_net.train()
                neg_net.train()

        pos_net.eval()
        neg_net.eval()
        up_total, down_total = 0.0, 0.0
        with torch.no_grad():
            for Xp, Xn, Zp, Zn in test_loader:
                wd_p = pos_net(Zp)
                wd_n = neg_net(Zn)
                up_total += (pos_net.log_likelihood_loss(Xp, wd_p) + neg_net.log_likelihood_loss(Xn, wd_n)).item()
                down_total += (Xp.sum() + Xn.sum()).item()
        test_ppl = np.exp(up_total / down_total)
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

            our_ppl, pos_net, neg_net = self._run_our_model_single_with_net(verbose=False)
            our_ppls.append(our_ppl)

            plsa_ppl = self._run_plsa_single(verbose=False)
            plsa_ppls.append(plsa_ppl)

            dmm_ppl = self._run_dmm_single(pos_net, neg_net, verbose=False)
            dmm_ppls.append(dmm_ppl)

        avg_our = np.mean(our_ppls)
        avg_plsa = np.mean(plsa_ppls)
        avg_dmm = np.mean(dmm_ppls)

        self.print_metrics("Our Model (average)", avg_our)
        self.print_metrics("pLSA (average)", avg_plsa)
        self.print_metrics("DMM (average)", avg_dmm)

        print("\nAll models finished.")

if __name__ == '__main__':
    evaluator = ItemEvaluator()
    evaluator.run_all(n_runs=20)