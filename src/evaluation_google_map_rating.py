import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import cvxpy as cp
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

BASE_DIR = os.path.join('..', 'intermediate', 'google_map', 'evaluation', 'rating')
INPUT_DIR = os.path.join(BASE_DIR, 'input')
ORIGINAL_DIR = os.path.join(BASE_DIR, 'original')

try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

class CompleteUserNetwork(nn.Module):
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

class CompleteRatingNetwork(nn.Module):
    def __init__(self, num_user, latent_dim, character_dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn((num_user, latent_dim, character_dim)), requires_grad=True)

    def forward(self, user_idx, motivation_proportion, user_purchase, beta, character_vector):
        likelihood = beta.T[user_purchase]
        posterior = motivation_proportion.unsqueeze(-2) * likelihood
        norm_posterior = posterior / torch.sum(posterior, dim=-1, keepdim=True)
        u = self.u[user_idx]
        weighted_u = torch.einsum('ijkp, ipl->ijkl', norm_posterior, u)
        select_character = character_vector[user_purchase]
        inferred_rating = torch.sum(weighted_u * select_character, dim=-1)
        return inferred_rating

    def rating_mask_loss(self, user_rating, inferred_rating, valid_len):
        mask = torch.arange(user_rating.shape[-1])[None, None, :] <= valid_len[:, :, None] - 1
        diff = torch.square(user_rating - inferred_rating)
        return (diff * mask.float()).sum()

    def norm_loss(self, user_idx):
        return torch.abs(self.u[user_idx]).sum()

class Model1UserNetwork(nn.Module):
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
        H = self.dropout(torch.relu(self.bn(self.fc(bag_of_item))))
        mean = self.fc_mean(H)
        log_var = self.fc_log_var(H)
        theta = self.reparameterize(mean, log_var)
        word_distribution = torch.mm(theta, torch.softmax(self.beta, dim=-1))
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

class Model1RatingNetwork(nn.Module):
    def __init__(self, num_user, latent_dim, character_dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn((num_user, latent_dim, character_dim)), requires_grad=True)

    def forward(self, user_idx, motivation_proportion, user_purchase, beta, character_vector):
        likelihood = beta.T[user_purchase]
        posterior = motivation_proportion.unsqueeze(-2) * likelihood
        norm_posterior = posterior / torch.sum(posterior, dim=-1, keepdim=True)
        u = self.u[user_idx]
        weighted_u = torch.einsum('ikp, ipl->ikl', norm_posterior, u)
        select_character = character_vector[user_purchase]
        inferred_rating = torch.sum(weighted_u * select_character, dim=-1)
        return inferred_rating

    def rating_mask_loss(self, user_rating, inferred_rating, valid_len):
        mask = torch.arange(user_rating.shape[-1])[None, :] <= valid_len[:, None] - 1
        diff = torch.square(user_rating - inferred_rating)
        return (diff * mask.float()).sum()

    def norm_loss(self, user_idx):
        return torch.abs(self.u[user_idx]).sum()

class Model2RatingNetwork(nn.Module):
    def __init__(self, num_user, character_dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn((num_user, character_dim)), requires_grad=True)

    def forward(self, user_idx, user_purchase, character_vector):
        u = self.u[user_idx].unsqueeze(-2)
        select_character = character_vector[user_purchase]
        inferred_rating = torch.sum(u * select_character, dim=-1)
        return inferred_rating

    def rating_mask_loss(self, user_rating, inferred_rating, valid_len):
        mask = torch.arange(user_rating.shape[-1])[None, :] <= valid_len[:, None] - 1
        diff = torch.square(user_rating - inferred_rating)
        return (diff * mask.float()).sum()

    def norm_loss(self, user_idx):
        return torch.abs(self.u[user_idx]).sum()

class PMFRatingNetwork(nn.Module):
    def __init__(self, num_user, num_item, character_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_user, character_dim)
        self.item_embedding = nn.Embedding(num_item, character_dim)

    def forward(self, behavior):
        user_vector = self.user_embedding(behavior[:, 0].to(torch.long))
        item_vector = self.item_embedding(behavior[:, 1].to(torch.long))
        inferred_rating = torch.sum(user_vector * item_vector, dim=-1)
        return inferred_rating

    def rating_loss(self, behavior, inferred_rating):
        diff = torch.square(behavior[:, -1] - inferred_rating)
        return diff.sum()

class ModelEvaluator:
    def __init__(self):
        self.character_vector = torch.load(os.path.join('..', 'intermediate', 'google_map', 'character_vector.pt'))

    def print_metrics(self, name, train_mae, train_rmse, train_num, test_mae, test_rmse, test_num):
        print(f"\n========== {name} ==========")
        print(f"Train set: {train_num} behaviors")
        print(f"  MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Test set: {test_num} behaviors")
        print(f"  MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

    def _run_complete_model_single(self, verbose=False):
        user_tensor = torch.load(os.path.join(INPUT_DIR, 'complete_model_train_user_tensor.pt'))
        train_purchase = torch.load(os.path.join(INPUT_DIR, 'complete_model_train_user_purchase_tensor.pt')).long()
        train_rating = torch.load(os.path.join(INPUT_DIR, 'complete_model_train_rating_tensor.pt'))
        train_valid_len = torch.load(os.path.join(INPUT_DIR, 'complete_model_train_valid_len_tensor.pt')).long()
        test_purchase = torch.load(os.path.join(INPUT_DIR, 'complete_model_test_user_purchase_tensor.pt')).long()
        test_rating = torch.load(os.path.join(INPUT_DIR, 'complete_model_test_rating_tensor.pt'))
        test_valid_len = torch.load(os.path.join(INPUT_DIR, 'complete_model_test_valid_len_tensor.pt')).long()

        num_user = user_tensor.shape[0]
        num_item = user_tensor.shape[2]
        latent_dim = 4
        hidden_dim = 512
        character_dim = 9
        dropout = 0.2

        user_net = CompleteUserNetwork(num_item, hidden_dim, latent_dim, dropout)
        optimizer = torch.optim.Adam(user_net.parameters(), lr=0.005)
        dataloader = DataLoader(user_tensor, batch_size=512, shuffle=False)
        for epoch in range(200):
            total_loss = 0
            for batch in dataloader:
                mean, log_var, wd = user_net(batch)
                loss = user_net.log_likelihood_loss(batch, wd) + user_net.kl_loss(mean, log_var)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(user_net.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch}, loss = {total_loss:.2f}")

        beta = torch.softmax(user_net.beta, dim=-1).detach()
        user_motivation = []
        dataloader = DataLoader(user_tensor, batch_size=512, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                mean, _, _ = user_net(batch)
                user_motivation.append(torch.softmax(mean, dim=-1))
        user_motivation = torch.cat(user_motivation, dim=0)

        rating_net = CompleteRatingNetwork(num_user, latent_dim, character_dim)
        optimizer = torch.optim.Adam(rating_net.parameters(), lr=0.03)
        idx_tensor = torch.arange(num_user)
        train_dataset = TensorDataset(idx_tensor, user_motivation, train_purchase, train_rating, train_valid_len)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
        test_dataset = TensorDataset(idx_tensor, user_motivation, test_purchase, test_rating, test_valid_len)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        for epoch in range(140):
            total_loss = 0
            rating_net.train()
            for idx, mot, pur, rat, leng in train_loader:
                inferred = rating_net(idx, mot, pur, beta, self.character_vector)
                loss = rating_net.rating_mask_loss(rat, inferred, leng) + 0.01 * rating_net.norm_loss(idx)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(rating_net.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch}, loss = {total_loss:.2f}")

        rating_net.eval()
        train_mae, train_mse, train_cnt = 0, 0, 0
        test_mae, test_mse, test_cnt = 0, 0, 0
        with torch.no_grad():
            for idx, mot, pur, rat, leng in train_loader:
                inferred = rating_net(idx, mot, pur, beta, self.character_vector)
                mask = torch.arange(rat.shape[-1])[None, None, :] <= leng[:, :, None] - 1
                mae = (torch.abs(rat - inferred) * mask.float()).sum()
                mse = (torch.square(rat - inferred) * mask.float()).sum()
                cnt = mask.float().sum()
                train_mae += mae.item()
                train_mse += mse.item()
                train_cnt += cnt.item()
            for idx, mot, pur, rat, leng in test_loader:
                inferred = rating_net(idx, mot, pur, beta, self.character_vector)
                mask = torch.arange(rat.shape[-1])[None, None, :] <= leng[:, :, None] - 1
                mae = (torch.abs(rat - inferred) * mask.float()).sum()
                mse = (torch.square(rat - inferred) * mask.float()).sum()
                cnt = mask.float().sum()
                test_mae += mae.item()
                test_mse += mse.item()
                test_cnt += cnt.item()

        train_mae_avg = train_mae / train_cnt
        train_rmse = np.sqrt(train_mse / train_cnt)
        test_mae_avg = test_mae / test_cnt
        test_rmse = np.sqrt(test_mse / test_cnt)

        return {
            'train_mae': train_mae_avg,
            'train_rmse': train_rmse,
            'train_cnt': train_cnt,
            'test_mae': test_mae_avg,
            'test_rmse': test_rmse,
            'test_cnt': test_cnt
        }

    def _run_model1_single(self, verbose=False):
        user_tensor = torch.load(os.path.join(INPUT_DIR, 'model1_train_user_tensor.pt'))
        train_purchase = torch.load(os.path.join(INPUT_DIR, 'model1_train_user_purchase_tensor.pt')).long()
        train_rating = torch.load(os.path.join(INPUT_DIR, 'model1_train_rating_tensor.pt'))
        train_valid_len = torch.load(os.path.join(INPUT_DIR, 'model1_train_valid_len_tensor.pt')).long()
        test_purchase = torch.load(os.path.join(INPUT_DIR, 'model1_test_user_purchase_tensor.pt')).long()
        test_rating = torch.load(os.path.join(INPUT_DIR, 'model1_test_rating_tensor.pt'))
        test_valid_len = torch.load(os.path.join(INPUT_DIR, 'model1_test_valid_len_tensor.pt')).long()

        num_user = user_tensor.shape[0]
        num_item = user_tensor.shape[1]
        latent_dim = 4
        hidden_dim = 512
        character_dim = 9
        dropout = 0.2

        user_net = Model1UserNetwork(num_item, hidden_dim, latent_dim, dropout)
        optimizer = torch.optim.Adam(user_net.parameters(), lr=0.005)
        dataloader = DataLoader(user_tensor, batch_size=512, shuffle=False)
        for epoch in range(200):
            total_loss = 0
            for batch in dataloader:
                mean, log_var, wd = user_net(batch)
                loss = user_net.log_likelihood_loss(batch, wd) + user_net.kl_loss(mean, log_var)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(user_net.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch}, loss = {total_loss:.2f}")

        beta = torch.softmax(user_net.beta, dim=-1).detach()
        user_motivation = []
        dataloader = DataLoader(user_tensor, batch_size=512, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                mean, _, _ = user_net(batch)
                user_motivation.append(torch.softmax(mean, dim=-1))
        user_motivation = torch.cat(user_motivation, dim=0)

        rating_net = Model1RatingNetwork(num_user, latent_dim, character_dim)
        optimizer = torch.optim.Adam(rating_net.parameters(), lr=0.03)
        idx_tensor = torch.arange(num_user)
        train_dataset = TensorDataset(idx_tensor, user_motivation, train_purchase, train_rating, train_valid_len)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
        test_dataset = TensorDataset(idx_tensor, user_motivation, test_purchase, test_rating, test_valid_len)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        for epoch in range(110):
            total_loss = 0
            rating_net.train()
            for idx, mot, pur, rat, leng in train_loader:
                inferred = rating_net(idx, mot, pur, beta, self.character_vector)
                loss = rating_net.rating_mask_loss(rat, inferred, leng) + 0.01 * rating_net.norm_loss(idx)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(rating_net.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch}, loss = {total_loss:.2f}")

        rating_net.eval()
        train_mae, train_mse, train_cnt = 0, 0, 0
        test_mae, test_mse, test_cnt = 0, 0, 0
        with torch.no_grad():
            for idx, mot, pur, rat, leng in train_loader:
                inferred = rating_net(idx, mot, pur, beta, self.character_vector)
                mask = torch.arange(rat.shape[-1])[None, :] <= leng[:, None] - 1
                mae = (torch.abs(rat - inferred) * mask.float()).sum()
                mse = (torch.square(rat - inferred) * mask.float()).sum()
                cnt = mask.float().sum()
                train_mae += mae.item()
                train_mse += mse.item()
                train_cnt += cnt.item()
            for idx, mot, pur, rat, leng in test_loader:
                inferred = rating_net(idx, mot, pur, beta, self.character_vector)
                mask = torch.arange(rat.shape[-1])[None, :] <= leng[:, None] - 1
                mae = (torch.abs(rat - inferred) * mask.float()).sum()
                mse = (torch.square(rat - inferred) * mask.float()).sum()
                cnt = mask.float().sum()
                test_mae += mae.item()
                test_mse += mse.item()
                test_cnt += cnt.item()

        train_mae_avg = train_mae / train_cnt
        train_rmse = np.sqrt(train_mse / train_cnt)
        test_mae_avg = test_mae / test_cnt
        test_rmse = np.sqrt(test_mse / test_cnt)

        return {
            'train_mae': train_mae_avg,
            'train_rmse': train_rmse,
            'train_cnt': train_cnt,
            'test_mae': test_mae_avg,
            'test_rmse': test_rmse,
            'test_cnt': test_cnt
        }

    def _run_model2_single(self, verbose=False):
        train_purchase = torch.load(os.path.join(INPUT_DIR, 'model2_train_user_purchase_tensor.pt')).long()
        train_rating = torch.load(os.path.join(INPUT_DIR, 'model2_train_rating_tensor.pt'))
        train_valid_len = torch.load(os.path.join(INPUT_DIR, 'model2_train_valid_len_tensor.pt')).long()
        test_purchase = torch.load(os.path.join(INPUT_DIR, 'model2_test_user_purchase_tensor.pt')).long()
        test_rating = torch.load(os.path.join(INPUT_DIR, 'model2_test_rating_tensor.pt'))
        test_valid_len = torch.load(os.path.join(INPUT_DIR, 'model2_test_valid_len_tensor.pt')).long()

        num_user = train_purchase.shape[0]
        character_dim = 9

        rating_net = Model2RatingNetwork(num_user, character_dim)
        optimizer = torch.optim.Adam(rating_net.parameters(), lr=0.019)
        idx_tensor = torch.arange(num_user)
        train_dataset = TensorDataset(idx_tensor, train_purchase, train_rating, train_valid_len)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
        test_dataset = TensorDataset(idx_tensor, test_purchase, test_rating, test_valid_len)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        for epoch in range(130):
            total_loss = 0
            rating_net.train()
            for idx, pur, rat, leng in train_loader:
                inferred = rating_net(idx, pur, self.character_vector)
                loss = rating_net.rating_mask_loss(rat, inferred, leng) + 0.01 * rating_net.norm_loss(idx)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(rating_net.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch}, loss = {total_loss:.2f}")

        rating_net.eval()
        train_mae, train_mse, train_cnt = 0, 0, 0
        test_mae, test_mse, test_cnt = 0, 0, 0
        with torch.no_grad():
            for idx, pur, rat, leng in train_loader:
                inferred = rating_net(idx, pur, self.character_vector)
                mask = torch.arange(rat.shape[-1])[None, :] <= leng[:, None] - 1
                mae = (torch.abs(rat - inferred) * mask.float()).sum()
                mse = (torch.square(rat - inferred) * mask.float()).sum()
                cnt = mask.float().sum()
                train_mae += mae.item()
                train_mse += mse.item()
                train_cnt += cnt.item()
            for idx, pur, rat, leng in test_loader:
                inferred = rating_net(idx, pur, self.character_vector)
                mask = torch.arange(rat.shape[-1])[None, :] <= leng[:, None] - 1
                mae = (torch.abs(rat - inferred) * mask.float()).sum()
                mse = (torch.square(rat - inferred) * mask.float()).sum()
                cnt = mask.float().sum()
                test_mae += mae.item()
                test_mse += mse.item()
                test_cnt += cnt.item()

        train_mae_avg = train_mae / train_cnt
        train_rmse = np.sqrt(train_mse / train_cnt)
        test_mae_avg = test_mae / test_cnt
        test_rmse = np.sqrt(test_mse / test_cnt)

        return {
            'train_mae': train_mae_avg,
            'train_rmse': train_rmse,
            'train_cnt': train_cnt,
            'test_mae': test_mae_avg,
            'test_rmse': test_rmse,
            'test_cnt': test_cnt
        }

    def _run_pmf_single(self, verbose=False):
        train_behavior = torch.load(os.path.join(INPUT_DIR, 'pmf_train_rating.pt'))
        test_behavior = torch.load(os.path.join(INPUT_DIR, 'pmf_test_rating.pt'))

        num_user = int(train_behavior[:, 0].max().item()) + 1
        num_item = int(train_behavior[:, 1].max().item()) + 1
        character_dim = 9

        net = PMFRatingNetwork(num_user, num_item, character_dim)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        train_loader = DataLoader(train_behavior, batch_size=1024, shuffle=False)
        test_loader = DataLoader(test_behavior, batch_size=1024, shuffle=False)
        for epoch in range(150):
            total_loss = 0
            net.train()
            for batch in train_loader:
                inferred = net(batch)
                loss = net.rating_loss(batch, inferred)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()
            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch}, loss = {total_loss:.2f}")

        net.eval()
        train_mae, train_mse, train_cnt = 0, 0, 0
        test_mae, test_mse, test_cnt = 0, 0, 0
        with torch.no_grad():
            for batch in train_loader:
                inferred = net(batch)
                mae = torch.abs(batch[:, -1] - inferred).sum().item()
                mse = torch.square(batch[:, -1] - inferred).sum().item()
                cnt = batch.shape[0]
                train_mae += mae
                train_mse += mse
                train_cnt += cnt
            for batch in test_loader:
                inferred = net(batch)
                mae = torch.abs(batch[:, -1] - inferred).sum().item()
                mse = torch.square(batch[:, -1] - inferred).sum().item()
                cnt = batch.shape[0]
                test_mae += mae
                test_mse += mse
                test_cnt += cnt

        train_mae_avg = train_mae / train_cnt
        train_rmse = np.sqrt(train_mse / train_cnt)
        test_mae_avg = test_mae / test_cnt
        test_rmse = np.sqrt(test_mse / test_cnt)

        return {
            'train_mae': train_mae_avg,
            'train_rmse': train_rmse,
            'train_cnt': train_cnt,
            'test_mae': test_mae_avg,
            'test_rmse': test_rmse,
            'test_cnt': test_cnt
        }

    def _run_utadis_single(self, verbose=False):
        with open(os.path.join(INPUT_DIR, 'utadis_train_input_coefficient.json'), 'r') as f:
            train_dict = json.load(f)
        with open(os.path.join(INPUT_DIR, 'utadis_test_input_coefficient.json'), 'r') as f:
            test_dict = json.load(f)

        num_user = 4936
        character_dim = 9

        train_mae_sum, train_mse_sum, train_behavior = 0, 0, 0
        test_mae_sum, test_mse_sum, test_behavior = 0, 0, 0

        for i in range(num_user):
            str_i = str(i)
            if str_i not in train_dict:
                continue
            temp_train = train_dict[str_i]
            num_behavior = len(temp_train)
            u = cp.Variable(character_dim, nonneg=True)
            bias = cp.Variable((num_behavior, 2), nonneg=True)
            constraints = []
            for j in range(num_behavior):
                constraints.append(u @ temp_train[j][:-1] + bias[j][0] - bias[j][1] == temp_train[j][-1])
            objective = cp.Minimize(cp.sum(bias))
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=False)

            if u.value is None:
                u_opt = np.zeros(character_dim)
            else:
                u_opt = u.value

            cor = np.array(temp_train)[:, :-1]
            inferred = cor @ u_opt
            true = np.array(temp_train)[:, -1]
            train_mae_sum += np.abs(inferred - true).sum()
            train_mse_sum += np.square(inferred - true).sum()
            train_behavior += num_behavior

            if str_i in test_dict:
                temp_test = test_dict[str_i]
                num_test = len(temp_test)
                cor_test = np.array(temp_test)[:, :-1]
                inferred_test = cor_test @ u_opt
                true_test = np.array(temp_test)[:, -1]
                test_mae_sum += np.abs(inferred_test - true_test).sum()
                test_mse_sum += np.square(inferred_test - true_test).sum()
                test_behavior += num_test

        train_mae = train_mae_sum / train_behavior
        train_rmse = np.sqrt(train_mse_sum / train_behavior)
        test_mae = test_mae_sum / test_behavior if test_behavior > 0 else 0
        test_rmse = np.sqrt(test_mse_sum / test_behavior) if test_behavior > 0 else 0

        return {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_cnt': train_behavior,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_cnt': test_behavior
        }

    def _run_nsa_single(self, verbose=False):
        train_df = pd.read_csv(os.path.join(ORIGINAL_DIR, 'train.csv'))
        test_df = pd.read_csv(os.path.join(ORIGINAL_DIR, 'test.csv'))

        def compute_metrics(df):
            text_list = df['text'].astype(str).tolist()
            rating_list = df['rating'].tolist()
            mae, mse = 0, 0
            for i in range(len(text_list)):
                review = text_list[i]
                rating = rating_list[i]
                score = 1 + (sia.polarity_scores(review)['compound'] + 1) * 2
                mae += abs(score - rating)
                mse += (score - rating) ** 2
            cnt = len(text_list)
            return mae, mse, cnt

        train_mae, train_mse, train_cnt = compute_metrics(train_df)
        test_mae, test_mse, test_cnt = compute_metrics(test_df)

        train_mae_avg = train_mae / train_cnt
        train_rmse = np.sqrt(train_mse / train_cnt)
        test_mae_avg = test_mae / test_cnt
        test_rmse = np.sqrt(test_mse / test_cnt)

        return {
            'train_mae': train_mae_avg,
            'train_rmse': train_rmse,
            'train_cnt': train_cnt,
            'test_mae': test_mae_avg,
            'test_rmse': test_rmse,
            'test_cnt': test_cnt
        }

    def run_all(self, n_runs=20):
        start_time = time.time()
        print("Current time:", datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'))
        print(f"Running each model {n_runs} times and averaging results...\n")

        models = [
            ("Complete Model", self._run_complete_model_single),
            ("Model 1", self._run_model1_single),
            ("Model 2", self._run_model2_single),
            ("PMF", self._run_pmf_single),
            ("UTADIS", self._run_utadis_single),
            ("NSA", self._run_nsa_single)
        ]

        for name, func in models:
            print(f"Processing {name}...")
            agg = {
                'train_mae': 0.0,
                'train_rmse': 0.0,
                'train_cnt': 0,
                'test_mae': 0.0,
                'test_rmse': 0.0,
                'test_cnt': 0
            }
            for run in range(n_runs):
                if run % 5 == 0:
                    print(f"  Run {run+1}/{n_runs}")
                res = func(verbose=False)
                agg['train_mae'] += res['train_mae']
                agg['train_rmse'] += res['train_rmse']
                agg['train_cnt'] += res['train_cnt']
                agg['test_mae'] += res['test_mae']
                agg['test_rmse'] += res['test_rmse']
                agg['test_cnt'] += res['test_cnt']

            avg_train_mae = agg['train_mae'] / n_runs
            avg_train_rmse = agg['train_rmse'] / n_runs
            avg_test_mae = agg['test_mae'] / n_runs
            avg_test_rmse = agg['test_rmse'] / n_runs
            avg_train_cnt = agg['train_cnt'] / n_runs
            avg_test_cnt = agg['test_cnt'] / n_runs

            self.print_metrics(f"{name} (average over {n_runs} runs)",
                               avg_train_mae, avg_train_rmse, avg_train_cnt,
                               avg_test_mae, avg_test_rmse, avg_test_cnt)

        print("\nAll models finished.")

if __name__ == '__main__':
    evaluator = ModelEvaluator()
    evaluator.run_all(n_runs=20)