import os
import json
import collections
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

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

class DataPrepare:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data', 'cell_p')
        self.output_dir = os.path.join(self.base_dir, 'intermediate', 'cell_p')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def derive_item_FOP_tensor(self):
        pos_path = os.path.join(self.data_dir, 'pos_FOP_text.json')
        if not os.path.exists(pos_path):
            print(f'File not found: {pos_path}')
            return
        with open(pos_path, 'r') as f:
            pos_dict = json.load(f)
        pos_list = [pos_dict[str(i)] for i in range(len(pos_dict))]
        vocab_pos = Vocab(pos_list, min_freq=3)
        pos_idx_list = vocab_pos[pos_list]
        pos_tensor = torch.zeros((len(pos_list), len(vocab_pos)))
        for i in tqdm(range(len(pos_list)), desc='Processing pos_FOP'):
            temp_idx = torch.tensor(pos_idx_list[i])
            onehot = nn.functional.one_hot(temp_idx, len(vocab_pos))
            pos_tensor[i] = onehot.sum(dim=0)
        torch.save(pos_tensor, os.path.join(self.output_dir, 'pos_tensor.pt'))

        neg_path = os.path.join(self.data_dir, 'neg_FOP_text.json')
        if not os.path.exists(neg_path):
            print(f'File not found: {neg_path}')
            return
        with open(neg_path, 'r') as f:
            neg_dict = json.load(f)
        neg_list = [neg_dict[str(i)] for i in range(len(neg_dict))]
        vocab_neg = Vocab(neg_list, min_freq=3)
        neg_idx_list = vocab_neg[neg_list]
        neg_tensor = torch.zeros((len(neg_list), len(vocab_neg)))
        for i in tqdm(range(len(neg_list)), desc='Processing neg_FOP'):
            temp_idx = torch.tensor(neg_idx_list[i])
            onehot = nn.functional.one_hot(temp_idx, len(vocab_neg))
            neg_tensor[i] = onehot.sum(dim=0)
        torch.save(neg_tensor, os.path.join(self.output_dir, 'neg_tensor.pt'))

    def derive_user_tensor(self):
        user_path = os.path.join(self.data_dir, 'user.csv')
        if not os.path.exists(user_path):
            print(f'File not found: {user_path}')
            return
        df = pd.read_csv(user_path)
        user_ids = df['user_id'].unique()
        time_periods = df['time_period'].unique()
        item_ids = df['item_id'].unique()
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        time_to_idx = {t: i for i, t in enumerate(time_periods)}
        num_users = len(user_ids)
        num_times = len(time_periods)
        num_items = len(item_ids)
        user_tensor = torch.zeros((num_users, num_times, num_items))
        grouped = df.groupby('user_id')
        for user_id, group in tqdm(grouped, desc='Building user_tensor'):
            u_idx = user_to_idx[user_id]
            for time_period, sub_group in group.groupby('time_period'):
                t_idx = time_to_idx[time_period]
                items = torch.tensor(sub_group['item_id'].tolist())
                onehot = nn.functional.one_hot(items, num_items)
                user_tensor[u_idx, t_idx] = onehot.sum(dim=0)
        torch.save(user_tensor, os.path.join(self.output_dir, 'user_tensor.pt'))
        print(f'user_tensor shape: {user_tensor.shape}')

    def derive_rating(self):
        user_path = os.path.join(self.data_dir, 'user.csv')
        if not os.path.exists(user_path):
            print(f'File not found: {user_path}')
            return
        df = pd.read_csv(user_path)
        user_ids = df['user_id'].unique()
        time_periods = df['time_period'].unique()
        num_users = len(user_ids)
        num_times = len(time_periods)
        max_behaviors = df.groupby(['user_id', 'time_period']).size().max()
        user_purchase = torch.zeros((num_users, num_times, max_behaviors), dtype=torch.long)
        rating = torch.zeros((num_users, num_times, max_behaviors), dtype=torch.float)
        valid_len = torch.zeros((num_users, num_times), dtype=torch.long)
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        time_to_idx = {t: i for i, t in enumerate(time_periods)}
        grouped = df.groupby('user_id')
        for user_id, group in tqdm(grouped, desc='Building rating tensors'):
            u_idx = user_to_idx[user_id]
            for time_period, sub_group in group.groupby('time_period'):
                t_idx = time_to_idx[time_period]
                purchases = sub_group['item_id'].tolist()
                ratings = sub_group['rating'].tolist()
                length = len(purchases)
                user_purchase[u_idx, t_idx, :length] = torch.tensor(purchases)
                rating[u_idx, t_idx, :length] = torch.tensor(ratings)
                valid_len[u_idx, t_idx] = length
        torch.save(user_purchase, os.path.join(self.output_dir, 'user_purchase_tensor.pt'))
        torch.save(rating, os.path.join(self.output_dir, 'rating_tensor.pt'))
        torch.save(valid_len, os.path.join(self.output_dir, 'valid_len_tensor.pt'))
        print('Rating tensors saved.')

    def main(self):
        self.derive_item_FOP_tensor()
        self.derive_user_tensor()
        self.derive_rating()

if __name__ == '__main__':
    dp = DataPrepare()
    dp.main()