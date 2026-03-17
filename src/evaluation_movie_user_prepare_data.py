import os
import random
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

class GenerateMovieUserInput:
    def __init__(self):
        self.base_dir = os.path.join('..', 'intermediate', 'movie', 'evaluation', 'user')
        self.original_dir = os.path.join(self.base_dir, 'original')
        self.input_dir = os.path.join(self.base_dir, 'input')
        os.makedirs(self.input_dir, exist_ok=True)

        self.csv_path = os.path.join(self.original_dir, 'user.csv')
        self.df = pd.read_csv(self.csv_path)

        self.unique_user = self.df['user_id'].nunique()
        self.unique_time_period = self.df['time_period'].nunique()
        self.unique_item = self.df['item_id'].nunique()

    def generate(self):
        user_tensor = torch.zeros((self.unique_user, self.unique_time_period, self.unique_item))

        grouped = self.df.groupby('user_id')
        for user_id, group in tqdm(grouped, desc="Building user tensor"):
            time_grouped = group.groupby('time_period')
            for time_period, sub_group in time_grouped:
                item_list = torch.tensor(sub_group['item_id'].tolist())
                onehot = nn.functional.one_hot(item_list, self.unique_item)
                user_tensor[user_id, time_period] = onehot.sum(dim=0)

        time_one_hot = nn.functional.one_hot(torch.arange(self.unique_time_period)).float()
        time_one_hot = time_one_hot.unsqueeze(0).repeat(self.unique_user, 1, 1)
        user_tensor = torch.cat([user_tensor, time_one_hot], dim=-1)

        indices = list(range(self.unique_user))
        random.shuffle(indices)
        train_size = int(self.unique_user * 0.9)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        train_tensor = user_tensor[train_idx]
        test_tensor = user_tensor[test_idx]

        torch.save(train_tensor, os.path.join(self.input_dir, 'user_train.pt'))
        torch.save(test_tensor, os.path.join(self.input_dir, 'user_test.pt'))

        print(f"Saved train tensor of shape {train_tensor.shape} to {self.input_dir}/user_train.pt")
        print(f"Saved test tensor of shape {test_tensor.shape} to {self.input_dir}/user_test.pt")

if __name__ == '__main__':
    generator = GenerateMovieUserInput()
    generator.generate()