import os
import json
import pandas as pd
import torch
import torch.nn as nn

class GenerateMovieInput:
    def __init__(self):
        self.base_dir = os.path.join('..', 'intermediate', 'movie', 'evaluation', 'rating')
        self.original_dir = os.path.join(self.base_dir, 'original')
        self.input_dir = os.path.join(self.base_dir, 'input')
        os.makedirs(self.input_dir, exist_ok=True)

        self.train_csv = os.path.join(self.original_dir, 'train.csv')
        self.test_csv = os.path.join(self.original_dir, 'test.csv')
        self.character_vector_path = os.path.join('..', 'intermediate', 'movie', 'character_vector.pt')

        self.unique_user = 12347
        self.unique_time_period = 12
        self.unique_item = 7643

    def _save_tensor(self, tensor, model_name, suffix):
        path = os.path.join(self.input_dir, f"{model_name}_{suffix}.pt")
        torch.save(tensor, path)
        print(f"Saved {path}")

    def _save_json(self, data, model_name, suffix):
        path = os.path.join(self.input_dir, f"{model_name}_{suffix}.json")
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Saved {path}")

    def generate_complete_model(self):
        df_train = pd.read_csv(self.train_csv)
        user_tensor = torch.zeros((self.unique_user, self.unique_time_period, self.unique_item))
        grouped = df_train.groupby('user_id')
        for user_id, group in grouped:
            time_grouped = group.groupby('time_period')
            for time_period, sub_group in time_grouped:
                item_list = torch.tensor(sub_group['item_id'].tolist())
                onehot = nn.functional.one_hot(item_list, self.unique_item)
                user_tensor[user_id, time_period] = onehot.sum(dim=0)
        self._save_tensor(user_tensor, 'complete_model', 'train_user_tensor')

        count_series = df_train.groupby(['user_id', 'time_period']).size()
        max_behavior_per_period = int(count_series.max())
        train_purchase = torch.zeros((self.unique_user, self.unique_time_period, max_behavior_per_period), dtype=torch.long)
        train_rating = torch.zeros((self.unique_user, self.unique_time_period, max_behavior_per_period))
        train_valid_len = torch.zeros((self.unique_user, self.unique_time_period), dtype=torch.long)

        grouped = df_train.groupby('user_id')
        for user_id, group in grouped:
            time_grouped = group.groupby('time_period')
            for time_period, sub_group in time_grouped:
                purchase = sub_group['item_id'].tolist()
                rating = sub_group['rating'].tolist()
                valid_len = len(purchase)
                train_purchase[user_id, time_period, :valid_len] = torch.tensor(purchase)
                train_rating[user_id, time_period, :valid_len] = torch.tensor(rating)
                train_valid_len[user_id, time_period] = valid_len

        self._save_tensor(train_purchase, 'complete_model', 'train_user_purchase_tensor')
        self._save_tensor(train_rating, 'complete_model', 'train_rating_tensor')
        self._save_tensor(train_valid_len, 'complete_model', 'train_valid_len_tensor')

        df_test = pd.read_csv(self.test_csv)
        count_series_test = df_test.groupby(['user_id', 'time_period']).size()
        max_behavior_per_period_test = int(count_series_test.max()) if not count_series_test.empty else 0
        test_purchase = torch.zeros((self.unique_user, self.unique_time_period, max_behavior_per_period_test), dtype=torch.long)
        test_rating = torch.zeros((self.unique_user, self.unique_time_period, max_behavior_per_period_test))
        test_valid_len = torch.zeros((self.unique_user, self.unique_time_period), dtype=torch.long)

        grouped_test = df_test.groupby('user_id')
        for user_id, group in grouped_test:
            time_grouped = group.groupby('time_period')
            for time_period, sub_group in time_grouped:
                purchase = sub_group['item_id'].tolist()
                rating = sub_group['rating'].tolist()
                valid_len = len(purchase)
                test_purchase[user_id, time_period, :valid_len] = torch.tensor(purchase)
                test_rating[user_id, time_period, :valid_len] = torch.tensor(rating)
                test_valid_len[user_id, time_period] = valid_len

        self._save_tensor(test_purchase, 'complete_model', 'test_user_purchase_tensor')
        self._save_tensor(test_rating, 'complete_model', 'test_rating_tensor')
        self._save_tensor(test_valid_len, 'complete_model', 'test_valid_len_tensor')

    def generate_model1(self):
        df_train = pd.read_csv(self.train_csv)
        df_test = pd.read_csv(self.test_csv)

        user_tensor = torch.zeros((self.unique_user, self.unique_item))
        grouped = df_train.groupby('user_id')
        for user_id, group in grouped:
            item_list = torch.tensor(group['item_id'].tolist())
            onehot = nn.functional.one_hot(item_list, self.unique_item)
            user_tensor[user_id] = onehot.sum(dim=0)
        self._save_tensor(user_tensor, 'model1', 'train_user_tensor')

        max_behavior_train = int(df_train['user_id'].value_counts().max())
        train_purchase = torch.zeros((self.unique_user, max_behavior_train), dtype=torch.long)
        train_rating = torch.zeros((self.unique_user, max_behavior_train))
        train_valid_len = torch.zeros((self.unique_user,), dtype=torch.long)

        grouped = df_train.groupby('user_id')
        for user_id, group in grouped:
            purchase = group['item_id'].tolist()
            rating = group['rating'].tolist()
            valid_len = len(purchase)
            train_purchase[user_id, :valid_len] = torch.tensor(purchase)
            train_rating[user_id, :valid_len] = torch.tensor(rating)
            train_valid_len[user_id] = valid_len

        self._save_tensor(train_purchase, 'model1', 'train_user_purchase_tensor')
        self._save_tensor(train_rating, 'model1', 'train_rating_tensor')
        self._save_tensor(train_valid_len, 'model1', 'train_valid_len_tensor')

        max_behavior_test = int(df_test['user_id'].value_counts().max()) if not df_test.empty else 0
        test_purchase = torch.zeros((self.unique_user, max_behavior_test), dtype=torch.long)
        test_rating = torch.zeros((self.unique_user, max_behavior_test))
        test_valid_len = torch.zeros((self.unique_user,), dtype=torch.long)

        grouped_test = df_test.groupby('user_id')
        for user_id, group in grouped_test:
            purchase = group['item_id'].tolist()
            rating = group['rating'].tolist()
            valid_len = len(purchase)
            test_purchase[user_id, :valid_len] = torch.tensor(purchase)
            test_rating[user_id, :valid_len] = torch.tensor(rating)
            test_valid_len[user_id] = valid_len

        self._save_tensor(test_purchase, 'model1', 'test_user_purchase_tensor')
        self._save_tensor(test_rating, 'model1', 'test_rating_tensor')
        self._save_tensor(test_valid_len, 'model1', 'test_valid_len_tensor')

    def generate_model2(self):
        df_train = pd.read_csv(self.train_csv)
        df_test = pd.read_csv(self.test_csv)

        max_behavior_train = int(df_train['user_id'].value_counts().max())
        train_purchase = torch.zeros((self.unique_user, max_behavior_train), dtype=torch.long)
        train_rating = torch.zeros((self.unique_user, max_behavior_train))
        train_valid_len = torch.zeros((self.unique_user,), dtype=torch.long)

        grouped = df_train.groupby('user_id')
        for user_id, group in grouped:
            purchase = group['item_id'].tolist()
            rating = group['rating'].tolist()
            valid_len = len(purchase)
            train_purchase[user_id, :valid_len] = torch.tensor(purchase)
            train_rating[user_id, :valid_len] = torch.tensor(rating)
            train_valid_len[user_id] = valid_len

        self._save_tensor(train_purchase, 'model2', 'train_user_purchase_tensor')
        self._save_tensor(train_rating, 'model2', 'train_rating_tensor')
        self._save_tensor(train_valid_len, 'model2', 'train_valid_len_tensor')

        max_behavior_test = int(df_test['user_id'].value_counts().max()) if not df_test.empty else 0
        test_purchase = torch.zeros((self.unique_user, max_behavior_test), dtype=torch.long)
        test_rating = torch.zeros((self.unique_user, max_behavior_test))
        test_valid_len = torch.zeros((self.unique_user,), dtype=torch.long)

        grouped_test = df_test.groupby('user_id')
        for user_id, group in grouped_test:
            purchase = group['item_id'].tolist()
            rating = group['rating'].tolist()
            valid_len = len(purchase)
            test_purchase[user_id, :valid_len] = torch.tensor(purchase)
            test_rating[user_id, :valid_len] = torch.tensor(rating)
            test_valid_len[user_id] = valid_len

        self._save_tensor(test_purchase, 'model2', 'test_user_purchase_tensor')
        self._save_tensor(test_rating, 'model2', 'test_rating_tensor')
        self._save_tensor(test_valid_len, 'model2', 'test_valid_len_tensor')

    def generate_pmf(self):
        df_train = pd.read_csv(self.train_csv)
        df_test = pd.read_csv(self.test_csv)

        train_rating = torch.zeros((len(df_train), 3))
        train_rating[:, 0] = torch.tensor(df_train['user_id'].tolist())
        train_rating[:, 1] = torch.tensor(df_train['item_id'].tolist())
        train_rating[:, 2] = torch.tensor(df_train['rating'].tolist())
        self._save_tensor(train_rating, 'pmf', 'train_rating')

        test_rating = torch.zeros((len(df_test), 3))
        test_rating[:, 0] = torch.tensor(df_test['user_id'].tolist())
        test_rating[:, 1] = torch.tensor(df_test['item_id'].tolist())
        test_rating[:, 2] = torch.tensor(df_test['rating'].tolist())
        self._save_tensor(test_rating, 'pmf', 'test_rating')

    def generate_utadis(self):
        char_vec = torch.load(self.character_vector_path)
        df_train = pd.read_csv(self.train_csv)
        df_test = pd.read_csv(self.test_csv)

        train_dict = {}
        grouped_train = df_train.groupby('user_id')
        for user_id, group in grouped_train:
            item_list = group['item_id'].tolist()
            rating_list = torch.tensor(group['rating'].tolist()).unsqueeze(-1)
            item_vecs = char_vec[torch.tensor(item_list)]
            combined = torch.cat([item_vecs, rating_list], dim=-1)
            train_dict[int(user_id)] = combined.tolist()
        self._save_json(train_dict, 'utadis', 'train_input_coefficient')

        test_dict = {}
        grouped_test = df_test.groupby('user_id')
        for user_id, group in grouped_test:
            item_list = group['item_id'].tolist()
            rating_list = torch.tensor(group['rating'].tolist()).unsqueeze(-1)
            item_vecs = char_vec[torch.tensor(item_list)]
            combined = torch.cat([item_vecs, rating_list], dim=-1)
            test_dict[int(user_id)] = combined.tolist()
        self._save_json(test_dict, 'utadis', 'test_input_coefficient')

    def generate_all(self):
        self.generate_complete_model()
        self.generate_model1()
        self.generate_model2()
        self.generate_pmf()
        self.generate_utadis()

if __name__ == '__main__':
    generator = GenerateMovieInput()
    generator.generate_all()