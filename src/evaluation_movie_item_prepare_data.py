import os
import json
import random
import collections
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

class GenerateMovieItemInput:
    def __init__(self):
        self.base_dir = os.path.join('..', 'intermediate', 'movie', 'evaluation', 'item')
        self.original_dir = os.path.join(self.base_dir, 'original')
        self.input_dir = os.path.join(self.base_dir, 'input')
        os.makedirs(self.input_dir, exist_ok=True)

        self.pos_json = os.path.join(self.original_dir, 'FOP_pos.json')
        self.neg_json = os.path.join(self.original_dir, 'FOP_neg.json')

        self.total_num = 7643
        self.train_num = 6878

    def _load_json_as_dict(self, path):
        with open(path, 'r') as f:
            data = f.read()
        return json.loads(data)

    def _build_vocab_and_indices(self, data_dict, min_freq):
        item_ids = sorted(data_dict.keys(), key=lambda x: int(x))
        text_lists = [data_dict[str(i)] for i in item_ids]
        vocab = Vocab(text_lists, min_freq=min_freq)
        idx_lists = vocab[text_lists]
        return vocab, idx_lists, len(item_ids)

    def _tensor_from_indices(self, idx_lists, vocab_size):
        num_items = len(idx_lists)
        tensor = torch.zeros((num_items, vocab_size))
        for i, idx_list in enumerate(tqdm(idx_lists, desc="Building tensor")):
            temp_idx = torch.tensor(idx_list)
            onehot = nn.functional.one_hot(temp_idx, vocab_size)
            tensor[i] = onehot.sum(dim=0)
        return tensor

    def generate_all(self):
        print("Processing positive FOP text...")
        pos_dict = self._load_json_as_dict(self.pos_json)
        vocab_pos, pos_idx_lists, pos_num = self._build_vocab_and_indices(pos_dict, min_freq=5)
        pos_tensor_full = self._tensor_from_indices(pos_idx_lists, len(vocab_pos))

        print("Processing negative FOP text...")
        neg_dict = self._load_json_as_dict(self.neg_json)
        vocab_neg, neg_idx_lists, neg_num = self._build_vocab_and_indices(neg_dict, min_freq=3)
        neg_tensor_full = self._tensor_from_indices(neg_idx_lists, len(vocab_neg))

        # random train/test split
        indices = list(range(self.total_num))
        random.shuffle(indices)
        train_idx = indices[:self.train_num]
        test_idx = indices[self.train_num:]

        pos_train = pos_tensor_full[train_idx]
        pos_test = pos_tensor_full[test_idx]
        neg_train = neg_tensor_full[train_idx]
        neg_test = neg_tensor_full[test_idx]

        torch.save(pos_train, os.path.join(self.input_dir, 'pos_train.pt'))
        torch.save(pos_test, os.path.join(self.input_dir, 'pos_test.pt'))
        torch.save(neg_train, os.path.join(self.input_dir, 'neg_train.pt'))
        torch.save(neg_test, os.path.join(self.input_dir, 'neg_test.pt'))

        print(f"Saved tensors to {self.input_dir}")

if __name__ == '__main__':
    generator = GenerateMovieItemInput()
    generator.generate_all()