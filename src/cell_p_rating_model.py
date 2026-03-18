import os
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class RatingModel(nn.Module):
    def __init__(self, num_user, latent_dim, character_dim, beta, character_vector):
        super().__init__()
        self.beta = beta
        self.character_vector = character_vector
        self.u = nn.Parameter(torch.randn((num_user, latent_dim, character_dim)), requires_grad=True)

    def forward(self, user_idx, motivation_proportion, user_purchase):
        likelihood = self.beta.T[user_purchase]
        posterior = motivation_proportion.unsqueeze(-2) * likelihood
        norm_posterior = posterior / torch.sum(posterior, dim=-1, keepdim=True)

        u = self.u[user_idx]
        weighted_u = torch.einsum('ijkp, ipl->ijkl', norm_posterior, u)
        select_character = self.character_vector[user_purchase]

        inferred_rating = torch.sum(weighted_u * select_character, dim=-1)
        return inferred_rating

    def rating_mask_loss(self, user_rating, inferred_rating, valid_len):
        mask = torch.arange(user_rating.shape[-1], device=valid_len.device)[None, None, :] <= valid_len[:, :, None] - 1
        diff = torch.square(user_rating - inferred_rating)
        masked_loss = diff * mask.float()
        return masked_loss.sum()

    def norm_loss(self, user_idx):
        u = self.u[user_idx]
        return torch.abs(u).sum()

class RatingModelTrainer:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.intermediate_dir = os.path.join(self.base_dir, 'intermediate', 'cell_p')
        os.makedirs(self.intermediate_dir, exist_ok=True)

        self.latent_dim = 4
        self.num_user = 8629
        self.character_dim = 8
        self.lr = 0.007
        self.num_epochs = 200
        self.batch_size = 1024

        self.character_vector = torch.load(
            os.path.join(self.intermediate_dir, 'character_vector.pt'),
            map_location=self.device
        )
        self.beta = torch.load(
            os.path.join(self.intermediate_dir, 'beta.pt'),
            map_location=self.device
        )

    def load_data(self):
        motivation = torch.load(os.path.join(self.intermediate_dir, 'user_motivation.pt'))
        purchase = torch.load(os.path.join(self.intermediate_dir, 'user_purchase_tensor.pt')).to(torch.long)
        rating = torch.load(os.path.join(self.intermediate_dir, 'rating_tensor.pt'))
        valid_len = torch.load(os.path.join(self.intermediate_dir, 'valid_len_tensor.pt'))
        return motivation, purchase, rating, valid_len

    def train(self):
        start_time = time.time()
        print(datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'))

        network = RatingModel(
            self.num_user, self.latent_dim, self.character_dim,
            self.beta, self.character_vector
        ).to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)

        motivation, purchase, rating, valid_len = self.load_data()
        idx_tensor = torch.arange(self.num_user)
        dataset = TensorDataset(idx_tensor, motivation, purchase, rating, valid_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        network.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in dataloader:
                idx_b, mot_b, pur_b, rat_b, len_b = [x.to(self.device) for x in batch]

                inferred = network(idx_b, mot_b, pur_b)
                loss = network.rating_mask_loss(rat_b, inferred, len_b) + 3 * network.norm_loss(idx_b)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(network.parameters(), 10.0)
                optimizer.step()

                total_loss += loss.item()
            if epoch % 1 == 0:
                print(f"epoch {epoch}, loss = {total_loss:.2f}")

        model_path = os.path.join(self.intermediate_dir, 'rating_model.pt')
        torch.save(network.state_dict(), model_path)
        end_time = time.time()
        print(datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'))
        print(f"Training time: {end_time - start_time:.2f}s")
        return network

    def u_insight(self):
        network = RatingModel(
            self.num_user, self.latent_dim, self.character_dim,
            self.beta, self.character_vector
        ).to(self.device)
        model_path = os.path.join(self.intermediate_dir, 'rating_model.pt')
        network.load_state_dict(torch.load(model_path, map_location=self.device))
        network.eval()

        u = network.u.detach().cpu()
        torch.set_printoptions(sci_mode=False, precision=3)

        user_motivation = torch.load(os.path.join(self.intermediate_dir, 'user_motivation.pt'))
        avg_motivation = user_motivation.mean(dim=1)
        threshold = 0.3

        for k in range(self.latent_dim):
            mask = avg_motivation[:, k] > threshold
            if mask.any():
                selected_u = u[mask]
                avg_u = selected_u.mean(dim=0)[k]
                print(f"{avg_u.tolist()}")

    def run(self):
        self.train()
        self.u_insight()

if __name__ == '__main__':
    trainer = RatingModelTrainer(device='cuda:0')
    trainer.run()