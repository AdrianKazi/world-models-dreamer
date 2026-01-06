import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.replay_buffer import ReplayBuffer
from src.envs.make_env import make_env


# ---------- config ----------
CAPACITY   = 100_000
BATCH_SIZE = 32
SEQ_LEN    = 16
LR         = 3e-4
STEPS      = 10_000
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- world model (MINIMAL) ----------
class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.dynamics = nn.GRU(
            input_size=latent_dim + action_dim,
            hidden_size=latent_dim,
            batch_first=True
        )

        self.decoder = nn.Linear(latent_dim, obs_dim)
        self.reward_head = nn.Linear(latent_dim, 1)

    def forward(self, obs, actions):
        # obs: [B, T, obs_dim]
        B, T, _ = obs.shape

        z = self.encoder(obs.view(B * T, -1))
        z = z.view(B, T, -1)

        x = torch.cat([z, actions], dim=-1)
        h, _ = self.dynamics(x)

        obs_hat = self.decoder(h)
        reward_hat = self.reward_head(h).squeeze(-1)

        return obs_hat, reward_hat


# ---------- training ----------
def main():
    env = make_env()
    obs, _ = env.reset()

    obs_dim = obs.shape[0]
    action_sample = env.action_space.sample()
    action_dim = 1 if np.isscalar(action_sample) else action_sample.shape[0]

    buffer = ReplayBuffer(CAPACITY, (obs_dim,), (action_dim,))

    # fill buffer with random data
    for _ in range(5_000):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    model = WorldModel(obs_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    for step in range(STEPS):
        obs_b, act_b, rew_b, _ = buffer.sample(BATCH_SIZE, SEQ_LEN)

        obs_b = torch.tensor(obs_b, dtype=torch.float32, device=DEVICE)
        act_b = torch.tensor(act_b, dtype=torch.float32, device=DEVICE)
        rew_b = torch.tensor(rew_b, dtype=torch.float32, device=DEVICE)

        obs_hat, rew_hat = model(obs_b, act_b)

        obs_loss = mse(obs_hat, obs_b)
        rew_loss = mse(rew_hat, rew_b)

        loss = obs_loss + rew_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"[{step}] loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
