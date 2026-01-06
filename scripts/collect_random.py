import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import numpy as np
from src.envs.make_env import make_env
from src.data.replay_buffer import ReplayBuffer

# config
CAPACITY = 10000
STEPS = 1000

env = make_env()
obs, info = env.reset()

obs_shape = obs.shape
action_shape = (1,) if np.isscalar(env.action_space.sample()) else env.action_space.sample().shape

buffer = ReplayBuffer(CAPACITY, obs_shape, action_shape)

for _ in range(STEPS):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    buffer.add(obs, action, reward, done)
    obs = next_obs

    if done:
        obs, info = env.reset()

# test sample
batch = buffer.sample(batch_size=4, seq_len=8)
print("Sample OK:", [b.shape for b in batch])
