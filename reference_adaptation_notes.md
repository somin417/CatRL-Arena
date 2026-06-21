# Reference Adaptation Notes — CatBreak DQN (Phase 4)

## Inspected reference files

- `ref/IE_Assignment3/DQNAgent.py` (primary source)

## Copied / adapted components

| Reference | New project file | Notes |
|-----------|------------------|-------|
| `ReplayBuffer` | `replay_buffer.py` | Uniform replay; local `np.random.default_rng` |
| `PrioritizedReplayBuffer` | `replay_buffer.py` | PER with alpha/beta; safe fallback to uniform |
| `MLP_QNet` | `networks.py` | Simplified 3-hidden-layer MLP (spec layout) |
| `DuelingQNet` | `networks.py` | V + (A - mean A) dueling head |
| `DQNAgent.train()` | `agent_dqn.py` | Huber loss, Double DQN target, PER weights |
| `hard_update` / `soft_update` | `agent_dqn.py` | Hard update by frequency or soft tau |
| epsilon-greedy `getAction` | `agent_dqn.py` `act()` | Local RNG, no global `np.random` |

## Removed CartPole / gym-specific parts

- `gymnasium` / `gym` environment creation (`CartPole-v1`)
- `make_envs`, `render_env`, `close_envs`
- `angleReward`, `EXPLORING_START`, `initialize_state`
- `runEpisode`, `runMany`, `runTest`, `full_score`
- CartPole plotting helpers
- Global `device` CUDA assumption (CatBreak DQN defaults to CPU)
- Reference global flags `DOUBLE_Q`, `DUEL`, `PER` (now constructor args)

## CatBreak-specific changes

- Environment: `CatBreakEnv` (custom, headless-safe)
- `obs_dim` and `n_actions=3` from env metadata
- Normalized vector observations (ball, paddle, brick bitmap, lives, steps)
- Training script logs CatBreak metrics (broken bricks, clear, lives, etc.)
- Fair eval seeds: `env_seed = base_seed + episode`
- Checkpoints saved as full dict (`dqn_best.pt`, `dqn_last.pt`)
- `DQNPolicyAgent` integrates trained model into `agents.py` and `ui_duel.py`
- No Pygame window during training or evaluation

## Not implemented in Phase 4

- PPO, CEM, DDPG, TD3
- CNN / pixel observations
- n-step DQN (optional experiments in reference extras)
