# CatBreak RL Arena

**KAIST IE540 Term Project** — Human-vs-agent Breakout-style reinforcement learning environment.

Two identical game panels run side by side in Pygame (human vs agent). The same `CatBreakEnv` supports headless training and evaluation for baselines, DQN, CEM-MPC, and CEM-Aim.

## Highlights

| Agent | Type | Runtime cost | Notes |
|-------|------|--------------|-------|
| **Random** | Baseline | Instant | Uniform random actions |
| **FollowBall** | Heuristic | Instant | Tracks ball x-position; strong survival prior |
| **CEM-MPC** | Model-based planner | Slow (CPU rollouts) | Cross-entropy MPC over known physics |
| **CEM-Aim v3** | Learned (12 params) | Instant | FollowBall base + safety-gated residual aiming |
| **DQN** | Deep RL | Instant at inference | MLP / Dueling / Double-DQN / PER options |

Pre-trained **CEM-Aim v3** checkpoint is included:

```
runs/cem_aim/v3_win_hunt_seed1/cem_aim_val_best.npz
```

## Requirements

- Python **3.10+**
- macOS / Linux / Windows (Pygame UI; PyTorch for DQN training)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Mac, you can also use the helper script:

```bash
bash scripts/setup_mac.sh
```

For NVIDIA GPU training, install PyTorch with the CUDA wheel matching your driver (see comment in `requirements.txt`).

## GitHub cat image

Place the Octocat PNG at:

```
assets/github_cat.png
```

Training, evaluation, and the interactive demo default to the **cat outline layout** (20×20 grid, ~82 outline bricks). Use `--layout rect` for the smaller rectangular grid ablation.

## Interactive demo

```bash
python ui_duel.py
```

Defaults: **CEM-Aim v3** on the right panel, **seed 6**, checkpoint `runs/cem_aim/v3_win_hunt_seed1/cem_aim_val_best.npz`.

Override on the command line:

```bash
python ui_duel.py --agent follow --seed 0
python ui_duel.py --agent dqn --model runs/dqn/dqn_best.pt --seed 42
python ui_duel.py --agent cem_mpc
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `←` / `→` or `A` / `D` | Move human paddle (left panel) |
| `SPACE` | Pause / unpause |
| `R` | Reset both games with the same seed |
| `N` | Reset with a new random seed |
| `1` | Right panel: `RandomAgent` |
| `2` | Right panel: `FollowBallAgent` |
| `8` | Right panel: `CEM-Aim` (bundled checkpoint) |
| `5` | Right panel: `CEM-MPC` (reduced planner settings for UI) |
| `4` | Right panel: trained DQN |
| `9` | Right panel: `RLPolicyAgent` |
| `ESC` | Quit |

## Quick evaluation

Episode `k` uses `env_seed = base_seed + k` for fair cross-agent comparison.

```bash
# Baselines
python evaluate.py --agent random  --episodes 20 --seed 0
python evaluate.py --agent follow   --episodes 20 --seed 0

# CEM-Aim (bundled checkpoint)
python evaluate_cem_aim.py \
  --model runs/cem_aim/v3_win_hunt_seed1/cem_aim_val_best.npz \
  --seeds 0:19

# DQN
python evaluate_dqn.py --model runs/dqn/dqn_best.pt --episodes 50 --seed 0

# Compare Random vs FollowBall
python compare_baselines.py --episodes 50 --seed 0
```

CSV logs and plots are written under `runs/` (created at runtime).

## Agents

### FollowBall baseline

`FollowBallAgent` moves the paddle toward the ball's current x-position (threshold `0.035` in normalized obs units). It ignores brick information. On the cat layout it clears ~45% of episodes and is the main benchmark to beat.

### CEM-MPC planner

Model-based **Cross-Entropy Method + Model Predictive Control** using known CatBreak physics. Samples discrete action sequences, evaluates them in a copied env, selects elites, and executes only the first action.

```bash
python evaluate_cem_mpc.py --episodes 50 --seed 0
python diagnose_physics.py --episodes 3 --seed 0
python plot_cem_mpc.py --csv runs/cem_mpc/cem_mpc_episodes_<timestamp>.csv
```

Fast smoke test:

```bash
python evaluate_cem_mpc.py --episodes 2 --seed 0 --horizon 5 --population-size 8 --iterations 1
```

Save demonstration trajectories (for DQN / CEM-Aim warm-start):

```bash
python evaluate_cem_mpc.py --episodes 30 --seed 0 --save-demo --save-planning-log
python analyze_mpc_to_aim.py --planning-log runs/cem_mpc/planning_log_<timestamp>.csv
```

See `literature_cem_mpc_notes.md` for design rationale.

### CEM-Aim v3 (main learned policy)

CEM-Aim is a **fast** 12-parameter policy optimized by CEM. It does **not** simulate futures at runtime.

- **Base controller:** exact FollowBall reproduction
- **Learned layer:** safety-gated residual aiming toward brick clusters and committed shot macros
- **Training objective:** pairwise wins vs FollowBall plus opportunity/endgame snapshot rewards

Train:

```bash
python train_cem_aim.py \
  --policy-version cem_aim_v3_residual_option \
  --generations 20 \
  --population-size 64 \
  --save-dir runs/cem_aim/my_run
```

Quick smoke:

```bash
python train_cem_aim.py --quick --save-dir runs/cem_aim_smoke
```

Evaluate:

```bash
python evaluate_cem_aim.py \
  --model runs/cem_aim/v3_win_hunt_seed1/cem_aim_val_best.npz \
  --seeds 0:19
```

Plot success summary:

```bash
python scripts/plot_cem_aim_success.py --run-dir runs/cem_aim/v3_win_hunt_seed1
python scripts/plot_agent_leaderboard_table.py --seeds 0:19
```

### DQN

DQN uses a Q-network, epsilon-greedy exploration, replay buffer, and target network. Optional: `--double-dqn`, `--dueling`, `--per`.

Adapted from course Assignment3 `DQNAgent.py` (see `reference_adaptation_notes.md`).

Train:

```bash
python train_dqn.py --episodes 500 --seed 0 --double-dqn --dueling
```

Training can open `ui_duel.py` with the best checkpoint afterward. Use `--no-demo` for headless runs.

Quick smoke:

```bash
python train_dqn.py --episodes 2 --quick --save-dir runs/dqn_smoke
```

Outputs under `runs/dqn/`: `train_episodes.csv`, `eval_history.csv`, `config.json`, `dqn_last.pt`, `dqn_best.pt`

Compare vs baselines:

```bash
python compare_dqn_vs_baselines.py --model runs/dqn/dqn_best.pt --episodes 50 --seed 0
python plot_dqn.py --run-dir runs/dqn
```

### PPO / generic CEM (placeholders)

```bash
python train_ppo.py --episodes 5 --seed 0
python train_cem.py --episodes 5 --seed 0
```

## Environment API

```python
from catbreak_env import CatBreakEnv

env = CatBreakEnv()  # default: cat outline; or config={"layout": "rect"}
obs = env.reset(seed=0)
obs, reward, done, info = env.step(action)    # action in {0, 1, 2}
```

| Method | Description |
|--------|-------------|
| `reset(seed)` | Reset episode; returns `obs` |
| `step(action)` | Returns `(obs, reward, done, info)` |
| `get_obs()` / `parse_obs(obs)` | Current observation / named fields |
| `get_state_dict()` / `set_state_dict()` | Snapshot / restore full state |
| `clone()` | Deep copy for planning rollouts |
| `render_surface(surface, rect, title)` | Draw state (Pygame only) |
| `close()` | Cleanup |

**Observation** (float32, length `obs_dim`):

1. `ball_x`, `ball_y`, `ball_vx`, `ball_vy` (normalized)
2. `paddle_x`, `paddle_vx` (normalized)
3. Flattened brick-alive bitmap
4. `lives` normalized, `step_count` normalized

**Actions:** `0` = LEFT, `1` = STAY, `2` = RIGHT

**Rewards:** +200 per brick, −0.01 per step (more bricks first; same bricks → fewer steps wins)

## Plug in custom agents

Subclass `BaseAgent` in `agents.py`:

```python
class MyAgent(BaseAgent):
    name = "MyAgent"

    def reset(self, seed=None):
        ...

    def act(self, obs, info=None, env=None):
        return S.ACTION_STAY
```

Register via `make_agent("my_agent")` or use an existing adapter (`DQNPolicyAgent`, `CEMAimPolicyAgent`, `CEMMPCPolicyAgent`).

## Tests

```bash
python tests/test_env_smoke.py
python tests/test_follow_agent.py
python tests/test_dqn_smoke.py
python tests/test_cem_mpc_smoke.py
python tests/test_cem_aim_smoke.py
python tests/test_cnn_dqn_smoke.py
```

## Project structure

```
settings.py              # paths, physics, display, run directories
cat_layout.py            # rectangular + cat outline layouts
catbreak_env.py          # Gym-like environment (headless-safe)
agents.py                # BaseAgent + all policy adapters
agent_dqn.py             # CatBreak DQN agent
networks.py              # MLP / Dueling Q-networks
replay_buffer.py         # uniform + prioritized replay
cem_mpc.py               # CEM-MPC planner
cem_mpc_parallel.py      # parallel rollout helpers
cem_mpc_teacher.py       # teacher search utilities
cem_aim_policy.py        # CEM-Aim v2/v3 policy + features
cem_aim_v3.py            # v3 training, snapshots, pairwise eval
train_cem_aim.py         # CEM-Aim CEM training CLI
evaluate_cem_aim.py      # CEM-Aim vs baselines
analyze_mpc_to_aim.py    # MPC planning log analysis
diagnose_physics.py      # paddle collision diagnostics
evaluate_cem_mpc.py      # CEM-MPC evaluation
plot_cem_mpc.py          # CEM-MPC plots
ui_duel.py               # interactive human vs agent demo
evaluate.py              # headless baseline evaluation
evaluate_dqn.py          # DQN checkpoint evaluation
compare_baselines.py     # Random vs FollowBall
compare_dqn_vs_baselines.py
plot_results.py
plot_baselines.py
plot_dqn.py
plot_policy_comparison.py
train_dqn.py             # DQN training
train_ppo.py             # PPO placeholder
train_cem.py             # CEM placeholder
run_teacher_search.py    # CEM-MPC teacher hyperparameter search
torch_utils.py           # device / parallel worker helpers
scripts/
  setup_mac.sh
  plot_cem_aim_success.py
  plot_agent_leaderboard_table.py
  eval_cem_mpc_seeds_parallel.py
  recover_v3_win_hunt_checkpoint.py
tests/
assets/github_cat.png
runs/                    # checkpoints, CSVs, plots (partially committed)
reference_adaptation_notes.md
literature_cem_mpc_notes.md
```

## Notes

- Action space is **discrete**: LEFT / STAY / RIGHT.
- DDPG / TD3 are reserved for a future **continuous-action** paddle variant.
- Octocat outline asset is for course demo use; GitHub owns the Octocat trademark.
