# Literature Notes: CEM-MPC for CatBreak

## 1. Cross-Entropy Method (CEM)

CEM is a derivative-free, population-based optimization method. Instead of computing gradients, it:

1. Maintains a distribution over candidate solutions (here: categorical distributions over discrete action sequences).
2. Samples a population of candidates from that distribution.
3. Evaluates each candidate via rollout (simulated return or lexicographic score).
4. Selects elite candidates (top fraction by score).
5. Updates the distribution toward elite statistics (action frequencies at each horizon step).

Premature convergence is a known risk: the distribution can collapse onto a suboptimal mode too early. Mitigations used in CatBreak:

- **Probability floor** (`prob_floor`): each action retains minimum mass at every horizon step.
- **Smoothing**: convex combination of old and elite-fitted distributions.
- **Deterministic seed sequences**: FollowBall, all-STAY, all-LEFT, etc. are injected so the planner never forgets strong reactive baselines.
- **Temperature** (optional): softens sampling before selection.

Reference: *A Tutorial on the Cross-Entropy Method* (de Boer et al.).

## 2. Model Predictive Control (MPC)

MPC plans over a finite horizon \(H\), solves for a sequence of actions, executes **only the first action**, then re-plans from the new state. This receding-horizon loop handles model error and stochastic transitions without committing to a full-episode open-loop plan.

For CatBreak, the "model" is the known `CatBreakEnv` physics (not learned). Each planning step copies the current state, rolls out candidate action sequences in simulation, ranks them, and returns the first action of the best sequence.

Reference: *Model-Predictive Control via Cross-Entropy and Gradient-Based Optimization* (Bharadhwaj et al.).

## 3. Why categorical CEM for CatBreak

CatBreak actions are discrete: LEFT (0), STAY (1), RIGHT (2). Gaussian CEM over continuous controls does not apply directly. We use a **categorical distribution** at each horizon timestep: a probability vector over the three actions. Elite updates set each row toward the empirical action frequencies of top sequences.

Reference: *Sample-efficient Cross-Entropy Method for Real-time Planning* (Pinneri et al.) — discusses efficient CEM for planning with discrete or structured action spaces.

## 4. Why CEM-MPC can beat FollowBall

`FollowBallAgent` is reactive: it moves the paddle toward the ball's current normalized x-position. It does not:

- Aim the ball via paddle offset to open side tunnels.
- Plan multi-step paddle motion to intercept future ball trajectories.
- Trade survival for faster brick clearing when safe.

CEM-MPC searches over short action sequences in a faithful physics simulator. If paddle angle control (`PADDLE_ANGLE_CONTROL`) allows horizontal deflection based on hit offset and paddle velocity, sequences that edge-hit the ball toward brick clusters can outscore pure tracking — especially on the cat outline layout where geometry matters.

If CEM-MPC cannot beat FollowBall even with adequate horizon and population size, the physics may lack controllability, the horizon may be too short, or the lexicographic scorer may need tuning — and model-free DQN is unlikely to do better without demonstrations or architecture changes.

## 5. How this helps DQN later

CEM-MPC rollouts produce **demonstration trajectories** (`--save-demo` in `evaluate_cem_mpc.py`):

- `(obs, action, reward, next_obs, done)` tuples saved as NPZ/CSV.
- Can prefill the DQN replay buffer with higher-quality transitions than random exploration.
- Can support **DQfD** (Deep Q-learning from Demonstrations) later: margin loss that keeps the Q-network close to demonstrated actions on expert states.

Reference: *Deep Q-learning from Demonstrations* (Hester et al., DQfD).

Do **not** implement DQfD in the CEM-MPC phase — only collect clean demo data.

## 6. Related: learned models (context only)

*Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (Schrittwieser et al., MuZero) combines learned dynamics with tree search / planning. CatBreak uses a **known** model instead, which is appropriate when physics are cheap to simulate and fully observable.

## Summary

| Component | CatBreak implementation |
|-----------|-------------------------|
| Optimizer | Categorical CEM over action sequences |
| Model | `CatBreakEnv` copy + `simulate_sequence` |
| Control | MPC: plan H steps, execute action 0, warm-start |
| Objective | Lexicographic: bricks > clear > survival > speed |
| Baseline injection | FollowBall sequence in every CEM iteration |
| Output | Eval CSVs, optional planning log, demo NPZ for DQN |
