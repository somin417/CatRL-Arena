"""Central configuration for CatBreak RL Arena."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSET_CAT_IMAGE = "assets/github_cat.png"
CAT_IMAGE_PATH = ASSETS_DIR / "github_cat.png"
RUNS_DIR = PROJECT_ROOT / "runs"

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
FPS = 60
FIXED_DT = 1.0 / 60.0

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FOOTER_HEIGHT = 28
HEADER_HEIGHT = 80
DIVIDER_WIDTH = 4
PANEL_PADDING = 12

# Per-panel playfield (logical game coordinates)
FIELD_WIDTH = 480
FIELD_HEIGHT = 620

# Layout modes
LAYOUT_RECT = "rect"
LAYOUT_CAT = "cat"
DEFAULT_LAYOUT = LAYOUT_CAT  # Octocat outline — training, eval, and UI

# Rectangular brick grid (fast fallback / ablations)
BRICK_ROWS = 6
BRICK_COLS = 10
BRICK_WIDTH = 36
BRICK_HEIGHT = 14
BRICK_GAP = 3
BRICK_TOP_MARGIN = 32

# Octocat outline layout (interactive demo / optional)
CAT_GRID_SIZE = 20
CAT_BRICK_PIXEL = 20
CAT_BRICK_GAP = 0
CAT_LUMINANCE_THRESHOLD = 248.0
CAT_TOP_MARGIN = 10
CAT_BOTTOM_GAP = 130
CAT_OUTLINE_WHITE = (230, 237, 243)
CAT_OUTLINE_GRAY = (139, 148, 158)

# ---------------------------------------------------------------------------
# GitHub dark-theme palette
# ---------------------------------------------------------------------------
COLOR_BG = (13, 17, 23)
COLOR_PANEL = (22, 27, 34)
COLOR_BORDER = (48, 54, 61)
COLOR_ACCENT = (56, 139, 253)
COLOR_TEXT = (230, 237, 243)
COLOR_TEXT_DIM = (139, 148, 158)
COLOR_DIVIDER = (48, 54, 61)
COLOR_BALL = (240, 246, 252)
COLOR_PADDLE = (48, 54, 61)

BRICK_COLORS = [
    (248, 81, 73),
    (227, 179, 65),
    (63, 185, 80),
    (56, 139, 253),
    (188, 140, 255),
]

# ---------------------------------------------------------------------------
# Game physics
# ---------------------------------------------------------------------------
BALL_RADIUS = 6
PADDLE_WIDTH = 72
PADDLE_HEIGHT = 14
PADDLE_SPEED = 320.0
PADDLE_Y_OFFSET = 36

# Paddle bounce physics (not reward shaping — enables trajectory control)
PADDLE_ANGLE_CONTROL = True
PADDLE_MAX_BOUNCE_ANGLE_DEG = 65
PADDLE_SPIN_STRENGTH = 0.25

INITIAL_LIVES = 1
MAX_STEPS = 10000

BALL_SPEED_MIN = 220.0
BALL_SPEED_MAX = 280.0
INITIAL_BALL_ANGLE_RANGE = (-0.55, 0.55)

# Observation modes (Phase 4)
OBS_MODE_VECTOR = "vector"
OBS_MODE_GRID = "grid"
OBS_MODE_HYBRID = "hybrid"
DEFAULT_OBS_MODE = OBS_MODE_VECTOR

GRID_H = 24
GRID_W = 24
GRID_CHANNELS = 5  # brick, ball, ball_prev, paddle, valid_area
HYBRID_VECTOR_DIM = 8

# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
#   return ≈ broken_bricks * REWARD_BRICK + steps * REWARD_STEP
#       + REWARD_DEATH if no_lives + REWARD_CLEAR if cleared
# REWARD_BRICK must exceed MAX_STEPS * |REWARD_STEP| so +1 brick always beats any time gap.
# With MAX_STEPS=10000 and REWARD_STEP=-0.01 → need REWARD_BRICK > 100.
# REWARD_DEATH (-50) is safe: 200 > 50 + 150. Keep REWARD_CLEAR <= 40 so clear bonus
# cannot beat +1 brick. REWARD_LIFE_LOST stays 0 with INITIAL_LIVES=1 (death already penalized).
REWARD_BRICK = 200.0
REWARD_CLEAR = 0.0
REWARD_STEP = -0.01
REWARD_LIFE_LOST = 0.0
REWARD_DEATH = -50.0

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
ACTION_LEFT = 0
ACTION_STAY = 1
ACTION_RIGHT = 2
N_ACTIONS = 3
NUM_ACTIONS = N_ACTIONS  # alias

ACTION_NAMES = {
    0: "LEFT",
    1: "STAY",
    2: "RIGHT",
}

# FollowBall baseline threshold (normalized obs units)
FOLLOW_BALL_THRESHOLD = 0.035
HEURISTIC_THRESHOLD_NORM = FOLLOW_BALL_THRESHOLD  # backward-compat alias

# Evaluation output
BASELINES_DIR = RUNS_DIR / "baselines"
BASELINE_EPISODES_CSV = BASELINES_DIR / "baseline_episodes.csv"
BASELINE_SUMMARY_CSV = BASELINES_DIR / "baseline_summary.csv"
BASELINE_PLOTS_DIR = BASELINES_DIR / "plots"
BASELINE_COMPARISON_PNG = BASELINE_PLOTS_DIR / "baseline_comparison.png"

DQN_DIR = RUNS_DIR / "dqn"
DQN_PER_DIR = RUNS_DIR / "dqn_per"
DQN_CNN_DIR = RUNS_DIR / "dqn_cnn"
DQN_CNN_PER_DIR = RUNS_DIR / "dqn_cnn_per"
DQN_BEST_CKPT = DQN_DIR / "dqn_best.pt"
DQN_LAST_CKPT = DQN_DIR / "dqn_last.pt"
DQN_EVAL_DIR = RUNS_DIR / "dqn_eval"
DQN_EVAL_CSV = DQN_EVAL_DIR / "dqn_eval.csv"

CEM_MPC_DIR = RUNS_DIR / "cem_mpc"
CEM_MPC_DEMOS_DIR = CEM_MPC_DIR / "demos"
CEM_MPC_LOGS_DIR = CEM_MPC_DIR / "logs"
CEM_MPC_SNAPSHOTS_DIR = CEM_MPC_DIR / "snapshots"
CEM_MPC_PLOTS_DIR = CEM_MPC_DIR / "plots"
TEACHER_STRESS_SEEDS = (10, 11, 13, 14, 19)
CEM_AIM_DIR = RUNS_DIR / "cem_aim"
CEM_AIM_V3_WIN_HUNT_DIR = CEM_AIM_DIR / "v3_win_hunt_seed1"
CEM_AIM_V3_VAL_BEST = CEM_AIM_V3_WIN_HUNT_DIR / "cem_aim_val_best.npz"
DIAGNOSTICS_DIR = RUNS_DIR / "diagnostics"

COMPARISON_DIR = RUNS_DIR / "comparison"
COMPARISON_EPISODES_CSV = COMPARISON_DIR / "dqn_vs_baselines_episodes.csv"
COMPARISON_SUMMARY_CSV = COMPARISON_DIR / "dqn_vs_baselines_summary.csv"

PLOTS_DIR = RUNS_DIR / "plots"
