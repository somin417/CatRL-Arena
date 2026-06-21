"""Agent interface and baselines for CatBreak RL Arena."""

from __future__ import annotations

import abc
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

import settings as S
from catbreak_env import obs_vector_for_agent

SUPPORTED_AGENT_NAMES = (
    "random",
    "follow",
    "followball",
    "heuristic",
    "keyboard",
    "dqn",
    "rl",
    "cem_mpc",
    "cem-mpc",
    "mpc",
    "cem_aim",
    "cem-aim",
)


def find_latest_dqn_checkpoint() -> Optional[Path]:
    if S.DQN_BEST_CKPT.exists():
        return S.DQN_BEST_CKPT
    if S.DQN_LAST_CKPT.exists():
        return S.DQN_LAST_CKPT
    return None


class BaseAgent(abc.ABC):
    """Base interface for CatBreak agents."""

    name: str = "BaseAgent"

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Called when the environment resets."""

    @abc.abstractmethod
    def act(
        self,
        obs: np.ndarray,
        info: Optional[dict] = None,
        env: Optional[object] = None,
    ) -> int:
        """Return a discrete action in {0: left, 1: stay, 2: right}."""

    def load(self, path: str) -> None:
        raise NotImplementedError(f"{self.name} does not support load() yet.")


class RandomAgent(BaseAgent):
    name = "Random"

    def __init__(self) -> None:
        self._rng: Optional[np.random.Generator] = None

    def reset(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        obs: np.ndarray,
        info: Optional[dict] = None,
        env: Optional[object] = None,
    ) -> int:
        assert self._rng is not None
        return int(self._rng.integers(0, S.N_ACTIONS))


def followball_action_from_norm(
    ball_x_norm: float,
    paddle_x_norm: float,
    threshold: float = S.FOLLOW_BALL_THRESHOLD,
) -> int:
    """FollowBall policy from normalized [0,1] field coordinates."""
    delta = float(ball_x_norm) - float(paddle_x_norm)
    if delta < -threshold:
        return S.ACTION_LEFT
    if delta > threshold:
        return S.ACTION_RIGHT
    return S.ACTION_STAY


class FollowBallAgent(BaseAgent):
    name = "FollowBall"

    def __init__(self, threshold: float = S.FOLLOW_BALL_THRESHOLD) -> None:
        self.threshold = float(threshold)

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(
        self,
        obs: np.ndarray,
        info: Optional[dict] = None,
        env: Optional[object] = None,
    ) -> int:
        vec = obs_vector_for_agent(obs)
        return followball_action_from_norm(float(vec[0]), float(vec[4]), self.threshold)


HeuristicAgent = FollowBallAgent


class KeyboardAgent(BaseAgent):
    name = "Keyboard"

    def __init__(self) -> None:
        self._action = S.ACTION_STAY

    def reset(self, seed: Optional[int] = None) -> None:
        self._action = S.ACTION_STAY

    def set_action(self, action: int) -> None:
        self._action = int(action)

    @property
    def current_action(self) -> int:
        return self._action

    def act(
        self,
        obs: np.ndarray,
        info: Optional[dict] = None,
        env: Optional[object] = None,
    ) -> int:
        return self._action


class DQNPolicyAgent(BaseAgent):
    """Greedy policy from a trained CatBreak DQN checkpoint."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        epsilon: float = 0.0,
        fallback_to_follow: bool = True,
    ) -> None:
        self.model_path = model_path
        self.epsilon = float(epsilon)
        self._fallback_to_follow = fallback_to_follow
        self._dqn = None
        self._fallback = FollowBallAgent()
        self._using_fallback = False
        self._load_model()

    @property
    def name(self) -> str:
        if self._using_fallback:
            return "DQN(fallback=FollowBall)"
        return "DQN"

    def _resolve_model_path(self) -> Optional[Path]:
        if self.model_path:
            path = Path(self.model_path)
            if path.exists():
                return path
        return find_latest_dqn_checkpoint()

    def _load_model(self) -> None:
        path = self._resolve_model_path()
        if path is None:
            self._using_fallback = self._fallback_to_follow
            if self._using_fallback:
                warnings.warn(
                    "DQNPolicyAgent: no checkpoint found; using FollowBallAgent fallback.",
                    stacklevel=2,
                )
            return
        try:
            import torch
            from agent_dqn import CatBreakDQNAgent
            from torch_utils import get_device

            device = get_device()
            self._dqn = CatBreakDQNAgent.from_checkpoint(path, device=device)
            self.model_path = str(path)
            self._using_fallback = False
            extra = torch.load(path, map_location=device, weights_only=False).get("extra") or {}
            self._dqn._last_extra = extra
        except ImportError as exc:
            self._dqn = None
            self._using_fallback = self._fallback_to_follow
            if self._using_fallback:
                warnings.warn(
                    f"DQNPolicyAgent: torch is required to load checkpoints ({exc}). "
                    "Using FollowBallAgent fallback.",
                    stacklevel=2,
                )
            else:
                raise
        except Exception as exc:
            self._dqn = None
            self._using_fallback = self._fallback_to_follow
            if self._using_fallback:
                warnings.warn(
                    f"DQNPolicyAgent: failed to load checkpoint ({exc}). "
                    "Using FollowBallAgent fallback.",
                    stacklevel=2,
                )
            else:
                raise

    def reset(self, seed: Optional[int] = None) -> None:
        self._fallback.reset(seed=seed)

    def load(self, path: str) -> None:
        self.model_path = path
        self._load_model()

    def act(
        self,
        obs: np.ndarray,
        info: Optional[dict] = None,
        env: Optional[object] = None,
    ) -> int:
        if self._dqn is None:
            if self._fallback_to_follow:
                return self._fallback.act(obs, info, env)
            raise RuntimeError("DQNPolicyAgent: no loaded model and fallback disabled.")
        return self._dqn.act(obs, epsilon=self.epsilon)


class CEMAimPolicyAgent(BaseAgent):
    """Fast CEM-trained aiming policy (no runtime simulation)."""

    name = "CEM-Aim"

    def __init__(self, model_path: Optional[str] = None) -> None:
        from cem_aim_policy import CEMAimPolicy, load_cem_aim_policy

        self.model_path = model_path
        if model_path:
            self.policy = load_cem_aim_policy(model_path)
        else:
            self.policy = CEMAimPolicy(CEMAimPolicy.prior_follow_like())

    def reset(self, seed: Optional[int] = None) -> None:
        if hasattr(self.policy, "reset_episode"):
            self.policy.reset_episode(seed)

    def note_step(self, info: Optional[dict], env: Optional[object] = None) -> None:
        if hasattr(self.policy, "note_step_after_env_step"):
            self.policy.note_step_after_env_step(info, env)

    def load(self, path: str) -> None:
        from cem_aim_policy import load_cem_aim_policy

        self.model_path = path
        self.policy = load_cem_aim_policy(path)

    def act(
        self,
        obs: np.ndarray,
        info: Optional[dict] = None,
        env: Optional[object] = None,
    ) -> int:
        return self.policy.act(obs, info=info, env=env)


class CEMMPCPolicyAgent(BaseAgent):
    """CEM-MPC model-based planner (requires env in act())."""

    name = "CEM-MPC"

    def __init__(self, **planner_kwargs) -> None:
        mode = planner_kwargs.pop("mode", "safe_eval")
        layout = planner_kwargs.pop("layout", None)
        if layout and "env_config" not in planner_kwargs:
            planner_kwargs["env_config"] = {"layout": layout}

        if mode == "teacher_search":
            from cem_mpc_teacher import TeacherSearchPlanner

            if layout:
                planner_kwargs.setdefault("layout", layout)
            self.planner = TeacherSearchPlanner(**planner_kwargs)
        else:
            from cem_mpc import CEMMPCPlanner

            planner_kwargs.setdefault("mode", "safe_eval")
            self.planner = CEMMPCPlanner(**planner_kwargs)

    def reset(self, seed: Optional[int] = None) -> None:
        self.planner.reset(seed=seed)

    def act(
        self,
        obs: np.ndarray,
        info: Optional[dict] = None,
        env: Optional[object] = None,
    ) -> int:
        if env is None:
            raise ValueError("CEM-MPC requires env for planning.")
        return self.planner.act(obs, info=info, env=env)


class RLPolicyAgent(BaseAgent):
    """Placeholder that prefers DQN if available, otherwise FollowBall."""

    name = "RLPolicy"

    def __init__(
        self,
        model_path: Optional[str] = None,
        fallback_to_follow: bool = True,
    ) -> None:
        self._delegate = DQNPolicyAgent(
            model_path=model_path,
            fallback_to_follow=fallback_to_follow,
        )

    @property
    def name(self) -> str:
        return self._delegate.name

    def reset(self, seed: Optional[int] = None) -> None:
        self._delegate.reset(seed=seed)

    def load(self, path: str) -> None:
        self._delegate.load(path)

    def act(
        self,
        obs: np.ndarray,
        info: Optional[dict] = None,
        env: Optional[object] = None,
    ) -> int:
        return self._delegate.act(obs, info, env)


def _normalize_agent_name(name: str) -> str:
    key = name.lower().strip()
    if key in ("heuristic", "followball"):
        return "follow"
    if key in ("cem-mpc", "mpc"):
        return "cem_mpc"
    if key in ("cem-aim",):
        return "cem_aim"
    return key


def make_agent(name: str, seed: Optional[int] = None, **kwargs) -> BaseAgent:
    key = _normalize_agent_name(name)
    if key == "random":
        agent: BaseAgent = RandomAgent()
    elif key == "follow":
        agent = FollowBallAgent()
    elif key == "keyboard":
        agent = KeyboardAgent()
    elif key == "dqn":
        agent = DQNPolicyAgent(
            model_path=kwargs.get("model_path"),
            fallback_to_follow=kwargs.get("fallback_to_follow", True),
        )
    elif key == "rl":
        agent = RLPolicyAgent(
            model_path=kwargs.get("model_path"),
            fallback_to_follow=kwargs.get("fallback_to_follow", True),
        )
    elif key == "cem_mpc":
        planner_kwargs = dict(kwargs.get("planner_kwargs") or {})
        if "env_config" not in planner_kwargs and kwargs.get("layout"):
            planner_kwargs["env_config"] = {"layout": kwargs["layout"]}
        agent = CEMMPCPolicyAgent(**planner_kwargs)
    elif key == "cem_aim":
        agent = CEMAimPolicyAgent(model_path=kwargs.get("model_path"))
    else:
        supported = ", ".join(SUPPORTED_AGENT_NAMES)
        raise ValueError(f"Unknown agent '{name}'. Supported: {supported}")
    if seed is not None:
        agent.reset(seed=seed)
    return agent


Agent = BaseAgent
PolicyAgent = DQNPolicyAgent
