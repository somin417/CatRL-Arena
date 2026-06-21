"""Gym-like Breakout environment for CatBreak RL Arena (headless-safe)."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

import settings as S
from cat_layout import BrickLayout, get_layout


def obs_dim_for_layout(layout: str) -> int:
    """Observation size for a layout mode (8 scalars + brick bitmap)."""
    layout_obj = get_layout(layout)
    return 8 + layout_obj.rows * layout_obj.cols


def layout_for_obs_dim(obs_dim: int) -> str:
    """Map checkpoint obs_dim back to a known layout mode."""
    rect_dim = obs_dim_for_layout(S.LAYOUT_RECT)
    cat_dim = obs_dim_for_layout(S.LAYOUT_CAT)
    if obs_dim == rect_dim:
        return S.LAYOUT_RECT
    if obs_dim == cat_dim:
        return S.LAYOUT_CAT
    raise ValueError(
        f"obs_dim={obs_dim} does not match rect ({rect_dim}) or cat ({cat_dim})."
    )


def obs_vector_for_agent(obs) -> np.ndarray:
    """Extract a flat vector for legacy agents (FollowBall, etc.)."""
    if isinstance(obs, dict):
        return np.asarray(obs["vector"], dtype=np.float32)
    return np.asarray(obs, dtype=np.float32)


def layout_from_checkpoint(path: str | Path) -> str:
    """Read layout from checkpoint metadata, or infer from obs_dim."""
    import torch
    from pathlib import Path as PathType

    path = PathType(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    extra = payload.get("extra") or {}
    layout = extra.get("layout")
    if layout in (S.LAYOUT_RECT, S.LAYOUT_CAT):
        return layout
    obs_dim = payload.get("obs_dim")
    if obs_dim is not None:
        return layout_for_obs_dim(int(obs_dim))
    return S.DEFAULT_LAYOUT


def obs_mode_from_checkpoint(path: str | Path) -> str:
    import torch
    from pathlib import Path as PathType

    path = PathType(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    extra = payload.get("extra") or {}
    return payload.get("obs_mode") or extra.get("obs_mode") or S.DEFAULT_OBS_MODE


def _circle_rect_collision(
    cx: float, cy: float, radius: float, rect: Tuple[int, int, int, int]
) -> bool:
    rx, ry, rw, rh = rect
    nearest_x = max(rx, min(cx, rx + rw))
    nearest_y = max(ry, min(cy, ry + rh))
    dx = cx - nearest_x
    dy = cy - nearest_y
    return dx * dx + dy * dy <= radius * radius


class CatBreakEnv:
    """Single-player CatBreak environment with deterministic fixed-step physics."""

    def __init__(
        self,
        config: Optional[dict] = None,
        render_mode: Optional[str] = None,
        name: str = "CatBreak",
    ) -> None:
        config = config or {}
        layout_mode = config.get("layout", S.DEFAULT_LAYOUT)
        self._layout_mode = layout_mode
        self.obs_mode = config.get("obs_mode", S.DEFAULT_OBS_MODE)
        self.grid_h = int(config.get("grid_h", S.GRID_H))
        self.grid_w = int(config.get("grid_w", S.GRID_W))
        self.grid_channels = int(config.get("grid_channels", S.GRID_CHANNELS))
        self._layout: BrickLayout = config.get("brick_layout") or get_layout(layout_mode)
        self.render_mode = render_mode
        self.name = name

        self.brick_rows = self._layout.rows
        self.brick_cols = self._layout.cols
        self.n_actions = S.N_ACTIONS
        brick_n = self.brick_rows * self.brick_cols
        self.obs_dim = 8 + brick_n
        self.hybrid_vector_dim = S.HYBRID_VECTOR_DIM
        self.grid_shape = (self.grid_channels, self.grid_h, self.grid_w)
        self._initial_brick_count = self._layout.total_bricks
        self.obs_indices = {
            "ball_x": 0,
            "ball_y": 1,
            "ball_vx": 2,
            "ball_vy": 3,
            "paddle_x": 4,
            "paddle_vx": 5,
            "bricks_start": 6,
            "bricks_end": 6 + brick_n,
            "lives": 6 + brick_n,
            "step_count": 7 + brick_n,
        }

        self._seed: Optional[int] = None
        self.rng = np.random.default_rng()
        self._bricks = np.zeros((self.brick_rows, self.brick_cols), dtype=bool)

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self._ball_prev_x = 0.0
        self._ball_prev_y = 0.0
        self.paddle_x = 0.0
        self.paddle_vx = 0.0
        self.paddle_y = float(S.FIELD_HEIGHT - S.PADDLE_Y_OFFSET)

        self.score = 0
        self.broken_bricks = 0
        self.lives = S.INITIAL_LIVES
        self.step_count = 0
        self.done = False
        self.terminal_reason: Optional[str] = None
        self.last_action = S.ACTION_STAY
        self.last_reward = 0.0
        self.last_info: dict = {}
        self._last_paddle_collision: Optional[dict] = None
        self._paddle_hit_this_step: bool = False

        self._cat_mascot = None
        self._cat_mascot_missing = False

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._seed = int(seed)
        self.rng = np.random.default_rng(self._seed)

        self._bricks = self._layout.mask.copy()
        self.paddle_x = S.FIELD_WIDTH / 2.0
        self.paddle_vx = 0.0
        self.paddle_y = float(S.FIELD_HEIGHT - S.PADDLE_Y_OFFSET)
        self.score = 0
        self.broken_bricks = 0
        self.lives = S.INITIAL_LIVES
        self.step_count = 0
        self.done = False
        self.terminal_reason = None
        self.last_action = S.ACTION_STAY
        self.last_reward = 0.0

        self._reset_ball()
        self._ball_prev_x = self.ball_x
        self._ball_prev_y = self.ball_y
        self.last_info = self._build_info(
            bricks_broken_this_step=0, life_lost=False, clear=False
        )
        return self.get_obs()

    def _step_core(self, action: int) -> tuple[float, bool, dict]:
        """Advance physics and return (reward, done, info) without building obs."""
        if action not in (S.ACTION_LEFT, S.ACTION_STAY, S.ACTION_RIGHT):
            raise ValueError(f"Invalid action {action}; expected 0, 1, or 2.")

        if self.done:
            self.last_info = self._build_info(0, False, False)
            return 0.0, True, self.last_info

        ball_prev_x, ball_prev_y = self.ball_x, self.ball_y
        self.last_action = int(action)
        reward = S.REWARD_STEP
        bricks_broken_this_step = 0
        life_lost = False
        cleared = False
        self._paddle_hit_this_step = False

        self._apply_paddle_action(action)
        self._integrate_ball()
        bricks_broken_this_step = self._resolve_collisions()
        if bricks_broken_this_step:
            reward += S.REWARD_BRICK * bricks_broken_this_step
            self.broken_bricks += bricks_broken_this_step
            self.score += bricks_broken_this_step

        if self._remaining_bricks() == 0:
            reward += S.REWARD_CLEAR
            self.score += int(S.REWARD_CLEAR)
            self.done = True
            cleared = True
            self.terminal_reason = "cleared"

        if not cleared and self._ball_fell_below():
            self.lives -= 1
            reward += S.REWARD_LIFE_LOST
            life_lost = True
            if self.lives <= 0:
                self.done = True
                self.terminal_reason = "no_lives"
                reward += S.REWARD_DEATH
            else:
                self._reset_ball()

        self.step_count += 1
        if self.step_count >= S.MAX_STEPS and not self.done:
            self.done = True
            self.terminal_reason = "max_steps"

        self._ball_prev_x = ball_prev_x
        self._ball_prev_y = ball_prev_y
        self.last_reward = reward
        self.last_info = self._build_info(
            bricks_broken_this_step=bricks_broken_this_step,
            life_lost=life_lost,
            clear=cleared,
        )
        return reward, self.done, self.last_info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        reward, done, info = self._step_core(action)
        return self._package_obs(), reward, done, info

    def step_fast(self, action: int) -> tuple[float, bool, dict]:
        """Rollout-only step: skips observation construction."""
        return self._step_core(action)

    def _package_obs(self):
        if self.obs_mode == S.OBS_MODE_GRID:
            return self.get_grid_obs()
        if self.obs_mode == S.OBS_MODE_HYBRID:
            return {"grid": self.get_grid_obs(), "vector": self.get_hybrid_vector()}
        return self.get_vector_obs()

    def get_obs(self):
        """Return observation in the configured obs_mode format."""
        return self._package_obs()

    def get_vector_obs(self) -> np.ndarray:
        """Legacy flattened vector: 8 scalars + brick bitmap."""
        brick_flat = self._bricks.astype(np.float32).flatten()
        scalars = np.array(
            [
                self.ball_x / S.FIELD_WIDTH,
                self.ball_y / S.FIELD_HEIGHT,
                self.ball_vx / S.BALL_SPEED_MAX,
                self.ball_vy / S.BALL_SPEED_MAX,
                self.paddle_x / S.FIELD_WIDTH,
                self.paddle_vx / S.PADDLE_SPEED,
                self.lives / S.INITIAL_LIVES,
                self.step_count / S.MAX_STEPS,
            ],
            dtype=np.float32,
        )
        return np.concatenate([scalars, brick_flat])

    def get_hybrid_vector(self) -> np.ndarray:
        """Compact vector for CNN hybrid mode (no brick bitmap)."""
        remaining = self._remaining_bricks()
        denom = max(1, self._initial_brick_count)
        return np.array(
            [
                self.ball_x / S.FIELD_WIDTH,
                self.ball_y / S.FIELD_HEIGHT,
                self.ball_vx / S.BALL_SPEED_MAX,
                self.ball_vy / S.BALL_SPEED_MAX,
                self.paddle_x / S.FIELD_WIDTH,
                self.paddle_vx / S.PADDLE_SPEED,
                remaining / denom,
                self.step_count / S.MAX_STEPS,
            ],
            dtype=np.float32,
        )

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(np.clip(x / S.FIELD_WIDTH * self.grid_w, 0, self.grid_w - 1))
        gy = int(np.clip(y / S.FIELD_HEIGHT * self.grid_h, 0, self.grid_h - 1))
        return gy, gx

    def _stamp_point(self, channel: np.ndarray, x: float, y: float, value: float = 1.0) -> None:
        gy, gx = self._world_to_grid(x, y)
        channel[gy, gx] = max(channel[gy, gx], value)
        if gy > 0:
            channel[gy - 1, gx] = max(channel[gy - 1, gx], value * 0.5)
        if gy + 1 < self.grid_h:
            channel[gy + 1, gx] = max(channel[gy + 1, gx], value * 0.5)
        if gx > 0:
            channel[gy, gx - 1] = max(channel[gy, gx - 1], value * 0.5)
        if gx + 1 < self.grid_w:
            channel[gy, gx + 1] = max(channel[gy, gx + 1], value * 0.5)

    def _rasterize_rect(self, channel: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        rx, ry, rw, rh = rect
        x0 = int(np.clip(rx / S.FIELD_WIDTH * self.grid_w, 0, self.grid_w - 1))
        y0 = int(np.clip(ry / S.FIELD_HEIGHT * self.grid_h, 0, self.grid_h - 1))
        x1 = int(np.clip((rx + rw) / S.FIELD_WIDTH * self.grid_w, 0, self.grid_w - 1))
        y1 = int(np.clip((ry + rh) / S.FIELD_HEIGHT * self.grid_h, 0, self.grid_h - 1))
        channel[y0 : y1 + 1, x0 : x1 + 1] = 1.0

    def get_grid_obs(self) -> np.ndarray:
        """Object-channel grid [C, H, W] built directly from state (no pygame)."""
        grid = np.zeros(self.grid_shape, dtype=np.float32)
        brick_ch, ball_ch, ball_prev_ch, paddle_ch, valid_ch = range(self.grid_channels)

        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if self._bricks[row, col]:
                    self._rasterize_rect(grid[brick_ch], self._layout.brick_rect(row, col))

        self._stamp_point(grid[ball_ch], self.ball_x, self.ball_y, 1.0)
        self._stamp_point(grid[ball_prev_ch], self._ball_prev_x, self._ball_prev_y, 1.0)

        paddle_rect = (
            int(self.paddle_x - S.PADDLE_WIDTH / 2),
            int(self.paddle_y - S.PADDLE_HEIGHT / 2),
            S.PADDLE_WIDTH,
            S.PADDLE_HEIGHT,
        )
        self._rasterize_rect(grid[paddle_ch], paddle_rect)

        margin_x = S.BALL_RADIUS
        margin_y = S.BALL_RADIUS
        vx0, vy0 = self._world_to_grid(margin_x, margin_y)
        vx1, vy1 = self._world_to_grid(
            S.FIELD_WIDTH - margin_x, S.FIELD_HEIGHT - margin_y
        )
        grid[valid_ch, vy0 : vy1 + 1, vx0 : vx1 + 1] = 1.0

        return grid

    def parse_obs(self, obs: np.ndarray) -> dict:
        """Parse a 1D observation vector into named fields."""
        idx = self.obs_indices
        bricks = obs[idx["bricks_start"] : idx["bricks_end"]]
        return {
            "ball_x": float(obs[idx["ball_x"]]),
            "ball_y": float(obs[idx["ball_y"]]),
            "ball_vx": float(obs[idx["ball_vx"]]),
            "ball_vy": float(obs[idx["ball_vy"]]),
            "paddle_x": float(obs[idx["paddle_x"]]),
            "paddle_vx": float(obs[idx["paddle_vx"]]),
            "bricks": bricks.copy(),
            "lives": float(obs[idx["lives"]]),
            "step_count": float(obs[idx["step_count"]]),
        }

    def get_config_dict(self) -> dict:
        return {
            "layout": self._layout_mode,
            "obs_mode": self.obs_mode,
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
            "grid_channels": self.grid_channels,
        }

    def get_state_dict(self) -> dict:
        return {
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_vx": self.ball_vx,
            "ball_vy": self.ball_vy,
            "ball_prev_x": self._ball_prev_x,
            "ball_prev_y": self._ball_prev_y,
            "paddle_x": self.paddle_x,
            "paddle_vx": self.paddle_vx,
            "paddle_y": self.paddle_y,
            "bricks": self._bricks.copy(),
            "score": self.score,
            "broken_bricks": self.broken_bricks,
            "lives": self.lives,
            "step_count": self.step_count,
            "done": self.done,
            "terminal_reason": self.terminal_reason,
            "seed": self._seed,
            "rng_state": self.rng.bit_generator.state,
            "last_action": self.last_action,
            "last_reward": self.last_reward,
        }

    def clone(self) -> "CatBreakEnv":
        new_env = CatBreakEnv(
            config=self.get_config_dict(),
            render_mode=None,
            name=self.name + "_clone",
        )
        new_env.set_state_dict(self.get_state_dict())
        return new_env

    def set_state_dict(self, state: dict) -> None:
        self.ball_x = float(state["ball_x"])
        self.ball_y = float(state["ball_y"])
        self.ball_vx = float(state["ball_vx"])
        self.ball_vy = float(state["ball_vy"])
        self._ball_prev_x = float(state.get("ball_prev_x", self.ball_x))
        self._ball_prev_y = float(state.get("ball_prev_y", self.ball_y))
        self.paddle_x = float(state["paddle_x"])
        self.paddle_vx = float(state["paddle_vx"])
        self.paddle_y = float(state.get("paddle_y", S.FIELD_HEIGHT - S.PADDLE_Y_OFFSET))
        self._bricks = np.array(state["bricks"], dtype=bool)
        self.score = int(state["score"])
        self.broken_bricks = int(state["broken_bricks"])
        self.lives = int(state["lives"])
        self.step_count = int(state["step_count"])
        self.done = bool(state["done"])
        self.terminal_reason = state.get("terminal_reason")
        self._seed = state.get("seed")
        self.last_action = int(state.get("last_action", S.ACTION_STAY))
        self.last_reward = float(state.get("last_reward", 0.0))
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]

    def get_state(self) -> dict:
        """Alias for get_state_dict (CEM-Aim snapshot training)."""
        return self.get_state_dict()

    def set_state(self, state: dict) -> None:
        """Alias for set_state_dict."""
        self.set_state_dict(state)

    def clone_state(self) -> dict:
        return self.get_state_dict()

    def restore_state(self, state: dict) -> None:
        self.set_state_dict(state)

    def close(self) -> None:
        self._cat_mascot = None

    # ------------------------------------------------------------------
    # Rendering (pygame imported lazily)
    # ------------------------------------------------------------------

    def render_surface(
        self,
        surface,
        rect,
        title: str = "CatBreak",
        show_debug: bool = True,
        current_action: Optional[int] = None,
    ) -> None:
        import pygame

        surface.fill(S.COLOR_BG, rect)
        pygame.draw.rect(surface, S.COLOR_PANEL, rect, border_radius=0)
        pygame.draw.rect(surface, S.COLOR_BORDER, rect, width=3)

        inner = rect.inflate(-S.PANEL_PADDING * 2, -S.PANEL_PADDING * 2)
        scale = min(
            inner.width / S.FIELD_WIDTH,
            (inner.height - 60) / S.FIELD_HEIGHT,
        )
        field_w = int(S.FIELD_WIDTH * scale)
        field_h = int(S.FIELD_HEIGHT * scale)
        field_rect = pygame.Rect(0, 0, field_w, field_h)
        field_rect.centerx = inner.centerx
        field_rect.top = inner.top + 52

        field_surf = pygame.Surface((field_w, field_h))
        field_surf.fill(S.COLOR_BG)
        pygame.draw.rect(field_surf, S.COLOR_BORDER, field_surf.get_rect(), width=2)

        self._draw_bricks(field_surf, scale, pygame)
        self._draw_ball(field_surf, scale, pygame)
        self._draw_paddle(field_surf, scale, pygame)
        self._draw_mascot(field_surf, scale, pygame)

        surface.blit(field_surf, field_rect.topleft)

        font_title = pygame.font.SysFont("monospace", 18, bold=True)
        font_metrics = pygame.font.SysFont("monospace", 13)
        font_small = pygame.font.SysFont("monospace", 11)

        surface.blit(font_title.render(title, True, S.COLOR_TEXT), (inner.left, inner.top))

        if show_debug:
            action = current_action if current_action is not None else self.last_action
            action_label = f"  action={S.ACTION_NAMES.get(action, '?')}"
            status = self.terminal_reason or "playing"
            metrics = (
                f"score={self.score}  broken={self.broken_bricks}  "
                f"remain={self._remaining_bricks()}  steps={self.step_count}  "
                f"lives={self.lives}  status={status}{action_label}"
            )
            surface.blit(
                font_metrics.render(metrics, True, S.COLOR_TEXT_DIM),
                (inner.left, inner.top + 24),
            )
            seed_text = f"seed={self._seed}" if self._seed is not None else "seed=?"
            seed_surf = font_small.render(seed_text, True, S.COLOR_TEXT_DIM)
            surface.blit(seed_surf, (inner.right - seed_surf.get_width(), inner.top + 24))

    def _ensure_mascot(self, pygame) -> None:
        if self._cat_mascot is not None or self._cat_mascot_missing:
            return
        if not S.CAT_IMAGE_PATH.exists():
            print(f"WARNING: Cat image not found at {S.CAT_IMAGE_PATH}. Using placeholder.")
            self._cat_mascot_missing = True
            return
        try:
            self._cat_mascot = pygame.image.load(str(S.CAT_IMAGE_PATH)).convert()
        except pygame.error as exc:
            print(f"WARNING: Failed to load cat image: {exc}")
            self._cat_mascot_missing = True

    def _draw_mascot(self, surf, scale: float, pygame) -> None:
        self._ensure_mascot(pygame)
        target_h = max(24, int(36 * scale))
        if self._cat_mascot is not None:
            mw, mh = self._cat_mascot.get_size()
            target_w = int(target_h * mw / mh)
            scaled = pygame.transform.scale(self._cat_mascot, (target_w, target_h))
            surf.blit(scaled, (4, 4))
        elif self._cat_mascot_missing:
            placeholder = pygame.Rect(4, 4, target_h, target_h)
            pygame.draw.rect(surf, S.COLOR_PADDLE, placeholder)
            pygame.draw.rect(surf, (248, 81, 73), placeholder, width=2)

    def _draw_bricks(self, surf, scale: float, pygame) -> None:
        draw_border = scale >= 0.85
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if not self._bricks[row, col]:
                    continue
                rx, ry, rw, rh = self._layout.brick_rect(row, col)
                sw = max(1, int(rw * scale))
                sh = max(1, int(rh * scale))
                scaled = pygame.Rect(int(rx * scale), int(ry * scale), sw, sh)
                rgb = self._layout.colors[row, col]
                pygame.draw.rect(surf, (int(rgb[0]), int(rgb[1]), int(rgb[2])), scaled)
                if draw_border and sw >= 3:
                    pygame.draw.rect(surf, S.COLOR_BORDER, scaled, width=1)

    def _draw_ball(self, surf, scale: float, pygame) -> None:
        cx = int(self.ball_x * scale)
        cy = int(self.ball_y * scale)
        radius = max(2, int(S.BALL_RADIUS * scale))
        pygame.draw.circle(surf, S.COLOR_BALL, (cx, cy), radius)

    def _draw_paddle(self, surf, scale: float, pygame) -> None:
        pw = max(4, int(S.PADDLE_WIDTH * scale))
        ph = max(3, int(S.PADDLE_HEIGHT * scale))
        px = int(self.paddle_x * scale) - pw // 2
        py = int(self.paddle_y * scale) - ph // 2
        paddle_rect = pygame.Rect(px, py, pw, ph)
        pygame.draw.rect(surf, S.COLOR_PADDLE, paddle_rect)
        pygame.draw.rect(surf, S.COLOR_BORDER, paddle_rect, width=2)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _build_info(
        self,
        bricks_broken_this_step: int,
        life_lost: bool,
        clear: bool,
    ) -> dict:
        return {
            "score": self.score,
            "broken_bricks": self.broken_bricks,
            "remaining_bricks": self._remaining_bricks(),
            "bricks_broken_this_step": bricks_broken_this_step,
            "life_lost": life_lost,
            "lives": self.lives,
            "step_count": self.step_count,
            "clear": clear,
            "terminal_reason": self.terminal_reason,
            "action_name": S.ACTION_NAMES.get(self.last_action, "?"),
            "seed": self._seed,
            "name": self.name,
            "paddle_hit": int(self._paddle_hit_this_step),
            "hit_offset": float(
                self._last_paddle_collision.get("hit_offset", 0.0)
            ) if self._paddle_hit_this_step and self._last_paddle_collision else 0.0,
        }

    def _remaining_bricks(self) -> int:
        return int(self._bricks.sum())

    def _reset_ball(self) -> None:
        self.ball_x = self.paddle_x
        self.ball_y = self.paddle_y - S.PADDLE_HEIGHT / 2 - S.BALL_RADIUS - 8
        angle = float(self.rng.uniform(*S.INITIAL_BALL_ANGLE_RANGE))
        speed = float(self.rng.uniform(S.BALL_SPEED_MIN, S.BALL_SPEED_MAX))
        self.ball_vx = speed * math.sin(angle)
        self.ball_vy = -abs(speed * math.cos(angle))

    def _apply_paddle_action(self, action: int) -> None:
        prev_x = self.paddle_x
        if action == S.ACTION_LEFT:
            self.paddle_x -= S.PADDLE_SPEED * S.FIXED_DT
        elif action == S.ACTION_RIGHT:
            self.paddle_x += S.PADDLE_SPEED * S.FIXED_DT
        half = S.PADDLE_WIDTH / 2.0
        self.paddle_x = float(np.clip(self.paddle_x, half, S.FIELD_WIDTH - half))
        self.paddle_vx = (self.paddle_x - prev_x) / S.FIXED_DT

    def _integrate_ball(self) -> None:
        self.ball_x += self.ball_vx * S.FIXED_DT
        self.ball_y += self.ball_vy * S.FIXED_DT

    def _ball_fell_below(self) -> bool:
        return self.ball_y - S.BALL_RADIUS > S.FIELD_HEIGHT

    def _resolve_collisions(self) -> int:
        hits = 0
        r = S.BALL_RADIUS

        if self.ball_x - r < 0:
            self.ball_x = r
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + r > S.FIELD_WIDTH:
            self.ball_x = S.FIELD_WIDTH - r
            self.ball_vx = -abs(self.ball_vx)

        if self.ball_y - r < 0:
            self.ball_y = r
            self.ball_vy = abs(self.ball_vy)

        paddle = (
            int(self.paddle_x - S.PADDLE_WIDTH / 2),
            int(self.paddle_y - S.PADDLE_HEIGHT / 2),
            S.PADDLE_WIDTH,
            S.PADDLE_HEIGHT,
        )
        px, py, pw, ph = paddle
        if _circle_rect_collision(self.ball_x, self.ball_y, r, paddle) and self.ball_vy > 0:
            self._paddle_hit_this_step = True
            self.ball_y = py - r - 0.5
            paddle_half = S.PADDLE_WIDTH / 2.0
            offset = (self.ball_x - self.paddle_x) / paddle_half
            offset = float(np.clip(offset, -1.0, 1.0))
            ball_vx_before = self.ball_vx
            ball_vy_before = self.ball_vy
            speed = math.hypot(self.ball_vx, self.ball_vy)
            if speed < 1e-6:
                speed = (S.BALL_SPEED_MIN + S.BALL_SPEED_MAX) / 2.0
            if S.PADDLE_ANGLE_CONTROL:
                max_angle = math.radians(S.PADDLE_MAX_BOUNCE_ANGLE_DEG)
                angle = offset * max_angle
                self.ball_vx = speed * math.sin(angle) + S.PADDLE_SPIN_STRENGTH * self.paddle_vx
                self.ball_vy = -abs(speed * math.cos(angle))
            else:
                angle = offset * 0.75
                self.ball_vx = speed * math.sin(angle)
                self.ball_vy = -abs(speed * math.cos(angle))
            new_speed = math.hypot(self.ball_vx, self.ball_vy)
            if new_speed > 1e-6:
                scale = speed / new_speed
                self.ball_vx *= scale
                self.ball_vy *= scale
            self._last_paddle_collision = {
                "step": self.step_count,
                "ball_x": self.ball_x,
                "paddle_center_x": self.paddle_x,
                "hit_offset": offset,
                "paddle_vx": self.paddle_vx,
                "ball_vx_before": ball_vx_before,
                "ball_vy_before": ball_vy_before,
                "ball_vx_after": self.ball_vx,
                "ball_vy_after": self.ball_vy,
            }

        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if not self._bricks[row, col]:
                    continue
                brick = self._layout.brick_rect(row, col)
                if _circle_rect_collision(self.ball_x, self.ball_y, r, brick):
                    self._bricks[row, col] = False
                    hits += 1
                    self._bounce_off_rect(brick)
                    return hits
        return hits

    def _bounce_off_rect(self, rect: Tuple[int, int, int, int]) -> None:
        cx, cy = self.ball_x, self.ball_y
        r = S.BALL_RADIUS
        rx, ry, rw, rh = rect
        left, right, top, bottom = rx, rx + rw, ry, ry + rh
        overlaps = {
            "left": abs((cx + r) - left),
            "right": abs(right - (cx - r)),
            "top": abs((cy + r) - top),
            "bottom": abs(bottom - (cy - r)),
        }
        side = min(overlaps, key=overlaps.get)
        if side in ("left", "right"):
            self.ball_vx = -self.ball_vx
            self.ball_x = (left - r - 0.5) if side == "left" else (right + r + 0.5)
        else:
            self.ball_vy = -self.ball_vy
            self.ball_y = (top - r - 0.5) if side == "top" else (bottom + r + 0.5)


def _smoke_test() -> None:
    env = CatBreakEnv()
    obs = env.reset(seed=0)
    print(f"obs shape: {obs.shape}, obs_dim: {env.obs_dim}")
    rng = np.random.default_rng(0)
    for _ in range(10):
        action = int(rng.integers(0, S.N_ACTIONS))
        obs, reward, done, info = env.step(action)
        if done:
            break
    print(f"final info: {info}")
    env.close()


if __name__ == "__main__":
    _smoke_test()
