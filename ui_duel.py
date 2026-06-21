"""CatBreak RL Arena — human vs agent side-by-side demo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pygame

import settings as S
from agents import BaseAgent, DQNPolicyAgent, KeyboardAgent, find_latest_dqn_checkpoint, make_agent
from catbreak_env import CatBreakEnv, layout_from_checkpoint, obs_mode_from_checkpoint


def keys_to_action(keys: pygame.key.ScancodeWrapper) -> int:
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    if left and not right:
        return S.ACTION_LEFT
    if right and not left:
        return S.ACTION_RIGHT
    return S.ACTION_STAY


def draw_divider(screen: pygame.Surface, x: int, top: int, bottom: int) -> None:
    pygame.draw.rect(screen, S.COLOR_DIVIDER, pygame.Rect(x, top, S.DIVIDER_WIDTH, bottom - top))


def draw_header(
    screen: pygame.Surface,
    font: pygame.font.Font,
    agent: BaseAgent,
    seed: int,
    paused: bool,
    agent_action: int,
    layout_mode: str,
) -> None:
    screen.blit(font.render("CatBreak RL Arena", True, S.COLOR_TEXT), (24, 12))
    subtitle = (
        f"Human (left)  vs  {agent.name} (right)  |  seed={seed}"
        + f"  |  layout={layout_mode}"
        + f"  |  agent_action={S.ACTION_NAMES.get(agent_action, '?')}"
        + ("  [PAUSED]" if paused else "")
    )
    screen.blit(font.render(subtitle, True, S.COLOR_TEXT_DIM), (24, 38))


def draw_footer(screen: pygame.Surface, font: pygame.font.Font) -> None:
    text = "Course demo. Octocat asset belongs to GitHub."
    surf = font.render(text, True, S.COLOR_TEXT_DIM)
    rect = surf.get_rect()
    rect.midbottom = (S.WINDOW_WIDTH // 2, S.WINDOW_HEIGHT - 6)
    screen.blit(surf, rect)


def draw_controls_hint(screen: pygame.Surface, font: pygame.font.Font) -> None:
    hint = (
        "Arrows/A-D: move  |  SPACE: pause  |  R: reset  |  N: new seed  |  "
        "1: Random  |  2: FollowBall  |  8: CEM-Aim  |  5: CEM-MPC  |  "
        "4: DQN  |  9: RLPolicy  |  ESC: quit"
    )
    screen.blit(font.render(hint, True, S.COLOR_TEXT_DIM), (24, S.WINDOW_HEIGHT - S.FOOTER_HEIGHT - 20))


def panel_rects() -> tuple[pygame.Rect, pygame.Rect]:
    top = S.HEADER_HEIGHT
    bottom = S.WINDOW_HEIGHT - S.FOOTER_HEIGHT - 24
    height = bottom - top
    half_w = (S.WINDOW_WIDTH - S.DIVIDER_WIDTH) // 2
    left = pygame.Rect(0, top, half_w, height)
    right = pygame.Rect(half_w + S.DIVIDER_WIDTH, top, half_w, height)
    return left, right


def make_env_pair(layout_mode: str, agent_obs_mode: str = S.DEFAULT_OBS_MODE) -> tuple[CatBreakEnv, CatBreakEnv]:
    human_config = {"layout": layout_mode, "obs_mode": S.OBS_MODE_VECTOR}
    agent_config = {"layout": layout_mode, "obs_mode": agent_obs_mode}
    return (
        CatBreakEnv(config=human_config, name="Human"),
        CatBreakEnv(config=agent_config, name="Agent"),
    )


def resolve_layout(
    agent_name: str,
    model_path: str | None,
    agent: BaseAgent,
) -> str:
    """Pick env layout so observations match the right-side policy."""
    if isinstance(agent, DQNPolicyAgent) and agent._dqn is not None:
        extra = getattr(agent._dqn, "_last_extra", None)
        if isinstance(extra, dict) and extra.get("layout") in (S.LAYOUT_RECT, S.LAYOUT_CAT):
            return extra["layout"]

    if agent_name in ("dqn", "rl"):
        path = model_path
        if path is None:
            found = find_latest_dqn_checkpoint()
            path = str(found) if found else None
        if path and Path(path).exists():
            return layout_from_checkpoint(path)

    return S.DEFAULT_LAYOUT


def resolve_agent_obs_mode(
    agent_name: str,
    model_path: str | None,
    agent: BaseAgent,
) -> str:
    if isinstance(agent, DQNPolicyAgent) and agent._dqn is not None:
        return agent._dqn.obs_mode
    if agent_name in ("dqn", "rl"):
        path = model_path
        if path is None:
            found = find_latest_dqn_checkpoint()
            path = str(found) if found else None
        if path and Path(path).exists():
            return obs_mode_from_checkpoint(path)
    return S.DEFAULT_OBS_MODE


def reset_duel(
    seed: int,
    human_env: CatBreakEnv,
    agent_env: CatBreakEnv,
    human_agent: KeyboardAgent,
    right_agent: BaseAgent,
) -> None:
    human_env.reset(seed=seed)
    agent_env.reset(seed=seed)
    human_agent.reset(seed=seed)
    right_agent.reset(seed=seed + 1)


def main(
    agent_name: str = "cem_aim",
    model_path: str | None = str(S.CEM_AIM_V3_VAL_BEST),
    seed: int = 6,
) -> None:
    pygame.init()
    screen = pygame.display.set_mode((S.WINDOW_WIDTH, S.WINDOW_HEIGHT))
    pygame.display.set_caption("CatBreak RL Arena")
    clock = pygame.time.Clock()

    font_header = pygame.font.SysFont("monospace", 22, bold=True)
    font_footer = pygame.font.SysFont("monospace", 12)

    human_agent = KeyboardAgent()
    right_agent: BaseAgent = make_agent(agent_name, model_path=model_path)
    layout_mode = resolve_layout(agent_name, model_path, right_agent)
    agent_obs_mode = resolve_agent_obs_mode(agent_name, model_path, right_agent)
    human_env, agent_env = make_env_pair(layout_mode, agent_obs_mode=agent_obs_mode)
    if layout_mode != S.LAYOUT_CAT:
        print(
            f"Using layout={layout_mode} obs_mode={agent_obs_mode} "
            f"(obs_dim={agent_env.obs_dim}) to match checkpoint."
        )

    paused = False
    agent_action = S.ACTION_STAY
    reset_duel(seed, human_env, agent_env, human_agent, right_agent)

    def switch_agent(name: str, ckpt: str | None = None) -> None:
        nonlocal right_agent, human_env, agent_env, layout_mode, agent_obs_mode, agent_action
        if name in ("cem_mpc", "cem-mpc", "mpc"):
            print(
                "CEM-MPC planning is CPU-heavy in UI; using reduced horizon=10, "
                "population=48, iterations=2."
            )
            right_agent = make_agent(
                "cem_mpc",
                planner_kwargs={
                    "horizon": 10,
                    "population_size": 48,
                    "iterations": 2,
                    "env_config": {"layout": layout_mode},
                },
            )
        else:
            right_agent = make_agent(name, model_path=ckpt)
        new_layout = resolve_layout(name, ckpt, right_agent)
        new_obs_mode = resolve_agent_obs_mode(name, ckpt, right_agent)
        if new_layout != layout_mode or new_obs_mode != agent_obs_mode:
            human_env.close()
            agent_env.close()
            layout_mode = new_layout
            agent_obs_mode = new_obs_mode
            human_env, agent_env = make_env_pair(layout_mode, agent_obs_mode=agent_obs_mode)
            print(
                f"Switched layout={layout_mode} obs_mode={agent_obs_mode} "
                f"(obs_dim={agent_env.obs_dim})."
            )
        agent_action = S.ACTION_STAY
        reset_duel(seed, human_env, agent_env, human_agent, right_agent)

    running = True
    while running:
        clock.tick(S.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_duel(seed, human_env, agent_env, human_agent, right_agent)
                elif event.key == pygame.K_n:
                    seed = int(np.random.default_rng().integers(0, 1_000_000))
                    reset_duel(seed, human_env, agent_env, human_agent, right_agent)
                elif event.key == pygame.K_1:
                    switch_agent("random")
                elif event.key == pygame.K_2:
                    switch_agent("follow")
                elif event.key == pygame.K_8:
                    switch_agent("cem_aim", model_path)
                elif event.key == pygame.K_5:
                    switch_agent("cem_mpc")
                elif event.key == pygame.K_4:
                    switch_agent("dqn", model_path)
                elif event.key == pygame.K_9:
                    switch_agent("rl", model_path)

        if not paused:
            human_agent.set_action(keys_to_action(pygame.key.get_pressed()))
            if not human_env.done:
                human_env.step(human_agent.act(human_env.get_obs(), env=human_env))
            if not agent_env.done:
                obs = agent_env.get_obs()
                agent_action = right_agent.act(obs, agent_env.last_info, env=agent_env)
                agent_env.step(agent_action)

        screen.fill(S.COLOR_BG)
        draw_header(
            screen, font_header, right_agent, seed, paused, agent_action, layout_mode
        )
        draw_controls_hint(screen, font_footer)

        left_rect, right_rect = panel_rects()
        draw_divider(screen, left_rect.right, left_rect.top, left_rect.bottom)

        human_env.render_surface(
            screen, left_rect, title="HUMAN", current_action=human_agent.current_action
        )
        agent_env.render_surface(
            screen,
            right_rect,
            title=f"AGENT ({right_agent.name})",
            current_action=agent_action,
        )
        draw_footer(screen, font_footer)
        pygame.display.flip()

    human_env.close()
    agent_env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CatBreak human vs agent demo.")
    parser.add_argument("--agent", type=str, default="cem_aim", help="right panel agent")
    parser.add_argument(
        "--model",
        type=str,
        default=str(S.CEM_AIM_V3_VAL_BEST),
        help="checkpoint for cem_aim/dqn/rl",
    )
    parser.add_argument("--seed", type=int, default=6)
    cli = parser.parse_args()
    main(agent_name=cli.agent, model_path=cli.model, seed=cli.seed)
