"""Brick layouts for CatBreak — rectangular default and optional Octocat outline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import settings as S


@dataclass(frozen=True)
class BrickLayout:
    rows: int
    cols: int
    mask: np.ndarray
    colors: np.ndarray
    brick_width: float
    brick_height: float
    brick_gap: float
    origin_x: float
    origin_y: float
    image_missing: bool = False

    @property
    def total_bricks(self) -> int:
        return int(self.mask.sum())

    def brick_rect(self, row: int, col: int) -> Tuple[int, int, int, int]:
        x = self.origin_x + col * (self.brick_width + self.brick_gap)
        y = self.origin_y + row * (self.brick_height + self.brick_gap)
        return int(x), int(y), int(self.brick_width), int(self.brick_height)


def make_rectangular_layout() -> BrickLayout:
    rows, cols = S.BRICK_ROWS, S.BRICK_COLS
    mask = np.ones((rows, cols), dtype=bool)
    colors = np.zeros((rows, cols, 3), dtype=np.uint8)
    for r in range(rows):
        colors[r, :, :] = S.BRICK_COLORS[r % len(S.BRICK_COLORS)]

    gap = float(S.BRICK_GAP)
    bw, bh = float(S.BRICK_WIDTH), float(S.BRICK_HEIGHT)
    grid_w = cols * bw + (cols - 1) * gap
    origin_x = (S.FIELD_WIDTH - grid_w) / 2.0
    origin_y = float(S.BRICK_TOP_MARGIN)

    return BrickLayout(
        rows=rows,
        cols=cols,
        mask=mask,
        colors=colors,
        brick_width=bw,
        brick_height=bh,
        brick_gap=gap,
        origin_x=origin_x,
        origin_y=origin_y,
        image_missing=False,
    )


def _load_cat_pixels() -> Optional[np.ndarray]:
    """Load cat image pixels; pygame is imported lazily."""
    if not S.CAT_IMAGE_PATH.exists():
        return None
    try:
        import pygame

        if not pygame.get_init():
            pygame.init()
        if pygame.display.get_surface() is None:
            try:
                pygame.display.set_mode((1, 1), pygame.HIDDEN)
            except (pygame.error, TypeError):
                pygame.display.set_mode((1, 1))
        surface = pygame.image.load(str(S.CAT_IMAGE_PATH)).convert()
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
    except Exception:
        return None


def _sample_silhouette(pixels: np.ndarray, grid_size: int, threshold: float) -> np.ndarray:
    img_h, img_w = pixels.shape[:2]
    silhouette = np.zeros((grid_size, grid_size), dtype=bool)
    row_centers = np.clip(
        (np.arange(grid_size) * 2 + 1) * img_h // (2 * grid_size), 0, img_h - 1
    )
    col_centers = np.clip(
        (np.arange(grid_size) * 2 + 1) * img_w // (2 * grid_size), 0, img_w - 1
    )
    for r in range(grid_size):
        cy = int(row_centers[r])
        for c in range(grid_size):
            cx = int(col_centers[c])
            if float(pixels[cy, cx].mean()) < threshold:
                silhouette[r, c] = True
    return silhouette


def _silhouette_to_outline(silhouette: np.ndarray) -> np.ndarray:
    rows, cols = silhouette.shape
    outline = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if not silhouette[r, c]:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or not silhouette[nr, nc]:
                    outline[r, c] = True
                    break
    return outline


def _outline_colors(outline: np.ndarray) -> np.ndarray:
    rows, cols = outline.shape
    colors = np.zeros((rows, cols, 3), dtype=np.uint8)
    white = np.array(S.CAT_OUTLINE_WHITE, dtype=np.uint8)
    gray = np.array(S.CAT_OUTLINE_GRAY, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if outline[r, c]:
                colors[r, c] = white if (r + c) % 2 == 0 else gray
    return colors


def make_cat_layout() -> BrickLayout:
    """Octocat outline from github_cat.png; falls back to rectangular grid."""
    pixels = _load_cat_pixels()
    if pixels is None:
        layout = make_rectangular_layout()
        return BrickLayout(
            rows=layout.rows,
            cols=layout.cols,
            mask=layout.mask,
            colors=layout.colors,
            brick_width=layout.brick_width,
            brick_height=layout.brick_height,
            brick_gap=layout.brick_gap,
            origin_x=layout.origin_x,
            origin_y=layout.origin_y,
            image_missing=True,
        )

    grid_size = S.CAT_GRID_SIZE
    silhouette = _sample_silhouette(pixels, grid_size, S.CAT_LUMINANCE_THRESHOLD)
    mask = _silhouette_to_outline(silhouette)
    colors = _outline_colors(mask)

    gap = float(S.CAT_BRICK_GAP)
    brick_size = float(S.CAT_BRICK_PIXEL)
    grid_w = grid_size * brick_size + (grid_size - 1) * gap
    grid_h = grid_size * brick_size + (grid_size - 1) * gap
    origin_x = (S.FIELD_WIDTH - grid_w) / 2.0
    paddle_top = S.FIELD_HEIGHT - S.PADDLE_Y_OFFSET - S.PADDLE_HEIGHT / 2.0
    origin_y = max(float(S.CAT_TOP_MARGIN), paddle_top - S.CAT_BOTTOM_GAP - grid_h)

    return BrickLayout(
        rows=grid_size,
        cols=grid_size,
        mask=mask,
        colors=colors,
        brick_width=brick_size,
        brick_height=brick_size,
        brick_gap=gap,
        origin_x=origin_x,
        origin_y=origin_y,
        image_missing=False,
    )


def get_layout(mode: str = S.DEFAULT_LAYOUT) -> BrickLayout:
    if mode == S.LAYOUT_CAT:
        return make_cat_layout()
    return make_rectangular_layout()
