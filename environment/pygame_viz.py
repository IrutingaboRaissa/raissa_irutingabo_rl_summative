"""
Pygame-based GUI Visualization for NutriVision Environment
Warm food-inspired palette (kitchen / plate / fresh produce) for intuitive nutrition context.
"""

from __future__ import annotations

import math
import pygame
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

from environment.custom_env import NutriVisionEnv

_MACRO_ICONS = {
    "Calories": "\N{FIRE}",
    "Protein": "\N{CUT OF MEAT}",
    "Carbs": "\N{COOKED RICE}",
    "Fats": "\N{AVOCADO}",
}

_CLIENT_NAMES = (
    "Amara",
    "Kwame",
    "Zola",
    "Thandi",
    "Kofi",
    "Nia",
    "Jelani",
    "Eshe",
)


@dataclass(frozen=True)
class _UILayout:
    pad: int
    gap: int
    header_h: int
    content_top: int
    panel_h: int
    person_w: int
    w_macro: int
    w_food: int
    w_reward: int
    x_person: int
    x_macro: int
    x_food: int
    x_reward: int
    legend_y: int
    legend_h: int
    timeline_y: int
    timeline_pad: int


def _lerp_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


class NutriVisionVisualizer:
    """Real-time 2D visualization of NutriVision environment using Pygame."""

    def __init__(self, width: int = 1280, height: int = 720):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("RL meal suggestions · nutrition goals")

        self.clock = pygame.time.Clock()
        self.anim_frame = 0
        self.font_headline = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 26)
        self.font_medium = pygame.font.Font(None, 21)
        self.font_small = pygame.font.Font(None, 17)
        self.font_tiny = pygame.font.Font(None, 14)
        self.font_metric = pygame.font.Font(None, 30)
        self.font_mono = pygame.font.SysFont("consolas", 13)
        if self.font_mono is None or self.font_mono.get_height() < 8:
            self.font_mono = pygame.font.SysFont("couriernew", 13)
        if self.font_mono is None or self.font_mono.get_height() < 8:
            self.font_mono = self.font_small

        # Warm “kitchen & plate”: espresso backdrop, cream type, honey trim, produce semantics
        self.colors = {
            "bg_top": (58, 44, 38),
            "bg_bottom": (32, 26, 22),
            "card": (48, 42, 38),
            "card_alt": (54, 48, 44),
            "card_top": (52, 46, 42),
            "card_bottom": (40, 36, 32),
            "card_shadow": (12, 10, 8),
            "card_border": (200, 155, 88),
            "positive": (98, 178, 108),
            "negative": (212, 98, 108),
            "neutral": (118, 162, 205),
            "warning": (232, 178, 72),
            "text": (252, 246, 236),
            "muted": (168, 156, 142),
            "white": (255, 255, 255),
            "gold": (240, 198, 108),
            "title_outline": (24, 18, 14),
            "stripe": (218, 108, 82),
            "track": (58, 50, 46),
            "timeline_bg": (44, 38, 34),
            "timeline_fill": (218, 165, 72),
            "grid_line": (120, 108, 98),
            "spark_line": (140, 205, 130),
            "projected_cool": (108, 148, 188),
            "header_headline": (255, 250, 245),
            "header_metric": (255, 248, 200),
            "macro_on_target": (88, 188, 118),
            "macro_warn": (228, 188, 72),
            "macro_over": (218, 86, 92),
            "log_highlight": (200, 230, 255),
            "transition_dim": (140, 132, 122),
        }

        self.macro_bar_colors = {
            "Calories": (232, 118, 72),
            "Protein": (200, 112, 132),
            "Carbs": (228, 188, 88),
            "Fats": (132, 158, 88),
        }

        self.goal_themes = {
            0: {
                "name": "Weight Loss",
                "header_left": (52, 118, 88),
                "header_right": (92, 168, 128),
                "accent": (120, 198, 140),
            },
            1: {
                "name": "Weight Gain",
                "header_left": (168, 88, 58),
                "header_right": (228, 148, 72),
                "accent": (245, 178, 92),
            },
            2: {
                "name": "Maintenance",
                "header_left": (98, 118, 158),
                "header_right": (148, 138, 178),
                "accent": (168, 188, 220),
            },
        }

        self.action_legend = [
            ("0", "Accept (macro reward)", self.colors["positive"]),
            ("1", "Lower cal (−0.5)", self.colors["warning"]),
            ("2", "More protein (−0.5)", self.colors["neutral"]),
            ("3", "Skip (0)", (118, 108, 98)),
        ]

        self._reward_history: List[float] = []
        self._cum_reward_history: List[float] = []
        self._alignment_history: List[float] = []
        self._viz_reset_seen = False
        self._client_idx = 0
        self._client_name = _CLIENT_NAMES[0]
        self._last_hist_key: tuple[int, int, float] | None = None
        self._macro_fill_smooth: Dict[str, float] = {}
        self._viz_obs_before_step: np.ndarray | None = None
        self._transition_log: Deque[dict] = deque(maxlen=3)
        self._recent_actions: Deque[int] = deque(maxlen=12)
        self._bar_lerp = 0.38

        self.running = True

    def _compute_layout(self) -> _UILayout:
        """Fit all panels + legend + timeline inside the current window."""
        pad = max(6, min(12, self.width // 120))
        gap = max(10, min(16, self.width // 110))
        hdr = min(86, max(64, int(self.height * 0.105)))
        leg_h = min(128, max(88, int(self.height * 0.155)))
        tl_reserve = 30
        panel_h = self.height - hdr - leg_h - tl_reserve - 8
        panel_h = max(168, min(288, panel_h))

        content_top = hdr + 2
        legend_y = content_top + panel_h + 6
        timeline_y = self.height - 16

        inner = self.width - 2 * pad
        person_w = max(136, min(196, int(inner * 0.145)))
        rest = inner - person_w - 3 * gap
        cw = rest // 3
        rem = rest - cw * 3
        w_macro = cw + (1 if rem >= 1 else 0)
        w_food = cw + (1 if rem >= 2 else 0)
        w_reward = rest - w_macro - w_food

        x_person = pad
        x_macro = x_person + person_w + gap
        x_food = x_macro + w_macro + gap
        x_reward = x_food + w_food + gap

        return _UILayout(
            pad=pad,
            gap=gap,
            header_h=hdr,
            content_top=content_top,
            panel_h=panel_h,
            person_w=person_w,
            w_macro=w_macro,
            w_food=w_food,
            w_reward=w_reward,
            x_person=x_person,
            x_macro=x_macro,
            x_food=x_food,
            x_reward=x_reward,
            legend_y=legend_y,
            legend_h=leg_h,
            timeline_y=timeline_y,
            timeline_pad=max(10, pad + 4),
        )

    def _on_fresh_episode(self, env: NutriVisionEnv) -> bool:
        return (
            env.step_count == 0
            and env.num_meals_logged == 0
            and env.daily_calories < 1e-6
            and env.episode_reward == 0.0
        )

    def _sync_client_and_history(self, env: NutriVisionEnv) -> None:
        if self._on_fresh_episode(env):
            if not self._viz_reset_seen:
                self._reward_history.clear()
                self._cum_reward_history.clear()
                self._alignment_history.clear()
                self._macro_fill_smooth.clear()
                self._transition_log.clear()
                self._recent_actions.clear()
                self._last_hist_key = None
                self._viz_obs_before_step = None
                self._client_idx = (self._client_idx + 1) % len(_CLIENT_NAMES)
                self._client_name = _CLIENT_NAMES[self._client_idx]
                self._viz_reset_seen = True
        else:
            self._viz_reset_seen = False

    @staticmethod
    def _fmt_obs_vec(obs: np.ndarray, dims: int = 5) -> str:
        flat = np.asarray(obs, dtype=np.float64).ravel()
        parts = [f"{float(flat[i]):.0f}" for i in range(min(dims, flat.size))]
        return "[" + ",".join(parts) + (",…]" if flat.size > dims else "]")

    def _macro_ratio_color(self, ratio: float) -> Tuple[int, int, int]:
        """Green on target, yellow approaching, red over limit (vs daily target)."""
        g = self.colors["macro_on_target"]
        y = self.colors["macro_warn"]
        r = self.colors["macro_over"]
        if ratio < 0.82:
            return _lerp_color(y, g, max(0.0, ratio / 0.82))
        if ratio <= 1.0:
            return _lerp_color(y, g, 0.35 + 0.65 * ((ratio - 0.82) / max(0.18, 1e-6)))
        if ratio <= 1.12:
            return _lerp_color(g, y, (ratio - 1.0) / 0.12)
        if ratio <= 1.35:
            return _lerp_color(y, r, (ratio - 1.12) / max(0.23, 1e-6))
        return r

    def _note_step_for_viz(
        self,
        env: NutriVisionEnv,
        action: int | None,
        reward: float,
        obs_before: np.ndarray | None,
        obs_after: np.ndarray,
    ) -> None:
        if action is None:
            return
        key = (env.step_count, int(action), round(float(reward), 4))
        if key == self._last_hist_key:
            return
        self._last_hist_key = key
        self._reward_history.append(float(reward))
        if len(self._reward_history) > 36:
            self._reward_history = self._reward_history[-36:]
        self._cum_reward_history.append(float(env.episode_reward))
        if len(self._cum_reward_history) > 36:
            self._cum_reward_history = self._cum_reward_history[-36:]
        self._alignment_history.append(float(self._wellbeing_logged(env)))
        if len(self._alignment_history) > 36:
            self._alignment_history = self._alignment_history[-36:]
        self._recent_actions.append(int(action))
        if obs_before is not None:
            self._transition_log.append(
                {
                    "step": env.step_count,
                    "s": self._fmt_obs_vec(obs_before),
                    "a": int(action),
                    "r": float(reward),
                    "sp": self._fmt_obs_vec(obs_after),
                }
            )

    def _avg_macro_deviation(
        self,
        cal: float,
        p: float,
        c: float,
        f: float,
        env: NutriVisionEnv,
    ) -> float:
        cal_r = cal / max(env.calorie_target, 1e-6)
        pr = p / max(env.protein_target, 1e-6)
        cr = c / max(env.carbs_target, 1e-6)
        fr = f / max(env.fats_target, 1e-6)
        return (abs(cal_r - 1.0) + abs(pr - 1.0) + abs(cr - 1.0) + abs(fr - 1.0)) / 4.0

    def _wellbeing_logged(self, env: NutriVisionEnv) -> float:
        """0..1 from current logged day vs targets (same spirit as env macro ratios)."""
        d = self._avg_macro_deviation(
            env.daily_calories,
            env.daily_protein,
            env.daily_carbs,
            env.daily_fats,
            env,
        )
        return max(0.0, min(1.0, 1.0 - min(d, 1.0)))

    def _wellbeing_if_accept(self, env: NutriVisionEnv) -> float:
        food = env.current_food
        return self._wellbeing_from_totals(
            env.daily_calories + food["cal"],
            env.daily_protein + food["protein"],
            env.daily_carbs + food["carbs"],
            env.daily_fats + food["fat"],
            env,
        )

    def _wellbeing_from_totals(
        self,
        cal: float,
        p: float,
        c: float,
        f: float,
        env: NutriVisionEnv,
    ) -> float:
        d = self._avg_macro_deviation(cal, p, c, f, env)
        return max(0.0, min(1.0, 1.0 - min(d, 1.0)))

    def _draw_client_icon(self, cx: int, cy: int, wellbeing: float, phase: float) -> None:
        """Small user glyph: supersampled outline + soft badge (reads clearly on dark cards)."""
        k = 2
        W, H = 34 * k, 40 * k
        sfc = pygame.Surface((W, H), pygame.SRCALPHA)
        ox = W // 2
        oy = H // 2 + int(k * math.sin(phase * 2.0))

        honey = (228, 178, 92)
        basil = (120, 185, 125)
        rim = _lerp_color(honey, basil, wellbeing)
        ink = (28, 22, 18)
        fill_badge = (*_lerp_color((58, 48, 42), (52, 72, 55), wellbeing), 150)
        glow_ring = (*_lerp_color((180, 140, 75), rim, 0.5), 95)

        pygame.draw.circle(sfc, fill_badge, (ox, oy), 15 * k)
        pygame.draw.circle(sfc, glow_ring, (ox, oy), 15 * k, k)

        head_r = 5 * k
        head_cy = oy - 8 * k
        pygame.draw.circle(sfc, ink, (ox, head_cy), head_r + k, 3 * k)
        pygame.draw.circle(sfc, rim, (ox, head_cy), head_r, 2 * k)

        aw, ah = 24 * k, 12 * k
        arc_top = oy - 2 * k
        arc_rect = pygame.Rect(ox - aw // 2, arc_top, aw, ah)
        pygame.draw.arc(sfc, ink, arc_rect.inflate(2 * k, 2 * k), math.pi, 2 * math.pi, 3 * k)
        pygame.draw.arc(sfc, rim, arc_rect, math.pi, 2 * math.pi, 2 * k)

        shelf_y = arc_rect.bottom - k
        pygame.draw.line(
            sfc,
            rim,
            (arc_rect.left + 3 * k, shelf_y),
            (arc_rect.right - 3 * k, shelf_y),
            2 * k,
        )

        out_w, out_h = W // k, H // k
        small = pygame.transform.smoothscale(sfc, (out_w, out_h))
        self.screen.blit(small, small.get_rect(center=(cx, cy)))

    def _draw_reward_sparkline(self, rect: pygame.Rect, color: Tuple[int, int, int] | None = None) -> None:
        hist = self._reward_history
        if len(hist) < 2:
            return
        lo = min(hist)
        hi = max(hist)
        span = max(hi - lo, 0.5)
        col = color if color is not None else self.colors["spark_line"]
        pts = []
        for i, v in enumerate(hist):
            t = i / max(len(hist) - 1, 1)
            px = rect.x + int(t * (rect.w - 4)) + 2
            nv = (v - lo) / span
            py = rect.bottom - 4 - int(nv * (rect.h - 8))
            pts.append((px, py))
        pygame.draw.lines(self.screen, col, False, pts, 2)

    def _draw_scalar_series_sparkline(
        self,
        rect: pygame.Rect,
        hist: List[float],
        color: Tuple[int, int, int],
        *,
        pad_y: int = 4,
    ) -> None:
        if len(hist) < 2:
            return
        lo = min(hist)
        hi = max(hist)
        span = max(hi - lo, 1e-6)
        pts = []
        for i, v in enumerate(hist):
            t = i / max(len(hist) - 1, 1)
            px = rect.x + int(t * (rect.w - 4)) + 2
            nv = (v - lo) / span
            py = rect.bottom - pad_y - int(nv * (rect.h - 2 * pad_y))
            pts.append((px, py))
        pygame.draw.lines(self.screen, color, False, pts, 2)

    def _draw_obs_dim_strip(self, x: int, y: int, obs: np.ndarray, max_w: int) -> None:
        flat = np.asarray(obs, dtype=np.float64).ravel()
        n = min(15, flat.size)
        if n <= 0:
            return
        cell = max(4, min(8, max_w // max(n, 1)))
        gap = max(1, cell // 5)
        for i in range(n):
            t = max(0.0, min(1.0, float(flat[i]) / 5000.0))
            cx = x + i * (cell + gap) + cell // 2
            base = _lerp_color(self.colors["track"], self.colors["timeline_fill"], t)
            pygame.draw.circle(self.screen, base, (cx, y), max(2, cell // 2))
            pygame.draw.circle(self.screen, self.colors["grid_line"], (cx, y), max(2, cell // 2), 1)

    def draw_person_section(
        self,
        env: NutriVisionEnv,
        lay: _UILayout,
        obs_vec: np.ndarray,
    ) -> None:
        x, y = lay.x_person, lay.content_top
        section_w, section_h = lay.person_w, lay.panel_h
        rect = pygame.Rect(x, y, section_w, section_h)
        self._draw_panel_card(rect)

        ix = max(8, section_w // 18)
        title = self.font_medium.render("Client", True, self.colors["text"])
        self.screen.blit(title, (x + ix, y + 6))

        name_surf = self.font_large.render(self._client_name, True, self.colors["gold"])
        self.screen.blit(name_surf, (x + ix, y + 28))

        gtheme = self.goal_themes.get(env.goal_type, self.goal_themes[2])
        gnames = ["Weight loss", "Weight gain", "Maintenance"]
        gshort = ["Loss", "Gain", "Maint."]
        icon_reserve = 52
        gline = f"Goal: {gnames[env.goal_type]}"
        goal_surf = self.font_small.render(gline, True, self.colors["muted"])
        if goal_surf.get_width() > section_w - 2 * ix - icon_reserve:
            gline = f"Goal: {gshort[env.goal_type]}"
            goal_surf = self.font_small.render(gline, True, self.colors["muted"])
        dot_c = (x + ix + 5, y + 59)
        pygame.draw.circle(self.screen, gtheme["accent"], dot_c, 6)
        pygame.draw.circle(self.screen, self.colors["title_outline"], dot_c, 6, 1)
        self.screen.blit(goal_surf, (x + ix + 16, y + 54))

        wb_now = self._wellbeing_logged(env)
        wb_acc = self._wellbeing_if_accept(env)

        phase = self.anim_frame * 0.07
        icon_cx = x + section_w - ix - 8
        icon_cy = y + 28
        self._draw_client_icon(icon_cx, icon_cy, wb_now, phase)

        bar_w = max(60, section_w - 2 * ix)
        by = y + int(section_h * 0.38)
        by = min(by, y + section_h - 100)
        self.screen.blit(self.font_tiny.render("Today", True, self.colors["muted"]), (x + ix, by))
        track = pygame.Rect(x + ix, by + 12, bar_w, 8)
        pygame.draw.rect(self.screen, self.colors["track"], track, border_radius=5)
        if wb_now > 0:
            pygame.draw.rect(
                self.screen,
                _lerp_color(self.colors["warning"], self.colors["positive"], wb_now),
                pygame.Rect(track.x, track.y, int(bar_w * wb_now), track.h),
                border_radius=5,
            )
        self.screen.blit(
            self.font_tiny.render(f"{int(wb_now * 100)}%", True, self.colors["muted"]),
            (x + ix, by + 22),
        )

        self.screen.blit(self.font_tiny.render("+ meal", True, self.colors["muted"]), (x + ix, by + 32))
        track2 = pygame.Rect(x + ix, by + 44, bar_w, 8)
        pygame.draw.rect(self.screen, self.colors["track"], track2, border_radius=5)
        if wb_acc > 0:
            pygame.draw.rect(
                self.screen,
                _lerp_color(self.colors["projected_cool"], self.colors["timeline_fill"], wb_acc),
                pygame.Rect(track2.x, track2.y, int(bar_w * wb_acc), track2.h),
                border_radius=5,
            )
        self.screen.blit(
            self.font_tiny.render(f"{int(wb_acc * 100)}%", True, self.colors["muted"]),
            (x + ix, by + 54),
        )

        ly = by + 62

        bw = section_w - 2 * ix
        spark_h = max(8, min(11, section_h // 22))
        label_h = 11
        bottom_pad = 6
        spark2_top = y + section_h - bottom_pad - spark_h
        spark1_top = spark2_top - spark_h - label_h - 2
        spark1_top = max(spark1_top, ly + 4)
        spark2_top = max(spark2_top, spark1_top + spark_h + label_h + 2)
        obs_label_y = spark1_top - label_h - 12
        if obs_label_y < ly + 2:
            obs_label_y = ly + 2
            spark1_top = obs_label_y + label_h + 12
            spark2_top = max(spark2_top, spark1_top + spark_h + label_h + 2)
        if spark2_top + spark_h > y + section_h - bottom_pad:
            spark2_top = y + section_h - bottom_pad - spark_h
            spark1_top = min(spark1_top, spark2_top - spark_h - label_h - 4)

        self.screen.blit(
            self.font_mono.render("obs ×15", True, self.colors["muted"]),
            (x + ix, obs_label_y),
        )
        self._draw_obs_dim_strip(x + ix, obs_label_y + 12, obs_vec, bw)

        self.screen.blit(
            self.font_tiny.render("Align", True, self.colors["muted"]),
            (x + ix, spark1_top - label_h),
        )
        spark_a = pygame.Rect(x + ix, spark1_top, bw, spark_h)
        pygame.draw.rect(self.screen, self.colors["track"], spark_a, border_radius=5)
        self._draw_scalar_series_sparkline(
            spark_a, self._alignment_history, self.colors["macro_on_target"], pad_y=3
        )

        self.screen.blit(
            self.font_tiny.render("Reward", True, self.colors["muted"]),
            (x + ix, spark2_top - label_h),
        )
        spark_r = pygame.Rect(x + ix, spark2_top, bw, spark_h)
        pygame.draw.rect(self.screen, self.colors["track"], spark_r, border_radius=5)
        self._draw_scalar_series_sparkline(
            spark_r, self._cum_reward_history, self.colors["gold"], pad_y=3
        )
        self._draw_reward_sparkline(spark_r, color=self.colors["spark_line"])

    def _blit_text_outline(
        self,
        surf: pygame.Surface,
        font: pygame.font.Font,
        text: str,
        pos: Tuple[int, int],
        fg: Tuple[int, int, int],
        outline: Tuple[int, int, int] | None = None,
        width: int = 2,
    ) -> None:
        if outline is None:
            outline = self.colors["title_outline"]
        x, y = pos
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                if dx == 0 and dy == 0:
                    continue
                t = font.render(text, True, outline)
                surf.blit(t, (x + dx, y + dy))
        surf.blit(font.render(text, True, fg), (x, y))

    def _draw_arcade_stars(self, t: float) -> None:
        for i in range(36):
            sx = (i * 137 + int(t * 14)) % (self.width + 40) - 20
            sy = 105 + (i * 73) % (self.height - 130)
            tw = 0.5 + 0.5 * math.sin(t * 2.4 + i * 0.65)
            r = int(1 + tw * 2)
            c = (255, 228, 168) if tw > 0.55 else (180, 200, 150)
            pygame.draw.circle(self.screen, c, (sx, sy), r)

    def _step_reward_explanation(self, action_idx: int | None) -> tuple[str, str]:
        """Two short lines tied to NutriVisionEnv._compute_reward."""
        if action_idx is None:
            return (
                "No move yet — reward shows after each env step.",
                "Bars = logged intake so far (after accepts).",
            )
        if action_idx == 0:
            return (
                "Accept: ~0-5 from macro fit vs targets; +2 extra on loss/gain when rules match.",
                "Uses projected day totals if this meal is logged.",
            )
        if action_idx in (1, 2):
            return (
                "Alternative: fixed −0.5 — meal not logged; new dish next step.",
                "Same penalty for lower-cal or higher-protein requests.",
            )
        return ("Skip: reward 0 — neutral; meal not logged.", "")

    def _fill_vertical_gradient(self, rect: pygame.Rect, top: Tuple[int, int, int], bottom: Tuple[int, int, int]) -> None:
        x, y, w, h = rect.x, rect.y, rect.w, rect.h
        if h <= 0:
            return
        for row in range(h):
            t = row / max(h - 1, 1)
            c = _lerp_color(top, bottom, t)
            pygame.draw.line(self.screen, c, (x, y + row), (x + w - 1, y + row))

    def _rounded_rect(
        self,
        rect: pygame.Rect,
        color: Tuple[int, int, int],
        *,
        radius: int = 14,
        border: int = 0,
        border_color: Tuple[int, int, int] | None = None,
    ) -> None:
        if border > 0 and border_color:
            pygame.draw.rect(self.screen, border_color, rect, width=border, border_radius=radius)
            inner = rect.inflate(-2 * border, -2 * border)
            pygame.draw.rect(self.screen, color, inner, border_radius=max(radius - border, 0))
        else:
            pygame.draw.rect(self.screen, color, rect, border_radius=radius)

    def _panel_shadow(self, rect: pygame.Rect, radius: int = 14, alpha: int = 100) -> None:
        shadow = rect.move(4, 5)
        s = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.colors["card_shadow"], alpha), s.get_rect(), border_radius=radius)
        self.screen.blit(s, shadow.topleft)
        shadow2 = rect.move(2, 2)
        s2 = pygame.Surface((shadow2.w, shadow2.h), pygame.SRCALPHA)
        pygame.draw.rect(s2, (*self.colors["card_shadow"], min(55, alpha // 2)), s2.get_rect(), border_radius=radius)
        self.screen.blit(s2, shadow2.topleft)

    def _draw_panel_card(self, rect: pygame.Rect, *, radius: int = 14) -> None:
        self._panel_shadow(rect, radius=radius, alpha=108)
        pygame.draw.rect(self.screen, self.colors["card_border"], rect, width=2, border_radius=radius)
        inner = rect.inflate(-4, -4)
        if inner.w > 0 and inner.h > 0:
            self._fill_vertical_gradient(inner, self.colors["card_top"], self.colors["card_bottom"])

    def draw_header(
        self,
        lay: _UILayout,
        goal_type: int,
        step: int,
        max_steps: int,
        *,
        episode_done: bool = False,
        step_reward: float = 0.0,
    ) -> None:
        theme = self.goal_themes.get(goal_type, self.goal_themes[2])
        hh = lay.header_h
        header = pygame.Rect(0, 0, self.width, hh)
        self._fill_vertical_gradient(
            header,
            theme["header_left"],
            theme["header_right"],
        )

        tx = lay.pad + 8
        ty0 = max(5, hh // 20)
        pill_y = hh - 24
        max_w = self.width - lay.pad * 2 - 140
        line_a = "RL agent suggests meals under a nutrition goal:"
        line_b = "weight loss, gain, or maintenance."
        full_txt = f"{line_a} {line_b}"
        short_txt = "RL agent suggests meals for loss, gain, or maintenance."

        font_h = self.font_headline
        fg = self.colors["header_headline"]
        line_skip = font_h.get_height() - 3

        def blit_head(s: str, y: int, f: pygame.font.Font) -> None:
            self._blit_text_outline(self.screen, f, s, (tx, y), fg, width=2)

        if hh < 70:
            blit_head(short_txt, ty0, self.font_medium)
        elif font_h.size(full_txt)[0] <= max_w and ty0 + line_skip <= pill_y - 8:
            blit_head(full_txt, ty0, font_h)
        elif font_h.size(line_a)[0] <= max_w and ty0 + line_skip * 2 <= pill_y - 6:
            blit_head(line_a, ty0, font_h)
            blit_head(line_b, ty0 + line_skip, font_h)
        elif font_h.size(short_txt)[0] <= max_w:
            blit_head(short_txt, ty0, font_h)
        else:
            blit_head(short_txt, ty0, self.font_medium)

        pill = f"Today: {theme['name']}"
        pill_surf = self.font_medium.render(pill, True, self.colors["white"])
        self.screen.blit(pill_surf, (tx, pill_y))

        rx = self.width - lay.pad - 8
        step_label = self.font_small.render("Step", True, self.colors["muted"])
        step_nums = f"{step} / {max_steps}"
        step_main = self.font_metric.render(step_nums, True, self.colors["header_metric"])
        self.screen.blit(step_label, (rx - step_label.get_width(), pill_y - 22))
        self.screen.blit(step_main, (rx - step_main.get_width(), pill_y - 6))

        rw_c = (
            self.colors["positive"]
            if step_reward > 0
            else (self.colors["negative"] if step_reward < 0 else self.colors["muted"])
        )
        rw_line = self.font_small.render(f"r: {step_reward:+.2f}", True, rw_c)
        self.screen.blit(rw_line, (rx - rw_line.get_width(), pill_y - 40))

        if episode_done:
            done_s = self.font_small.render("Episode done", True, self.colors["header_metric"])
            bx = rx - done_s.get_width()
            by = ty0
            pad = 4
            db = pygame.Rect(bx - pad, by - pad, done_s.get_width() + 2 * pad, done_s.get_height() + 2 * pad)
            pygame.draw.rect(self.screen, self.colors["title_outline"], db, border_radius=6)
            pygame.draw.rect(
                self.screen,
                _lerp_color(theme["header_left"], theme["header_right"], 0.5),
                db.inflate(-2, -2),
                border_radius=5,
            )
            self.screen.blit(done_s, (bx, by))

        accent = theme["accent"]
        bar_h = max(3, min(6, hh // 14))
        pygame.draw.rect(self.screen, accent, (0, hh, self.width, bar_h))

    def draw_macros_section(self, daily: Dict, targets: Dict, lay: _UILayout) -> None:
        x, y = lay.x_macro, lay.content_top
        section_w, section_h = lay.w_macro, lay.panel_h
        rect = pygame.Rect(x, y, section_w, section_h)
        self._draw_panel_card(rect)

        mx = max(8, section_w // 22)
        title = self.font_medium.render("Today's intake", True, self.colors["text"])
        self.screen.blit(title, (x + mx, y + 8))

        hint = self.font_tiny.render(
            "Green = on target · Yellow = approaching · Red = over limit",
            True,
            self.colors["muted"],
        )
        self.screen.blit(hint, (x + mx, y + 30))

        macros = [
            ("Calories", daily["calories"], targets["calorie_target"], "kcal"),
            ("Protein", daily["protein"], targets["protein_target"], "g"),
            ("Carbs", daily["carbs"], targets["carbs_target"], "g"),
            ("Fats", daily["fats"], targets["fats_target"], "g"),
        ]

        label_w = min(92, section_w // 4)
        bar_x = x + mx + label_w
        bar_w = max(40, section_w - label_w - mx * 2)
        bar_h = max(14, min(20, section_h // 16))
        row_y = y + 50
        row_gap = max(34, (section_h - 52 - bar_h - 20) // 4)

        for name, actual, target, unit in macros:
            icon = _MACRO_ICONS.get(name, "•")
            dot_rect = pygame.Rect(x + mx, row_y + 2, 14, 14)
            ic = self.font_small.render(icon, True, self.colors["text"])
            self.screen.blit(ic, (x + mx, row_y))

            label = self.font_small.render(name, True, self.colors["text"])
            self.screen.blit(label, (x + mx + 18, row_y + 1))

            track = pygame.Rect(bar_x, row_y, bar_w, bar_h)
            pygame.draw.rect(self.screen, self.colors["track"], track, border_radius=10)

            ratio = float(actual) / max(float(target), 1e-6)
            target_fill = min(ratio / 1.5, 1.0)
            prev = self._macro_fill_smooth.get(name, target_fill)
            smooth = prev + (target_fill - prev) * self._bar_lerp
            self._macro_fill_smooth[name] = smooth
            fill_w = max(0, int(bar_w * smooth))

            fill_color = self._macro_ratio_color(ratio)
            if fill_w > 0:
                fill_rect = pygame.Rect(bar_x, row_y, fill_w, bar_h)
                pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=10)

            target_marker_x = bar_x + int(bar_w / 1.5)
            pygame.draw.line(
                self.screen,
                self.colors["grid_line"],
                (target_marker_x, row_y - 2),
                (target_marker_x, row_y + bar_h + 2),
                2,
            )

            val_c = self._macro_ratio_color(ratio)
            val = self.font_tiny.render(f"{actual:.0f} / {target:.0f} {unit}", True, val_c)
            self.screen.blit(val, (bar_x + 6, row_y + 2))

            row_y += row_gap

        key = self.font_tiny.render("Vertical marker ≈ 100% of daily target (logged).", True, self.colors["muted"])
        self.screen.blit(key, (x + mx, y + section_h - 20))

    def draw_food_section(
        self,
        food_name: str,
        food_info: Dict,
        action: str,
        step_reward: float,
        lay: _UILayout,
        action_idx: int | None = None,
    ) -> None:
        x, y = lay.x_food, lay.content_top
        section_w, section_h = lay.w_food, lay.panel_h
        rect = pygame.Rect(x, y, section_w, section_h)
        self._draw_panel_card(rect)

        fx = max(8, section_w // 20)
        stripe_w = max(5, min(8, section_w // 50))
        stripe = pygame.Rect(x, y, stripe_w, section_h)
        pygame.draw.rect(self.screen, self.colors["stripe"], stripe, border_radius=3)

        tx = x + fx + stripe_w + 4
        title = self.font_medium.render("Suggested meal", True, self.colors["text"])
        self.screen.blit(title, (tx, y + 8))
        who = self.font_tiny.render(f"For {self._client_name}", True, self.colors["muted"])
        self.screen.blit(who, (tx, y + 30))

        food_label = self.font_large.render(food_name, True, self.colors["neutral"])
        self.screen.blit(food_label, (tx, y + 46))

        nutrition_y = y + 76
        n_gap = max(18, (section_h - 100) // 6)
        nutrition = [
            (f"{_MACRO_ICONS['Calories']} Cal ", f"{food_info['cal']} kcal"),
            (f"{_MACRO_ICONS['Protein']} Prot", f"{food_info['protein']} g"),
            (f"{_MACRO_ICONS['Carbs']} Carb", f"{food_info['carbs']} g"),
            (f"{_MACRO_ICONS['Fats']} Fat ", f"{food_info['fat']} g"),
        ]
        for left, right in nutrition:
            self.screen.blit(self.font_small.render(left, True, self.colors["text"]), (tx, nutrition_y))
            rv = self.font_small.render(right, True, self.colors["gold"])
            self.screen.blit(rv, (tx + 88, nutrition_y))
            nutrition_y += n_gap

        agent_line = f"Last action: {action}"
        action_color = self.colors["positive"] if action == "Accept" else self.colors["warning"]
        if action == "Skip":
            action_color = self.colors["muted"]
        elif action == "Higher Protein":
            action_color = self.colors["neutral"]

        bot = y + section_h
        hi = action_idx if action_idx is not None else -1
        if 0 <= hi <= 3:
            glow = pygame.Rect(tx - 4, bot - 60, section_w - fx - stripe_w - 8, 30)
            pygame.draw.rect(self.screen, action_color, glow, width=2, border_radius=8)
        agent_surf = self.font_large.render(agent_line, True, action_color)
        self.screen.blit(agent_surf, (tx, bot - 56))

        rw_color = (
            self.colors["positive"]
            if step_reward > 0
            else (self.colors["negative"] if step_reward < 0 else self.colors["muted"])
        )
        rw_surf = self.font_medium.render(f"Step reward: {step_reward:+.2f}", True, rw_color)
        self.screen.blit(rw_surf, (tx, bot - 30))

        ex1, _ex2 = self._step_reward_explanation(action_idx)
        ex_surf = self.font_tiny.render(ex1, True, self.colors["muted"])
        ex_y = min(nutrition_y + 2, bot - 64 - ex_surf.get_height())
        ex_y = max(ex_y, y + 82)
        self.screen.blit(ex_surf, (tx, ex_y))

    def draw_reward_section(self, episode_reward: float, step_reward: float, num_meals: int, lay: _UILayout) -> None:
        x, y = lay.x_reward, lay.content_top
        section_w, section_h = lay.w_reward, lay.panel_h
        rect = pygame.Rect(x, y, section_w, section_h)
        self._draw_panel_card(rect)

        rx = max(8, section_w // 22)
        title = self.font_medium.render("Rewards", True, self.colors["text"])
        self.screen.blit(title, (x + rx, y + 8))
        sub = self.font_mono.render("G_return = sum_t r_t", True, self.colors["muted"])
        self.screen.blit(sub, (x + rx, y + 30))

        metrics_y = y + 48
        m_step = max(54, min(66, (section_h - 52) // 3))
        last_col = (
            self.colors["muted"]
            if abs(step_reward) < 1e-6
            else (self.colors["positive"] if step_reward > 0 else self.colors["negative"])
        )
        ep_col = (
            self.colors["positive"]
            if episode_reward > 0
            else (self.colors["negative"] if episode_reward < 0 else self.colors["muted"])
        )
        blocks = [
            ("Episode total reward", episode_reward, ep_col),
            ("This step", step_reward, last_col),
            ("Meals logged (accepts)", float(num_meals), self.colors["gold"]),
        ]

        for label, value, color in blocks:
            lab = self.font_small.render(label, True, self.colors["muted"])
            self.screen.blit(lab, (x + rx, metrics_y))
            if isinstance(value, float):
                val_text = f"{value:+.2f}" if label != "Meals logged (accepts)" else f"{int(value)}"
            else:
                val_text = str(value)
            val_surf = self.font_metric.render(val_text, True, color)
            self.screen.blit(val_surf, (x + rx, metrics_y + 20))
            metrics_y += m_step

    def draw_action_legend(self, lay: _UILayout, last_action: int | None = None) -> None:
        y = lay.legend_y
        bar = pygame.Rect(lay.pad, y, self.width - 2 * lay.pad, lay.legend_h)
        self._draw_panel_card(bar)

        lx = lay.pad + 8
        label = self.font_medium.render("Actions 0–3 (Gymnasium Discrete)", True, self.colors["text"])
        self.screen.blit(label, (lx, y + 4))

        cx = lx
        row1_y = y + 22
        pill_w = 24
        row_max = self.width - lay.pad - 140
        for key, desc, col in self.action_legend:
            short = desc.replace(" (macro reward)", "").replace(" (−0.5)", "-.5").replace(" (0)", "")
            txt = self.font_small.render(short, True, self.colors["text"])
            need = pill_w + 4 + txt.get_width() + 10
            if cx + need > row_max and cx > lx:
                cx = lx
                row1_y += 22
            ki = int(key)
            key_rect = pygame.Rect(cx, row1_y, pill_w, 20)
            br_col = col
            if last_action is not None and ki == last_action:
                br_col = tuple(min(255, c + 45) for c in col)
                pygame.draw.rect(self.screen, self.colors["white"], key_rect.inflate(4, 4), width=2, border_radius=7)
            pygame.draw.rect(self.screen, br_col, key_rect, border_radius=5)
            if last_action is not None and ki == last_action:
                pygame.draw.rect(self.screen, self.colors["white"], key_rect, width=2, border_radius=5)
            key_surf = self.font_tiny.render(key, True, self.colors["white"])
            self.screen.blit(key_surf, (cx + 7, row1_y + 3))
            cx += pill_w + 4
            self.screen.blit(txt, (cx, row1_y + 1))
            cx += txt.get_width() + 10

        heat_x = bar.right - 118
        heat_y = y + 20
        self.screen.blit(
            self.font_mono.render("last actions", True, self.colors["muted"]),
            (heat_x, y + 6),
        )
        counts = [0, 0, 0, 0]
        for a in self._recent_actions:
            if 0 <= a <= 3:
                counts[a] += 1
        total = max(sum(counts), 1)
        hm_h = 28
        for i in range(4):
            frac = counts[i] / total
            bx = heat_x + i * 28
            base = pygame.Rect(bx, heat_y + 14, 22, hm_h)
            pygame.draw.rect(self.screen, self.colors["track"], base, border_radius=4)
            fill_h = max(2, int(hm_h * frac))
            fill = pygame.Rect(bx, heat_y + 14 + hm_h - fill_h, 22, fill_h)
            _, _, c = self.action_legend[i]
            pygame.draw.rect(self.screen, c, fill, border_radius=4)
            self.screen.blit(self.font_mono.render(str(i), True, self.colors["muted"]), (bx + 6, heat_y + 2))

        trans_y = max(row1_y + 24, y + 48)
        self.screen.blit(
            self.font_mono.render("Transition (last 3):  s — a — r — s'", True, self.colors["muted"]),
            (lx, trans_y),
        )
        trans_y += 15
        rows = list(self._transition_log)
        for i, tr in enumerate(rows):
            is_last = i == len(rows) - 1
            fg = self.colors["log_highlight"] if is_last else self.colors["transition_dim"]
            rw = f"{tr['r']:+.2f}"
            line = f"{tr['step']:02d}  {tr['s']}  a={tr['a']}  r={rw}  {tr['sp']}"
            if self.font_mono.size(line)[0] > bar.w - 24:
                line = f"{tr['step']:02d}  a={tr['a']}  r={rw}  {tr['s']}→{tr['sp']}"
            self.screen.blit(self.font_mono.render(line, True, fg), (lx, trans_y))
            trans_y += 14

        cheat_y = min(trans_y + 4, y + lay.legend_h - 28)
        cheat_y = max(cheat_y, trans_y)
        r1 = "reward: accept≈0..5 (+2 rules) · alt −0.5 · skip 0"
        self.screen.blit(self.font_mono.render(r1, True, self.colors["muted"]), (lx, cheat_y))

    def draw_day_timeline(self, lay: _UILayout, step: int, max_steps: int) -> None:
        pad = lay.timeline_pad
        tw = self.width - 2 * pad
        th = 12
        ty = lay.timeline_y
        track = pygame.Rect(pad, ty, tw, th)
        pygame.draw.rect(self.screen, self.colors["timeline_bg"], track, border_radius=7)
        frac = 0.0 if max_steps <= 0 else min(step / max_steps, 1.0)
        if frac > 0:
            fill = pygame.Rect(pad, ty, int(tw * frac), th)
            pygame.draw.rect(self.screen, self.colors["timeline_fill"], fill, border_radius=7)
        cap = self.font_mono.render("t / T", True, self.colors["muted"])
        self.screen.blit(cap, (pad, ty - 15))

    def render_episode(
        self,
        env: NutriVisionEnv,
        action: int | None = None,
        reward: float = 0.0,
        show_fps: bool = True,
        *,
        episode_done: bool = False,
    ) -> None:
        bg = pygame.Rect(0, 0, self.width, self.height)
        self._fill_vertical_gradient(bg, self.colors["bg_top"], self.colors["bg_bottom"])
        t = pygame.time.get_ticks() / 1000.0
        self._draw_arcade_stars(t)

        daily = {
            "calories": env.daily_calories,
            "protein": env.daily_protein,
            "carbs": env.daily_carbs,
            "fats": env.daily_fats,
        }
        targets = {
            "calorie_target": env.calorie_target,
            "protein_target": env.protein_target,
            "carbs_target": env.carbs_target,
            "fats_target": env.fats_target,
        }

        action_names = ["Accept", "Lower Cal", "Higher Protein", "Skip"]
        action_text = action_names[action] if action is not None else "—"

        curr_obs = np.asarray(env._get_observation(), dtype=np.float32).reshape(-1)

        self._sync_client_and_history(env)
        self._note_step_for_viz(env, action, reward, self._viz_obs_before_step, curr_obs)

        lay = self._compute_layout()
        self.draw_header(
            lay,
            env.goal_type,
            env.step_count,
            env.max_steps,
            episode_done=episode_done,
            step_reward=reward,
        )
        self.draw_person_section(env, lay, curr_obs)
        self.draw_macros_section(daily, targets, lay)
        self.draw_food_section(
            env.current_food["name"],
            env.current_food,
            action_text,
            reward,
            lay,
            action_idx=action,
        )
        self.draw_reward_section(
            env.episode_reward,
            reward,
            env.num_meals_logged,
            lay,
        )
        self.draw_action_legend(lay, last_action=action)
        self.draw_day_timeline(lay, env.step_count, env.max_steps)

        self._viz_obs_before_step = np.array(curr_obs, copy=True)

        if show_fps:
            fps_text = self.font_tiny.render(f"FPS {int(self.clock.get_fps())}", True, self.colors["muted"])
            self.screen.blit(fps_text, (self.width - fps_text.get_width() - lay.pad, lay.timeline_y - 18))

        pygame.display.flip()
        self.clock.tick(30)
        self.anim_frame += 1

    def play_episode_interactive(self, env: NutriVisionEnv) -> None:
        env.reset()
        action_map = {
            pygame.K_0: 0,
            pygame.K_1: 1,
            pygame.K_2: 2,
            pygame.K_3: 3,
        }

        print("\n" + "=" * 80)
        print("INTERACTIVE EPISODE - Control Agent with Keys 0-3")
        print("=" * 80)
        print("0: Accept recommendation")
        print("1: Request lower-calorie alternative")
        print("2: Request higher-protein alternative")
        print("3: Skip meal")
        print("Close window or press ESC to stop")
        print("=" * 80 + "\n")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in action_map:
                        act = action_map[event.key]
                        obs, rew, terminated, truncated, info = env.step(act)
                        done = terminated or truncated
                        self.render_episode(env, act, rew, episode_done=done)
                        print(
                            f"Action: {['Accept', 'Lower Cal', 'Higher Protein', 'Skip'][act]} | "
                            f"Reward: {rew:+.2f} | Food: {info['food_name']}"
                        )
                        if done:
                            print(f"\nEpisode ended! Total reward: {info['episode_reward']:+.2f}")
                            return

            self.render_episode(env)

        self.close()

    def record_episode_video(
        self,
        env: NutriVisionEnv,
        model,
        algorithm: str,
        output_path: str = "visualizations/agent_demo.avi",
        *,
        show_fps: bool = False,
        verbose: bool = True,
        max_frames: int | None = None,
    ) -> None:
        import cv2

        obs, _ = env.reset()

        path_lower = output_path.lower()
        if path_lower.endswith(".mp4"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (self.width, self.height))
        if not out.isOpened():
            raise RuntimeError(
                f"Could not open VideoWriter for {output_path!r}. "
                "Try .avi with MJPG or install a full OpenCV build."
            )

        frame_count = 0

        while True:
            if algorithm == "reinforce":
                import torch

                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    probs = model(state_tensor)
                act = torch.argmax(probs, dim=1).item()
            else:
                act, _ = model.predict(obs, deterministic=True)

            obs, rew, terminated, truncated, info = env.step(act)
            self.render_episode(
                env,
                act,
                rew,
                show_fps=show_fps,
                episode_done=terminated or truncated,
            )

            if verbose:
                action_names = ["Accept", "Lower Cal", "Higher Protein", "Skip"]
                food_name = info.get("food_name", "?")
                print(
                    f"Step {env.step_count:02d} | "
                    f"Action: {action_names[act]} | "
                    f"Reward: {rew:+.2f} | "
                    f"Food: {food_name}"
                )

            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
            if terminated or truncated:
                break

        out.release()
        print(f"[OK] Video saved to {output_path} ({frame_count} frames)")

    def close(self) -> None:
        pygame.quit()


def demo_visualization() -> None:
    env = NutriVisionEnv()
    visualizer = NutriVisionVisualizer()
    try:
        visualizer.play_episode_interactive(env)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        visualizer.close()


if __name__ == "__main__":
    demo_visualization()
