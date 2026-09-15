# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared post-move overlay primitives for the grasp and place tools.

Both ``grasp/debug.py::vis_post_move_grasp`` and
``place/debug.py::vis_post_place`` draw the same two things on the after-move
image:

  * a **parallel-jaw gripper schematic** anchored at a contact / release point
    and oriented by a rotation matrix (predicted vs actual), and
  * a **compact dark translucent info panel** with a consistent 2-colour
    legend (predicted = yellow, actual = cyan) and grouped labelled rows.

Keeping these here means the two tools never drift apart cosmetically.  The
functions take an explicit ``project_world`` callable so each tool keeps its
own camera-projection convention (grasp clamps differently than place).
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Single 2-colour scheme, reused everywhere (BGR) ──────────────────────
C_PRED = (255, 255, 0)     # cyan   — predicted / target gripper
C_ACTUAL = (0, 255, 255)   # yellow — actual gripper
C_TEXT = (245, 245, 245)   # near-white values
C_LABEL = (180, 180, 180)  # dim-grey field names

_F = cv2.FONT_HERSHEY_SIMPLEX

# Parallel-jaw gripper geometry (metres), expressed in the grasp frame:
#   X = finger-opening axis, Y = perpendicular, Z = approach (toward object).
# The anchor is the fingertip / release contact point (midpoint between the
# two pads), so the schematic sits where the fingers close — not at the flange.
_JAW_HALF_W = 0.041   # half the max Robotiq opening (~82 mm)
_FINGER_LEN = 0.05    # pad → knuckle, back along -Z
_STEM_LEN = 0.03      # knuckle → base, further back along -Z


def draw_gripper(vis, project_world, anchor_world, R_world, color, thick):
    """Draw a parallel-jaw gripper claw outline at a contact point.

    Two pads at ±X·half_w, fingers extending back along -Z to a cross bar,
    then a short centre stem — a recognisable claw.

    Args:
        vis: image to draw on (mutated in place).
        project_world: callable ``world_pt(3,) -> (u, v) | None``.
        anchor_world: (3,) contact point in world frame.
        R_world: (3, 3) grasp rotation, columns = X/Y/Z axes in world frame.
        color: BGR line colour.
        thick: line thickness.
    """
    c = np.asarray(anchor_world)[:3]
    X, Z = R_world[:, 0], R_world[:, 2]
    pads = {
        "lp": c + X * _JAW_HALF_W,                     # left pad tip
        "rp": c - X * _JAW_HALF_W,                     # right pad tip
        "lk": c + X * _JAW_HALF_W - Z * _FINGER_LEN,   # left knuckle
        "rk": c - X * _JAW_HALF_W - Z * _FINGER_LEN,   # right knuckle
        "base": c - Z * (_FINGER_LEN + _STEM_LEN),     # base
        "mid": c - Z * _FINGER_LEN,                    # cross-bar midpoint
    }
    px = {k: project_world(v) for k, v in pads.items()}
    segs = [("lp", "lk"), ("rp", "rk"), ("lk", "rk"), ("mid", "base")]
    for a, b in segs:
        if px[a] is not None and px[b] is not None:
            cv2.line(vis, px[a], px[b], color, thick)


def draw_info_panel(
    vis, fscale, fthick, line_h, *,
    title, rows,
    legend=("predicted", "actual"),
    c_pred=C_PRED, c_actual=C_ACTUAL, c_text=C_TEXT, c_label=C_LABEL,
):
    """Draw a compact dark-translucent readout card in the top-left corner.

    Args:
        title: card title string.
        rows: ordered list of ``(kind, payload)`` after the legend:
            * ``("big", (name, value))``  — large headline value (white).
            * ``("kv",  (name, value))``  — dim-grey name + white value.
        legend: ``(pred_label, actual_label)`` shown with colour swatches;
            pass ``None`` to omit the legend row entirely.
    """
    pad = int(round(10 * line_h / 28))
    x0, y0 = pad, pad

    render_rows: list = [("title", title)]
    if legend is not None:
        render_rows.append(("legend", legend))
    render_rows.extend(rows)

    title_scale = fscale * 0.85
    big_scale = fscale * 1.15
    small_scale = fscale * 0.78

    def _row_h(kind):
        if kind == "big":
            return int(round(line_h * 1.35))
        if kind == "title":
            return int(round(line_h * 1.05))
        return line_h

    def _row_w(kind, payload):
        if kind == "legend":
            pl, al = payload
            (w1, _h), _b = cv2.getTextSize(pl + "  ", _F, small_scale, 1)
            (w2, _h), _b = cv2.getTextSize(al, _F, small_scale, 1)
            return int(120 * fscale) + w1 + w2
        if kind == "title":
            (w, _h), _b = cv2.getTextSize(payload, _F, title_scale, fthick)
            return w
        name, val = payload
        v_scale = big_scale if kind == "big" else small_scale
        (w1, _h), _b = cv2.getTextSize(name + "   ", _F, small_scale, 1)
        (w2, _h), _b = cv2.getTextSize(val, _F, v_scale, fthick)
        return w1 + w2

    total_h = sum(_row_h(k) for k, _ in render_rows) + 2 * pad
    total_w = max(_row_w(k, p) for k, p in render_rows) + 2 * pad

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + total_w, y0 + total_h),
                  (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
    cv2.rectangle(vis, (x0, y0), (x0 + total_w, y0 + total_h),
                  (90, 90, 90), 1)

    tx, ty = x0 + pad, y0 + pad
    for kind, payload in render_rows:
        rh = _row_h(kind)
        base = ty + int(rh * 0.72)
        if kind == "title":
            cv2.putText(vis, payload, (tx, base), _F, title_scale,
                        c_text, fthick)
        elif kind == "legend":
            pl, al = payload
            r = max(4, fthick * 3)
            cv2.circle(vis, (tx + r, base - r // 2), r, c_pred, -1)
            cv2.putText(vis, pl, (tx + 3 * r, base),
                        _F, small_scale, c_label, 1)
            (lw, _h), _b = cv2.getTextSize(pl, _F, small_scale, 1)
            x2 = tx + 3 * r + lw + int(20 * fscale)
            cv2.circle(vis, (x2 + r, base - r // 2), r, c_actual, -1)
            cv2.putText(vis, al, (x2 + 3 * r, base),
                        _F, small_scale, c_label, 1)
        elif kind == "big":
            name, val = payload
            cv2.putText(vis, name, (tx, base - int(rh * 0.18)),
                        _F, small_scale, c_label, 1)
            (nw, _h), _b = cv2.getTextSize(name + "  ", _F, small_scale, 1)
            cv2.putText(vis, val, (tx + nw, base), _F, big_scale,
                        c_text, fthick + 1)
        else:  # kv
            name, val = payload
            cv2.putText(vis, name, (tx, base), _F, small_scale, c_label, 1)
            (nw, _h), _b = cv2.getTextSize(name + "   ", _F, small_scale, 1)
            cv2.putText(vis, val, (tx + nw, base), _F, small_scale,
                        c_text, max(1, fthick))
        ty += rh
