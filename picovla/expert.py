"""Scripted expert. Stateless: the action depends only on the current scene,
so a cloned policy sees one consistent answer per state."""
import math
from .sim import DIRS, RB, RT


def _unit(x, y):
    n = math.sqrt(x * x + y * y)
    return (x / n, y / n) if n > 1e-9 else (0.0, 0.0)


def _goto(tip, target, gain=1.0 / 0.04):
    dx, dy = target[0] - tip[0], target[1] - tip[1]
    k = min(1.0, gain * math.sqrt(dx * dx + dy * dy))
    ux, uy = _unit(dx, dy)
    return ux * k, uy * k


def _push(env, b, ux, uy):
    tip = env.tip
    px, py = -uy, ux
    rx, ry = tip[0] - b[0], tip[1] - b[1]
    along, lat = rx * ux + ry * uy, rx * px + ry * py
    stand = RB + RT + 0.03
    if along < -(RB + RT) * 0.6 and abs(lat) < 0.03:
        # behind the block and lined up: push through, steer the contact point back onto the line
        return _unit(ux - 4.0 * lat * px, uy - 4.0 * lat * py)
    behind = (b[0] - ux * stand, b[1] - uy * stand)
    if along > -(RB + RT) * 0.6 and abs(lat) < RB + RT + 0.04:
        # on the wrong side or touching: go round, keep clear of the block
        side = 1.0 if lat >= 0 else -1.0
        back = stand if along < -0.02 else 0.0  # from the front, first step out sideways
        way = (b[0] + px * side * (stand + 0.06) - ux * back, b[1] + py * side * (stand + 0.06) - uy * back)
        return _goto(tip, way, gain=1e9)
    return _goto(tip, behind)


def act(env):
    t = env.task
    b = env.blocks[t.color]
    if t.kind == "touch":
        return _goto(env.tip, b)
    if t.kind == "push":
        return _push(env, b, *DIRS[t.arg])
    o = env.blocks[t.arg]
    return _push(env, b, *_unit(o[0] - b[0], o[1] - b[1]))
