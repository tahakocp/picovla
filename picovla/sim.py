"""Top-down 2D tabletop: a planar two-link arm pushes coloured discs.

The physics uses only + - * / and sqrt so web/sim.js reproduces it bit for bit.
The arm is drawn with IK but only the fingertip collides with blocks.
"""
import math
import numpy as np

COLORS = {"red": (220, 50, 47), "green": (80, 170, 60), "blue": (40, 110, 220), "yellow": (235, 190, 30)}
BG = (238, 232, 220)
ARM = (90, 90, 100)
TIP = (30, 30, 35)
BASE = (0.0, -1.25)
L1, L2 = 1.2, 1.05
RB, RT = 0.09, 0.05          # block radius, fingertip radius
SPEED = 0.04                 # fingertip travel per step at |action| = 1
SUB = 4                      # physics substeps per control step
MAX_STEPS = 120
PUSH_DIST = 0.25
DIRS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, 1.0), "down": (0.0, -1.0)}


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def reachable(x, y):
    dx, dy = x - BASE[0], y - BASE[1]
    d = math.sqrt(dx * dx + dy * dy)
    return 0.25 < d < L1 + L2 - 0.02


class Task:
    """kind: touch | push | toward. color: target block. arg: direction or second colour."""

    def __init__(self, kind, color, arg=None):
        self.kind, self.color, self.arg = kind, color, arg

    def __repr__(self):
        return f"Task({self.kind!r}, {self.color!r}, {self.arg!r})"


class Env:
    def __init__(self, blocks, tip, task):
        self.blocks = {c: [float(x), float(y)] for c, (x, y) in blocks.items()}
        self.tip = [float(tip[0]), float(tip[1])]
        self.task = task
        self.start = {c: tuple(p) for c, p in self.blocks.items()}
        self.t = 0

    @staticmethod
    def random(rng, task=None):
        """Random scene with 3 of the 4 colours. task=None picks a random task."""
        colors = list(rng.permutation(list(COLORS))[:3])
        blocks = {}
        while len(blocks) < 3:
            x, y = rng.uniform(-0.6, 0.6), rng.uniform(-0.35, 0.5)
            if all((x - a) ** 2 + (y - b) ** 2 > 0.32 ** 2 for a, b in blocks.values()):
                blocks[colors[len(blocks)]] = (x, y)
        while True:
            tip = (rng.uniform(-0.6, 0.6), rng.uniform(-0.75, -0.45))
            if all((tip[0] - a) ** 2 + (tip[1] - b) ** 2 > 0.25 ** 2 for a, b in blocks.values()):
                break
        if task is None:
            kind = rng.choice(["touch", "push", "toward"], p=[0.25, 0.5, 0.25])
            c = colors[rng.integers(3)]
            if kind == "touch":
                task = Task("touch", c)
            elif kind == "push":
                task = Task("push", c, list(DIRS)[rng.integers(4)])
            else:
                task = Task("toward", c, [o for o in colors if o != c][rng.integers(2)])
        return Env(blocks, tip, task)

    def step(self, action):
        ax, ay = float(action[0]), float(action[1])
        n = math.sqrt(ax * ax + ay * ay)
        if n > 1.0:
            ax, ay = ax / n, ay / n
        for _ in range(SUB):
            nx = _clamp(self.tip[0] + ax * SPEED / SUB, -1.0, 1.0)
            ny = _clamp(self.tip[1] + ay * SPEED / SUB, -1.0, 1.0)
            if reachable(nx, ny):
                self.tip = [nx, ny]
            self._resolve()
        self.t += 1
        return self.success()

    def _resolve(self):
        names = sorted(self.blocks)
        for _ in range(3):
            for c in names:
                b = self.blocks[c]
                dx, dy = b[0] - self.tip[0], b[1] - self.tip[1]
                d = math.sqrt(dx * dx + dy * dy)
                if 1e-9 < d < RB + RT:
                    k = (RB + RT - d) / d
                    b[0] += dx * k
                    b[1] += dy * k
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = self.blocks[names[i]], self.blocks[names[j]]
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    d = math.sqrt(dx * dx + dy * dy)
                    if 1e-9 < d < 2 * RB:
                        k = (2 * RB - d) / d * 0.5
                        a[0] -= dx * k
                        a[1] -= dy * k
                        b[0] += dx * k
                        b[1] += dy * k
            for b in self.blocks.values():
                b[0] = _clamp(b[0], -1.0 + RB, 1.0 - RB)
                b[1] = _clamp(b[1], -1.0 + RB, 1.0 - RB)

    def success(self):
        t = self.task
        b = self.blocks[t.color]
        if t.kind == "touch":
            dx, dy = b[0] - self.tip[0], b[1] - self.tip[1]
            return math.sqrt(dx * dx + dy * dy) < RB + RT + 0.015
        s = self.start[t.color]
        if t.kind == "push":
            ux, uy = DIRS[t.arg]
            dx, dy = b[0] - s[0], b[1] - s[1]
            along = dx * ux + dy * uy
            across = abs(-dx * uy + dy * ux)
            return along >= PUSH_DIST and across < 0.12
        o = self.blocks[t.arg]
        dx, dy = b[0] - o[0], b[1] - o[1]
        return math.sqrt(dx * dx + dy * dy) < 2 * RB + 0.05

    def state(self):
        return np.array(self.tip, dtype=np.float32)


def elbow(tip):
    """Elbow position for drawing (elbow-right IK solution)."""
    px, py = tip[0] - BASE[0], tip[1] - BASE[1]
    d = _clamp(math.sqrt(px * px + py * py), 1e-6, L1 + L2)
    c = _clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1.0, 1.0)
    th = math.atan2(py, px) - math.acos(c)
    return BASE[0] + L1 * math.cos(th), BASE[1] + L1 * math.sin(th)


def _seg_dist2(px, py, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = px - a[0], py - a[1]
    t = np.clip((wx * vx + wy * vy) / (vx * vx + vy * vy), 0.0, 1.0)
    ex, ey = wx - t * vx, wy - t * vy
    return ex * ex + ey * ey


_GRID = {}


def render(env, size=64):
    """RGB uint8 (size, size, 3). Row 0 is y = +1."""
    if size not in _GRID:
        c = (np.arange(size) + 0.5) / size * 2.0 - 1.0
        _GRID[size] = np.meshgrid(c, -c)
    gx, gy = _GRID[size]
    img = np.empty((size, size, 3), np.uint8)
    img[:] = BG
    for c in sorted(env.blocks):
        b = env.blocks[c]
        img[(gx - b[0]) ** 2 + (gy - b[1]) ** 2 < RB * RB] = COLORS[c]
    e = elbow(env.tip)
    img[_seg_dist2(gx, gy, BASE, e) < 0.045 ** 2] = ARM
    img[_seg_dist2(gx, gy, e, env.tip) < 0.035 ** 2] = ARM
    img[(gx - env.tip[0]) ** 2 + (gy - env.tip[1]) ** 2 < RT * RT] = TIP
    return img
