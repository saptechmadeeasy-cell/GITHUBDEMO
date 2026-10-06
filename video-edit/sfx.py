"""Procedural sound effects (no external samples, royalty-free by construction).

Each generator returns a float32 stereo array (n, 2) at SR, peak-normalised to 1.0.
"""
import numpy as np

SR = 48000
rng = np.random.default_rng(7)


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _norm(x):
    p = np.max(np.abs(x))
    return x / p if p > 0 else x


def _stereo(mono, pan=None):
    if pan is None:
        return np.stack([mono, mono], axis=1)
    # pan: array in [-1, 1] per sample (or scalar)
    pan = np.broadcast_to(pan, mono.shape)
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    return np.stack([mono * l, mono * r], axis=1)


def _lowpass_sweep(x, cutoffs):
    """One-pole low-pass with a per-sample cutoff (Hz)."""
    a = 1 - np.exp(-2 * np.pi * cutoffs / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def whoosh(dur=0.5, lo=300, hi=5000, direction=1):
    t = _t(dur)
    p = t / dur
    noise = rng.standard_normal(len(t))
    cut = lo + (hi - lo) * np.sin(np.pi * p) ** 2
    y = _lowpass_sweep(noise, cut)
    y -= _lowpass_sweep(y, np.full(len(t), 150.0))  # high-pass the rumble away
    env = np.sin(np.pi * p ** 0.7) ** 2
    pan = direction * (p * 1.4 - 0.7)
    return _norm(_stereo(_norm(y * env), pan))


def pop(f0=1100, f1=350, dur=0.12):
    t = _t(dur)
    f = f1 + (f0 - f1) * np.exp(-t * 45)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t * 28)
    y[: int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    return _norm(_stereo(y))


def ding(f=1318.5, dur=1.2):
    t = _t(dur)
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t * 4)
         + 0.5 * np.sin(2 * np.pi * f * 2.01 * t) * np.exp(-t * 6)
         + 0.25 * np.sin(2 * np.pi * f * 3.02 * t) * np.exp(-t * 9)
         + 0.6 * np.sin(2 * np.pi * f * 1.5 * t) * np.exp(-t * 5))
    y[: int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))
    return _norm(_stereo(y))


def impact(dur=1.4):
    t = _t(dur)
    f = 38 + 80 * np.exp(-t * 9)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    noise = rng.standard_normal(len(t))
    crack = _lowpass_sweep(noise, 200 + 6000 * np.exp(-t * 18)) * np.exp(-t * 14)
    y = sub + 0.8 * _norm(crack)
    y[: int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    return _norm(_stereo(np.tanh(1.6 * y)))


def riser(dur=1.6):
    t = _t(dur)
    p = t / dur
    noise = rng.standard_normal(len(t))
    y = _lowpass_sweep(noise, 300 + 9000 * p ** 2)
    y -= _lowpass_sweep(y, np.full(len(t), 200.0))
    tone = np.sin(2 * np.pi * np.cumsum(180 + 700 * p ** 2) / SR)
    env = p ** 2.2
    env[-int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))
    return _norm(_stereo(_norm(y) * env + 0.35 * tone * env))


def tick(dur=0.05):
    t = _t(dur)
    y = np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 120)
    return _norm(_stereo(y))


def chime_up(dur=0.9):
    """Two-note rising chime for positive moments."""
    a = ding(1046.5, dur)
    b = ding(1568.0, dur)
    off = int(0.09 * SR)
    out = np.zeros((len(a) + off, 2))
    out[: len(a)] += a
    out[off:] += b * 0.9
    return _norm(out)


def error_buzz(dur=0.35):
    t = _t(dur)
    y = np.sign(np.sin(2 * np.pi * 140 * t)) * 0.5 + np.sin(2 * np.pi * 147 * t)
    y = _lowpass_sweep(y, np.full(len(t), 1800.0)) * np.exp(-t * 7)
    return _norm(_stereo(y))


GENERATORS = {
    "whoosh": lambda: whoosh(0.5),
    "whoosh_short": lambda: whoosh(0.32, 400, 7000, -1),
    "pop": lambda: pop(),
    "pop_hi": lambda: pop(1500, 600, 0.1),
    "ding": lambda: ding(),
    "chime": lambda: chime_up(),
    "impact": lambda: impact(),
    "riser": lambda: riser(),
    "tick": lambda: tick(),
    "buzz": lambda: error_buzz(),
}


def build_track(events, duration):
    """events: list of (time_s, name, gain_db). Returns (n, 2) float32."""
    cache = {}
    out = np.zeros((int(duration * SR) + SR, 2), np.float32)
    for t0, name, gain_db in events:
        if name not in cache:
            cache[name] = GENERATORS[name]().astype(np.float32)
        s = cache[name] * (10 ** (gain_db / 20))
        i = int(t0 * SR)
        j = min(len(out), i + len(s))
        out[i:j] += s[: j - i]
    return out[: int(duration * SR)]
