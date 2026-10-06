"""Add motion graphics, keyword callouts, punch-in zooms and sound effects
to the "RAP ABAP" explainer video.

Usage:
    python3 render.py INPUT.mov OUTPUT.mp4 [--preview START END]

Pipeline: ffmpeg decodes raw RGB frames -> PIL composites animated cards ->
ffmpeg encodes H.264 and mixes the original voice with a synthesized SFX track.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfx  # noqa: E402

W, H, FPS = 1080, 1920, 30

FONT_DIR = "/usr/share/fonts/opentype/inter"
F_BLACK = f"{FONT_DIR}/Inter-Black.otf"
F_XBOLD = f"{FONT_DIR}/Inter-ExtraBold.otf"
F_BOLD = f"{FONT_DIR}/Inter-Bold.otf"
F_SEMI = f"{FONT_DIR}/Inter-SemiBold.otf"

# Palette matched to the "pankajkumaram" watermark (navy + amber swoosh).
NAVY = (27, 42, 107)
INK = (17, 24, 39)
AMBER = (245, 158, 11)
BLUE = (37, 99, 235)
RED = (220, 38, 38)
GREEN = (22, 163, 74)
LIGHT = (232, 240, 254)
GRAY = (100, 116, 139)
WHITE = (255, 255, 255)

CARD_W = 960
CARD_TOP = 160          # sits in the bookshelf band, under the watermark, above the head
CONTENT_W = CARD_W - 80

# Hard cuts in the source (talking head <-> slide). Used for cut-zoom + whoosh.
CUTS = [12.5, 13.5, 14.75, 18.75, 24.0, 26.25, 30.75, 41.0, 121.0, 131.25]

# Punch-in zooms on emphasis words (start, end) - talking-head shots only.
PUNCH_INS = [(3.25, 5.4), (44.0, 46.3), (57.25, 59.5), (63.0, 67.2),
             (83.0, 86.5), (170.5, 174.5)]
ZOOM_CENTER = (540, 640)  # keeps the watermark in frame while pushing in
ZOOM_AMOUNT = 0.055


def font(path, size, _cache={}):
    key = (path, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(path, size)
    return _cache[key]


# ---------------------------------------------------------------- easing ---
def clamp01(x):
    return max(0.0, min(1.0, x))


def ease_out_cubic(p):
    return 1 - (1 - p) ** 3


def ease_out_back(p, s=1.9):
    p -= 1
    return 1 + p * p * ((s + 1) * p + s)


def ease_in_out(p):
    return 0.5 - 0.5 * math.cos(math.pi * p)


# --------------------------------------------------------------- sprites ---
def text_size(txt, f, tracking=0):
    l, t, r, b = f.getbbox(txt)
    return r - l + tracking * max(0, len(txt) - 1), b - t, l, t


def draw_text(txt, f, color, tracking=0, pad=4):
    w, h, l, t = text_size(txt, f, tracking)
    im = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if tracking:
        x = pad - l
        for ch in txt:
            d.text((x, pad - t), ch, font=f, fill=color)
            x += f.getlength(ch) + tracking
    else:
        d.text((pad - l, pad - t), txt, font=f, fill=color)
    return im


def pill(txt, f, fg, bg, padx=28, pady=16, radius=None, icon=None):
    tw, th, l, t = text_size(txt, f)
    icon_w = 0
    if icon:
        icon_w = th + 14
    w, h = tw + icon_w + 2 * padx, th + 2 * pady
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius or h // 2, fill=bg)
    if icon:
        cx, cy, r = padx + th // 2, h // 2, th // 2
        if icon == "check":
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=WHITE)
            d.line([(cx - r * .45, cy), (cx - r * .1, cy + r * .4), (cx + r * .5, cy - r * .35)],
                   fill=bg, width=max(4, th // 7), joint="curve")
        elif icon == "cross":
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=WHITE)
            k = r * .4
            d.line([(cx - k, cy - k), (cx + k, cy + k)], fill=bg, width=max(4, th // 7))
            d.line([(cx - k, cy + k), (cx + k, cy - k)], fill=bg, width=max(4, th // 7))
        elif icon == "bolt":
            d.polygon([(cx + r * .2, cy - r), (cx - r * .6, cy + r * .15), (cx - r * .02, cy + r * .15),
                       (cx - r * .25, cy + r), (cx + r * .6, cy - r * .15), (cx + r * .05, cy - r * .15)],
                      fill=fg)
    d.text((padx + icon_w - l, pady - t), txt, font=f, fill=fg)
    return im


def box(txt, f, fg=WHITE, bg=NAVY, padx=30, pady=24):
    lines = txt.split("\n")
    asc, desc = f.getmetrics()
    lh = asc + desc
    gap = int(lh * 0.02)
    widths = [int(f.getlength(s)) for s in lines]
    w = max(widths) + 2 * padx
    h = lh * len(lines) + gap * (len(lines) - 1) + 2 * pady - 6
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=22, fill=bg)
    d.rounded_rectangle((0, h - 8, w - 1, h - 1), radius=4, fill=AMBER)
    d.rectangle((0, h - 12, w - 1, h - 8), fill=bg)
    y = pady - 4
    for s, sw in zip(lines, widths):
        d.text(((w - sw) // 2, y), s, font=f, fill=fg)
        y += lh + gap
    return im


def arrow(size=56, color=AMBER):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = size
    d.polygon([(s * .08, s * .38), (s * .55, s * .38), (s * .55, s * .16), (s * .95, s * .5),
               (s * .55, s * .84), (s * .55, s * .62), (s * .08, s * .62)], fill=color)
    return im


def multiline(txt, f, color, spacing=1.12, accent_words=()):
    lines = txt.split("\n")
    asc, desc = f.getmetrics()
    lh = int((asc + desc) * spacing)
    widths = [f.getlength(s) for s in lines]
    w = int(max(widths)) + 8
    h = lh * len(lines)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, s in enumerate(lines):
        x = (w - widths[i]) / 2
        for word in s.split(" "):
            c = AMBER if word.strip(",.!?") in accent_words else color
            d.text((x, i * lh), word, font=f, fill=c)
            x += f.getlength(word + " ")
    return im


def acronym_word(word, f):
    w = int(f.getlength(word)) + 8
    asc, desc = f.getmetrics()
    im = Image.new("RGBA", (w, asc + desc), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((0, 0), word[0], font=f, fill=AMBER)
    d.text((f.getlength(word[0]), 0), word[1:], font=f, fill=NAVY)
    return im


def big_word(txt):
    f = font(F_BLACK, 210)
    im = draw_text(txt, f, NAVY, tracking=10)
    w, h = im.size
    out = Image.new("RGBA", (w, h + 26), (0, 0, 0, 0))
    out.alpha_composite(im, (0, 0))
    d = ImageDraw.Draw(out)
    d.rounded_rectangle((w * .18, h + 6, w * .82, h + 22), radius=8, fill=AMBER)
    return out


# ------------------------------------------------------------------ card ---
class Item:
    def __init__(self, img, t, x, y, anim="pop", sound=None):
        self.img, self.t, self.x, self.y, self.anim, self.sound = img, t, x, y, anim, sound


class Card:
    """A white rounded card that animates in at t_in and out at t_out.

    rows: list of (kind, payload). Every element carries its own appear time so
    words/chips land exactly when they're spoken.
    """

    PAD_Y = 34
    ROW_GAP = 20

    def __init__(self, t_in, t_out, rows, accent=AMBER, sound_in="whoosh"):
        self.t_in, self.t_out, self.accent = t_in, t_out, accent
        self.sound_in = sound_in
        self.items = []
        y = self.PAD_Y + 6
        for kind, payload in rows:
            built = self._build_row(kind, payload)  # list of (img, t, anim, sound, dx)
            row_h = max(b[0].size[1] for b in built)
            gap = 18
            if kind == "chips":
                # wrap chips onto multiple lines
                lines, cur, cur_w = [], [], 0
                for b in built:
                    bw = b[0].size[0]
                    if cur and cur_w + gap + bw > CONTENT_W:
                        lines.append(cur)
                        cur, cur_w = [], 0
                    cur_w += (gap if cur else 0) + bw
                    cur.append(b)
                lines.append(cur)
                for line in lines:
                    lw = sum(b[0].size[0] for b in line) + gap * (len(line) - 1)
                    x = (CARD_W - lw) // 2
                    lh = max(b[0].size[1] for b in line)
                    for img, t, anim, snd, _ in line:
                        self.items.append(Item(img, t, x, y + (lh - img.size[1]) // 2, anim, snd))
                        x += img.size[0] + gap
                    y += lh + 14
                y += self.ROW_GAP - 14
                continue
            spacing = 22 if kind == "flow" else (14 if kind == "acronym" else 0)
            total_w = sum(b[0].size[0] for b in built) + spacing * (len(built) - 1)
            x = (CARD_W - total_w) // 2
            for img, t, anim, snd, _ in built:
                self.items.append(Item(img, t, x, y + (row_h - img.size[1]) // 2, anim, snd))
                x += img.size[0] + spacing
            y += row_h + self.ROW_GAP
        self.h = y - self.ROW_GAP + self.PAD_Y
        self.bg = self._build_bg()
        for it in self.items:
            it.t = max(it.t, t_in + 0.12)

    def _build_row(self, kind, p):
        if kind == "label":
            txt, t, color = (p + (self.accent,))[:3] if len(p) == 2 else p
            f = font(F_BOLD, 30)
            img = draw_text(txt.upper(), f, color, tracking=5)
            dot = Image.new("RGBA", (img.size[0] + 34, img.size[1]), (0, 0, 0, 0))
            d = ImageDraw.Draw(dot)
            cy = img.size[1] // 2
            d.ellipse((0, cy - 9, 18, cy + 9), fill=color)
            dot.alpha_composite(img, (34, 0))
            return [(dot, t, "fade", None, 0)]
        if kind == "main":
            txt, t = p[0], p[1]
            opts = p[2] if len(p) > 2 else {}
            f = font(F_XBOLD, opts.get("size", 68))
            img = multiline(txt, f, opts.get("color", INK), accent_words=opts.get("accent", ()))
            return [(img, t, opts.get("anim", "rise"), opts.get("sound"), 0)]
        if kind == "chips":
            out = []
            for c in p:
                txt, t = c[0], c[1]
                opts = c[2] if len(c) > 2 else {}
                style = opts.get("style", "light")
                fg, bg = {"light": (NAVY, LIGHT), "amber": (INK, AMBER), "red": (WHITE, RED),
                          "green": (WHITE, GREEN), "navy": (WHITE, NAVY), "blue": (WHITE, BLUE)}[style]
                f = font(F_BOLD, opts.get("size", 38))
                img = pill(txt, f, fg, bg, icon=opts.get("icon"))
                out.append((img, t, "pop", opts.get("sound", "pop"), 0))
            return out
        if kind == "flow":
            out = []
            f = font(F_BOLD, p.get("size", 40)) if isinstance(p, dict) else font(F_BOLD, 40)
            steps = p["steps"] if isinstance(p, dict) else p
            for i, (txt, t) in enumerate(steps):
                if i:
                    out.append((arrow(), t, "slide", None, 0))
                bg = NAVY if i % 2 == 0 else BLUE
                out.append((box(txt, f, bg=bg), t, "pop", "pop", 0))
            return out
        if kind == "big":
            txt, t = p
            return [(big_word(txt), t, "stamp", "impact", 0)]
        if kind == "acronym":
            f = font(F_XBOLD, 48)
            return [(acronym_word(w, f), t, "pop", "pop_hi", 0) for w, t in p]
        if kind == "note":
            txt, t = p
            img = multiline(txt, font(F_SEMI, 30), GRAY)
            return [(img, t, "fade", None, 0)]
        raise ValueError(kind)

    def _build_bg(self):
        m = 40
        im = Image.new("RGBA", (CARD_W + 2 * m, self.h + 2 * m), (0, 0, 0, 0))
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((m, m + 12, m + CARD_W, m + self.h + 12),
                                             radius=40, fill=(10, 20, 60, 90))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(18)))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((m, m, m + CARD_W, m + self.h), radius=40, fill=(255, 255, 255, 240))
        # amber top accent bar
        d.rounded_rectangle((m + CARD_W * .38, m, m + CARD_W * .62, m + 10), radius=5, fill=self.accent)
        self.margin = m
        return im

    def sounds(self):
        ev = []
        if self.sound_in:
            ev.append((self.t_in - 0.08, self.sound_in, -17))
        for it in self.items:
            if it.sound:
                gain = {"pop": -15, "pop_hi": -16, "impact": -9, "ding": -14,
                        "chime": -14, "buzz": -18, "tick": -18}.get(it.sound, -15)
                ev.append((it.t, it.sound, gain))
        ev.append((self.t_out - 0.3, "whoosh_short", -22))
        return ev

    def active(self, t):
        return self.t_in <= t <= self.t_out

    def render(self, t):
        """Return (rgba_image, x, y) for time t, or None."""
        if not self.active(t):
            return None
        canvas = self.bg.copy()
        m = self.margin
        for it in self.items:
            dt = t - it.t
            if dt < 0:
                continue
            p = clamp01(dt / (0.42 if it.anim == "stamp" else 0.32))
            img = it.img
            x, y = it.x + m, it.y + m
            alpha = 1.0
            if it.anim == "pop":
                s = max(0.01, 0.55 + 0.45 * ease_out_back(p))
                alpha = clamp01(p * 2.5)
            elif it.anim == "stamp":
                s = 1.0 + 0.9 * (1 - ease_out_cubic(p))
                alpha = clamp01(p * 3)
            elif it.anim == "rise":
                s = 1.0
                y += int(30 * (1 - ease_out_cubic(p)))
                alpha = ease_out_cubic(p)
            elif it.anim == "slide":
                s = 1.0
                x -= int(30 * (1 - ease_out_cubic(p)))
                alpha = p
            else:  # fade
                s = 1.0
                alpha = p
            if s != 1.0:
                nw, nh = max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))
                img2 = img.resize((nw, nh), Image.BILINEAR)
                x += (img.size[0] - nw) // 2
                y += (img.size[1] - nh) // 2
                img = img2
            if alpha < 1:
                img = with_alpha(img, alpha)
            paste(canvas, img, x, y)
        # card-level enter / exit
        pin = clamp01((t - self.t_in) / 0.38)
        pout = clamp01((self.t_out - t) / 0.28)
        s = 0.88 + 0.12 * ease_out_back(pin, 1.4)
        a = ease_out_cubic(pin) * ease_out_cubic(pout)
        dy = int(-50 * (1 - ease_out_cubic(pin)) - 40 * (1 - pout))
        img = canvas
        if s != 1.0:
            nw, nh = int(canvas.size[0] * s), int(canvas.size[1] * s)
            img = canvas.resize((nw, nh), Image.BILINEAR)
        if a < 1:
            img = with_alpha(img, a)
        x = (W - img.size[0]) // 2
        y = CARD_TOP - m + (canvas.size[1] - img.size[1]) // 2 + dy
        return img, x, y


def with_alpha(img, a):
    r, g, b, al = img.split()
    al = al.point(lambda v: int(v * a))
    return Image.merge("RGBA", (r, g, b, al))


def paste(dst, src, x, y):
    """alpha_composite that tolerates negative / out-of-bounds offsets."""
    sx0, sy0 = max(0, -x), max(0, -y)
    x0, y0 = max(0, x), max(0, y)
    w = min(src.size[0] - sx0, dst.size[0] - x0)
    h = min(src.size[1] - sy0, dst.size[1] - y0)
    if w <= 0 or h <= 0:
        return
    dst.alpha_composite(src, (x0, y0), (sx0, sy0, sx0 + w, sy0 + h))


# -------------------------------------------------------------- timeline ---
# Times come from the narration (word-level, read off the burned-in captions).
# Cards only appear on talking-head shots; the slide segments stay untouched.
def timeline():
    C = Card
    return [
        C(0.15, 5.35, [
            ("label", ("SAP ABAP Developers", 0.15)),
            ("main", ("Still learning only\nclassic ABAP?", 0.3, {"accent": ("classic",)})),
            ("chips", [("You're missing a key part of SAP", 2.75, {"style": "red", "icon": "cross", "sound": "buzz"})]),
        ]),
        C(5.9, 12.3, [
            ("label", ("The traditional way", 5.9)),
            ("main", ("Classic ABAP development", 6.3, {"size": 58})),
            ("chips", [("ECC", 8.0, {"style": "navy"}), ("S/4HANA", 9.5, {"style": "navy"})]),
        ]),
        C(18.9, 23.85, [
            ("label", ("Classic ABAP objects", 18.9)),
            ("chips", [("Reports", 19.2), ("Module Pool", 19.5), ("BDC", 19.8),
                       ("Function Modules", 20.25), ("Enhancements", 21.25)]),
        ]),
        C(26.4, 30.6, [
            ("label", ("Then came", 26.4)),
            ("main", ("ABAP on HANA", 27.9, {"size": 84, "anim": "pop", "sound": "pop", "accent": ("HANA",)})),
        ]),
        C(41.2, 50.6, [
            ("label", ("What's next?", 41.2)),
            ("main", ("SAP's powerful\nnew technology", 44.0, {"accent": ("powerful",), "sound": "ding"})),
            ("chips", [("Must-learn in the current era", 48.5, {"style": "amber", "icon": "bolt"})]),
        ]),
        C(51.2, 55.6, [
            ("label", ("Warning", 51.2, RED)),
            ("main", ("Hard to survive\nwithout it", 52.2, {"accent": ("survive",)})),
        ], accent=RED),
        C(56.0, 59.6, [
            ("label", ("Also known as", 56.0)),
            ("main", ("CLEAN CORE", 57.25, {"size": 104, "anim": "stamp", "sound": "impact", "color": NAVY})),
            ("main", ("development", 58.25, {"size": 50, "color": GRAY})),
        ]),
        C(60.0, 67.2, [
            ("label", ("The technology is...", 60.0)),
            ("big", ("RAP", 63.0)),
            ("acronym", [("RESTful", 65.25), ("Application", 65.75), ("Programming", 66.75)]),
        ]),
        C(67.4, 76.1, [
            ("label", ("What is RAP?", 67.4)),
            ("main", ("Framework for\nnew ABAP development", 67.6, {"size": 62, "accent": ("new",)})),
            ("chips", [("Transactional apps", 74.0, {"style": "navy"})]),
        ]),
        C(77.0, 88.9, [
            ("label", ("Know RAP? Then you can build", 77.0)),
            ("flow", [("RAP", 77.6), ("Fiori Elements\napps", 80.5)]),
            ("chips", [("No front-end developer needed", 83.75, {"style": "green", "icon": "check", "sound": "chime"})]),
            ("note", ("for a typical application", 87.75)),
        ]),
        C(89.5, 99.9, [
            ("label", ("Of course...", 89.5)),
            ("main", ("Heavy front-end\ncustomization?", 91.75, {"size": 62, "accent": ("customization?",)})),
            ("chips", [("Fiori developer builds the extension", 95.75, {"style": "blue"})]),
        ]),
        C(100.1, 109.9, [
            ("label", ("RAP + Fiori Elements", 100.1)),
            ("main", ("Speeds up app\ndevelopment", 101.25, {"accent": ("Speeds", "up")})),
            ("chips", [("Standard Fiori Elements", 107.5, {"style": "navy"})]),
        ]),
        C(110.2, 118.8, [
            ("label", ("The workflow", 110.2)),
            ("flow", [("Build in\nRAP", 111.0), ("Use as a\nFiori app", 117.25)]),
        ]),
        C(119.25, 120.85, [
            ("label", ("Inside RAP", 119.25)),
            ("main", ("The building blocks", 119.4, {"size": 60})),
        ]),
        C(131.4, 138.7, [
            ("label", ("CDS view layers", 131.4)),
            ("flow", {"steps": [("Interface\nCDS view", 133.0), ("Projection\nCDS view", 137.0)], "size": 38}),
        ]),
        C(139.0, 145.9, [
            ("label", ("Behavior definition layers", 139.0)),
            ("flow", {"steps": [("Interface\nBDEF", 141.25), ("Projection\nBDEF", 144.0)], "size": 38}),
        ]),
        C(146.4, 156.0, [
            ("label", ("Service layer", 146.4)),
            ("chips", [("No interface / projection split", 148.5, {"style": "light", "size": 32})]),
            ("flow", {"steps": [("Service\ndefinition", 153.25), ("Service\nbinding", 155.5)], "size": 38}),
        ]),
        C(156.3, 166.9, [
            ("label", ("Service binding types", 156.3)),
            ("chips", [("OData V2", 159.0, {"style": "navy", "size": 54}),
                       ("OData V4", 161.25, {"style": "blue", "size": 54})]),
            ("note", ("Both can be generated from RAP", 164.25)),
        ]),
        C(167.2, 178.6, [
            ("label", ("ABAP developer?", 167.2)),
            ("main", ("Grow your career:\nstart learning RAP", 170.5, {"size": 62, "accent": ("RAP",), "sound": "chime"})),
            ("chips", [("Start today", 173.75, {"style": "amber", "icon": "bolt"})]),
        ]),
    ]


# ----------------------------------------------------------------- frame ---
def zoom_factor(t):
    z = 1.0
    for a, b in PUNCH_INS:
        if a - 0.05 <= t <= b + 0.35:
            pin = ease_in_out(clamp01((t - a) / 0.3))
            pout = ease_in_out(clamp01((b + 0.35 - t) / 0.35))
            z = max(z, 1 + ZOOM_AMOUNT * min(pin, pout))
    for c in CUTS:  # tiny settle-in after every hard cut
        if c <= t < c + 0.3:
            z = max(z, 1 + 0.035 * (1 - ease_out_cubic((t - c) / 0.3)))
    return z


def apply_zoom(img, z):
    if z <= 1.0001:
        return img
    cx, cy = ZOOM_CENTER
    w, h = W / z, H / z
    x0 = min(max(cx - w * cx / W, 0), W - w)
    y0 = min(max(cy - h * cy / H, 0), H - h)
    return img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + w, y0 + h))


def draw_progress(img, t, duration):
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 9), fill=(27, 42, 107, 110))
    d.rectangle((0, 0, int(W * t / duration), 9), fill=AMBER + (255,))


def sfx_events(cards):
    ev = []
    for c in cards:
        ev += c.sounds()
    for c in CUTS:
        ev.append((c - 0.12, "whoosh", -19))
    ev.append((61.4, "riser", -20))  # build-up into the RAP reveal at 63.0
    return ev


def write_wav(path, data):
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sfx.SR)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--preview", nargs=2, type=float, metavar=("START", "END"))
    ap.add_argument("--crf", type=int, default=19)
    args = ap.parse_args()

    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", args.input]))
    start, end = (args.preview if args.preview else (0.0, dur))

    cards = timeline()
    tmp = tempfile.mkdtemp()
    sfx_path = os.path.join(tmp, "sfx.wav")
    write_wav(sfx_path, sfx.build_track(sfx_events(cards), dur))

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(end - start), "-i", args.input,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-ss", str(start), "-t", str(end - start), "-i", args.input,
         "-ss", str(start), "-t", str(end - start), "-i", sfx_path,
         "-filter_complex",
         "[1:a]aresample=48000[v];[v][2:a]amix=inputs=2:normalize=0:duration=first,"
         "alimiter=limit=0.95:level=false[a]",
         "-map", "0:v", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", args.output],
        stdin=subprocess.PIPE)

    fsize = W * H * 3
    n = 0
    total = int(round((end - start) * FPS))
    while True:
        buf = dec.stdout.read(fsize)
        if len(buf) < fsize:
            break
        t = start + n / FPS
        img = Image.frombuffer("RGB", (W, H), buf, "raw", "RGB", 0, 1)
        img = apply_zoom(img, zoom_factor(t)).convert("RGBA")
        for c in cards:
            r = c.render(t)
            if r:
                paste(img, *r)
        draw_progress(img, t, dur)
        enc.stdin.write(img.convert("RGB").tobytes())
        n += 1
        if n % 300 == 0:
            print(f"  {n}/{total} frames", file=sys.stderr, flush=True)
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"done: {n} frames -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
