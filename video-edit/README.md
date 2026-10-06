# Explainer video: motion graphics + SFX

`render.py` takes the source explainer (`video_final_1.mov`) and adds:

- Animated callout cards synced to the narration (hook, ECC/S4HANA, ABAP on HANA,
  Clean Core, RAP reveal + RESTful/Application/Programming, RAP -> Fiori Elements flow,
  CDS / BDEF interface -> projection layers, service definition -> binding, OData V2/V4, CTA)
- Punch-in zooms on emphasis words and a settle-zoom on every hard cut
- A progress bar along the top
- Procedurally generated sound effects (`sfx.py`: whoosh, pop, ding, chime, impact, riser,
  buzz) mixed under the original voice with a limiter. No third-party samples.

Cards only appear on talking-head shots; the existing slide segments are left as-is.

```
python3 render.py INPUT.mov output/video_final_motion_graphics.mp4
python3 render.py INPUT.mov preview.mp4 --preview 56 68   # quick check of a section
```

Requires ffmpeg, Python 3, Pillow, numpy and the Inter font (`/usr/share/fonts/opentype/inter`).
The timeline (text + timings) lives in `timeline()` in `render.py`.
