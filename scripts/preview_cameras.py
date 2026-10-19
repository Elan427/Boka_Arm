"""Render each scene camera to a PNG to verify framing.

Run: python scripts/preview_cameras.py
"""
import mujoco
from PIL import Image

from so101lib import paths, scene

paths.OUT_DIR.mkdir(parents=True, exist_ok=True)

sim = scene.load()
m, d = sim.m, sim.d

cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
print(f"{m.ncam} cameras: {cams}")

# One renderer, reused across cameras (per-render renderers leak GL context).
with mujoco.Renderer(m, 480, 640) as r:
    for cam in cams:
        r.update_scene(d, camera=cam)
        out = paths.OUT_DIR / f"preview_{cam}.png"
        Image.fromarray(r.render()).save(out)
        print(f"  {cam:8s} -> {out}")

print("\nwrist view should show the jaws and open space in front;")
print("top view should show the full workspace with the cube in frame.")
