"""Preview camera renders to verify framing."""
import mujoco
from PIL import Image
from pathlib import Path

# Ensure output directory exists
Path("out").mkdir(exist_ok=True)

m = mujoco.MjModel.from_xml_path("assets/so101/scene_lift_cube.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

print(f"Model loaded: {m.ncam} cameras")
print("Camera positions:")
for i in range(m.ncam):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
    print(f"  {name}: pos={d.cam_xpos[i]}, mat={d.cam_xmat[i][:3]}")

with mujoco.Renderer(m, 480, 640) as r:
    for cam in ["top", "wrist"]:
        r.update_scene(d, camera=cam)
        img = Image.fromarray(r.render())
        img.save(f"out/preview_{cam}.png")
        print(f"{cam:8s} -> out/preview_{cam}.png")

print("\nInspect out/preview_top.png and out/preview_wrist.png")
print("Wrist view should show jaws and space in front")
print("Top view should show full workspace with cube in frame")
