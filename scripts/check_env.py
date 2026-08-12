import importlib.util
from importlib.metadata import PackageNotFoundError, version

import torch


def v(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


print("torch     ", torch.__version__)
print("cuda avail", torch.cuda.is_available())
print("device    ", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability())
for pkg in [
    "lerobot",
    "mujoco",
    "mink",
    "datasets",
    "torchcodec",
    "av",
    "opencv-python",
    "wandb",
    "diffusers",
    "transformers",
]:
    print(f"{pkg:16s}", v(pkg))

print("\nvideo backends:")
for mod in ["torchcodec", "av", "cv2"]:
    print(f"  {mod:12s}", bool(importlib.util.find_spec(mod)))

print("\noffscreen render:")
import mujoco

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="box" size=".1 .1 .1" rgba=".8 .3 .3 1"/>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 640)
r.update_scene(d)
img = r.render()
print("  ", img.shape, img.dtype, "mean:", img.mean())
