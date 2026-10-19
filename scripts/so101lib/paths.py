"""Repo-relative paths.

Every path used by a script is derived from REPO_ROOT rather than assumed to be
relative to the current working directory, so scripts run the same from anywhere
(project-plan.md: "use pathlib everywhere and avoid hardcoded separators or
absolute paths in scripts").
"""
from pathlib import Path

# scripts/so101lib/paths.py -> parents[0]=so101lib, [1]=scripts, [2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

ASSETS_DIR = REPO_ROOT / "assets" / "so101"
SCENE_XML = ASSETS_DIR / "scene_lift_cube.xml"
MODEL_XML = ASSETS_DIR / "so101_new_calib.xml"

CONFIG_DIR = REPO_ROOT / "config"
CALIB_JSON = CONFIG_DIR / "so101_calib.json"

OUT_DIR = REPO_ROOT / "out"
