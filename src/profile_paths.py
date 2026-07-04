"""Profile-aware paths for datasets, models, and reports."""

from __future__ import annotations

from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
PROFILES_ROOT = ROOT_PATH / "profiles"

DEFAULT_PROFILE = "default"
LEGACY_MODEL_PATH = ROOT_PATH / "models" / "gesture_mlp.joblib"
LEGACY_DATASET_PATH = ROOT_PATH / "data" / "gesture_landmarks.jsonl"


def _sanitize_profile_name(profile: str) -> str:
    cleaned = "".join(c for c in profile.strip().lower() if c.isalnum() or c in {"-", "_"})
    return cleaned or DEFAULT_PROFILE


def profile_dir(profile: str) -> Path:
    name = _sanitize_profile_name(profile)
    return PROFILES_ROOT / name


def model_path_for_profile(profile: str) -> Path:
    if _sanitize_profile_name(profile) == DEFAULT_PROFILE:
        return LEGACY_MODEL_PATH
    return profile_dir(profile) / "gesture_mlp.joblib"


def dataset_path_for_profile(profile: str) -> Path:
    if _sanitize_profile_name(profile) == DEFAULT_PROFILE:
        return LEGACY_DATASET_PATH
    return profile_dir(profile) / "gesture_landmarks.jsonl"


def reports_dir_for_profile(profile: str) -> Path:
    if _sanitize_profile_name(profile) == DEFAULT_PROFILE:
        return ROOT_PATH / "reports" / DEFAULT_PROFILE
    return profile_dir(profile) / "reports"
