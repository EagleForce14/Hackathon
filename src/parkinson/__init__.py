"""Parkinson motor score prediction — package root.

Exposes `PROJECT_ROOT`: the absolute path to the project root,
derived from this file's location. Import this constant from any
module that needs to resolve a project-relative path (data files,
reports, fixtures) so scripts run correctly from any working directory.

Requires editable install: `pip install -e .`
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
