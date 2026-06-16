from __future__ import annotations

import os
from pathlib import Path


DEFAULT_APPWORLD_WORK_DIR = '/tmp/workfile'


def prepare_appworld_work_dir(work_dir: str | None = None) -> Path:
    """Route AppWorld/LazyMind relative runtime files away from the repo root."""
    raw = str(work_dir or os.getenv('LAZYMIND_APPWORLD_WORK_DIR') or DEFAULT_APPWORLD_WORK_DIR).strip()
    target = Path(raw).expanduser()
    workspace = target / 'workspace'
    temp_dir = target / 'tmp'
    sqlite_temp_dir = target / 'sqlite_tmp'

    for path in (target, workspace, temp_dir, sqlite_temp_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ['LAZYMIND_APPWORLD_WORK_DIR'] = str(target)
    os.environ['LAZYMIND_AGENTIC_WORKSPACE'] = str(workspace)
    os.environ['TMPDIR'] = str(temp_dir)
    os.environ['SQLITE_TMPDIR'] = str(sqlite_temp_dir)
    os.chdir(target)
    return target
