from __future__ import annotations

import random
from pathlib import Path


SUPPORTED_DATASETS = ('train', 'dev', 'test_normal', 'test_challenge')


def load_task_ids(data_root: str, dataset: str = 'dev') -> list[str]:
    dataset_name = str(dataset or '').strip()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f'Unsupported AppWorld dataset {dataset_name!r}. '
            f'Expected one of: {", ".join(SUPPORTED_DATASETS)}.'
        )

    dataset_file = Path(data_root).expanduser() / 'data' / 'datasets' / f'{dataset_name}.txt'
    if not dataset_file.exists():
        raise FileNotFoundError(f'AppWorld dataset file not found: {dataset_file}')

    task_ids = [
        line.strip().split(':', 1)[0]
        for line in dataset_file.read_text(encoding='utf-8').splitlines()
    ]
    return [task_id for task_id in task_ids if task_id]


def plan_task_ids(
    *,
    data_root: str,
    dataset: str = 'dev',
    episodes: int = 1,
    task_ids: list[str] | None = None,
    seed: int | None = None,
) -> list[str]:
    explicit = [str(task_id).strip() for task_id in (task_ids or []) if str(task_id).strip()]
    if explicit:
        return explicit[:episodes] if episodes > 0 else explicit
    if episodes <= 0:
        return []
    candidates = load_task_ids(data_root, dataset)
    if seed is not None:
        random.Random(seed).shuffle(candidates)
    return candidates[:episodes]
