from __future__ import annotations

import random
from pathlib import Path
from typing import Any


SUPPORTED_SPLITS = ('train', 'eval_in_distribution', 'eval_out_of_distribution')


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'ALFWorld config file not found: {path}')

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            'PyYAML is required to load ALFWorld config files. '
            'Install it with `pip install pyyaml` or `pip install alfworld`.'
        ) from exc

    with path.open('r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ValueError(f'ALFWorld config must be a YAML mapping: {path}')
    return config


def init_alfworld_env(
    config_path: str,
    split: str = 'eval_out_of_distribution',
    batch_size: int = 1,
    seed: int | None = None,
) -> Any:
    """Initialize an ALFWorld environment from a base_config.yaml file.

    ALFWorld's official entrypoint creates an environment class from
    ``config["env"]["type"]`` and then calls ``init_env(batch_size=...)``.
    """
    if batch_size != 1:
        raise ValueError('This benchmark wrapper currently supports batch_size=1 only.')
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f'Unsupported ALFWorld split {split!r}. Expected one of: {", ".join(SUPPORTED_SPLITS)}.')

    config = _load_yaml_config(config_path)
    env_config = config.get('env')
    if not isinstance(env_config, dict) or not env_config.get('type'):
        raise ValueError('ALFWorld config must include env.type.')
    env_type = env_config['type']

    try:
        from alfworld.agents.environment import get_environment
    except ImportError as exc:
        raise RuntimeError(
            'ALFWorld is not installed. Install the text environment with '
            '`pip install alfworld` and download data with `alfworld-download`.'
        ) from exc

    env_cls = get_environment(env_type)
    env = env_cls(config, train_eval=split)
    if seed is not None and hasattr(env, 'game_files'):
        random.Random(seed).shuffle(env.game_files)
        env.num_games = len(env.game_files)
    return env.init_env(batch_size=batch_size)
