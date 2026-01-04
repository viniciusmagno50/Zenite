from __future__ import annotations

from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TrainingDefaults:
    # Padrões (UI)
    epochs_default: int = 20
    lr_default: float = 0.01
    lr_mult_default: float = 0.95
    lr_min_default: float = 0.00001

    data_update_default: int = 1000
    eval_limit_default: int = 0  # 0 = sem limite

    # Limites razoáveis para evitar valores absurdos
    epochs_min: int = 1
    epochs_max: int = 1_000_000

    lr_min: float = 1e-12
    lr_max: float = 1e3

    data_update_min: int = 1
    data_update_max: int = 10_000_000

    eval_limit_min: int = 0
    eval_limit_max: int = 1_000_000


def _find_config_ini(start: Optional[Path] = None) -> Optional[Path]:
    """
    Procura config.ini subindo diretórios, começando em:
    - start (se fornecido)
    - cwd
    """
    candidates = []
    if start:
        candidates.append(start)
    candidates.append(Path.cwd())

    for base in candidates:
        base = base.resolve()
        for _ in range(8):
            p = base / "config.ini"
            if p.exists():
                return p
            if base.parent == base:
                break
            base = base.parent
    return None


def _get_int(cp: ConfigParser, section: str, key: str, default: int) -> int:
    try:
        return cp.getint(section, key, fallback=default)
    except Exception:
        return default


def _get_float(cp: ConfigParser, section: str, key: str, default: float) -> float:
    try:
        return cp.getfloat(section, key, fallback=default)
    except Exception:
        return default


def load_training_defaults(config_path: Optional[str | Path] = None) -> TrainingDefaults:
    """
    Carrega defaults de treino. Se config.ini não tiver [training], usa fallback.

    Exemplo opcional no config.ini:

    [training]
    epochs_default = 20
    lr_default = 0.01
    lr_mult_default = 0.95
    lr_min_default = 0.00001
    data_update_default = 1000
    eval_limit_default = 0
    """
    cfg = TrainingDefaults()

    path = Path(config_path).resolve() if config_path else _find_config_ini()
    if not path or not path.exists():
        return cfg

    cp = ConfigParser()
    try:
        cp.read(path, encoding="utf-8")
    except Exception:
        return cfg

    if not cp.has_section("training"):
        return cfg

    epochs = _get_int(cp, "training", "epochs_default", cfg.epochs_default)
    epochs = max(cfg.epochs_min, min(cfg.epochs_max, epochs))

    lr = _get_float(cp, "training", "lr_default", cfg.lr_default)
    lr = max(cfg.lr_min, min(cfg.lr_max, lr))

    lr_mult = _get_float(cp, "training", "lr_mult_default", cfg.lr_mult_default)
    lr_min = _get_float(cp, "training", "lr_min_default", cfg.lr_min_default)

    data_update = _get_int(cp, "training", "data_update_default", cfg.data_update_default)
    data_update = max(cfg.data_update_min, min(cfg.data_update_max, data_update))

    eval_limit = _get_int(cp, "training", "eval_limit_default", cfg.eval_limit_default)
    eval_limit = max(cfg.eval_limit_min, min(cfg.eval_limit_max, eval_limit))

    return TrainingDefaults(
        epochs_default=epochs,
        lr_default=lr,
        lr_mult_default=lr_mult,
        lr_min_default=lr_min,
        data_update_default=data_update,
        eval_limit_default=eval_limit,
        epochs_min=cfg.epochs_min,
        epochs_max=cfg.epochs_max,
        lr_min=cfg.lr_min,
        lr_max=cfg.lr_max,
        data_update_min=cfg.data_update_min,
        data_update_max=cfg.data_update_max,
        eval_limit_min=cfg.eval_limit_min,
        eval_limit_max=cfg.eval_limit_max,
    )
