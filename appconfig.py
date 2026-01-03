# appconfig.py
from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.ini"

DEFAULTS = {
    "paths": {
        "style_path": "style/general_style.qss",
    },
    "startup": {
        "toolbar": "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
        "data": "pages.wd_create_analise:WdCreateAnalisePane",
        # legado (se algum ini antigo existir)
        "toolbar_module": "pages.individual.toolbar.tb_ind_toolbar",
        "toolbar_class": "TbIndToolbar",
        "data_module": "pages.wd_create_analise",
        "data_class": "WdCreateAnalisePane",
    },
}


def _ensure_absolute(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p).resolve()


def _to_ini_text(cfg: ConfigParser) -> str:
    from io import StringIO

    buf = StringIO()
    cfg.write(buf)
    return buf.getvalue()


def ensure_config() -> ConfigParser:
    cfg = ConfigParser()

    if CONFIG_PATH.exists():
        cfg.read(CONFIG_PATH, encoding="utf-8")
    else:
        cfg["paths"] = dict(DEFAULTS["paths"])
        cfg["startup"] = {
            "toolbar": DEFAULTS["startup"]["toolbar"],
            "data": DEFAULTS["startup"]["data"],
        }
        CONFIG_PATH.write_text(_to_ini_text(cfg), encoding="utf-8")
        return cfg

    if "paths" not in cfg:
        cfg["paths"] = {}
    if "startup" not in cfg:
        cfg["startup"] = {}

    if not cfg.has_option("paths", "style_path"):
        cfg.set("paths", "style_path", DEFAULTS["paths"]["style_path"])

    # Garantir novo formato
    if not cfg.has_option("startup", "toolbar"):
        tm = cfg.get("startup", "toolbar_module", fallback=DEFAULTS["startup"]["toolbar_module"])
        tc = cfg.get("startup", "toolbar_class", fallback=DEFAULTS["startup"]["toolbar_class"])
        cfg.set("startup", "toolbar", f"{tm}:{tc}")

    if not cfg.has_option("startup", "data"):
        dm = cfg.get("startup", "data_module", fallback=DEFAULTS["startup"]["data_module"])
        dc = cfg.get("startup", "data_class", fallback=DEFAULTS["startup"]["data_class"])
        cfg.set("startup", "data", f"{dm}:{dc}")

    # Persistir normalizado (ajuda a não voltar erro)
    try:
        CONFIG_PATH.write_text(_to_ini_text(cfg), encoding="utf-8")
    except Exception:
        pass

    return cfg


def get_style_path(cfg: ConfigParser) -> str:
    raw = cfg.get("paths", "style_path", fallback=DEFAULTS["paths"]["style_path"])
    p = _ensure_absolute(ROOT, raw)
    return str(p)


def get_startup_specs(cfg: ConfigParser) -> Tuple[str, str]:
    toolbar = cfg.get("startup", "toolbar", fallback=DEFAULTS["startup"]["toolbar"])
    data = cfg.get("startup", "data", fallback=DEFAULTS["startup"]["data"])
    return toolbar, data
