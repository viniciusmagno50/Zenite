# appconfig.py

from __future__ import annotations

from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path
from typing import Tuple

# Raiz do projeto (pasta onde está este arquivo)
ROOT = Path(__file__).resolve().parent

# Caminho absoluto para o config.ini
CONFIG_PATH = ROOT / "config.ini"

# Valores padrão (NUNCA remover chaves existentes; apenas complementar)
DEFAULTS = {
    "paths": {
        "style_path": "style/general_style.qss",
    },
    "startup": {
        "toolbar": "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
        "data": "pages.wd_create_analise:WdCreateAnalisePane",
        "toolbar_module": "pages.individual.toolbar.tb_ind_toolbar",
        "toolbar_class": "TbIndToolbar",
        "data_module": "pages.wd_create_analise",
        "data_class": "WdCreateAnalisePane",
    },
    "training": {
        "epochs_default": "10",
        "learning_rate_default": "0.01",
        "learning_rate_mult_default": "0.95",
        "learning_rate_min_default": "0.00001",
        "data_update_default": "1000",
        "eval_limit_default": "0",
        "shuffle_default": "1",
        "train_generations_default": "0",
        "generations_default": "3",
        "population_default": "10",
        "allow_add_layers_default": "1",
        "allow_remove_layers_default": "1",
        "allow_add_neurons_default": "1",
        "allow_remove_neurons_default": "1",
        "layer_delta_default": "1",
        "neuron_delta_default": "5",
        "show_top_default": "3",
        "show_original_default": "1",
        "output_index_default": "0",
        "plot_minmax_default": "1",
        "plot_loss_epoch_default": "1",
        "plot_acc_default": "1",
        "plot_dist_default": "1",
        "plot_lr_default": "1",
        "plot_perf_default": "1",
    },
}


def _ensure_absolute(root: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    return p


def _safe_get(cfg: ConfigParser, section: str, key: str, fallback: str) -> str:
    try:
        return cfg.get(section, key, fallback=fallback)
    except Exception:
        return fallback


def _get_int(cfg: ConfigParser, section: str, key: str, fallback: int) -> int:
    try:
        return int(_safe_get(cfg, section, key, str(fallback)))
    except Exception:
        return int(fallback)


def _get_float(cfg: ConfigParser, section: str, key: str, fallback: float) -> float:
    try:
        return float(_safe_get(cfg, section, key, str(fallback)))
    except Exception:
        return float(fallback)


def _get_bool(cfg: ConfigParser, section: str, key: str, fallback: bool) -> bool:
    raw = _safe_get(cfg, section, key, "1" if fallback else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def ensure_config() -> ConfigParser:
    """
    Garante que o config.ini exista e contenha ao menos os DEFAULTS.
    NÃO remove nem sobrescreve chaves existentes.
    Escreve de volta somente se algo foi adicionado.
    """
    cfg = ConfigParser()

    if CONFIG_PATH.exists():
        cfg.read(CONFIG_PATH, encoding="utf-8")

    changed = False

    # 1) garante seções e chaves padrão
    for section, values in DEFAULTS.items():
        if not cfg.has_section(section):
            cfg.add_section(section)
            changed = True
        for key, value in values.items():
            if not cfg.has_option(section, key):
                cfg.set(section, key, value)
                changed = True

    # 2) se tiver module/class mas não tiver spec (toolbar/data), cria spec automaticamente
    if cfg.has_section("startup"):
        tm = _safe_get(cfg, "startup", "toolbar_module", DEFAULTS["startup"]["toolbar_module"]).strip()
        tc = _safe_get(cfg, "startup", "toolbar_class", DEFAULTS["startup"]["toolbar_class"]).strip()
        dm = _safe_get(cfg, "startup", "data_module", DEFAULTS["startup"]["data_module"]).strip()
        dc = _safe_get(cfg, "startup", "data_class", DEFAULTS["startup"]["data_class"]).strip()

        if tm and tc and not cfg.has_option("startup", "toolbar"):
            cfg.set("startup", "toolbar", f"{tm}:{tc}")
            changed = True

        if dm and dc and not cfg.has_option("startup", "data"):
            cfg.set("startup", "data", f"{dm}:{dc}")
            changed = True

    # 3) salva se necessário
    if (not CONFIG_PATH.exists()) or changed:
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            cfg.write(f)

    return cfg


def get_style_path(cfg: ConfigParser) -> str:
    """
    Retorna o caminho ABSOLUTO do arquivo de estilo (QSS).
    """
    raw = _safe_get(cfg, "paths", "style_path", DEFAULTS["paths"]["style_path"])
    return str(_ensure_absolute(ROOT, raw))


# ----------------------------------------------------------------------
# Startup (duas formas: module/class OU spec)
# ----------------------------------------------------------------------
def get_startup_pages(cfg: ConfigParser) -> Tuple[str, str, str, str]:
    """
    Formato original: retorna (toolbar_module, toolbar_class, data_module, data_class).
    """
    tm = _safe_get(cfg, "startup", "toolbar_module", DEFAULTS["startup"]["toolbar_module"])
    tc = _safe_get(cfg, "startup", "toolbar_class", DEFAULTS["startup"]["toolbar_class"])
    dm = _safe_get(cfg, "startup", "data_module", DEFAULTS["startup"]["data_module"])
    dc = _safe_get(cfg, "startup", "data_class", DEFAULTS["startup"]["data_class"])
    return tm, tc, dm, dc


def get_startup_specs(cfg: ConfigParser) -> Tuple[str, str]:
    """
    Formato spec: retorna (toolbar_spec, data_spec) no padrão 'modulo:Classe'.
    Se não existir, monta a partir do module/class.
    """
    toolbar = _safe_get(cfg, "startup", "toolbar", "").strip()
    data = _safe_get(cfg, "startup", "data", "").strip()

    if toolbar and ":" in toolbar and data and ":" in data:
        return toolbar, data

    tm, tc, dm, dc = get_startup_pages(cfg)
    return f"{tm}:{tc}", f"{dm}:{dc}"


def set_startup_pages(
    cfg: ConfigParser,
    toolbar_module: str,
    toolbar_class: str,
    data_module: str,
    data_class: str,
) -> None:
    """
    Atualiza as páginas de startup nos dois formatos e persiste.
    """
    if not cfg.has_section("startup"):
        cfg.add_section("startup")

    cfg.set("startup", "toolbar_module", toolbar_module)
    cfg.set("startup", "toolbar_class", toolbar_class)
    cfg.set("startup", "data_module", data_module)
    cfg.set("startup", "data_class", data_class)

    cfg.set("startup", "toolbar", f"{toolbar_module}:{toolbar_class}")
    cfg.set("startup", "data", f"{data_module}:{data_class}")

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        cfg.write(f)


# ----------------------------------------------------------------------
# Treinamento (defaults centralizados em config.ini)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingDefaults:
    epochs_default: int
    lr_default: float
    lr_mult_default: float
    lr_min_default: float
    data_update_default: int
    eval_limit_default: int

    shuffle_default: bool

    train_generations_default: bool
    generations_default: int
    population_default: int

    allow_add_layers_default: bool
    allow_remove_layers_default: bool
    allow_add_neurons_default: bool
    allow_remove_neurons_default: bool
    layer_delta_default: int
    neuron_delta_default: int

    show_top_default: int
    show_original_default: bool
    output_index_default: int

    plot_minmax_default: bool
    plot_loss_epoch_default: bool
    plot_acc_default: bool
    plot_dist_default: bool
    plot_lr_default: bool
    plot_perf_default: bool


def get_training_defaults(cfg: ConfigParser) -> TrainingDefaults:
    """
    Lê defaults da seção [training].
    Se não existir, usa DEFAULTS.
    """
    d = DEFAULTS["training"]
    sec = "training"

    return TrainingDefaults(
        epochs_default=_get_int(cfg, sec, "epochs_default", int(d["epochs_default"])),
        lr_default=_get_float(cfg, sec, "learning_rate_default", float(d["learning_rate_default"])),
        lr_mult_default=_get_float(cfg, sec, "learning_rate_mult_default", float(d["learning_rate_mult_default"])),
        lr_min_default=_get_float(cfg, sec, "learning_rate_min_default", float(d["learning_rate_min_default"])),
        data_update_default=_get_int(cfg, sec, "data_update_default", int(d["data_update_default"])),
        eval_limit_default=_get_int(cfg, sec, "eval_limit_default", int(d["eval_limit_default"])),
        shuffle_default=_get_bool(cfg, sec, "shuffle_default", d["shuffle_default"] == "1"),
        train_generations_default=_get_bool(cfg, sec, "train_generations_default", d["train_generations_default"] == "1"),
        generations_default=_get_int(cfg, sec, "generations_default", int(d["generations_default"])),
        population_default=_get_int(cfg, sec, "population_default", int(d["population_default"])),
        allow_add_layers_default=_get_bool(cfg, sec, "allow_add_layers_default", d["allow_add_layers_default"] == "1"),
        allow_remove_layers_default=_get_bool(cfg, sec, "allow_remove_layers_default", d["allow_remove_layers_default"] == "1"),
        allow_add_neurons_default=_get_bool(cfg, sec, "allow_add_neurons_default", d["allow_add_neurons_default"] == "1"),
        allow_remove_neurons_default=_get_bool(cfg, sec, "allow_remove_neurons_default", d["allow_remove_neurons_default"] == "1"),
        layer_delta_default=_get_int(cfg, sec, "layer_delta_default", int(d["layer_delta_default"])),
        neuron_delta_default=_get_int(cfg, sec, "neuron_delta_default", int(d["neuron_delta_default"])),
        show_top_default=_get_int(cfg, sec, "show_top_default", int(d["show_top_default"])),
        show_original_default=_get_bool(cfg, sec, "show_original_default", d["show_original_default"] == "1"),
        output_index_default=_get_int(cfg, sec, "output_index_default", int(d["output_index_default"])),
        plot_minmax_default=_get_bool(cfg, sec, "plot_minmax_default", d["plot_minmax_default"] == "1"),
        plot_loss_epoch_default=_get_bool(cfg, sec, "plot_loss_epoch_default", d["plot_loss_epoch_default"] == "1"),
        plot_acc_default=_get_bool(cfg, sec, "plot_acc_default", d["plot_acc_default"] == "1"),
        plot_dist_default=_get_bool(cfg, sec, "plot_dist_default", d["plot_dist_default"] == "1"),
        plot_lr_default=_get_bool(cfg, sec, "plot_lr_default", d["plot_lr_default"] == "1"),
        plot_perf_default=_get_bool(cfg, sec, "plot_perf_default", d["plot_perf_default"] == "1"),
    )
