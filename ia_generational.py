from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from json_lib import build_paths, load_json, save_json


@dataclass
class MutationConfig:
    allow_add_layers: bool
    allow_remove_layers: bool
    max_layer_delta: int

    allow_add_neurons: bool
    allow_remove_neurons: bool
    max_neuron_delta: int


def _ensure_gen_dir(parent_folder: Path) -> Path:
    """
    Garante a pasta de geração para o parent:
        dados/{parent}/geracao

    Nota: recebemos parent_folder diretamente (evita recomputar build_paths e
    possíveis edge-cases de nomes com underscores/dígitos).
    """
    gen_dir = parent_folder / "geracao"
    gen_dir.mkdir(parents=True, exist_ok=True)
    return gen_dir


def _extract_structure_neurons(struct: Dict[str, Any]) -> Tuple[List[int], Optional[int]]:
    """Interpreta struct['neurons'] e tenta detectar se o último item é a saída fixa."""
    neurons = struct.get("neurons")

    if isinstance(neurons, str):
        parts = [p.strip() for p in neurons.split(",") if p.strip()]
        try:
            neurons = [int(p) for p in parts]
        except Exception:
            neurons = None

    if isinstance(neurons, (int, float)):
        neurons = [int(neurons)]

    if not isinstance(neurons, list) or not neurons:
        return [], None

    out_size = struct.get("output_size")
    if isinstance(out_size, (int, float)) and int(out_size) > 0:
        out_size = int(out_size)
    else:
        out_size = None

    if out_size is not None and int(neurons[-1]) == int(out_size):
        hidden = [int(x) for x in neurons[:-1]] or []
        return hidden, out_size

    return [int(x) for x in neurons], None


def _write_structure_neurons(struct: Dict[str, Any], hidden: List[int], output_fixed: Optional[int]) -> None:
    neurons_out: List[int] = [max(1, int(x)) for x in (hidden or [8])]
    if output_fixed is not None:
        neurons_out.append(int(output_fixed))

    struct["neurons"] = neurons_out
    struct["layers"] = len(neurons_out)


def _resize_weights_like(input_size: int, neurons: List[int], old_weights: Any) -> List[List[List[float]]]:
    """Ajusta weights para bater com input_size e neurons, preservando o máximo possível."""
    if input_size <= 0:
        input_size = 1

    if not isinstance(old_weights, list):
        old_weights = []

    new_weights: List[List[List[float]]] = []
    prev_in = int(input_size)

    for li, n_out in enumerate(neurons):
        n_out = max(1, int(n_out))
        old_layer = old_weights[li] if li < len(old_weights) and isinstance(old_weights[li], list) else []
        layer_rows: List[List[float]] = []

        for ni in range(n_out):
            if ni < len(old_layer) and isinstance(old_layer[ni], list) and len(old_layer[ni]) >= 1:
                row = old_layer[ni]
                try:
                    bias = float(row[0])
                except Exception:
                    bias = 0.0

                w_raw = row[1:] if len(row) > 1 else []
                w: List[float] = []
                for x in w_raw:
                    try:
                        w.append(float(x))
                    except Exception:
                        w.append(0.0)
            else:
                bias = random.uniform(-0.05, 0.05)
                w = [random.uniform(-0.05, 0.05) for _ in range(prev_in)]

            if len(w) < prev_in:
                w += [random.uniform(-0.05, 0.05) for _ in range(prev_in - len(w))]
            elif len(w) > prev_in:
                w = w[:prev_in]

            layer_rows.append([bias] + w)

        new_weights.append(layer_rows)
        prev_in = n_out

    return new_weights


def _sync_structure_after_mutation(struct: Dict[str, Any]) -> None:
    """Garante consistência mínima do structure após mutação."""
    input_size = int(struct.get("input_size") or 0)
    out_size = int(struct.get("output_size") or 0)

    neurons_raw = struct.get("neurons") or []
    if isinstance(neurons_raw, str):
        parts = [p.strip() for p in neurons_raw.split(",") if p.strip()]
        neurons = [int(p) for p in parts] if parts else []
    elif isinstance(neurons_raw, (int, float)):
        neurons = [int(neurons_raw)]
    else:
        neurons = [int(x) for x in neurons_raw] if isinstance(neurons_raw, list) else []

    if not neurons:
        neurons = [max(1, out_size or 1)]

    if out_size > 0 and int(neurons[-1]) != out_size:
        neurons[-1] = out_size

    neurons = [max(1, int(x)) for x in neurons]
    struct["neurons"] = neurons
    struct["layers"] = len(neurons)

    act = struct.get("activation") or []
    if isinstance(act, (str, type(None))):
        activation: List[Optional[str]] = [act] * len(neurons)
    elif isinstance(act, list):
        activation = [a if (a is None or isinstance(a, str)) else None for a in act]
    else:
        activation = []

    if len(activation) < len(neurons):
        activation += [None] * (len(neurons) - len(activation))
    elif len(activation) > len(neurons):
        activation = activation[: len(neurons)]
    struct["activation"] = activation

    struct["weights"] = _resize_weights_like(input_size, neurons, struct.get("weights"))


def mutate_manifest(manifest: Dict[str, Any], cfg: MutationConfig) -> Dict[str, Any]:
    """Aplica mutação no structure (camadas/neurônios), sem alterar input/output_size."""
    out = copy.deepcopy(manifest)
    struct = out.get("structure") or {}
    if not isinstance(struct, dict):
        struct = {}
    out["structure"] = struct

    hidden, output_fixed = _extract_structure_neurons(struct)
    if not hidden:
        hidden = [8]

    max_ld = max(0, int(cfg.max_layer_delta))
    if max_ld > 0:
        if cfg.allow_add_layers:
            add_n = random.randint(0, max_ld)
            for _ in range(add_n):
                base = int(sum(hidden) / len(hidden)) if hidden else 8
                new_n = max(1, int(base + random.randint(-max(1, base // 4), max(1, base // 4))))
                hidden.append(new_n)

        if cfg.allow_remove_layers:
            rem_n = random.randint(0, max_ld)
            for _ in range(rem_n):
                if len(hidden) <= 1:
                    break
                idx = random.randrange(0, len(hidden))
                hidden.pop(idx)

    max_nd = max(0, int(cfg.max_neuron_delta))
    if max_nd > 0 and hidden:
        idx = random.randrange(0, len(hidden))
        delta = random.randint(0, max_nd)

        if cfg.allow_add_neurons and delta > 0:
            hidden[idx] = max(1, hidden[idx] + delta)

        if cfg.allow_remove_neurons and delta > 0:
            hidden[idx] = max(1, hidden[idx] - delta)

    _write_structure_neurons(struct, hidden, output_fixed)
    _sync_structure_after_mutation(struct)
    return out


def _find_parent_weight_file(parent_folder: Path, parent_sanit: str) -> Optional[Path]:
    """Procura arquivo de pesos do parent (opcional)."""
    candidates = [
        parent_folder / f"{parent_sanit}.pth",
        parent_folder / f"{parent_sanit}.pt",
        parent_folder / f"{parent_sanit}.bin",
    ]
    for p in candidates:
        if p.exists():
            return p
    for suf in (".pth", ".pt", ".bin"):
        for p in parent_folder.glob(f"{parent_sanit}*{suf}"):
            if p.is_file():
                return p
    return None


def create_generation_individual(
    parent_name: str,
    generation_index: int,
    individual_id: int,
    cfg: MutationConfig,
) -> str:
    """
    Cria um indivíduo e salva em:
      dados/{parent}/geracao/{parent}_{geracao}_{id}.json

    Retorna o nome do indivíduo (para train_networks/build_engine).
    """
    parent_folder, parent_json, parent_sanit = build_paths(parent_name)
    gen_dir = _ensure_gen_dir(parent_folder)

    if not parent_json.exists():
        raise FileNotFoundError(f"Manifesto do parent não encontrado: {parent_json}")

    parent_data = load_json(parent_json)
    if not isinstance(parent_data, dict):
        raise ValueError(f"Manifesto do parent inválido (não é dict): {parent_json}")

    child_name = f"{parent_sanit}_{int(generation_index)}_{int(individual_id)}"
    child_data = copy.deepcopy(parent_data)

    ident = child_data.get("identification") or {}
    if not isinstance(ident, dict):
        ident = {}
    ident["parent"] = parent_sanit
    ident["generation"] = int(generation_index)
    ident["individual_id"] = int(individual_id)

    # Mantém compatibilidade: o projeto costuma guardar o nome como "arquivo"
    ident["name"] = f"{child_name}.json"
    child_data["identification"] = ident

    child_data = mutate_manifest(child_data, cfg)

    stats = child_data.get("stats") or {}
    if not isinstance(stats, dict):
        stats = {}
    stats.setdefault("accuracy", None)
    stats.setdefault("loss", None)
    stats.setdefault("last_train_time", None)
    child_data["stats"] = stats

    child_json = gen_dir / f"{child_name}.json"
    save_json(child_json, child_data)

    # Copia pesos do parent se existir (opcional)
    parent_weights = _find_parent_weight_file(parent_folder, parent_sanit)
    if parent_weights is not None and parent_weights.exists():
        ext = parent_weights.suffix or ".pth"
        child_weights = gen_dir / f"{child_name}{ext}"
        try:
            child_weights.write_bytes(parent_weights.read_bytes())
        except Exception:
            pass

    return child_name


def rank_top_k(results: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """
    Ordena resultados por:
      - accuracy/acc desc
      - loss/final_loss/loss_final asc

    Aceita chaves alternativas para manter compatibilidade entre versões.
    """
    kk = max(1, int(k))

    def _get_float(d: Dict[str, Any], keys: List[str], default: float) -> float:
        for key in keys:
            if key in d and d[key] is not None:
                try:
                    return float(d[key])
                except Exception:
                    pass
        return float(default)

    def key(r: Dict[str, Any]) -> Tuple[float, float]:
        acc = _get_float(r, ["accuracy", "acc"], default=-1.0)
        loss = _get_float(r, ["loss", "final_loss", "loss_final", "avg_loss", "loss_avg"], default=1e18)
        return (-acc, loss)

    return sorted(results, key=key)[:kk]
