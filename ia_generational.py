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


def _ensure_gen_dir(parent_name: str) -> Path:
    """Garante a pasta de geração SEMPRE em dados/{root_parent}/geracao.

    Suporta receber tanto:
      - nome do parent raiz (ex.: 'PETR4')
      - nome de um indivíduo geracional (ex.: 'PETR4_3_12')

    Isso evita criar estruturas aninhadas do tipo:
      dados/PETR4_3_12/geracao (INCORRETO)
    e garante que tudo fique em:
      dados/PETR4/geracao (CORRETO).
    """
    parent_folder, _, parent_sanit = build_paths(parent_name)

    # Se build_paths detectou que parent_name é geracional, parent_folder será:
    #   dados/{root_parent}/geracao
    # então o root_parent é o diretório pai.
    if parent_folder.name == "geracao":
        root_parent = parent_folder.parent.name
    else:
        root_parent = parent_sanit

    root_folder, _, root_sanit = build_paths(root_parent)
    # root_folder aqui é dados/{root_sanit}
    gen_dir = root_folder / "geracao"
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

    if not isinstance(neurons, list) or not neurons:
        return [], None

    clean: List[int] = []
    for v in neurons:
        try:
            iv = int(v)
        except Exception:
            continue
        if iv > 0:
            clean.append(iv)

    if len(clean) < 2:
        return clean, None

    # Heurística: se tiver pelo menos 2 camadas, assume que a última é saída fixa
    out_size = clean[-1]
    hidden = clean[:-1]
    return hidden, out_size


def _mutate_hidden_layers(hidden: List[int], cfg: MutationConfig) -> List[int]:
    """Aplica mutações nas camadas ocultas."""
    mutated = list(hidden)

    # Adiciona/remove camadas
    if cfg.allow_add_layers and cfg.max_layer_delta > 0:
        add_n = random.randint(0, cfg.max_layer_delta)
        for _ in range(add_n):
            # adiciona camada com tamanho similar às existentes (ou default)
            base = mutated[-1] if mutated else 8
            delta = random.randint(-max(1, cfg.max_neuron_delta), max(1, cfg.max_neuron_delta)) if cfg.max_neuron_delta > 0 else 0
            mutated.append(max(1, base + delta))

    if cfg.allow_remove_layers and cfg.max_layer_delta > 0 and mutated:
        rem_n = random.randint(0, min(cfg.max_layer_delta, len(mutated)))
        for _ in range(rem_n):
            if mutated:
                mutated.pop(random.randrange(0, len(mutated)))

    # Adiciona/remove neurônios em camadas existentes
    if cfg.max_neuron_delta > 0 and mutated:
        for i in range(len(mutated)):
            if cfg.allow_add_neurons and random.random() < 0.50:
                mutated[i] += random.randint(0, cfg.max_neuron_delta)
            if cfg.allow_remove_neurons and random.random() < 0.50:
                mutated[i] -= random.randint(0, cfg.max_neuron_delta)
            mutated[i] = max(1, mutated[i])

    return mutated


def _find_parent_weight_file(parent_folder: Path, parent_sanit: str) -> Optional[Path]:
    """Procura arquivo de pesos do parent (se existir)."""
    # tentativas comuns
    candidates = [
        parent_folder / f"{parent_sanit}.pth",
        parent_folder / f"{parent_sanit}.pt",
        parent_folder / f"{parent_sanit}.bin",
        parent_folder / f"{parent_sanit}.weights",
    ]
    for p in candidates:
        if p.exists():
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
      dados/{root_parent}/geracao/{root_parent}_{geracao}_{id}.json

    Retorna o nome do indivíduo (para train_networks/build_engine).
    """
    parent_folder, parent_json, parent_sanit = build_paths(parent_name)
    # Pasta de geração deve sempre ficar em dados/{root_parent}/geracao
    gen_dir = _ensure_gen_dir(parent_name)

    if not parent_json.exists():
        raise FileNotFoundError(f"Manifesto do parent não encontrado: {parent_json}")

    parent_data = load_json(parent_json)
    if not isinstance(parent_data, dict):
        raise ValueError(f"Manifesto do parent inválido (não é dict): {parent_json}")

    # IMPORTANTÍSSIMO:
    # Nome do novo indivíduo deve SEMPRE partir do root_parent (gen_dir.parent.name),
    # para não gerar cadeias aninhadas do tipo Pos_Neg_1_2_2_0.
    root_parent = gen_dir.parent.name
    child_name = f"{root_parent}_{int(generation_index)}_{int(individual_id)}"
    child_data = copy.deepcopy(parent_data)

    ident = child_data.get("identification") or {}
    if not isinstance(ident, dict):
        ident = {}
    ident["name"] = child_name
    ident["parent"] = parent_sanit  # mantém rastreio do parent real
    ident["generation"] = int(generation_index)
    ident["individual_id"] = int(individual_id)
    child_data["identification"] = ident

    struct = child_data.get("structure") or {}
    if not isinstance(struct, dict):
        struct = {}

    hidden, out_size = _extract_structure_neurons(struct)

    mutated_hidden = _mutate_hidden_layers(hidden, cfg)

    # Remonta neurons preservando saída fixa se detectada
    if out_size is not None:
        struct["neurons"] = list(mutated_hidden) + [int(out_size)]
    else:
        # fallback: se não detectou saída, mantém o que tiver ou coloca mutated_hidden
        struct["neurons"] = list(mutated_hidden) if mutated_hidden else struct.get("neurons", [])

    child_data["structure"] = struct

    # Salva JSON do indivíduo
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
    """Rank simples: prioriza maior acc e menor loss."""
    if not results:
        return []
    k = max(1, int(k))
    def _key(r: Dict[str, Any]) -> Tuple[float, float]:
        acc = float(r.get("acc", 0.0) or 0.0)
        loss = float(r.get("loss", 1e18) or 1e18)
        # sort: acc desc, loss asc
        return (-acc, loss)
    ranked = sorted(results, key=_key)
    return ranked[:k]
