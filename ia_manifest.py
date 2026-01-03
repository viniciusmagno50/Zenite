# ia_manifest.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Structure:
    """
    Representa a parte 'structure' do manifesto de IA.
    Convenção:
      - weights: [por camada][por neurônio][bias, w1..wN]
    """
    input_size: int
    output_size: int
    layers: int
    neurons: List[int]
    activation: List[Optional[str]]
    weights: List[List[List[float]]]


@dataclass
class IAManifest:
    """
    Manifesto completo de uma IA individual.
    """
    schema_version: int
    identification: Dict[str, Any]
    structure: Structure
    mutation: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    functions: Dict[str, Any] = field(default_factory=dict)

    # ----------------- Fábrica a partir de dict -----------------
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "IAManifest":
        """
        Constrói um IAManifest a partir de um dict (ex.: carregado do JSON).
        Faz conversões/ajustes leves para robustez.
        Levanta ValueError se algo essencial estiver ausente/inválido.

        Robustez adicionada:
          - aceita 'neurons' vazio (fluxo de criação/análise)
          - tenta inferir neurons a partir de 'weights'
          - fallback seguro: neurons = [output_size] (rede direta)
        """
        if not isinstance(data, dict):
            raise ValueError("Manifesto deve ser um dict.")

        s_raw = data.get("structure") or {}
        if not isinstance(s_raw, dict):
            raise ValueError("Campo 'structure' deve ser um dict.")

        input_size = int(s_raw.get("input_size") or 0)
        output_size = int(s_raw.get("output_size") or 0)
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size/output_size devem ser maiores que zero.")

        # ---------------- neurons (pode estar vazio no fluxo de criação) ----------------
        neurons_raw = s_raw.get("neurons")
        if neurons_raw is None:
            neurons = []
        elif isinstance(neurons_raw, int):
            neurons = [int(neurons_raw)]
        else:
            try:
                neurons = [int(n) for n in list(neurons_raw)]
            except Exception:
                neurons = []

        # ---------------- weights: garante que seja lista de listas de floats ----------------
        weights_conv: List[List[List[float]]] = []
        weights_raw = s_raw.get("weights") or []
        if not isinstance(weights_raw, list):
            weights_raw = []

        for layer in weights_raw:
            if not isinstance(layer, list):
                continue
            layer_rows: List[List[float]] = []
            for row in layer:
                if not isinstance(row, list):
                    continue
                try:
                    layer_rows.append([float(x) for x in row])
                except Exception:
                    # Se uma linha vier corrompida, ignora só aquela linha
                    continue
            weights_conv.append(layer_rows)

        # Se neurons estiver vazio, tenta inferir pelo nº de neurônios por camada (len(layer))
        if not neurons:
            inferred: List[int] = []
            for layer in weights_conv:
                # cada layer é uma lista de neurônios/linhas
                if isinstance(layer, list) and len(layer) > 0:
                    inferred.append(int(len(layer)))

            # Se inferiu algo, usa.
            # Observação: isso pode incluir a camada de saída; está ok para o app,
            # porque o painel precisa de tamanhos para construir inputs/outputs.
            if inferred:
                neurons = inferred
            else:
                # fallback seguro: rede direta input -> output
                neurons = [int(output_size)]

        # ---------------- layers: consistência com len(neurons) ----------------
        layers_val = int(s_raw.get("layers") or len(neurons))
        if layers_val != len(neurons):
            layers_val = len(neurons)

        # ---------------- activation: alinhada com neurons ----------------
        activ_raw = s_raw.get("activation") or []
        if isinstance(activ_raw, (str, type(None))):
            activation = [activ_raw] * len(neurons)
        else:
            activation = list(activ_raw)

        if len(activation) < len(neurons):
            activation += [None] * (len(neurons) - len(activation))
        elif len(activation) > len(neurons):
            activation = activation[:len(neurons)]

        structure = Structure(
            input_size=input_size,
            output_size=output_size,
            layers=layers_val,
            neurons=neurons,
            activation=activation,
            weights=weights_conv,
        )

        return IAManifest(
            schema_version=int(data.get("schema_version") or 1),
            identification=dict(data.get("identification") or {}),
            structure=structure,
            mutation=dict(data.get("mutation") or {}),
            stats=dict(data.get("stats") or {}),
            functions=dict(data.get("functions") or {}),
        )

    # ----------------- Conversão para dict -----------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "identification": dict(self.identification),
            "structure": {
                "input_size": int(self.structure.input_size),
                "output_size": int(self.structure.output_size),
                "layers": int(self.structure.layers),
                "neurons": list(self.structure.neurons),
                "activation": list(self.structure.activation),
                "weights": [
                    [[float(x) for x in row] for row in layer]
                    for layer in self.structure.weights
                ],
            },
            "mutation": dict(self.mutation),
            "stats": dict(self.stats),
            "functions": dict(self.functions),
        }
