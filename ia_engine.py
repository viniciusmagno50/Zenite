# ia_engine.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional, Dict, Any, Tuple

from json_lib import build_paths, load_json
from ia_manifest import IAManifest  # (mantido para compatibilidade/imports em outras partes)


# ---------------------------------------------------------
#  Detecção opcional de PyTorch (GPU / CPU)
# ---------------------------------------------------------
try:
    import torch  # type: ignore[attr-defined]
    _TORCH_AVAILABLE = True
    print("[IA] PyTorch detectado.")
except Exception:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    print("[IA] PyTorch não disponível. Usando engine em Python puro (CPU, sem treino).")


# ---------------------------------------------------------
#  Utilidades de manifesto
# ---------------------------------------------------------
def _resolve_manifest_path(name: str) -> Path:
    """
    Resolve o caminho do manifesto usando a convenção do projeto (json_lib.build_paths),
    incluindo suporte a redes geracionais:
      - dados/{nome}/{nome}.json
      - dados/{parent}/geracao/{parent}_{geracao}_{id}.json
    """
    _, arquivo, _ = build_paths(name)
    if arquivo.exists():
        return arquivo

    # Fallbacks antigos (compatibilidade)
    base = (name or "").strip()
    if base.endswith(".json"):
        base = base[:-5]

    candidates = [
        Path("data") / base / f"{base}.json",
        Path("dados") / base / f"{base}.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Manifesto da rede '{name}' não encontrado em:\n"
        + "\n".join([str(arquivo)] + [str(c) for c in candidates])
    )


def _load_manifest(name: str) -> Tuple[dict, Path]:
    path = _resolve_manifest_path(name)
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Manifesto de IA inválido (esperado dict): {path}")
    return data, path


# ---------------------------------------------------------
#  Softmax estável: garante 0..1 e soma = 1
# ---------------------------------------------------------
def softmax_stable(values: Sequence[float]) -> List[float]:
    v = [float(x) for x in values]
    if not v:
        return []
    if len(v) == 1:
        return [1.0]

    m = max(v)
    exps = [math.exp(x - m) for x in v]
    s = sum(exps)

    if not math.isfinite(s) or s <= 0.0:
        n = len(v)
        return [1.0 / n] * n

    out = [e / s for e in exps]

    # clamp + renormalização (garantia forte)
    out = [0.0 if (not math.isfinite(p) or p < 0.0) else p for p in out]
    total = sum(out)
    if total <= 0.0:
        n = len(out)
        return [1.0 / n] * n
    return [p / total for p in out]


# ---------------------------------------------------------
#  Normalização/Inicialização de pesos
#  Esperado: weights = [camada][neurônio][bias, w1..wN]
# ---------------------------------------------------------
def _make_zero_weights(input_size: int, neurons: List[int]) -> List[List[List[float]]]:
    weights: List[List[List[float]]] = []
    prev = int(input_size)
    for n in neurons:
        n = int(n)
        layer: List[List[float]] = []
        for _ in range(n):
            layer.append([0.0] + [0.0] * prev)  # bias + pesos
        weights.append(layer)
        prev = n
    return weights


def _normalize_weights(
    input_size: int,
    neurons: List[int],
    raw_weights: Any,
) -> List[List[List[float]]]:
    # Se weights vier int/None/str -> gera zeros (isso resolve sua fase de criação)
    if not isinstance(raw_weights, list) or not raw_weights:
        return _make_zero_weights(input_size, neurons)

    normalized: List[List[List[float]]] = []
    prev = int(input_size)

    for li, n_out in enumerate(neurons):
        n_out = int(n_out)
        layer_src = raw_weights[li] if li < len(raw_weights) else []
        if not isinstance(layer_src, list):
            layer_src = []

        layer_norm: List[List[float]] = []

        for row in layer_src:
            if not isinstance(row, list) or len(row) == 0:
                bias = 0.0
                w = [0.0] * prev
            else:
                try:
                    bias = float(row[0])
                except Exception:
                    bias = 0.0

                w_raw = row[1:]
                try:
                    w = [float(x) for x in w_raw]
                except Exception:
                    w = []

                if len(w) < prev:
                    w += [0.0] * (prev - len(w))
                elif len(w) > prev:
                    w = w[:prev]

            layer_norm.append([bias] + w)

        while len(layer_norm) < n_out:
            layer_norm.append([0.0] + [0.0] * prev)

        if len(layer_norm) > n_out:
            layer_norm = layer_norm[:n_out]

        normalized.append(layer_norm)
        prev = n_out

    return normalized


# ---------------------------------------------------------
#  Estruturas do engine
# ---------------------------------------------------------
@dataclass
class EngineInfo:
    name: str
    path: Path
    input_size: int
    output_size: int
    neurons: List[int]                 # camadas (sem input)
    activation: List[Optional[str]]    # por camada
    weights: List[List[List[float]]]   # normalizado


class BaseEngine:
    """
    forward() -> SEMPRE probabilidades (softmax)
    forward_logits() -> vetor bruto antes do softmax
    """

    def __init__(self, info: EngineInfo, manifest: Dict[str, Any]):
        self.info_data = info
        self.manifest = manifest

    @property
    def name(self) -> str:
        return self.info_data.name

    @property
    def input_size(self) -> int:
        return self.info_data.input_size

    @property
    def output_size(self) -> int:
        return self.info_data.output_size

    def forward_logits(self, inputs: Sequence[float]) -> List[float]:
        raise NotImplementedError

    def forward(self, inputs: Sequence[float]) -> List[float]:
        return softmax_stable(self.forward_logits(inputs))

    def info(self) -> str:
        return (
            f"Engine '{self.name}' | in={self.input_size} -> out={self.output_size} | "
            f"neurons={self.info_data.neurons}"
        )


# ---------------------------------------------------------
#  Engine em Python puro
# ---------------------------------------------------------
class PurePythonEngine(BaseEngine):
    def __init__(self, info: EngineInfo, manifest: Dict[str, Any]):
        super().__init__(info, manifest)
        self._weights = info.weights
        self._activs = info.activation or []

    @staticmethod
    def _apply_activation(x: List[float], func_name: Optional[str]) -> List[float]:
        if func_name is None:
            return x
        f = str(func_name).lower()
        if f == "tanh":
            return [math.tanh(v) for v in x]
        if f == "relu":
            return [v if v > 0.0 else 0.0 for v in x]
        if f == "sigmoid":
            out = []
            for v in x:
                try:
                    out.append(1.0 / (1.0 + math.exp(-v)))
                except OverflowError:
                    out.append(0.0 if v < 0 else 1.0)
            return out
        return x

    def forward_logits(self, inputs: Sequence[float]) -> List[float]:
        if len(inputs) != self.input_size:
            raise ValueError(
                f"Entrada com tamanho {len(inputs)} não compatível com input_size={self.input_size}"
            )

        x = list(map(float, inputs))

        if not self._weights:
            return [0.0] * max(1, int(self.output_size))

        for li, layer in enumerate(self._weights):
            out: List[float] = []
            for neuron in layer:
                bias = float(neuron[0]) if neuron else 0.0
                w = neuron[1:] if len(neuron) > 1 else []
                acc = bias
                for wi, xi in zip(w, x):
                    acc += float(wi) * float(xi)
                out.append(acc)

            act = self._activs[li] if li < len(self._activs) else None
            x = self._apply_activation(out, act)

        # garante tamanho
        if self.output_size > 0 and len(x) != self.output_size:
            if len(x) < self.output_size:
                x = x + [0.0] * (self.output_size - len(x))
            else:
                x = x[: self.output_size]
        return x


# ---------------------------------------------------------
#  Engine com PyTorch (opcional)
# ---------------------------------------------------------
class TorchEngine(BaseEngine):
    def __init__(self, info: EngineInfo, manifest: Dict[str, Any]):
        if not _TORCH_AVAILABLE or torch is None:  # type: ignore[truthy-function]
            raise RuntimeError("PyTorch não disponível no ambiente.")
        super().__init__(info, manifest)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn = torch.nn

        self.activs: List[Optional[str]] = info.activation or []

        weights = info.weights if isinstance(info.weights, list) else []
        if not weights:
            weights = _make_zero_weights(info.input_size, info.neurons)

        layers: List[torch.nn.Module] = []
        in_features = info.input_size

        for li, layer_matrix in enumerate(weights):
            out_features = len(layer_matrix)
            if out_features <= 0:
                continue

            linear = nn.Linear(in_features, out_features).to(self.device)

            W = []
            b = []
            for neuron in layer_matrix:
                bias = float(neuron[0]) if neuron else 0.0
                w = neuron[1:] if len(neuron) > 1 else []
                w = list(map(float, w))
                if len(w) < in_features:
                    w += [0.0] * (in_features - len(w))
                elif len(w) > in_features:
                    w = w[:in_features]
                W.append(w)
                b.append(bias)

            with torch.no_grad():
                linear.weight.copy_(torch.tensor(W, dtype=torch.float32, device=self.device))
                linear.bias.copy_(torch.tensor(b, dtype=torch.float32, device=self.device))

            layers.append(linear)
            in_features = out_features

        self.layers = layers
        self.model = nn.Sequential(*layers).to(self.device)

    def _apply_activation_tensor(self, x, func_name: Optional[str]):
        if func_name is None:
            return x
        f = str(func_name).lower()
        if f == "tanh":
            return torch.tanh(x)
        if f == "relu":
            return torch.relu(x)
        if f == "sigmoid":
            return torch.sigmoid(x)
        return x

    def _forward_tensor(self, inputs: Sequence[float]):
        if len(inputs) != self.input_size:
            raise ValueError(
                f"Entrada com tamanho {len(inputs)} não compatível com input_size={self.input_size}"
            )

        x = torch.tensor([list(map(float, inputs))], dtype=torch.float32, device=self.device)
        for li, layer in enumerate(self.layers):
            x = layer(x)
            act = self.activs[li] if li < len(self.activs) else None
            x = self._apply_activation_tensor(x, act)
        return x  # (1, out)

    def forward_logits(self, inputs: Sequence[float]) -> List[float]:
        x = self._forward_tensor(inputs)
        out = x.detach().cpu().numpy()[0].tolist()

        if self.output_size > 0 and len(out) != self.output_size:
            if len(out) < self.output_size:
                out = out + [0.0] * (self.output_size - len(out))
            else:
                out = out[: self.output_size]
        return out

    def parameters(self):
        return self.model.parameters()


# ---------------------------------------------------------
#  NeuralNetworkEngine (compatibilidade com suas páginas)
# ---------------------------------------------------------
_NeuralBase = TorchEngine if (_TORCH_AVAILABLE and torch is not None) else PurePythonEngine  # type: ignore[truthy-function]


class NeuralNetworkEngine(_NeuralBase):
    """
    Compatível com:
      - tb_ind.py / wd_create_analise.py  -> NeuralNetworkEngine.from_manifest(manifest)
      - build_engine(name)

    forward() retorna softmax (0..1 soma=1)
    forward_logits() retorna vetor bruto
    """

    @classmethod
    def from_manifest(cls, manifest: Any) -> "NeuralNetworkEngine":
        """
        Aceita:
          - IAManifest (com to_dict)
          - dict (json já carregado)

        Cria engine robusto (weights inválido -> inicializa zeros)
        """
        if isinstance(manifest, dict):
            data = manifest
        else:
            to_dict = getattr(manifest, "to_dict", None)
            if callable(to_dict):
                data = to_dict()
            else:
                raise TypeError("manifest deve ser IAManifest (com to_dict) ou dict.")

        ident = data.get("identification") or {}
        if not isinstance(ident, dict):
            ident = {}

        name_field = ident.get("name") or ident.get("id") or "unnamed"
        name = Path(str(name_field)).stem

        st = data.get("structure") or {}
        if not isinstance(st, dict):
            raise ValueError("Campo 'structure' inválido no manifesto.")

        input_size = int(st.get("input_size") or 0)
        output_size = int(st.get("output_size") or 0)

        neurons = st.get("neurons") or []
        if isinstance(neurons, int):
            neurons = [neurons]
        if not isinstance(neurons, list):
            neurons = []
        neurons = [int(x) for x in neurons] if neurons else []

        raw_weights = st.get("weights")
        if not neurons:
            if isinstance(raw_weights, list) and raw_weights:
                try:
                    neurons = [int(len(layer)) for layer in raw_weights if isinstance(layer, list) and len(layer) > 0]
                except Exception:
                    neurons = []
            if not neurons and output_size > 0:
                neurons = [output_size]

        activation = st.get("activation") or []
        if isinstance(activation, (str, type(None))):
            activation = [activation] * len(neurons)
        if not isinstance(activation, list):
            activation = []
        if len(activation) < len(neurons):
            activation = list(activation) + [None] * (len(neurons) - len(activation))
        if len(activation) > len(neurons):
            activation = list(activation)[: len(neurons)]

        weights = _normalize_weights(input_size, neurons, raw_weights)

        out_size = output_size if output_size > 0 else (neurons[-1] if neurons else 0)

        info = EngineInfo(
            name=name,
            path=Path("."),
            input_size=input_size,
            output_size=out_size,
            neurons=neurons,
            activation=[a for a in activation],
            weights=weights,
        )

        return cls(info, data)

    def forward(self, inputs: Sequence[float]) -> List[float]:
        logits = self.forward_logits(inputs)
        return softmax_stable(logits)


# ---------------------------------------------------------
#  build_engine (por nome)
# ---------------------------------------------------------
def build_engine(name: str) -> NeuralNetworkEngine:
    data, path = _load_manifest(name)
    engine = NeuralNetworkEngine.from_manifest(data)
    engine.info_data.path = path
    return engine
