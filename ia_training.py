# ia_training.py
# -----------------------------------------------------------
# Funções de treinamento e avaliação integradas com:
#   - IAManifest (estrutura JSON)
#
# Correção crítica:
#   - Normaliza manifesto conforme shape do dataset antes do treino/avaliação:
#       * input_size = len(x)
#       * output_size = n_classes (max(y)+1)
#       * neurons e weights redimensionados de forma segura (pad/truncate)
# -----------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import math
import random
import time

from json_lib import build_paths, load_json, save_json
from ia_manifest import IAManifest, Structure  # noqa: F401

# -----------------------------------------------------------
# Tentativa de importar PyTorch (opcional)
# -----------------------------------------------------------
try:  # pragma: no cover
    import torch  # type: ignore[attr-defined]
    import torch.nn as nn  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_TORCH_AVAILABLE: bool = torch is not None  # type: ignore[truthy-function]

if _TORCH_AVAILABLE:
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[union-attr]
else:
    _DEVICE = None  # type: ignore[assignment]


if _TORCH_AVAILABLE:
    try:
        torch.backends.cudnn.benchmark = True  # type: ignore[union-attr]
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# Cada amostra é (lista_de_entradas, classe_inteira)
Sample = Tuple[Sequence[float], int]


# -----------------------------------------------------------
# Utilitários de carga / salvamento de manifestos
# -----------------------------------------------------------
def load_manifest_by_name(name: str) -> Tuple[IAManifest, Path]:
    pasta, arquivo, nome_sanit = build_paths(name)
    if not arquivo.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado em:\n{arquivo}\n"
            f"(nome sanitizado: {nome_sanit})"
        )

    data = load_json(arquivo)
    manifest = IAManifest.from_dict(data)
    return manifest, arquivo


def save_manifest_to_path(manifest: IAManifest, path: Path) -> None:
    payload = manifest.to_dict()
    save_json(path, payload)


# -----------------------------------------------------------
# Helpers: inferência de shape do dataset
# -----------------------------------------------------------
def _infer_input_len(data_list: List[Sample]) -> int:
    if not data_list:
        return 0
    x0 = data_list[0][0]
    try:
        return max(1, int(len(x0)))
    except Exception:
        return 1


def _infer_num_classes(data_list: List[Sample]) -> int:
    if not data_list:
        return 1
    mx = 0
    for _, y in data_list:
        try:
            mx = max(mx, int(y))
        except Exception:
            pass
    return max(1, mx + 1)


def _fix_input_vector(x_vals: Sequence[float], target_len: int) -> List[float]:
    """
    Garante que cada entrada tenha exatamente target_len:
    - se menor, pad com 0.0
    - se maior, trunca
    """
    try:
        x = [float(v) for v in x_vals]
    except Exception:
        x = [0.0] * target_len

    if len(x) < target_len:
        x += [0.0] * (target_len - len(x))
    elif len(x) > target_len:
        x = x[:target_len]
    return x


# -----------------------------------------------------------
# Normalização do manifesto conforme dataset (CORREÇÃO DO ERRO)
# -----------------------------------------------------------
def _parse_neurons(neurons_raw: Any) -> List[int]:
    if isinstance(neurons_raw, str):
        parts = [p.strip() for p in neurons_raw.split(",") if p.strip()]
        out: List[int] = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                pass
        return out
    if isinstance(neurons_raw, (int, float)):
        return [int(neurons_raw)]
    if isinstance(neurons_raw, list):
        out = []
        for v in neurons_raw:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    return []


def _resize_weights_like(input_size: int, neurons: List[int], old_weights: Any) -> List[List[List[float]]]:
    """
    Ajusta weights para bater com input_size e neurons, preservando o máximo possível.
    Formato por camada:
      layer -> lista de neurônios
      neuron -> [bias, w1..wN]
    """
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


def _normalize_manifest_for_data(manifest: IAManifest, input_len: int, n_classes: int) -> bool:
    """
    Garante que manifest.structure esteja coerente com o dataset:
      - input_size == input_len
      - output_size == n_classes
      - neurons[-1] == n_classes
      - weights redimensionados conforme input/neurons
      - activation ajustado para o nº de camadas
    Retorna True se alterou algo.
    """
    changed = False
    st = manifest.structure

    # input/output sizes
    if getattr(st, "input_size", None) != int(input_len):
        st.input_size = int(input_len)
        changed = True

    if getattr(st, "output_size", None) != int(n_classes):
        st.output_size = int(n_classes)
        changed = True

    neurons = _parse_neurons(getattr(st, "neurons", None))
    if not neurons:
        # cria uma estrutura mínima: hidden default + output
        neurons = [max(2, input_len), int(n_classes)]
        changed = True
    else:
        # garante saída compatível
        if neurons[-1] != int(n_classes):
            neurons[-1] = int(n_classes)
            changed = True

    neurons = [max(1, int(x)) for x in neurons]
    if getattr(st, "neurons", None) != neurons:
        st.neurons = neurons
        changed = True

    # layers
    if getattr(st, "layers", None) != len(neurons):
        st.layers = len(neurons)
        changed = True

    # weights
    old_weights = getattr(st, "weights", None)
    new_weights = _resize_weights_like(int(input_len), neurons, old_weights)
    if old_weights != new_weights:
        st.weights = new_weights
        changed = True

    # activation
    act = getattr(st, "activation", None)
    if act is None:
        activation: List[Optional[str]] = [None] * len(neurons)
        st.activation = activation
        changed = True
    else:
        if isinstance(act, str):
            activation = [act] * len(neurons)
            st.activation = activation
            changed = True
        elif isinstance(act, list):
            activation2: List[Optional[str]] = []
            for a in act:
                if a is None or isinstance(a, str):
                    activation2.append(a)
                else:
                    activation2.append(None)
            if len(activation2) < len(neurons):
                activation2 += [None] * (len(neurons) - len(activation2))
                changed = True
            elif len(activation2) > len(neurons):
                activation2 = activation2[: len(neurons)]
                changed = True
            if activation2 != act:
                st.activation = activation2
                changed = True
        else:
            st.activation = [None] * len(neurons)
            changed = True

    return changed


# -----------------------------------------------------------
# Rede JsonNetwork (PyTorch)
# -----------------------------------------------------------
if _TORCH_AVAILABLE:

    class JsonNetwork(nn.Module):  # type: ignore[misc]
        """
        Constrói uma rede neural em PyTorch a partir de um IAManifest.
        weights = [camada][neurônio][bias, w1..wN]
        """

        def __init__(self, manifest: IAManifest):
            super().__init__()

            self.manifest = manifest
            st = manifest.structure

            weights_all = st.weights or []
            activ_all = list(st.activation or [])

            if len(activ_all) < len(weights_all):
                activ_all += [None] * (len(weights_all) - len(activ_all))
            elif len(activ_all) > len(weights_all):
                activ_all = activ_all[: len(weights_all)]

            self.layers = nn.ModuleList()
            self.activations: List[Optional[str]] = []

            for i, layer in enumerate(weights_all):
                if not layer:
                    raise ValueError(f"Camada {i} do manifesto está vazia.")

                out_features = len(layer)
                # in_features vem do tamanho do neurônio (bias + w...)
                in_features = len(layer[0]) - 1
                in_features = max(1, int(in_features))

                linear = nn.Linear(in_features, out_features)

                w_rows: List[List[float]] = []
                b_vals: List[float] = []
                for n in layer:
                    b_vals.append(float(n[0]))
                    w_rows.append([float(x) for x in n[1:]])

                W = torch.tensor(w_rows, dtype=torch.float32)
                b = torch.tensor(b_vals, dtype=torch.float32)
                with torch.no_grad():
                    linear.weight.copy_(W)
                    linear.bias.copy_(b)

                self.layers.append(linear)
                act_raw = activ_all[i]
                self.activations.append((act_raw or "").lower() if act_raw is not None else None)

        def forward(self, x):  # type: ignore[override]
            for linear, act in zip(self.layers, self.activations):
                x = linear(x)
                if act == "tanh":
                    x = torch.tanh(x)
                elif act == "relu":
                    x = torch.relu(x)
            return x

        def export_weights(self) -> List[List[List[float]]]:
            new_weights: List[List[List[float]]] = []
            for linear in self.layers:
                W = linear.weight.detach().cpu().numpy()
                b = linear.bias.detach().cpu().numpy()
                layer_data: List[List[float]] = []
                for i in range(W.shape[0]):
                    row = [float(b[i])]
                    row += [float(w) for w in W[i, :]]
                    layer_data.append(row)
                new_weights.append(layer_data)
            return new_weights

else:

    class JsonNetwork:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch não está disponível; JsonNetwork não pode ser instanciada.\n"
                "Instale torch na máquina de treino para usar o backprop."
            )

        def export_weights(self) -> List[List[List[float]]]:
            raise RuntimeError("PyTorch não está disponível; JsonNetwork não pode exportar pesos.")


# -----------------------------------------------------------
# Estrutura de retorno
# -----------------------------------------------------------
@dataclass
class TrainResult:
    name: str
    samples: int
    epochs: int
    final_loss: float
    avg_loss: float
    elapsed_seconds: float


@dataclass
class EvalResult:
    name: str
    accuracy: float
    avg_loss: Optional[float]
    avg_time_per_sample: float


# -----------------------------------------------------------
# Loss utils (modo python)
# -----------------------------------------------------------
def _softmax_stable(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(float(x) for x in logits)
    exps = [math.exp(float(x) - m) for x in logits]
    s = sum(exps)
    if s <= 0:
        return [0.0 for _ in exps]
    return [v / s for v in exps]


def _cross_entropy_from_logits(logits: Sequence[float], y_true: int) -> float:
    probs = _softmax_stable(logits)
    if not probs:
        return float("nan")
    if y_true < 0 or y_true >= len(probs):
        return float("nan")
    p = max(1e-12, float(probs[y_true]))
    return -math.log(p)


def _forward_python(
    weights: List[List[List[float]]],
    activation: List[Optional[str]],
    inputs: Sequence[float],
) -> List[float]:
    x = list(inputs)
    for layer_idx, layer in enumerate(weights):
        next_x: List[float] = []
        act = (activation[layer_idx] or "").lower() if layer_idx < len(activation) else ""
        for neuron in layer:
            bias = float(neuron[0])
            w = neuron[1:]
            s = bias
            for j, wj in enumerate(w):
                if j < len(x):
                    s += float(wj) * float(x[j])
            if act == "tanh":
                s = math.tanh(s)
            elif act == "relu":
                s = max(0.0, s)
            next_x.append(s)
        x = next_x
    return x


# -----------------------------------------------------------
# Treino
# -----------------------------------------------------------
ProgressCallback = Callable[[str, int, int, int, int, float], bool]


def train_networks(
    network_names: Sequence[str],
    data: Sequence[Sample],
    learning_rate: float = 0.01,
    n_epochs: int = 1,
    shuffle: bool = True,
    verbose: bool = True,
    batch_size: int = 1,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, TrainResult]:
    if not data:
        raise ValueError("Lista de dados de treino está vazia.")

    data_list: List[Sample] = list(data)

    input_len = _infer_input_len(data_list)
    n_classes = _infer_num_classes(data_list)

    results: Dict[str, TrainResult] = {}

    # ------------------------------
    # Modo sem PyTorch: no-op training (não derruba o app)
    # ------------------------------
    if not _TORCH_AVAILABLE:
        if verbose:
            print(
                "[TRAIN] PyTorch não está disponível. "
                "Treino com backprop desativado (no-op). "
                "Calculando loss por forward Python, sem atualizar pesos."
            )

        total_samples = len(data_list)
        total_steps = total_samples * max(1, int(n_epochs))
        bs = max(1, int(batch_size))

        for name in network_names:
            manifest, path_json = load_manifest_by_name(name)

            # NORMALIZA MANIFESTO PARA BATER COM DATA
            changed = _normalize_manifest_for_data(manifest, input_len=input_len, n_classes=n_classes)
            if changed:
                # salva já corrigido para eliminar o erro no próximo treino
                save_manifest_to_path(manifest, path_json)
                if verbose:
                    print(f"[TRAIN] Manifest '{name}' normalizado (input={input_len}, classes={n_classes}).")

            st = manifest.structure
            weights = st.weights or []
            activation = list(st.activation or [])

            losses_all: List[float] = []
            step_counter = 0
            stop_requested = False
            t0 = time.perf_counter()

            for epoch in range(max(1, int(n_epochs))):
                indices = list(range(total_samples))
                if shuffle:
                    random.shuffle(indices)

                for pos in range(0, len(indices), bs):
                    batch_idx = indices[pos : pos + bs]

                    batch_losses: List[float] = []
                    for idx in batch_idx:
                        x_vals, y_true = data_list[idx]
                        x_fix = _fix_input_vector(x_vals, input_len)
                        logits = _forward_python(weights, activation, x_fix)
                        batch_losses.append(_cross_entropy_from_logits(logits, int(y_true)))

                    loss_val = float(sum(batch_losses) / max(1, len(batch_losses)))
                    losses_all.append(loss_val)

                    step_counter += int(len(batch_idx))

                    if progress_callback is not None:
                        should_continue = progress_callback(
                            name,
                            step_counter,
                            total_steps,
                            epoch + 1,
                            max(1, int(n_epochs)),
                            loss_val,
                        )
                        if not should_continue:
                            stop_requested = True
                            break

                if stop_requested:
                    if verbose:
                        print(f"[TRAIN] No-op treino de '{name}' interrompido na época {epoch + 1}.")
                    break

            elapsed = time.perf_counter() - t0
            final_loss = float(losses_all[-1]) if losses_all else math.nan
            avg_loss = float(sum(losses_all) / float(len(losses_all))) if losses_all else math.nan

            results[name] = TrainResult(
                name=name,
                samples=total_samples,
                epochs=0,  # no-op
                final_loss=final_loss,
                avg_loss=avg_loss,
                elapsed_seconds=float(elapsed),
            )

        return results

    # ------------------------------
    # Modo com PyTorch: treino real
    # ------------------------------
    for name in network_names:
        manifest, path_json = load_manifest_by_name(name)

        # NORMALIZA MANIFESTO PARA BATER COM DATA (fix do erro matmul)
        changed = _normalize_manifest_for_data(manifest, input_len=input_len, n_classes=n_classes)
        if changed:
            save_manifest_to_path(manifest, path_json)
            if verbose:
                print(f"[TRAIN] Manifest '{name}' normalizado (input={input_len}, classes={n_classes}).")

        model = JsonNetwork(manifest).to(_DEVICE)  # type: ignore[arg-type]
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))  # type: ignore[union-attr]
        criterion = torch.nn.CrossEntropyLoss()  # type: ignore[union-attr]

        total_samples = len(data_list)
        total_steps = total_samples * int(n_epochs)
        step_counter = 0
        losses_all: List[float] = []

        t0 = time.perf_counter()
        stop_requested = False

        bs = max(1, int(batch_size))

        for epoch in range(int(n_epochs)):
            indices = list(range(total_samples))
            if shuffle:
                random.shuffle(indices)

            for pos in range(0, len(indices), bs):
                batch_idx = indices[pos : pos + bs]

                xs: List[List[float]] = []
                ys: List[int] = []
                for idx in batch_idx:
                    x_vals, target_cls = data_list[idx]
                    xs.append(_fix_input_vector(x_vals, input_len))
                    ys.append(int(target_cls))

                x_tensor = torch.tensor(xs, dtype=torch.float32, device=_DEVICE)  # type: ignore[union-attr]
                y_tensor = torch.tensor(ys, dtype=torch.long, device=_DEVICE)  # type: ignore[union-attr]

                optimizer.zero_grad()
                logits = model(x_tensor)
                loss = criterion(logits, y_tensor)
                loss.backward()
                optimizer.step()

                loss_val = float(loss.item())
                losses_all.append(loss_val)

                step_counter += int(len(batch_idx))

                if progress_callback is not None:
                    should_continue = progress_callback(
                        name,
                        step_counter,
                        total_steps,
                        epoch + 1,
                        int(n_epochs),
                        loss_val,
                    )
                    if not should_continue:
                        stop_requested = True
                        break

            if stop_requested:
                if verbose:
                    print(f"[TRAIN] Treinamento de '{name}' interrompido na época {epoch + 1}.")
                break

        elapsed = time.perf_counter() - t0
        final_loss = float(losses_all[-1]) if losses_all else math.nan
        avg_loss = float(sum(losses_all) / float(len(losses_all))) if losses_all else math.nan

        new_weights = model.export_weights()
        manifest.structure.weights = new_weights
        save_manifest_to_path(manifest, path_json)

        results[name] = TrainResult(
            name=name,
            samples=total_samples,
            epochs=int(n_epochs),
            final_loss=final_loss,
            avg_loss=avg_loss,
            elapsed_seconds=float(elapsed),
        )

    return results


# -----------------------------------------------------------
# Avaliação
# -----------------------------------------------------------
def evaluate_networks(
    network_names: Sequence[str],
    data: Sequence[Sample],
    limit: Optional[int] = None,
    verbose: bool = True,
) -> List[EvalResult]:
    data_list: List[Sample] = list(data)
    if not data_list:
        raise ValueError("Lista de dados para avaliação está vazia.")

    if limit is not None and limit > 0 and limit < len(data_list):
        data_list = random.sample(data_list, limit)

    input_len = _infer_input_len(data_list)
    n_classes = _infer_num_classes(data_list)

    total_data = len(data_list)
    results: List[EvalResult] = []

    for name in network_names:
        manifest, path_json = load_manifest_by_name(name)

        # NORMALIZA também na avaliação (evita crash por mismatch)
        changed = _normalize_manifest_for_data(manifest, input_len=input_len, n_classes=n_classes)
        if changed:
            save_manifest_to_path(manifest, path_json)
            if verbose:
                print(f"[EVAL] Manifest '{name}' normalizado (input={input_len}, classes={n_classes}).")

        st = manifest.structure
        weights = st.weights or []
        activation = list(st.activation or [])

        use_torch = _TORCH_AVAILABLE

        if use_torch:
            model = JsonNetwork(manifest).to(_DEVICE)  # type: ignore[arg-type]
            model.eval()
            criterion = torch.nn.CrossEntropyLoss()  # type: ignore[union-attr]
        else:
            model = None  # type: ignore[assignment]
            criterion = None  # type: ignore[assignment]

        correct = 0
        losses: List[float] = []
        t0 = time.perf_counter()

        for x_vals, target_cls in data_list:
            y_true = int(target_cls)
            x_fix = _fix_input_vector(x_vals, input_len)

            if use_torch:
                with torch.no_grad():  # type: ignore[union-attr]
                    x_tensor = torch.tensor(x_fix, dtype=torch.float32, device=_DEVICE).unsqueeze(0)  # type: ignore[union-attr]
                    logits = model(x_tensor)  # type: ignore[operator]
                    pred_cls = int(torch.argmax(logits, dim=-1).item())  # type: ignore[union-attr]
                    if criterion is not None:
                        loss = criterion(
                            logits,
                            torch.tensor([y_true], dtype=torch.long, device=_DEVICE),  # type: ignore[union-attr]
                        )
                        losses.append(float(loss.item()))
            else:
                out = _forward_python(weights, activation, x_fix)
                pred_cls = int(max(range(len(out)), key=lambda i: out[i])) if out else 0
                ce = _cross_entropy_from_logits(out, y_true)
                if not math.isnan(ce):
                    losses.append(float(ce))

            if pred_cls == y_true:
                correct += 1

        elapsed = time.perf_counter() - t0
        accuracy = correct / float(total_data)
        avg_loss = float(sum(losses) / float(len(losses))) if losses else None
        avg_time_per_sample = elapsed / float(total_data)

        results.append(
            EvalResult(
                name=name,
                accuracy=float(accuracy),
                avg_loss=avg_loss,
                avg_time_per_sample=float(avg_time_per_sample),
            )
        )

    results.sort(key=lambda r: (-r.accuracy, r.avg_time_per_sample, r.avg_loss if r.avg_loss is not None else float("inf")))
    return results
