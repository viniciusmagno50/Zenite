# ia_training.py
# -----------------------------------------------------------
# Funções de treinamento, avaliação  e (no futuro) mutação de redes,
# integradas com:
#   - IAManifest (estrutura de JSON moderna)
#
# Requisitos:
#   - Para TREINO com backprop, é necessário PyTorch.
#   - Se PyTorch não estiver disponível:
#       * inferência/avaliação continua funcionando via forward em Python
#       * treino NÃO deve derrubar o app: vira "no-op" com aviso (contrato ZeniteV3)
# -----------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import math
import random
import time

from json_lib import build_paths, load_json, save_json
from ia_manifest import IAManifest, Structure  # noqa: F401  (Structure pode ser útil depois)

# -----------------------------------------------------------
# Tentativa de importar PyTorch (opcional)
# -----------------------------------------------------------
try:  # pragma: no cover - depende do ambiente real
    import torch  # type: ignore[attr-defined]
    import torch.nn as nn  # type: ignore[attr-defined]
    import torch.nn.functional as F  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

_TORCH_AVAILABLE: bool = torch is not None  # type: ignore[truthy-function]

if _TORCH_AVAILABLE:
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[union-attr]
else:
    _DEVICE = None  # type: ignore[assignment]


if _TORCH_AVAILABLE:
    # Ajustes simples de performance (não alteram a API)
    try:
        torch.backends.cudnn.benchmark = True  # type: ignore[union-attr]
        # TF32 pode acelerar em GPUs recentes (sem quebrar em GPUs antigas)
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
    """
    Carrega um IAManifest a partir do 'name' (sem ou com .json),
    usando a mesma convenção de pastas de json_lib.build_paths:

      dados/{nome_sanitizado}/{nome_sanitizado}.json

    Retorna:
      (manifest, caminho_arquivo_json)
    """
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
    """
    Salva o IAManifest em um caminho específico.
    """
    payload = manifest.to_dict()
    save_json(path, payload)


# -----------------------------------------------------------
# Rede "JsonNetwork"
#   - se PyTorch estiver disponível: nn.Module real
#   - se NÃO estiver: stub que só levanta erro ao tentar usar
# -----------------------------------------------------------
if _TORCH_AVAILABLE:

    class JsonNetwork(nn.Module):  # type: ignore[misc]
        """
        Constrói uma rede neural em PyTorch a partir de um IAManifest.

        Convenções:
          - structure.weights = [camada][neurônio][bias, w1..wN]
          - structure.activation = ["tanh", None, ...] alinhado com 'neurons'
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
                in_features = len(layer[0]) - 1  # exclui bias

                linear = nn.Linear(in_features, out_features)
                # Preenche pesos/bias com os valores do JSON
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
                # None => saída linear
            return x

        # Exporta pesos atuais de volta para o formato do manifesto
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

    class JsonNetwork:  # pragma: no cover - usado só quando torch não existe
        """
        Stub de JsonNetwork para ambientes SEM PyTorch.

        Serve apenas para evitar erros de import. Qualquer tentativa
        de instanciar/usar vai levantar RuntimeError.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch não está disponível; JsonNetwork não pode ser instanciada.\n"
                "Instale torch na máquina de treino para usar o backprop."
            )

        def export_weights(self) -> List[List[List[float]]]:
            raise RuntimeError(
                "PyTorch não está disponível; JsonNetwork não pode exportar pesos."
            )


# -----------------------------------------------------------
# Estrutura de retorno de treino / avaliação
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
# Função de TREINO (uma ou várias redes)
# -----------------------------------------------------------
ProgressCallback = Callable[
    [str, int, int, int, int, float],
    bool,
]
# args:
#   name          -> nome da rede
#   step          -> passo global (1..total_steps)
#   total_steps   -> total de passos (samples * epochs)
#   epoch         -> época atual (1..n_epochs)
#   total_epochs  -> total de épocas
#   loss          -> loss do passo atual
#
# retorno:
#   True  -> continuar
#   False -> interromper o treinamento daquela rede


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
    """
    Loss cross-entropy (NLL) a partir de logits (saída linear do forward python).
    """
    probs = _softmax_stable(logits)
    if not probs:
        return float("nan")
    if y_true < 0 or y_true >= len(probs):
        return float("nan")
    p = max(1e-12, float(probs[y_true]))
    return -math.log(p)


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
    """
    Treina uma ou mais redes (por nome) em cima de uma lista de amostras:

        data = [(entrada, classe_alvo), ...]

    A arquitetura/pesos iniciais são lidos do JSON (IAManifest).
    Ao final, os pesos treinados são gravados de volta no JSON.

    Se progress_callback for fornecido, será chamado durante o treino (por batch)
    com informações de progresso e pode retornar False para interromper
    o treinamento da rede atual.

    IMPORTANTE (contrato ZeniteV3):
      - Sem PyTorch: não derruba o app.
        O treino vira "no-op" (sem atualizar pesos), mas retorna métricas/loss
        calculadas via forward Python para manter UI funcional.
    """
    if not data:
        raise ValueError("Lista de dados de treino está vazia.")

    data_list: List[Sample] = list(data)
    results: Dict[str, TrainResult] = {}

    # ------------------------------
    # Modo sem PyTorch: no-op training
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

                    # "no-op": apenas calcula loss média do batch
                    batch_losses: List[float] = []
                    for idx in batch_idx:
                        x_vals, y_true = data_list[idx]
                        logits = _forward_python(weights, activation, x_vals)
                        batch_losses.append(_cross_entropy_from_logits(logits, int(y_true)))

                    # loss do passo (batch) como média
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
                        print(
                            f"[TRAIN] No-op treino de '{name}' interrompido pelo usuário "
                            f"na época {epoch + 1}."
                        )
                    break

            elapsed = time.perf_counter() - t0
            if losses_all:
                final_loss = float(losses_all[-1])
                avg_loss = float(sum(losses_all) / float(len(losses_all)))
            else:
                final_loss = math.nan
                avg_loss = math.nan

            # Não altera pesos (no-op). Salva manifesto opcionalmente? Aqui NÃO precisa.
            # Mantemos o arquivo intacto para evitar falsa impressão de "treinou".

            if verbose:
                print(
                    f"[TRAIN] (no-op) '{name}': "
                    f"samples={total_samples}, req_epochs={int(n_epochs)}, "
                    f"steps={step_counter}/{total_steps}, "
                    f"final_loss={final_loss:.6g}, avg_loss={avg_loss:.6g}, "
                    f"time={elapsed:.2f}s"
                )

            results[name] = TrainResult(
                name=name,
                samples=total_samples,
                epochs=0,  # <- indica que NÃO houve backprop
                final_loss=final_loss,
                avg_loss=avg_loss,
                elapsed_seconds=float(elapsed),
            )

        return results

    # ------------------------------
    # Modo com PyTorch: treino real
    # ------------------------------
    results = {}

    for name in network_names:
        manifest, path_json = load_manifest_by_name(name)
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

        for epoch in range(int(n_epochs)):
            indices = list(range(total_samples))
            if shuffle:
                random.shuffle(indices)

            for pos in range(0, len(indices), max(1, int(batch_size))):
                batch_idx = indices[pos : pos + max(1, int(batch_size))]

                xs = []
                ys = []
                for idx in batch_idx:
                    x_vals, target_cls = data_list[idx]
                    xs.append(x_vals)
                    ys.append(int(target_cls))

                x_tensor = torch.tensor(  # type: ignore[union-attr]
                    xs, dtype=torch.float32, device=_DEVICE
                )
                y_tensor = torch.tensor(  # type: ignore[union-attr]
                    ys, dtype=torch.long, device=_DEVICE
                )

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
                    print(
                        f"[TRAIN] Treinamento de '{name}' interrompido pelo usuário "
                        f"na época {epoch + 1}."
                    )
                break

        elapsed = time.perf_counter() - t0

        if losses_all:
            final_loss = float(losses_all[-1])
            avg_loss = float(sum(losses_all) / float(len(losses_all)))
        else:
            final_loss = math.nan
            avg_loss = math.nan

        # Atualiza pesos no manifesto e salva
        new_weights = model.export_weights()
        manifest.structure.weights = new_weights
        save_manifest_to_path(manifest, path_json)

        total_used_steps = step_counter
        if verbose:
            print(
                f"[TRAIN] '{name}': "
                f"samples={total_samples}, epochs={int(n_epochs)}, "
                f"steps={total_used_steps}/{total_steps}, "
                f"final_loss={final_loss:.6g}, avg_loss={avg_loss:.6g}, "
                f"time={elapsed:.2f}s"
            )

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
# Forward em Python puro (sem PyTorch)
# -----------------------------------------------------------
def _forward_python(
    weights: List[List[List[float]]],
    activation: List[Optional[str]],
    inputs: Sequence[float],
) -> List[float]:
    """
    Forward pass simples em Python puro, usado quando PyTorch não está disponível.
    """
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
# Função de AVALIAÇÃO (uma ou várias redes)
# -----------------------------------------------------------
def evaluate_networks(
    network_names: Sequence[str],
    data: Sequence[Sample],
    limit: Optional[int] = None,
    verbose: bool = True,
) -> List[EvalResult]:
    """
    Avalia uma ou mais redes em cima de uma lista de amostras (entrada, classe).

    Retorna lista de EvalResult ordenada por:
      - maior accuracy
      - menor tempo médio por amostra
      - menor loss médio (se disponível)
    """
    data_list: List[Sample] = list(data)
    if not data_list:
        raise ValueError("Lista de dados para avaliação está vazia.")

    if limit is not None and limit > 0 and limit < len(data_list):
        data_list = random.sample(data_list, limit)

    total_data = len(data_list)
    results: List[EvalResult] = []

    for name in network_names:
        manifest, _ = load_manifest_by_name(name)
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

            if use_torch:
                with torch.no_grad():  # type: ignore[union-attr]
                    x_tensor = torch.tensor(  # type: ignore[union-attr]
                        x_vals, dtype=torch.float32, device=_DEVICE
                    ).unsqueeze(0)
                    logits = model(x_tensor)  # type: ignore[operator]
                    probs = torch.nn.functional.softmax(logits, dim=-1)  # type: ignore[union-attr]
                    pred_cls = int(torch.argmax(probs, dim=-1).item())  # type: ignore[union-attr]
                    if criterion is not None:
                        loss = criterion(
                            logits,
                            torch.tensor(  # type: ignore[union-attr]
                                [y_true], dtype=torch.long, device=_DEVICE
                            ),
                        )
                        losses.append(float(loss.item()))
            else:
                out = _forward_python(weights, activation, x_vals)
                max_idx = 0
                max_val = float("-inf")
                for i, v in enumerate(out):
                    if v > max_val:
                        max_val = v
                        max_idx = i
                pred_cls = max_idx

                # loss opcional no modo python (mantém avg_loss mais útil)
                ce = _cross_entropy_from_logits(out, y_true)
                if not math.isnan(ce):
                    losses.append(float(ce))

            if pred_cls == y_true:
                correct += 1

        elapsed = time.perf_counter() - t0
        accuracy = correct / float(total_data)
        avg_loss = float(sum(losses) / float(len(losses))) if losses else None
        avg_time_per_sample = elapsed / float(total_data)

        if verbose:
            print(
                f"[EVAL] '{name}': "
                f"accuracy={accuracy:.4f}, "
                f"avg_loss={avg_loss if avg_loss is not None else 'N/A'}, "
                f"time/sample={avg_time_per_sample*1000:.3f} ms"
            )

        results.append(
            EvalResult(
                name=name,
                accuracy=float(accuracy),
                avg_loss=avg_loss,
                avg_time_per_sample=float(avg_time_per_sample),
            )
        )

    def _sort_key(r: EvalResult):
        loss_key = r.avg_loss if r.avg_loss is not None else float("inf")
        return (-r.accuracy, r.avg_time_per_sample, loss_key)

    results.sort(key=_sort_key)
    return results
