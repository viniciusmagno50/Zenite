from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, Signal

from ia_engine import build_engine
from ia_generational import MutationConfig, create_generation_individual, rank_top_k
from ia_training import Sample, train_networks
from json_lib import build_paths, load_json


# -----------------------------------------------------------------------------
# Tipos de métricas (contrato UI <-> Worker)
# -----------------------------------------------------------------------------
@dataclass
class EpochMetrics:
    network_name: str
    neurons_total: int

    epoch_index: int
    total_epochs: int
    loss_final: float
    loss_avg: float
    accuracy: float
    samples_per_sec: float
    learning_rate: float

    class_counts: List[int]      # contagem de predições por classe (pred)
    class_hits: List[int]        # contagem de acertos por classe verdadeira (y_true)

    out_min: float               # min confiança (softmax) nos acertos
    out_max: float               # max confiança (softmax) nos acertos
    conf_hist_100: List[int]     # histograma 100 bins (0..1) nos acertos


# -----------------------------------------------------------------------------
# Utilidades locais
# -----------------------------------------------------------------------------
def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(float(x) for x in logits)
    exps = [math.exp(float(x) - m) for x in logits]
    s = sum(exps)
    if s <= 0.0:
        n = len(exps)
        return [1.0 / n] * n if n > 0 else []
    return [e / s for e in exps]


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
            bias = float(neuron[0]) if neuron else 0.0
            w = neuron[1:] if len(neuron) > 1 else []
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


def _neurons_total_from_manifest_dict(manifest: Dict[str, Any]) -> int:
    try:
        st = manifest.get("structure", {}) or {}
        neu = st.get("neurons", None)
        if isinstance(neu, list):
            return int(sum(int(v) for v in neu if isinstance(v, (int, float))))
    except Exception:
        pass
    return 0


def _load_weights_activation_from_json(network_name: str) -> Tuple[List[List[List[float]]], List[Optional[str]]]:
    _, manifest_path, _ = build_paths(network_name)
    data = load_json(manifest_path) or {}
    st = data.get("structure", {}) or {}

    weights = st.get("weights", None)
    if weights is None:
        weights = data.get("weights", [])
    if not isinstance(weights, list):
        weights = []

    activation = st.get("activation", None)
    if activation is None:
        activation = data.get("activation", [])
    if not isinstance(activation, list):
        activation = []

    act_norm: List[Optional[str]] = []
    for a in activation:
        act_norm.append(None if a is None else str(a))
    return weights, act_norm


def _ensure_len(arr: List[int], n: int) -> None:
    if n <= 0:
        return
    if len(arr) < n:
        arr.extend([0] * (n - len(arr)))


def _sample_to_xy(sample: Any) -> Tuple[Optional[Sequence[float]], Optional[Any]]:
    """
    Suporta:
      - Sample = (x_vals, y_int)  [seu formato real no ia_training.py]
      - objetos com atributos .x e .y
      - objetos com chaves ['x'], ['y']
    """
    # formato principal do projeto: tuple/list (x, y)
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], sample[1]

    # formato futuro: objeto .x/.y
    if hasattr(sample, "x") and hasattr(sample, "y"):
        try:
            return getattr(sample, "x"), getattr(sample, "y")
        except Exception:
            return None, None

    # fallback dict
    if isinstance(sample, dict):
        return sample.get("x"), sample.get("y")

    return None, None


def _y_to_class(y: Any) -> Optional[int]:
    """
    Aceita:
      - int / float
      - list/tuple one-hot/prob
      - numpy array (via list())
    """
    if y is None:
        return None
    if isinstance(y, bool):
        return int(y)
    if isinstance(y, int):
        return int(y)
    if isinstance(y, float):
        return int(y)

    if isinstance(y, (list, tuple)):
        if not y:
            return None
        try:
            return int(max(range(len(y)), key=lambda i: float(y[i])))
        except Exception:
            return None

    try:
        yy = list(y)  # numpy etc
        if not yy:
            return None
        return int(max(range(len(yy)), key=lambda i: float(yy[i])))
    except Exception:
        return None


def _eval_details_softmax(
    network_name: str,
    data: Sequence[Sample],
    limit: Optional[int],
) -> Tuple[float, float, float, List[int], List[int], List[int]]:
    """
    Avalia via forward Python + softmax.
    - Distribuição: conta TODAS as predições (mesmo errando)
    - Hist/confiança min/max: conta APENAS quando acerta (conforme sua regra)
    """
    data_list = list(data)
    if limit is not None and int(limit) > 0:
        data_list = data_list[: int(limit)]

    if not data_list:
        return 0.0, 0.0, 0.0, [], [], [0] * 100

    weights, activation = _load_weights_activation_from_json(network_name)

    # n_classes pelo tamanho da última camada
    n_classes = 0
    try:
        if weights and isinstance(weights[-1], list):
            n_classes = len(weights[-1])
    except Exception:
        n_classes = 0

    # fallback: pelo y max
    if n_classes <= 0:
        ys: List[int] = []
        for s in data_list:
            _, y = _sample_to_xy(s)
            yy = _y_to_class(y)
            if yy is not None:
                ys.append(yy)
        if ys:
            n_classes = max(ys) + 1

    class_counts: List[int] = [0] * max(0, n_classes)
    class_hits: List[int] = [0] * max(0, n_classes)

    correct = 0
    total = 0

    out_min = 1.0
    out_max = 0.0
    hist100 = [0] * 100

    for s in data_list:
        x, y = _sample_to_xy(s)
        if x is None:
            continue
        y_true = _y_to_class(y)
        if y_true is None:
            continue

        logits = _forward_python(weights, activation, list(x))
        probs = _softmax(logits)
        if not probs:
            continue

        _ensure_len(class_counts, len(probs))
        _ensure_len(class_hits, len(probs))

        pred = max(range(len(probs)), key=lambda i: probs[i])
        class_counts[pred] += 1  # SEMPRE

        total += 1
        if pred == y_true:
            correct += 1
            if 0 <= y_true < len(class_hits):
                class_hits[y_true] += 1

            p = float(probs[y_true]) if 0 <= y_true < len(probs) else 0.0
            b = int(p * 100.0)
            if b >= 100:
                b = 99
            elif b < 0:
                b = 0
            hist100[b] += 1

            out_min = min(out_min, p)
            out_max = max(out_max, p)

    acc = (correct / float(total)) if total > 0 else 0.0
    if correct <= 0:
        out_min, out_max = 0.0, 0.0

    return float(acc), float(out_min), float(out_max), list(class_counts), list(class_hits), list(hist100)


# -----------------------------------------------------------------------------
# Worker de treino (thread)
# -----------------------------------------------------------------------------
class TrainWorker(QObject):
    epoch_metrics = Signal(object)  # EpochMetrics
    progress_epochs = Signal(int, int)  # epoch_atual, total_epocas
    batch_progress = Signal(str, int, int, int, int, float)  # name, step_global, total_steps_global, epoch_index, total_epochs, loss

    stopped = Signal()
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        network_name: str,
        train_data: Sequence[Sample],
        test_data: Sequence[Sample],
        epochs: int,
        learning_rate: float,
        lr_mult: float,
        lr_min: float,
        eval_limit: Optional[int],
        shuffle: bool,
        progress_update_n: int = 1000,
        output_index: int = 0,   # mantido por compatibilidade (não usado aqui)
        batch_size: int = 16,
    ):
        super().__init__()
        self.network_name = str(network_name)
        self.train_data = list(train_data)
        self.test_data = list(test_data)
        self.epochs = max(1, int(epochs))
        self.learning_rate = float(learning_rate)
        self.lr_mult = float(lr_mult)
        self.lr_min = float(lr_min)

        v = int(eval_limit) if eval_limit is not None else 0
        self.eval_limit = v if v > 0 else None

        self.shuffle = bool(shuffle)
        self.progress_update_n = max(1, int(progress_update_n))
        self.batch_size = max(1, int(batch_size))

        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _neurons_total_from_manifest(self, network_name: str) -> int:
        try:
            _, manifest_path, _ = build_paths(network_name)
            data = load_json(manifest_path) or {}
            return _neurons_total_from_manifest_dict(data)
        except Exception:
            return 0

    def run(self) -> None:
        try:
            if not self.train_data:
                raise RuntimeError("Dataset de treino vazio.")

            lr = float(self.learning_rate)
            steps_per_epoch = max(1, len(self.train_data))
            total_steps_global = steps_per_epoch * self.epochs
            t0 = datetime.now().timestamp()

            for ep in range(self.epochs):
                if self._stop:
                    self.stopped.emit()
                    break

                epoch_base = ep * steps_per_epoch

                def _progress_cb(
                    name: str,
                    step_counter: int,
                    total_steps_cb: int,
                    epoch: int,
                    total_epochs: int,
                    loss_val: float,
                ) -> bool:
                    if self._stop:
                        return False

                    step_global = epoch_base + int(step_counter)
                    step_global = max(0, min(int(step_global), int(total_steps_global)))

                    if int(step_counter) % self.progress_update_n == 0:
                        self.batch_progress.emit(
                            str(name),
                            int(step_global),
                            int(total_steps_global),
                            int(ep),
                            int(self.epochs),
                            float(loss_val),
                        )
                    return True

                train_map = train_networks(
                    network_names=[self.network_name],
                    data=self.train_data,
                    learning_rate=lr,
                    n_epochs=1,
                    shuffle=self.shuffle,
                    verbose=False,
                    progress_callback=_progress_cb,
                    batch_size=self.batch_size,
                )
                tr = train_map.get(self.network_name)
                if tr is None:
                    raise RuntimeError("Treino retornou vazio para a rede.")

                acc, out_min, out_max, counts, hits, hist100 = _eval_details_softmax(
                    self.network_name,
                    self.test_data if self.test_data else self.train_data,
                    self.eval_limit,
                )

                loss_final = float(getattr(tr, "final_loss", 0.0) or 0.0)
                loss_avg = float(getattr(tr, "avg_loss", loss_final) or loss_final)

                elapsed = max(1e-6, datetime.now().timestamp() - t0)
                processed_steps = min(total_steps_global, (ep + 1) * steps_per_epoch)
                sps = float(processed_steps) / float(elapsed)

                neu_total = self._neurons_total_from_manifest(self.network_name)

                metrics = EpochMetrics(
                    network_name=self.network_name,
                    neurons_total=neu_total,
                    epoch_index=ep,
                    total_epochs=self.epochs,
                    loss_final=loss_final,
                    loss_avg=loss_avg,
                    accuracy=float(acc),
                    samples_per_sec=float(sps),
                    learning_rate=float(lr),
                    class_counts=list(counts),
                    class_hits=list(hits),
                    out_min=float(out_min),
                    out_max=float(out_max),
                    conf_hist_100=list(hist100) if hist100 else [0] * 100,
                )
                self.epoch_metrics.emit(metrics)
                self.progress_epochs.emit(ep + 1, self.epochs)

                lr = max(self.lr_min, lr * self.lr_mult)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()


# -----------------------------------------------------------------------------
# Worker de evolução/gerações (thread)
# -----------------------------------------------------------------------------
class GenerationalWorker(QObject):
    finished = Signal(list)  # tops
    error = Signal(str)
    log = Signal(str)
    progress = Signal(str, int, int, float)  # name, step, total_steps, loss
    stopped = Signal()

    def __init__(
        self,
        parent_name: str,
        train_data: Sequence[Sample],
        epochs_per_individual: int,
        learning_rate: float,
        lr_mult: float,
        lr_min: float,
        eval_limit: Optional[int],
        shuffle: bool,
        generations: int,
        population: int,
        mutation_cfg: MutationConfig,
        update_every_n: int = 1000,
        output_index: int = 0,  # compat
        batch_size: int = 16,
    ):
        super().__init__()
        self.parent_name = parent_name
        self.train_data = list(train_data)
        self.epochs_per_individual = max(1, int(epochs_per_individual))
        self.learning_rate = float(learning_rate)
        self.lr_mult = float(lr_mult)
        self.lr_min = float(lr_min)
        self.shuffle = bool(shuffle)
        v = int(eval_limit) if eval_limit is not None else 0
        self.eval_limit = v if v > 0 else None
        self.generations = max(1, int(generations))
        self.population = max(1, int(population))
        self.mutation_cfg = mutation_cfg
        self.update_every_n = max(1, int(update_every_n))
        self.batch_size = max(1, int(batch_size))
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            if not self.train_data:
                raise RuntimeError("Dataset de treino vazio (gerações).")

            _ = build_engine()
            lr = float(self.learning_rate)

            tops: List[Dict[str, Any]] = []
            step_counter = 0
            total_steps = (
                self.generations
                * self.population
                * self.epochs_per_individual
                * max(1, len(self.train_data))
            )

            def _progress_cb(
                name: str,
                step: int,
                total: int,
                epoch: int,
                total_epochs: int,
                loss_val: float
            ) -> bool:
                nonlocal step_counter
                if self._stop:
                    return False
                step_counter += 1
                if step_counter % self.update_every_n == 0:
                    self.progress.emit(str(name), step_counter, total_steps, float(loss_val))
                return True

            for g in range(1, self.generations + 1):
                if self._stop:
                    self.stopped.emit()
                    break

                individuals: List[str] = []
                for i in range(self.population):
                    if self._stop:
                        break
                    ind_name = create_generation_individual(
                        parent_name=self.parent_name,
                        generation=g,
                        idx=i,
                        cfg=self.mutation_cfg,
                    )
                    individuals.append(ind_name)

                    train_networks(
                        network_names=[ind_name],
                        data=self.train_data,
                        learning_rate=lr,
                        n_epochs=self.epochs_per_individual,
                        shuffle=self.shuffle,
                        verbose=False,
                        progress_callback=_progress_cb,
                        batch_size=self.batch_size,
                    )

                ranked = rank_top_k(
                    individuals=individuals,
                    data=self.train_data,
                    k=min(10, len(individuals)),
                    eval_limit=self.eval_limit,
                    output_index=0,
                )
                tops = ranked
                self.log.emit(f"[GEN] Geração {g}/{self.generations} concluída. Top={len(ranked)}")
                lr = max(self.lr_min, lr * self.lr_mult)

            self.finished.emit(tops)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit([])


# -----------------------------------------------------------------------------
# Worker de avaliação simples (thread)
# -----------------------------------------------------------------------------
class EvalWorker(QObject):
    finished = Signal(list)
    error = Signal(str)
    log = Signal(str)

    def __init__(
        self,
        network_name: str,
        data: Sequence[Sample],
        limit: Optional[int],
    ):
        super().__init__()
        self.network_name = str(network_name)
        self.data = list(data)
        v = int(limit) if limit is not None else 0
        self.limit = v if v > 0 else None

    def run(self) -> None:
        try:
            if not self.data:
                raise RuntimeError("Dataset vazio (avaliação).")

            acc, out_min, out_max, counts, hits, hist100 = _eval_details_softmax(
                self.network_name, self.data, self.limit
            )
            self.log.emit(
                f"[EVAL] '{self.network_name}': acc={acc:.4f}, min={out_min:.4f}, max={out_max:.4f}"
            )
            self.finished.emit([{
                "name": self.network_name,
                "accuracy": acc,
                "out_min": out_min,
                "out_max": out_max,
                "class_counts": counts,
                "class_hits": hits,
                "conf_hist_100": hist100,
            }])

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit([])
