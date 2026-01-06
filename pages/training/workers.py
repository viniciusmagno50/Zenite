from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import inspect
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

    class_counts: List[int]
    class_hits: List[int]

    out_min: float
    out_max: float
    conf_hist_100: List[int]


# -----------------------------------------------------------------------------
# Helpers numéricos
# -----------------------------------------------------------------------------
def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _prob_to_bin_100(p: float) -> Optional[int]:
    """
    Converte probabilidade [0..1] para bin [0..99].
    Retorna None se p for NaN/inf.
    """
    if not _is_finite(p):
        return None
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    b = int(p * 100.0)
    if b >= 100:
        b = 99
    elif b < 0:
        b = 0
    return b


# -----------------------------------------------------------------------------
# Utilidades locais
# -----------------------------------------------------------------------------
def _softmax(logits: Sequence[float]) -> List[float]:
    """
    Softmax defensivo:
      - Se logits contiver NaN/inf -> retorna [] (amostra é ignorada)
      - Se exp/soma der NaN/inf -> retorna []
    """
    if not logits:
        return []

    for x in logits:
        if not _is_finite(x):
            return []

    m = max(float(x) for x in logits)
    exps: List[float] = []
    for x in logits:
        e = math.exp(float(x) - m)
        if not math.isfinite(e):
            return []
        exps.append(e)

    s = sum(exps)
    if not math.isfinite(s) or s <= 0.0:
        return []

    probs = [e / s for e in exps]
    if any((not math.isfinite(p)) for p in probs):
        return []
    return probs


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
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], sample[1]

    if hasattr(sample, "x") and hasattr(sample, "y"):
        try:
            return getattr(sample, "x"), getattr(sample, "y")
        except Exception:
            return None, None

    if isinstance(sample, dict):
        return sample.get("x"), sample.get("y")

    return None, None


def _y_to_class(y: Any) -> Optional[int]:
    if y is None:
        return None
    if isinstance(y, bool):
        return int(y)
    if isinstance(y, int):
        return int(y)
    if isinstance(y, float):
        if not _is_finite(y):
            return None
        return int(y)

    if isinstance(y, (list, tuple)):
        if not y:
            return None
        try:
            vals = [float(v) for v in y]
            if any((not math.isfinite(v)) for v in vals):
                return None
            return int(max(range(len(vals)), key=lambda i: vals[i]))
        except Exception:
            return None

    try:
        yy = list(y)
        if not yy:
            return None
        vals = [float(v) for v in yy]
        if any((not math.isfinite(v)) for v in vals):
            return None
        return int(max(range(len(vals)), key=lambda i: vals[i]))
    except Exception:
        return None


def _eval_details_softmax(
    network_name: str,
    data: Sequence[Sample],
    limit: Optional[int],
) -> Tuple[float, float, float, List[int], List[int], List[int]]:
    """
    Avalia via forward Python + softmax.
    Defensivo: ignora amostras com NaN/inf em logits/probs.
    """
    data_list = list(data)
    if limit is not None and int(limit) > 0:
        data_list = data_list[: int(limit)]

    if not data_list:
        return 0.0, 0.0, 0.0, [], [], [0] * 100

    weights, activation = _load_weights_activation_from_json(network_name)

    n_classes = 0
    try:
        if weights and isinstance(weights[-1], list):
            n_classes = len(weights[-1])
    except Exception:
        n_classes = 0

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
        class_counts[pred] += 1

        total += 1
        if pred == y_true:
            correct += 1
            if 0 <= y_true < len(class_hits):
                class_hits[y_true] += 1

            p = float(probs[y_true]) if 0 <= y_true < len(probs) else 0.0
            b = _prob_to_bin_100(p)
            if b is not None:
                hist100[b] += 1
                out_min = min(out_min, p)
                out_max = max(out_max, p)

    acc = (correct / float(total)) if total > 0 else 0.0
    if correct <= 0:
        out_min, out_max = 0.0, 0.0

    return float(acc), float(out_min), float(out_max), list(class_counts), list(class_hits), list(hist100)


def _call_build_engine_compat(name: str) -> Any:
    """
    Compatibilidade:
      - build_engine(name)   (seu caso atual)
      - build_engine()       (caso antigo)
    """
    try:
        sig = inspect.signature(build_engine)
        params = list(sig.parameters.values())
        requires_name = False
        for p in params:
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                if p.default is inspect._empty:
                    requires_name = True
                break
        if requires_name:
            return build_engine(name)
        return build_engine()
    except TypeError:
        try:
            return build_engine(name)
        except Exception:
            return build_engine()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Worker de treino (thread)
# -----------------------------------------------------------------------------
class TrainWorker(QObject):
    epoch_metrics = Signal(object)  # EpochMetrics
    progress_epochs = Signal(int, int)
    batch_progress = Signal(str, int, int, int, int, float)

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
        output_index: int = 0,
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

            _call_build_engine_compat(self.network_name)

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

                    safe_loss = _safe_float(loss_val, default=0.0)

                    if int(step_counter) % self.progress_update_n == 0:
                        self.batch_progress.emit(
                            str(name),
                            int(step_global),
                            int(total_steps_global),
                            int(ep),
                            int(self.epochs),
                            float(safe_loss),
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

                loss_final = _safe_float(getattr(tr, "final_loss", 0.0), default=0.0)
                loss_avg = _safe_float(getattr(tr, "avg_loss", loss_final), default=loss_final)

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
    finished = Signal(list)  # tops (lista de dicts)
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
        output_index: int = 0,
        batch_size: int = 16,
        evolve_best: bool = False,
    ):
        super().__init__()
        self.parent_name = str(parent_name)
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
        self.evolve_best = bool(evolve_best)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            if not self.train_data:
                raise RuntimeError("Dataset de treino vazio (gerações).")

            _call_build_engine_compat(self.parent_name)

            lr = float(self.learning_rate)
            current_parent = str(self.parent_name)

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
                    self.progress.emit(str(name), step_counter, total_steps, _safe_float(loss_val, 0.0))
                return True

            tops: List[Dict[str, Any]] = []

            for g in range(1, self.generations + 1):
                if self._stop:
                    self.stopped.emit()
                    break

                self.log.emit(f"[GEN] Iniciando geração {g}/{self.generations}... parent={current_parent}")

                results: List[Dict[str, Any]] = []

                for i in range(self.population):
                    if self._stop:
                        break

                    ind_name = create_generation_individual(
                        parent_name=current_parent,
                        generation_index=g,
                        individual_id=i,
                        cfg=self.mutation_cfg,
                    )

                    train_map = train_networks(
                        network_names=[ind_name],
                        data=self.train_data,
                        learning_rate=lr,
                        n_epochs=self.epochs_per_individual,
                        shuffle=self.shuffle,
                        verbose=False,
                        progress_callback=_progress_cb,
                        batch_size=self.batch_size,
                    )
                    tr = train_map.get(ind_name)
                    if tr is None:
                        self.log.emit(f"[GEN] '{ind_name}': treino retornou vazio.")
                        continue

                    loss_avg = _safe_float(getattr(tr, "avg_loss", getattr(tr, "final_loss", 0.0)), default=float("inf"))

                    acc, _, _, _, _, _ = _eval_details_softmax(ind_name, self.train_data, self.eval_limit)

                    results.append({
                        "name": ind_name,
                        "acc": float(acc),
                        "loss": float(loss_avg),
                        "generation": int(g),
                        "individual_id": int(i),
                    })

                if not results:
                    self.log.emit(f"[GEN] Geração {g}: nenhum resultado válido.")
                    continue

                tops = rank_top_k(results, k=min(10, len(results)))

                self.log.emit(f"[GEN] Geração {g} concluída. Top={len(tops)}")
                for t in tops[:5]:
                    self.log.emit(f"   - {t.get('name')} | acc={t.get('acc')} | loss={t.get('loss')}")

                # ✅ NOVO: se marcado, usa o melhor como base da próxima geração
                if self.evolve_best and tops:
                    try:
                        best_name = str(tops[0].get("name"))
                        if best_name:
                            current_parent = best_name
                            self.log.emit(f"[GEN] Base evolutiva atualizada: {current_parent}")
                    except Exception:
                        pass

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
