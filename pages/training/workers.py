from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict, Any
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from ia_engine import build_engine
from ia_generational import MutationConfig, create_generation_individual, rank_top_k
from ia_training import (
    train_networks,
    evaluate_networks,
    Sample,
)
from json_lib import build_paths, load_json


@dataclass
class EpochMetrics:
    network_name: str
    neurons_total: int
    out_min: float
    out_max: float
    epoch_index: int
    total_epochs: int
    loss_final: float
    loss_avg: float
    accuracy: float
    samples_per_sec: float
    learning_rate: float
    class_counts: List[int]
    class_hits: List[int]


class TrainWorker(QObject):
    epoch_metrics = Signal(object)  # EpochMetrics
    progress_epochs = Signal(int, int)  # epoch_atual, total_epocas
    stopped = Signal()
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        network_name: str,
        train_data: List[Sample],
        test_data: List[Sample],
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
        self.output_index = max(0, int(output_index))
        self.batch_size = max(1, int(batch_size))

        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _neurons_total_from_manifest(self, network_name: str) -> int:
        try:
            _, manifest_path, _, _ = build_paths(network_name)
            manifest = load_json(manifest_path) or {}
            st = manifest.get("structure", {}) or {}
            input_size = int(st.get("input_size") or 0)
            output_size = int(st.get("output_size") or 0)
            neurons = st.get("neurons") or []
            hidden_total = 0
            if isinstance(neurons, list):
                for n in neurons:
                    try:
                        hidden_total += int(n)
                    except Exception:
                        pass
            return max(0, input_size + hidden_total + output_size)
        except Exception:
            return 0

    def _class_counts_hits(self, name: str, data: Sequence[Sample]) -> Tuple[List[int], List[int]]:
        try:
            eng = build_engine(name)
            out_size = getattr(eng, "output_size", 0)
            if not out_size:
                return [], []
            try:
                out_size = int(out_size)
            except Exception:
                return [], []
            if out_size <= 0:
                return [], []

            counts = [0] * out_size
            hits = [0] * out_size

            limit = self.eval_limit if self.eval_limit else len(data)
            limit = min(limit, len(data))

            for i in range(limit):
                x_vals, y = data[i]
                probs = eng.forward(x_vals)
                if not probs:
                    continue
                pred = int(max(range(len(probs)), key=lambda k: probs[k]))
                if 0 <= pred < out_size:
                    counts[pred] += 1
                    if int(y) == pred:
                        hits[pred] += 1
            return counts, hits
        except Exception:
            return [], []

    def run(self):
        try:
            if not self.train_data:
                raise RuntimeError("Dataset de treino vazio.")

            bs = self.batch_size
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():  # type: ignore[attr-defined]
                    bs = max(bs, 64)
            except Exception:
                pass

            lr = self.learning_rate

            def _progress_cb(
                name: str,
                step_counter: int,
                total_steps: int,
                epoch: int,
                total_epochs: int,
                loss_val: float
            ) -> bool:
                if self._stop:
                    return False
                if step_counter % self.progress_update_n == 0:
                    self.log.emit(
                        f"[TRAIN] {name} | epoch={epoch+1}/{total_epochs} | "
                        f"step={step_counter}/{total_steps} | loss={loss_val:.6f}"
                    )
                return True

            t0 = datetime.now().timestamp()
            total_samples = len(self.train_data) * self.epochs

            for ep in range(self.epochs):
                if self._stop:
                    self.stopped.emit()
                    break

                train_map = train_networks(
                    network_names=[self.network_name],
                    data=self.train_data,
                    learning_rate=lr,
                    n_epochs=1,
                    shuffle=self.shuffle,
                    verbose=False,
                    progress_callback=_progress_cb,
                    batch_size=bs,
                )
                tr = train_map.get(self.network_name)
                if tr is None:
                    raise RuntimeError(f"train_networks() não retornou resultado para '{self.network_name}'.")

                eval_data = self.test_data if self.test_data else self.train_data
                eval_list = evaluate_networks([self.network_name], eval_data, limit=self.eval_limit, verbose=False)
                er = eval_list[0] if eval_list else None

                loss_final = float(getattr(tr, "loss_final", 0.0) or 0.0)
                loss_avg = float(getattr(tr, "loss_avg", loss_final) or loss_final)

                acc = float(getattr(er, "accuracy", 0.0) or 0.0) if er else 0.0
                out_min = float(getattr(er, "out_min", 0.0) or 0.0) if er else 0.0
                out_max = float(getattr(er, "out_max", 0.0) or 0.0) if er else 0.0

                processed = min(total_samples, (ep + 1) * len(self.train_data))
                elapsed = max(1e-6, datetime.now().timestamp() - t0)
                sps = processed / elapsed

                counts, hits = self._class_counts_hits(self.network_name, eval_data)
                neu_total = self._neurons_total_from_manifest(self.network_name)

                metrics = EpochMetrics(
                    network_name=self.network_name,
                    neurons_total=neu_total,
                    out_min=out_min,
                    out_max=out_max,
                    epoch_index=ep,
                    total_epochs=self.epochs,
                    loss_final=loss_final,
                    loss_avg=loss_avg,
                    accuracy=acc,
                    samples_per_sec=sps,
                    learning_rate=lr,
                    class_counts=counts,
                    class_hits=hits,
                )
                self.epoch_metrics.emit(metrics)
                self.progress_epochs.emit(ep + 1, self.epochs)

                lr = max(self.lr_min, lr * self.lr_mult)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()


class GenerationalWorker(QObject):
    epoch_metrics = Signal(object)
    tops_updated = Signal(list)
    progress = Signal(str, int, int, float)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        parent_name: str,
        train_data: List[Sample],
        epochs_per_individual: int,
        learning_rate: float,
        lr_mult: float,
        lr_min: float,
        shuffle: bool,
        eval_limit: int,
        generations: int,
        population: int,
        mutation_cfg: MutationConfig,
        update_every_n: int = 1000,
        output_index: int = 0,
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
        self.output_index = max(0, int(output_index))
        self.batch_size = max(1, int(batch_size))
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self):
        try:
            if not self.train_data:
                raise RuntimeError("Dataset de treino vazio.")

            bs = self.batch_size
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():  # type: ignore[attr-defined]
                    bs = max(bs, 64)
            except Exception:
                pass

            base_lr = self.learning_rate

            individuals: List[str] = []
            for i in range(self.population):
                if self._stop:
                    break
                nm = create_generation_individual(self.parent_name, i, self.mutation_cfg)
                individuals.append(nm)

            self.log.emit(f"[GERAÇÃO 0] População criada: {len(individuals)} indivíduos.")

            total_steps = len(individuals) * self.epochs_per_individual * len(self.train_data)
            step_counter = 0

            def _progress_cb(name: str, step_i: int, steps_total_epoch: int, epoch: int, total_epochs: int, loss_val: float) -> bool:
                nonlocal step_counter
                if self._stop:
                    return False
                step_counter += 1
                if step_counter % self.update_every_n == 0:
                    self.progress.emit(name, step_counter, total_steps, float(loss_val))
                return True

            for g in range(1, self.generations + 1):
                if self._stop:
                    break

                self.log.emit(f"[GERAÇÃO {g}] Treinando {len(individuals)} indivíduos...")

                for nm in individuals:
                    if self._stop:
                        break

                    lr = base_lr
                    for ep in range(self.epochs_per_individual):
                        if self._stop:
                            break

                        res = train_networks(
                            network_names=[nm],
                            data=self.train_data,
                            learning_rate=lr,
                            n_epochs=1,
                            shuffle=self.shuffle,
                            verbose=False,
                            progress_callback=_progress_cb,
                            batch_size=bs,
                        )
                        tr = res.get(nm)
                        if tr is None:
                            continue

                        eval_list = evaluate_networks([nm], self.train_data, limit=self.eval_limit, verbose=False)
                        er = eval_list[0] if eval_list else None

                        loss_final = float(getattr(tr, "loss_final", 0.0) or 0.0)
                        loss_avg = float(getattr(tr, "loss_avg", loss_final) or loss_final)

                        acc = float(getattr(er, "accuracy", 0.0) or 0.0) if er else 0.0
                        out_min = float(getattr(er, "out_min", 0.0) or 0.0) if er else 0.0
                        out_max = float(getattr(er, "out_max", 0.0) or 0.0) if er else 0.0

                        try:
                            _, manifest_path, _, _ = build_paths(nm)
                            manifest = load_json(manifest_path) or {}
                            st = manifest.get("structure", {}) or {}
                            input_size = int(st.get("input_size") or 0)
                            output_size = int(st.get("output_size") or 0)
                            neurons = st.get("neurons") or []
                            hidden_total = 0
                            if isinstance(neurons, list):
                                for n in neurons:
                                    try:
                                        hidden_total += int(n)
                                    except Exception:
                                        pass
                            neu_total = max(0, input_size + hidden_total + output_size)
                        except Exception:
                            neu_total = 0

                        metrics = EpochMetrics(
                            network_name=nm,
                            neurons_total=neu_total,
                            out_min=out_min,
                            out_max=out_max,
                            epoch_index=(g - 1) * self.epochs_per_individual + ep,
                            total_epochs=self.generations * self.epochs_per_individual,
                            loss_final=loss_final,
                            loss_avg=loss_avg,
                            accuracy=acc,
                            samples_per_sec=0.0,
                            learning_rate=lr,
                            class_counts=[],
                            class_hits=[],
                        )
                        self.epoch_metrics.emit(metrics)

                        lr = max(self.lr_min, lr * self.lr_mult)

                top = rank_top_k(individuals, k=min(10, len(individuals)))
                self.tops_updated.emit(top)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()


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
        self.network_name = network_name
        self.data = list(data)
        self.limit = int(limit) if limit is not None and limit > 0 else None

    def run(self):
        try:
            if not self.data:
                self.log.emit("[ERRO] Nenhum dataset foi carregado.")
                self.finished.emit([])
                return

            self.log.emit(
                f"[EVAL] Avaliando rede '{self.network_name}' "
                f"em {len(self.data)} amostras (limite={self.limit})."
            )

            results = evaluate_networks(
                network_names=[self.network_name],
                data=self.data,
                limit=self.limit,
                verbose=False,
            )
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit([])
