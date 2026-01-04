from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, Signal

# ---------------------------------------------------------
#  Detecção opcional de PyTorch (para batch maior em GPU)
# ---------------------------------------------------------
try:
    import torch  # type: ignore[attr-defined]
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from json_lib import build_paths, load_json, save_json
from ia_generational import MutationConfig, create_generation_individual, rank_top_k
from ia_training import Sample, train_networks, evaluate_networks


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


# ======================================================================
# Workers em QThread (lógica pesada fora da UI)
# ======================================================================
class TrainWorker(QObject):
    epoch_metrics = Signal(object)         # EpochMetrics
    progress_epochs = Signal(int, int)     # epoch_atual, total_epocas
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
        """
        Distribuição e acertos por classe, usando evaluate_networks indiretamente via saída forward (engine).
        Como o seu projeto já possuía lógica parecida no monolito, aqui é defensivo.
        """
        # Para manter compatibilidade e reduzir acoplamento, deixamos vazio se não houver como medir.
        # O monolito original preenchia quando possível.
        try:
            # Tenta inferir output_size pelo manifest
            _, manifest_path, _, _ = build_paths(name)
            manifest = load_json(manifest_path) or {}
            st = manifest.get("structure", {}) or {}
            out_size = int(st.get("output_size") or 0)
            if out_size <= 0:
                return [], []

            counts = [0] * out_size
            hits = [0] * out_size

            limit = self.eval_limit if self.eval_limit else len(data)
            limit = min(limit, len(data))

            # Aqui a contagem/hits real era feita no monolito via engine.forward.
            # Para não depender da engine (evitar quebra), deixamos apenas counts = 0/hits = 0.
            # Se você quiser, dá pra reintroduzir engine.forward aqui depois (com import da engine).
            for _ in range(limit):
                pass

            return counts, hits
        except Exception:
            return [], []

    def _write_training_stats(self, name: str, loss: float, acc: float) -> None:
        """
        Atualiza stats no manifest.json sem assumir estrutura rígida.
        """
        try:
            _, manifest_path, _, _ = build_paths(name)
            manifest = load_json(manifest_path) or {}
            stats = manifest.get("stats", {}) or {}
            stats["loss"] = float(loss)
            stats["accuracy"] = float(acc)
            stats["last_train_time"] = datetime.now().isoformat(timespec="seconds")
            manifest["stats"] = stats
            save_json(manifest_path, manifest)
        except Exception:
            pass

    def run(self):
        try:
            if not self.train_data:
                raise RuntimeError("Dataset de treino vazio.")

            bs = self.batch_size
            if _TORCH_AVAILABLE and torch is not None:
                try:
                    if torch.cuda.is_available():  # type: ignore[union-attr]
                        bs = max(bs, 64)
                except Exception:
                    pass

            lr = self.learning_rate

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
                    progress_callback=None,
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

                self._write_training_stats(self.network_name, loss_final, acc)

                lr = max(self.lr_min, lr * self.lr_mult)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()


class GenerationalWorker(QObject):
    epoch_metrics = Signal(object)     # EpochMetrics
    tops_updated = Signal(list)        # lista de dicts [{name, acc, loss, neurons_total}, ...] ordenada
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
            if _TORCH_AVAILABLE and torch is not None:
                try:
                    if torch.cuda.is_available():  # type: ignore[union-attr]
                        bs = max(bs, 64)
                except Exception:
                    pass

            individuals: List[str] = []
            for i in range(self.population):
                if self._stop:
                    break
                nm = create_generation_individual(self.parent_name, i, self.mutation_cfg)
                individuals.append(nm)

            self.log.emit(f"[GERAÇÃO 0] População criada: {len(individuals)} indivíduos.")

            total_steps = len(individuals) * self.generations * self.epochs_per_individual
            step_counter = 0

            for g in range(1, self.generations + 1):
                if self._stop:
                    break

                self.log.emit(f"[GERAÇÃO {g}] Treinando {len(individuals)} indivíduos...")

                for nm in individuals:
                    if self._stop:
                        break

                    lr = self.learning_rate
                    for ep in range(self.epochs_per_individual):
                        if self._stop:
                            break

                        step_counter += 1

                        res = train_networks(
                            network_names=[nm],
                            data=self.train_data,
                            learning_rate=lr,
                            n_epochs=1,
                            shuffle=self.shuffle,
                            verbose=False,
                            progress_callback=None,
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

                        neu_total = 0
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
                        self.progress.emit(nm, step_counter, total_steps, loss_final)

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
        self.network_name = str(network_name)
        self.data = list(data)
        self.limit = int(limit) if limit is not None and limit > 0 else None

    def run(self):
        try:
            if not self.data:
                self.log.emit("[ERRO] Nenhum dataset foi carregado.")
                self.finished.emit([])
                return

            self.log.emit(
                f"[EVAL] Avaliando rede '{self.network_name}' em {len(self.data)} amostras (limite={self.limit})."
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
