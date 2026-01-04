from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict, Any, Callable
from datetime import datetime
import inspect

from PySide6.QtCore import QObject, Signal

from ia_engine import build_engine
from ia_generational import MutationConfig, create_generation_individual, rank_top_k
from ia_training import (
    train_networks,
    evaluate_networks,
    Sample,
)
from json_lib import build_paths, load_json


# =============================================================================
# Helpers internos (resiliência contra mudanças de assinatura)
# =============================================================================
def _filter_kwargs_for_callable(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filtra kwargs para não quebrar caso a assinatura do backend evolua.
    (O contrato diz que adapters.py deve fazer isso, mas aqui garantimos robustez
    mesmo antes dos adapters estarem 100% implementados.)
    """
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)
        allowed = set(params.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def _safe_call(fn: Callable[..., Any], **kwargs: Any) -> Any:
    return fn(**_filter_kwargs_for_callable(fn, kwargs))


def _extract_losses(train_result: Any) -> Tuple[float, float]:
    """
    Suporta múltiplos nomes de campos para loss.
    Preferência:
      - final_loss / avg_loss
      - loss_final / loss_avg
      - loss / avg_loss
      - fallback: 0.0
    """
    if train_result is None:
        return 0.0, 0.0

    def _get(*names: str, default: float = 0.0) -> float:
        for n in names:
            if hasattr(train_result, n):
                try:
                    v = getattr(train_result, n)
                    if v is None:
                        continue
                    return float(v)
                except Exception:
                    continue
        return float(default)

    final_loss = _get("final_loss", "loss_final", "loss", default=0.0)
    avg_loss = _get("avg_loss", "loss_avg", default=final_loss)
    return float(final_loss), float(avg_loss)


def _extract_accuracy(eval_result: Any) -> float:
    if eval_result is None:
        return 0.0
    try:
        v = getattr(eval_result, "accuracy", None)
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


# =============================================================================
# Métricas por época para UI
# =============================================================================
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
            # build_paths() -> (pasta, arquivo_json, nome_sanitizado)
            _, manifest_path, _ = build_paths(network_name)
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

    def _eval_stats_engine(
        self,
        name: str,
        data: Sequence[Sample],
        limit: Optional[int],
        output_index: int,
    ) -> Tuple[float, float, float, List[int], List[int]]:
        """
        Avalia via engine.forward() para obter:
          - accuracy
          - out_min/out_max da saída 'output_index'
          - counts/hits por classe (pred vs y)
        Isso independe do retorno de evaluate_networks e mantém o gráfico Min/Max estável.
        """
        try:
            eng = build_engine(name)
            out_size = getattr(eng, "output_size", 0)
            try:
                out_size = int(out_size)
            except Exception:
                out_size = 0

            if not data:
                return 0.0, 0.0, 0.0, [], []

            n = len(data)
            lim = limit if (limit is not None and limit > 0) else n
            lim = max(1, min(lim, n))

            # Se não souber o tamanho de saída, ainda dá pra pegar min/max
            counts: List[int] = [0] * out_size if out_size > 0 else []
            hits: List[int] = [0] * out_size if out_size > 0 else []

            correct = 0
            total = 0
            out_min = None
            out_max = None

            for i in range(lim):
                x_vals, y = data[i]
                probs = eng.forward(x_vals)
                if not probs:
                    continue

                # min/max da saída selecionada
                if 0 <= output_index < len(probs):
                    v = float(probs[output_index])
                    out_min = v if out_min is None else min(out_min, v)
                    out_max = v if out_max is None else max(out_max, v)

                # accuracy + distribuição
                if out_size > 0:
                    pred = int(max(range(len(probs)), key=lambda k: probs[k]))
                    if 0 <= pred < out_size:
                        counts[pred] += 1
                        if int(y) == pred:
                            hits[pred] += 1
                            correct += 1
                        total += 1

            acc = (correct / total) if total > 0 else 0.0
            return (
                float(acc),
                float(out_min if out_min is not None else 0.0),
                float(out_max if out_max is not None else 0.0),
                counts,
                hits,
            )
        except Exception:
            return 0.0, 0.0, 0.0, [], []

    def run(self):
        try:
            if not self.train_data:
                raise RuntimeError("Dataset de treino vazio.")

            # Heurística: se tiver torch+cuda, aumenta batch
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

                train_map = _safe_call(
                    train_networks,
                    network_names=[self.network_name],
                    data=self.train_data,
                    learning_rate=lr,
                    n_epochs=1,
                    shuffle=self.shuffle,
                    verbose=False,
                    progress_callback=_progress_cb,
                    batch_size=bs,
                )
                tr = train_map.get(self.network_name) if isinstance(train_map, dict) else None
                if tr is None:
                    raise RuntimeError(f"train_networks() não retornou resultado para '{self.network_name}'.")

                # Avaliação:
                eval_data = self.test_data if self.test_data else self.train_data

                # 1) tenta evaluate_networks (se existir)
                er = None
                try:
                    eval_list = _safe_call(
                        evaluate_networks,
                        network_names=[self.network_name],
                        data=eval_data,
                        limit=self.eval_limit,
                        verbose=False,
                    )
                    er = eval_list[0] if eval_list else None
                except Exception:
                    er = None

                # Loss compatível
                loss_final, loss_avg = _extract_losses(tr)

                # Accuracy preferencialmente do evaluate_networks, senão do engine
                acc = _extract_accuracy(er)

                # Min/Max + counts/hits sempre pelo engine (estável)
                acc2, out_min, out_max, counts, hits = self._eval_stats_engine(
                    self.network_name,
                    eval_data,
                    self.eval_limit,
                    self.output_index,
                )
                if acc == 0.0 and acc2 > 0.0:
                    acc = acc2  # fallback

                processed = min(total_samples, (ep + 1) * len(self.train_data))
                elapsed = max(1e-6, datetime.now().timestamp() - t0)
                sps = processed / elapsed

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
                    accuracy=float(acc),
                    samples_per_sec=float(sps),
                    learning_rate=float(lr),
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
    epoch_metrics = Signal(object)   # EpochMetrics
    tops_updated = Signal(list)      # lista de dicts
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

    def _neurons_total_from_manifest(self, network_name: str) -> int:
        try:
            _, manifest_path, _ = build_paths(network_name)
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

    def _eval_stats_engine(
        self,
        name: str,
        data: Sequence[Sample],
        limit: Optional[int],
        output_index: int,
    ) -> Tuple[float, float, float]:
        """
        Retorna (acc, out_min, out_max) via engine.forward().
        """
        try:
            eng = build_engine(name)
            out_size = getattr(eng, "output_size", 0)
            try:
                out_size = int(out_size)
            except Exception:
                out_size = 0

            if not data:
                return 0.0, 0.0, 0.0

            n = len(data)
            lim = limit if (limit is not None and limit > 0) else n
            lim = max(1, min(lim, n))

            correct = 0
            total = 0
            out_min = None
            out_max = None

            for i in range(lim):
                x_vals, y = data[i]
                probs = eng.forward(x_vals)
                if not probs:
                    continue

                if 0 <= output_index < len(probs):
                    v = float(probs[output_index])
                    out_min = v if out_min is None else min(out_min, v)
                    out_max = v if out_max is None else max(out_max, v)

                if out_size > 0:
                    pred = int(max(range(len(probs)), key=lambda k: probs[k]))
                    if int(y) == pred:
                        correct += 1
                    total += 1

            acc = (correct / total) if total > 0 else 0.0
            return float(acc), float(out_min if out_min is not None else 0.0), float(out_max if out_max is not None else 0.0)
        except Exception:
            return 0.0, 0.0, 0.0

    def run(self):
        try:
            if not self.train_data:
                raise RuntimeError("Dataset de treino vazio.")

            # Heurística: se tiver torch+cuda, aumenta batch
            bs = self.batch_size
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():  # type: ignore[attr-defined]
                    bs = max(bs, 64)
            except Exception:
                pass

            base_lr = self.learning_rate

            # total_steps aproximado para progress
            approx_steps_per_epoch = max(1, len(self.train_data) // max(1, bs))
            total_steps = self.generations * self.population * self.epochs_per_individual * approx_steps_per_epoch
            step_counter = 0

            def _progress_cb(name: str, step_i: int, steps_total_epoch: int, epoch: int, total_epochs: int, loss_val: float) -> bool:
                nonlocal step_counter
                if self._stop:
                    return False
                step_counter += 1
                if step_counter % self.update_every_n == 0:
                    self.progress.emit(name, step_counter, total_steps, float(loss_val))
                return True

            individuals: List[str] = []

            # Cria geração 0
            for i in range(self.population):
                if self._stop:
                    break
                # assinatura correta: (parent, generation_index, individual_id, cfg)
                nm = create_generation_individual(self.parent_name, 0, i, self.mutation_cfg)
                individuals.append(str(nm))

            self.log.emit(f"[GERAÇÃO 0] População criada: {len(individuals)} indivíduos.")

            # Guardar últimas métricas por indivíduo para rankear
            last_metrics: Dict[str, Dict[str, Any]] = {}

            total_epochs_global = self.generations * self.epochs_per_individual

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

                        res = _safe_call(
                            train_networks,
                            network_names=[nm],
                            data=self.train_data,
                            learning_rate=lr,
                            n_epochs=1,
                            shuffle=self.shuffle,
                            verbose=False,
                            progress_callback=_progress_cb,
                            batch_size=bs,
                        )

                        tr = res.get(nm) if isinstance(res, dict) else None
                        if tr is None:
                            continue

                        loss_final, loss_avg = _extract_losses(tr)

                        # Avalia (engine) para obter acc/min/max
                        acc, out_min, out_max = self._eval_stats_engine(
                            nm, self.train_data, self.eval_limit, self.output_index
                        )

                        neu_total = self._neurons_total_from_manifest(nm)

                        epoch_global = (g - 1) * self.epochs_per_individual + ep

                        metrics = EpochMetrics(
                            network_name=nm,
                            neurons_total=neu_total,
                            out_min=out_min,
                            out_max=out_max,
                            epoch_index=epoch_global,
                            total_epochs=total_epochs_global,
                            loss_final=loss_final,
                            loss_avg=loss_avg,
                            accuracy=acc,
                            samples_per_sec=0.0,
                            learning_rate=lr,
                            class_counts=[],
                            class_hits=[],
                        )
                        self.epoch_metrics.emit(metrics)

                        # guarda para ranking
                        last_metrics[nm] = {
                            "name": nm,
                            "loss": float(loss_final),
                            "loss_avg": float(loss_avg),
                            "accuracy": float(acc),
                            "out_min": float(out_min),
                            "out_max": float(out_max),
                            "neurons_total": int(neu_total),
                            "generation": int(g),
                        }

                        lr = max(self.lr_min, lr * self.lr_mult)

                # Ranking top-k
                payload = list(last_metrics.values())
                top: List[Dict[str, Any]] = []
                try:
                    # tenta usar rank_top_k como definido no projeto
                    top = rank_top_k(payload, k=min(10, len(payload)))
                except Exception:
                    # fallback: ordena por loss asc, depois accuracy desc
                    top = sorted(
                        payload,
                        key=lambda d: (float(d.get("loss", 1e18)), -float(d.get("accuracy", 0.0))),
                    )[: min(10, len(payload))]

                self.tops_updated.emit(top)

                # Prepara próxima geração (se houver)
                if g < self.generations:
                    individuals = []
                    for i in range(self.population):
                        if self._stop:
                            break
                        nm2 = create_generation_individual(self.parent_name, g, i, self.mutation_cfg)
                        individuals.append(str(nm2))
                    self.log.emit(f"[GERAÇÃO {g}] Próxima população criada: {len(individuals)} indivíduos.")

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
                f"[EVAL] Avaliando rede '{self.network_name}' "
                f"em {len(self.data)} amostras (limite={self.limit})."
            )

            results = _safe_call(
                evaluate_networks,
                network_names=[self.network_name],
                data=self.data,
                limit=self.limit,
                verbose=False,
            )
            self.finished.emit(results if results is not None else [])

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit([])
