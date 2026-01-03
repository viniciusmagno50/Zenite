from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict, Callable, Any
from datetime import datetime
import math

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QMessageBox,
    QComboBox,
    QSizePolicy,
    QSplitter,
)
import pyqtgraph as pg

# ----------------------------------------------------------------------
# Patch defensivo: pyqtgraph.ErrorBarItem às vezes assume numpy arrays e
# quebra quando recebe listas (TypeError: 'list' - 'list').
# Isso pode derrubar o processo (0xC0000005) dependendo do backend Qt.
# Aqui garantimos conversão automática para np.ndarray quando possível.
# ----------------------------------------------------------------------
def _patch_pyqtgraph_errorbaritem() -> None:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return

    try:
        # Import direto para evitar depender de atributos em pg
        from pyqtgraph.graphicsItems.ErrorBarItem import ErrorBarItem  # type: ignore
    except Exception:
        return

    if getattr(ErrorBarItem, "_zenite_patched", False):
        return

    _orig_setData = ErrorBarItem.setData

    def _setData_patched(self, *args, **kwargs):
        try:
            # kwargs style: x, y, top, bottom, width, beam, etc.
            for k in ("x", "y", "top", "bottom"):
                if k in kwargs and isinstance(kwargs[k], list):
                    kwargs[k] = np.asarray(kwargs[k], dtype=float)
            # args style: ErrorBarItem(x=?, y=?, top=?, bottom=?)
            # Normalmente args não são usados, mas deixamos como está.
        except Exception:
            pass
        return _orig_setData(self, *args, **kwargs)

    ErrorBarItem.setData = _setData_patched  # type: ignore[assignment]
    ErrorBarItem._zenite_patched = True  # type: ignore[attr-defined]

_patch_pyqtgraph_errorbaritem()

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
from ia_context import AIContext
from ia_training import train_networks, evaluate_networks, Sample, TrainResult, EvalResult
from ia_engine import build_engine
from ia_generational import MutationConfig, create_generation_individual, rank_top_k


# Defaults de LR quando não houver nada salvo no JSON
DEFAULT_LR = 0.01
DEFAULT_LR_MULT = 0.95
DEFAULT_LR_MIN = 0.000010000


# ----------------------------------------------------------------------
# Métricas agregadas por época (para os gráficos)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Workers em QThread
# ----------------------------------------------------------------------
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
        eval_limit: int,
        shuffle: bool,
        progress_update_n: int,
        output_index: int,
        batch_size: int = 32,
    ):
        super().__init__()
        self.network_name = network_name
        self.train_data = list(train_data or [])
        self.test_data = list(test_data or [])
        self.epochs = max(1, int(epochs))
        self.learning_rate = float(learning_rate)
        self.lr_mult = float(lr_mult)
        self.lr_min = float(lr_min)
        try:
            v = int(eval_limit) if eval_limit is not None else 0
        except Exception:
            v = 0
        self.eval_limit = v if v > 0 else None
        self.shuffle = bool(shuffle)
        self.progress_update_n = max(1, int(progress_update_n))
        self.output_index = max(0, int(output_index))
        self.batch_size = max(1, int(batch_size))
        self._stop = False

    def stop(self):
        self._stop = True

    def request_stop(self) -> None:
        """Compatibilidade com a UI (botão Parar)."""
        self.stop()

    @staticmethod
    def _neurons_total_from_manifest(name: str) -> int:
        try:
            _, jpath, _ = build_paths(name)
            data = load_json(jpath)
            if not isinstance(data, dict):
                return 0
            struct = data.get("structure") or {}
            if not isinstance(struct, dict):
                return 0
            neurons = struct.get("neurons") or []
            if isinstance(neurons, list):
                return int(sum(int(x) for x in neurons if isinstance(x, (int, float))))
        except Exception:
            pass
        return 0

    def _compute_out_minmax(self, name: str, data: List[Sample]) -> Tuple[float, float]:
        try:
            eng = build_engine(name)
            idx = min(self.output_index, max(0, eng.output_size - 1))
            out_min = 1.0
            out_max = 0.0
            limit = self.eval_limit if self.eval_limit > 0 else len(data)
            limit = min(limit, len(data))
            for i in range(limit):
                x_vals, _ = data[i]
                probs = eng.forward(x_vals)
                p = float(probs[idx]) if probs and idx < len(probs) else 0.0
                out_min = min(out_min, p)
                out_max = max(out_max, p)
            return out_min, out_max
        except Exception:
            return 0.0, 0.0

    def _compute_distribution(self, name: str, data: List[Sample]) -> Tuple[List[int], List[int]]:
        try:
            eng = build_engine(name)
            out_size = max(1, int(eng.output_size))
            counts = [0] * out_size
            hits = [0] * out_size

            limit = self.eval_limit if self.eval_limit > 0 else len(data)
            limit = min(limit, len(data))
            for i in range(limit):
                x_vals, y = data[i]
                probs = eng.forward(x_vals)
                if not probs:
                    continue
                pred = int(max(range(len(probs)), key=lambda k: probs[k]))
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

            # Ajusta batch automaticamente: GPU normalmente suporta maior
            bs = self.batch_size
            if _TORCH_AVAILABLE and torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[union-attr]
                bs = max(bs, 64)

            lr = self.learning_rate

            def _progress_cb(name: str, step_counter: int, total_steps: int, epoch: int, total_epochs: int, loss_val: float) -> bool:
                if self._stop:
                    return False
                # "Qnt. de dados" = emitir log/progresso em intervalos
                if step_counter % self.progress_update_n == 0 or step_counter == total_steps:
                    self.log.emit(f"[TREINO] {name} | epoch {epoch}/{total_epochs} | {step_counter}/{total_steps} | loss={loss_val:.4g}")
                return True

            for ep in range(self.epochs):
                if self._stop:
                    self.stopped.emit()
                    return

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

                # avaliação
                eval_data = self.test_data if self.test_data else self.train_data
                eval_list = evaluate_networks([self.network_name], eval_data, limit=self.eval_limit, verbose=False)
                er = eval_list[0] if eval_list else None

                acc = float(er.accuracy) if er is not None else 0.0
                loss_eval = float(er.avg_loss or 0.0) if er is not None else 0.0

                loss_final = float(tr.final_loss)
                loss_avg = float(tr.avg_loss)

                sps = (float(tr.samples) / float(tr.elapsed_seconds)) if (tr.elapsed_seconds and tr.elapsed_seconds > 0) else 0.0

                out_min, out_max = self._compute_out_minmax(self.network_name, eval_data)
                counts, hits = self._compute_distribution(self.network_name, eval_data)
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


class GenerationalWorker(QObject):
    epoch_metrics = Signal(object)     # EpochMetrics (1 por indivíduo, por geração)
    tops_updated = Signal(list)        # lista de dicts [{name, acc, loss, neurons_total}, ...] ordenada
    progress = Signal(str, int, int, float)  # name, processed, total, loss
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
        update_every_n: int,
        output_index: int,
        batch_size: int = 32,
    ):
        super().__init__()
        self.parent_name = parent_name
        self.train_data = train_data
        self.epochs_per_individual = int(epochs_per_individual)
        self.learning_rate = float(learning_rate)
        self.lr_mult = float(lr_mult)
        self.lr_min = float(lr_min)
        self.shuffle = bool(shuffle)
        try:
            v = int(eval_limit) if eval_limit is not None else 0
        except Exception:
            v = 0
        self.eval_limit = v if v > 0 else None
        self.generations = int(generations)
        self.population = int(population)
        self.mutation_cfg = mutation_cfg
        self.update_every_n = max(1, int(update_every_n))
        self.output_index = max(0, int(output_index))
        self.batch_size = max(1, int(batch_size))
        self._stop = False

    def stop(self):
        self._stop = True


    def request_stop(self) -> None:
        """Compatibilidade com a UI (botão Parar)."""
        self.stop()

    @staticmethod
    def _neurons_total_from_manifest(name: str) -> int:
        try:
            folder, jpath, _ = build_paths(name)
            data = load_json(jpath)
            if not isinstance(data, dict):
                return 0
            struct = data.get("structure") or {}
            if not isinstance(struct, dict):
                return 0
            neurons = struct.get("neurons") or []
            if isinstance(neurons, list):
                return int(sum(int(x) for x in neurons if isinstance(x, (int, float))))
        except Exception:
            pass
        return 0

    def _compute_out_minmax(self, name: str, data: List[Sample]) -> Tuple[float, float]:
        try:
            eng = build_engine(name)
            idx = min(self.output_index, max(0, eng.output_size - 1))
            out_min = 1.0
            out_max = 0.0
            limit = self.eval_limit if self.eval_limit > 0 else len(data)
            limit = min(limit, len(data))
            for i in range(limit):
                x_vals, _ = data[i]
                probs = eng.forward(x_vals)
                p = float(probs[idx]) if probs and idx < len(probs) else 0.0
                out_min = min(out_min, p)
                out_max = max(out_max, p)
            return out_min, out_max
        except Exception:
            return 0.0, 0.0

    def run(self):
        try:
            base_lr = self.learning_rate

            for g in range(self.generations):
                if self._stop:
                    break

                # cria população
                individuals: List[str] = []
                for i in range(self.population):
                    nm = create_generation_individual(
                        self.parent_name,
                        generation_index=g,
                        individual_id=i,
                        cfg=self.mutation_cfg,
                    )
                    individuals.append(nm)

                self.log.emit(f"[GERAÇÃO {g}] População criada: {len(individuals)} indivíduos.")

                # treino (todos os indivíduos no mesmo loop)
                # Para performance: batelada (batch_size) e 1 chamada por indivíduo (n_epochs = epochs_per_individual)
                def _progress_cb(name: str, step_counter: int, total_steps: int, epoch: int, total_epochs: int, loss_val: float) -> bool:
                    if self._stop:
                        return False
                    if step_counter % self.update_every_n == 0 or step_counter == total_steps:
                        self.progress.emit(name, step_counter, total_steps, float(loss_val))
                    return True

                # Treina cada indivíduo (sequencial; GPU normalmente não ganha com threads de modelos diferentes)
                results_train: Dict[str, TrainResult] = {}
                for nm in individuals:
                    if self._stop:
                        break

                    # LR schedule por indivíduo
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
                            batch_size=self.batch_size,
                        )
                        tr = res.get(nm)
                        if tr is not None:
                            results_train[nm] = tr
                        lr = max(self.lr_min, lr * self.lr_mult)

                # Avalia todos
                eval_list = evaluate_networks(individuals, self.train_data, limit=self.eval_limit, verbose=False)

                scored: List[Dict[str, Any]] = []
                for er in eval_list:
                    nm = er.name
                    acc = float(er.accuracy)
                    loss = float(er.avg_loss or 0.0)

                    n_total = self._neurons_total_from_manifest(nm)
                    out_min, out_max = self._compute_out_minmax(nm, self.train_data)

                    tr = results_train.get(nm)
                    loss_final = float(tr.final_loss) if tr is not None else loss
                    loss_avg = float(tr.avg_loss) if tr is not None else loss
                    sps = (float(tr.samples) / float(tr.elapsed_seconds)) if (tr is not None and tr.elapsed_seconds and tr.elapsed_seconds > 0) else 0.0

                    # emite métricas (1 ponto por geração)
                    m = EpochMetrics(
                        network_name=nm,
                        neurons_total=n_total,
                        out_min=out_min,
                        out_max=out_max,
                        epoch_index=g,
                        total_epochs=self.generations,
                        loss_final=loss_final,
                        loss_avg=loss_avg,
                        accuracy=acc,
                        samples_per_sec=sps,
                        learning_rate=float(base_lr),
                        class_counts=[],
                        class_hits=[],
                    )
                    self.epoch_metrics.emit(m)

                    scored.append({"name": nm, "acc": acc, "loss": loss, "neurons_total": n_total})

                tops = rank_top_k(scored, k=max(1, 3))
                self.tops_updated.emit(tops)

                self.log.emit(
                    f"[GERAÇÃO {g}] Top: "
                    + ", ".join(f"{t['name']} (acc={t.get('acc',0):.4f}, loss={t.get('loss',0):.4g})" for t in tops)
                )

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))

class EvalWorker(QObject):
    finished = Signal(list)  # List[EvalResult]
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

    def run(self) -> None:
        try:
            if not self.data:
                raise ValueError("Nenhum dado de avaliação foi carregado.")

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


# ----------------------------------------------------------------------
# Painel principal
# ----------------------------------------------------------------------
class WdTrainingPane(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Threads/ workers
        self._train_thread: Optional[QThread] = None
        self._train_worker: Optional[TrainWorker] = None
        self._eval_thread: Optional[QThread] = None
        self._eval_worker: Optional[EvalWorker] = None

        # Dados de treino/ avaliação
        self._train_data: Optional[List[Sample]] = None
        self._eval_data: Optional[List[Sample]] = None
        self._test_data: Optional[List[Sample]] = None  # alias p/ avaliações rápidas (UI)

        # Controle de cenário (train_datasets.py)
        self._dataset_funcs: Dict[str, Callable[[], Sequence[Sample]]] = {}
        self.cmb_dataset: Optional[QComboBox] = None

        # Configs adicionais
        self.spin_data_update: Optional[QSpinBox] = None  # Qnt. de dados (updates intra-época)
        self.chk_train_generations: Optional[QCheckBox] = None
        self.chk_train_loop: Optional[QCheckBox] = None
        self.spin_generations: Optional[QSpinBox] = None
        self.spin_population: Optional[QSpinBox] = None
        self.btn_evolve: Optional[QPushButton] = None

        self.chk_allow_add_layers: Optional[QCheckBox] = None
        self.chk_allow_remove_layers: Optional[QCheckBox] = None
        self.chk_allow_add_neurons: Optional[QCheckBox] = None
        self.chk_allow_remove_neurons: Optional[QCheckBox] = None
        self.spin_layer_delta: Optional[QSpinBox] = None
        self.spin_neuron_delta: Optional[QSpinBox] = None

        self.spin_show_top: Optional[QSpinBox] = None
        self.chk_show_original: Optional[QCheckBox] = None
        self.spin_output_index: Optional[QSpinBox] = None

        # Estado do modo geracional
        self._best_candidate: Optional[Dict[str, Any]] = None
        self._last_generation_tops: List[Dict[str, Any]] = []

        # Históricos para gráficos (por rede)
        # net_key -> { "x": [...], "loss_final": [...], "loss_avg": [...], "lr": [...], "perf": [...], "acc": [...],
        #             "out_min": [...], "out_max": [...], "class_counts": [...], "class_hits": [...] }
        self._net_hist: Dict[str, Dict[str, List[float]]] = {}
        self._net_epoch_x: Dict[str, List[int]] = {}

        # Curvas/itens por plot e por rede
        self._curves_loss_epoch: Dict[str, pg.PlotDataItem] = {}
        self._curves_lr: Dict[str, pg.PlotDataItem] = {}
        self._curves_perf: Dict[str, pg.PlotDataItem] = {}
        self._curves_acc: Dict[str, pg.PlotDataItem] = {}

        # Min/Max da saída selecionada por rede (barras min->max)
        self._minmax_bar: Optional[pg.BarGraphItem] = None
        self._minmax_scatter_min: Optional[pg.ScatterPlotItem] = None
        self._minmax_scatter_max: Optional[pg.ScatterPlotItem] = None

        # Distribuição (bars) por rede (somente para redes exibidas)
        self._dist_items: Dict[str, pg.BarGraphItem] = {}

        # Contagem de dados/ épocas para progresso
        self._total_train_samples: int = 0
        self._total_epochs: int = 0

        # Widgets de log/ progresso
        self.txt_log: Optional[QPlainTextEdit] = None
        self.lbl_progress: Optional[QLabel] = None
        self.lbl_progress_detail: Optional[QLabel] = None
        self.lbl_metrics: Optional[QLabel] = None
        self.progress_bar: Optional[QProgressBar] = None

        # Splitter vertical das linhas 2 e 3
        self.splitter: Optional[QSplitter] = None

        self._build_ui()

        # Ao criar o painel, já tenta carregar LR da IA atual (se houver)
        name = AIContext.get_name()
        if name:
            self._load_lr_from_json(name)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)  # espaçamento vertical mínimo

        # ======================== LINHA 1 ==============================
        row1 = QHBoxLayout()
        row1.setSpacing(2)  # espaçamento horizontal mínimo

        grp_flow = self._build_flow_group()
        grp_params = self._build_params_group()
        grp_graph_opts = self._build_graph_options_group()
        grp_gen_cfg = self._build_generation_config_group()
        grp_mut_cfg = self._build_mutation_config_group()
        grp_gen_view = self._build_generation_view_group()

        # 6 colunas com mesma largura
        for grp in (grp_flow, grp_params, grp_graph_opts, grp_gen_cfg, grp_mut_cfg, grp_gen_view):
            grp.setMinimumWidth(180)
            grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            row1.addWidget(grp, 1)

        root.addLayout(row1)

        # ===================== SPLITTER VERTICAL =======================
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(2)
        self.splitter.setChildrenCollapsible(False)

        # ---------------------- LINHA 2 -------------------------------
        row2 = QHBoxLayout()
        row2.setSpacing(2)

        self.plot_minmax = pg.PlotWidget(title="Min/Max da saída selecionada (prob)")
        self.plot_minmax.showGrid(x=True, y=True)
        self.plot_minmax.enableAutoRange("y", True)
        self.plot_minmax.setLabel("left", "Probabilidade")
        self.plot_minmax.setLabel("bottom", "Rede")

        # Itens (criados uma vez e atualizados)
        # Barra representa o intervalo [min, max] da saída selecionada para cada rede.
        self._minmax_bar = pg.BarGraphItem(x=[], height=[], width=0.6, y0=[])
        self.plot_minmax.addItem(self._minmax_bar)
        self._minmax_scatter_min = pg.ScatterPlotItem(x=[], y=[], size=8, symbol="t")
        self._minmax_scatter_max = pg.ScatterPlotItem(x=[], y=[], size=8, symbol="t1")
        self.plot_minmax.addItem(self._minmax_scatter_min)
        self.plot_minmax.addItem(self._minmax_scatter_max)

        self.plot_loss_epoch = pg.PlotWidget(title="Loss médio por época")
        self.plot_loss_epoch_curve = self.plot_loss_epoch.plot(pen=pg.mkPen(width=2))
        self.plot_loss_epoch.showGrid(x=True, y=True)
        self.plot_loss_epoch.enableAutoRange("y", True)

        self.plot_lr = pg.PlotWidget(title="Learning rate")
        self.plot_lr_curve = self.plot_lr.plot(pen=pg.mkPen(width=2))
        self.plot_lr.showGrid(x=True, y=True)
        self.plot_lr.enableAutoRange("y", True)

        self.plot_perf = pg.PlotWidget(title="Performance (samples/s)")
        self.plot_perf_curve = self.plot_perf.plot(pen=pg.mkPen(width=2))
        self.plot_perf.showGrid(x=True, y=True)
        self.plot_perf.enableAutoRange("y", True)

        for w in (self.plot_minmax, self.plot_loss_epoch, self.plot_lr, self.plot_perf):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            row2.addWidget(w, 1)

        row2_container = QWidget()
        row2_container.setLayout(row2)
        row2_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        row2_container.setMinimumHeight(0)

        # ---------------------- LINHA 3 -------------------------------
        row3 = QHBoxLayout()
        row3.setSpacing(2)

        self.plot_dist = pg.PlotWidget(title="Distribuição de saídas (predições)")
        self.plot_dist.showGrid(x=True, y=True)
        self.plot_dist.enableAutoRange("y", True)
        self._dist_item = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.plot_dist.addItem(self._dist_item)

        self.plot_acc = pg.PlotWidget(title="Acurácia por época")
        self.plot_acc_curve = self.plot_acc.plot(pen=pg.mkPen(width=2))
        self.plot_acc.showGrid(x=True, y=True)
        self.plot_acc.enableAutoRange("y", True)

        prog_group = QGroupBox("Progresso de treinamento")
        prog_layout = QVBoxLayout(prog_group)
        prog_layout.setSpacing(4)
        prog_layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_progress = QLabel("Aguardando início do treino...")
        self.lbl_progress_detail = QLabel("Dados processados: 0/0 (0%)")

        for lbl in (self.lbl_progress, self.lbl_progress_detail):
            lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.lbl_metrics = QLabel(
            "Acurácia: --\n"
            "Loss: --\n"
            "Perf: --\n"
            "LR: --"
        )
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        prog_layout.addWidget(self.lbl_progress)
        prog_layout.addWidget(self.lbl_progress_detail)
        prog_layout.addWidget(self.lbl_metrics)
        prog_layout.addWidget(self.progress_bar)
        prog_layout.addStretch(1)

        log_group = QGroupBox("Log de treinamento")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 6, 6, 6)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.txt_log.setMinimumHeight(0)
        log_layout.addWidget(self.txt_log)

        # Permitir que os grupos encolham o quanto for preciso
        prog_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        prog_group.setMinimumHeight(0)
        log_group.setMinimumHeight(0)

        for w in (self.plot_dist, self.plot_acc, prog_group, log_group):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            row3.addWidget(w, 1)

        row3_container = QWidget()
        row3_container.setLayout(row3)
        row3_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        row3_container.setMinimumHeight(0)

        self.splitter.addWidget(row2_container)
        self.splitter.addWidget(row3_container)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        root.addWidget(self.splitter, 1)

        # Inicializa datasets (combo)
        self._reload_dataset_functions()
        self._clear_histories()

    # ------------------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        self._adjust_splitter()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adjust_splitter()

    def _adjust_splitter(self):
        if not self.splitter:
            return
        h = self.splitter.height()
        if h <= 0:
            return
        half = h // 2
        self.splitter.setSizes([half, h - half])

    # ==================================================================
    # Construção dos grupos da LINHA 1
    # ==================================================================
    def _build_flow_group(self) -> QGroupBox:
        grp = QGroupBox("Fluxo de dados e treinamento")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)

        lbl_ds = QLabel("Cenário de dados (train_datasets.py):")
        self.cmb_dataset = QComboBox()
        layout.addWidget(lbl_ds)
        layout.addWidget(self.cmb_dataset)

        self.btn_load_data = QPushButton("1) Carregar dados")
        self.btn_prepare = QPushButton("2) Preparar treino")
        self.btn_train = QPushButton("3) Treinar rede")
        self.btn_eval = QPushButton("4) Avaliar rede")
        self.btn_stop = QPushButton("Parar")

        for btn in [
            self.btn_load_data,
            self.btn_prepare,
            self.btn_train,
            self.btn_eval,
            self.btn_stop,
        ]:
            btn.setFixedHeight(26)
            layout.addWidget(btn)

        self.btn_stop.setEnabled(False)
        layout.addStretch(1)

        self.btn_load_data.clicked.connect(self._on_load_data_clicked)
        self.btn_prepare.clicked.connect(self._on_prepare_clicked)
        self.btn_train.clicked.connect(self._on_train_clicked)
        self.btn_eval.clicked.connect(self._on_eval_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        return grp

    def _build_params_group(self) -> QGroupBox:
        grp = QGroupBox("Parâmetros de treinamento")
        layout = QFormLayout(grp)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignTop)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(2)
        grp.setContentsMargins(6, 6, 6, 6)

        # SpinBoxes compactos (um pouco mais largos)
        def compact_spinbox(spin: QSpinBox | QDoubleSpinBox, is_double: bool = False):
            spin.setMaximumWidth(110)
            if is_double:
                spin.setStyleSheet("QDoubleSpinBox { padding: 0 3px; }")
            else:
                spin.setStyleSheet("QSpinBox { padding: 0 3px; }")

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(10)
        compact_spinbox(self.spin_epochs, is_double=False)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(7)
        self.spin_lr.setRange(1e-9, 1.0)
        self.spin_lr.setSingleStep(0.0001)
        self.spin_lr.setValue(DEFAULT_LR)
        compact_spinbox(self.spin_lr, is_double=True)

        self.spin_lr_mult = QDoubleSpinBox()
        self.spin_lr_mult.setDecimals(7)
        self.spin_lr_mult.setRange(0.0, 2.0)
        self.spin_lr_mult.setSingleStep(0.0001)
        self.spin_lr_mult.setValue(DEFAULT_LR_MULT)
        self.spin_lr_mult.setToolTip(
            "Fator multiplicador do learning rate a cada época.\n"
            "Ex.: 0.9 = reduz 10% a cada época. 1.0 = mantém fixo."
        )
        compact_spinbox(self.spin_lr_mult, is_double=True)

        self.spin_lr_min = QDoubleSpinBox()
        self.spin_lr_min.setDecimals(7)
        self.spin_lr_min.setRange(0.0, 1.0)
        self.spin_lr_min.setSingleStep(1e-6)
        self.spin_lr_min.setValue(DEFAULT_LR_MIN)
        self.spin_lr_min.setToolTip(
            "Valor mínimo permitido para o learning rate.\n"
            "0.0 = sem piso (pode reduzir indefinidamente)."
        )
        compact_spinbox(self.spin_lr_min, is_double=True)

        # Qnt. de dados (atualizações intra-época)
        self.spin_data_update = QSpinBox()
        self.spin_data_update.setRange(1, 10_000_000)
        self.spin_data_update.setValue(1000)
        self.spin_data_update.setToolTip(
            "Atualiza progresso/log a cada N amostras processadas (sem esperar a época acabar)."
        )
        compact_spinbox(self.spin_data_update, is_double=False)

        self.spin_eval_limit = QSpinBox()
        self.spin_eval_limit.setRange(0, 1_000_000)
        self.spin_eval_limit.setValue(0)
        self.spin_eval_limit.setToolTip("0 = usar todos os dados na avaliação.")
        compact_spinbox(self.spin_eval_limit, is_double=False)

        layout.addRow("Épocas:", self.spin_epochs)
        layout.addRow("Learning rate:", self.spin_lr)
        layout.addRow("Fator LR/época:", self.spin_lr_mult)
        layout.addRow("LR mínimo:", self.spin_lr_min)
        layout.addRow("Qnt. de dados:", self.spin_data_update)
        layout.addRow("Limite p/ avaliação:", self.spin_eval_limit)

        self.chk_shuffle = QCheckBox("Embaralhar dados a cada época")
        self.chk_shuffle.setChecked(True)
        self.chk_shuffle.setStyleSheet("QCheckBox { padding: 0; }")
        layout.addRow("", self.chk_shuffle)

        return grp

    def _build_graph_options_group(self) -> QGroupBox:
        grp = QGroupBox("Gráficos em tempo real")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)

        self.chk_plot_minmax = QCheckBox("Min/Max da saída selecionada")
        self.chk_plot_minmax.setChecked(True)

        self.chk_plot_loss_epoch = QCheckBox("Loss médio por época")
        self.chk_plot_loss_epoch.setChecked(True)

        self.chk_plot_acc = QCheckBox("Acurácia por época")
        self.chk_plot_acc.setChecked(True)

        self.chk_plot_dist = QCheckBox("Distribuição de saídas")
        self.chk_plot_dist.setChecked(True)

        self.chk_plot_lr = QCheckBox("Learning rate")
        self.chk_plot_lr.setChecked(True)

        self.chk_plot_perf = QCheckBox("Performance")
        self.chk_plot_perf.setChecked(True)

        for chk in [
            self.chk_plot_minmax,
            self.chk_plot_loss_epoch,
            self.chk_plot_acc,
            self.chk_plot_dist,
            self.chk_plot_lr,
            self.chk_plot_perf,
        ]:
            layout.addWidget(chk)

        layout.addStretch(1)
        return grp

    def _build_generation_config_group(self) -> QGroupBox:
        grp = QGroupBox("Configuração de geração")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)

        self.chk_train_generations = QCheckBox("Treinar gerações")
        self.chk_train_generations.setChecked(False)
        self.chk_train_loop = QCheckBox("Treinar em loop (em breve)")
        self.chk_train_loop.setChecked(False)

        layout.addWidget(self.chk_train_generations)
        layout.addWidget(self.chk_train_loop)

        # Gerações
        lbl_g = QLabel("Gerações:")
        self.spin_generations = QSpinBox()
        self.spin_generations.setRange(1, 10_000)
        self.spin_generations.setValue(3)
        self.spin_generations.setMaximumWidth(110)

        # População
        lbl_p = QLabel("População:")
        self.spin_population = QSpinBox()
        self.spin_population.setRange(1, 10_000)
        self.spin_population.setValue(10)
        self.spin_population.setMaximumWidth(110)

        for lbl, spin in ((lbl_g, self.spin_generations), (lbl_p, self.spin_population)):
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            layout.addWidget(lbl)
            layout.addWidget(spin)

        self.btn_evolve = QPushButton("Evoluir (substituir pela melhor)")
        self.btn_evolve.setFixedHeight(26)
        self.btn_evolve.clicked.connect(self._on_evolve_clicked)
        layout.addWidget(self.btn_evolve)

        layout.addStretch(1)
        return grp


    def _build_mutation_config_group(self) -> QGroupBox:
        grp = QGroupBox("Regras de mutação")
        layout = QFormLayout(grp)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignTop)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(2)
        grp.setContentsMargins(6, 6, 6, 6)

        self.chk_allow_add_layers = QCheckBox("Permitir + camadas")
        self.chk_allow_remove_layers = QCheckBox("Permitir - camadas")
        self.chk_allow_add_neurons = QCheckBox("Permitir + neurônios")
        self.chk_allow_remove_neurons = QCheckBox("Permitir - neurônios")

        for chk in (
            self.chk_allow_add_layers,
            self.chk_allow_remove_layers,
            self.chk_allow_add_neurons,
            self.chk_allow_remove_neurons,
        ):
            chk.setChecked(True)

        self.spin_layer_delta = QSpinBox()
        self.spin_layer_delta.setRange(0, 1000)
        self.spin_layer_delta.setValue(1)
        self.spin_layer_delta.setMaximumWidth(110)

        self.spin_neuron_delta = QSpinBox()
        self.spin_neuron_delta.setRange(0, 10_000)
        self.spin_neuron_delta.setValue(5)
        self.spin_neuron_delta.setMaximumWidth(110)

        layout.addRow(self.chk_allow_add_layers)
        layout.addRow(self.chk_allow_remove_layers)
        layout.addRow(self.chk_allow_add_neurons)
        layout.addRow(self.chk_allow_remove_neurons)
        layout.addRow("Max +/- camadas:", self.spin_layer_delta)
        layout.addRow("Max +/- neurônios:", self.spin_neuron_delta)

        return grp

    def _build_generation_view_group(self) -> QGroupBox:
        grp = QGroupBox("Exibição (gerações)")
        layout = QFormLayout(grp)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignTop)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(2)
        grp.setContentsMargins(6, 6, 6, 6)

        self.spin_show_top = QSpinBox()
        self.spin_show_top.setRange(0, 50)
        self.spin_show_top.setValue(3)
        self.spin_show_top.setMaximumWidth(110)

        self.chk_show_original = QCheckBox("Mostrar original")
        self.chk_show_original.setChecked(True)

        self.spin_output_index = QSpinBox()
        self.spin_output_index.setRange(0, 10_000)
        self.spin_output_index.setValue(0)
        self.spin_output_index.setMaximumWidth(110)
        self.spin_output_index.setToolTip("Índice da saída para calcular min/max de probabilidade.")

        layout.addRow("Qtd redes no gráfico:", self.spin_show_top)
        layout.addRow("", self.chk_show_original)
        layout.addRow("Saída p/ min/max:", self.spin_output_index)

        # Se desativar tudo, desliga atualização de gráficos (desempenho)
        def _apply():
            self._apply_plot_enable_policy()
        self.spin_show_top.valueChanged.connect(_apply)
        self.chk_show_original.stateChanged.connect(_apply)

        return grp

    # ==================================================================
    # Suporte a train_datasets.py
    # ==================================================================
    def _reload_dataset_functions(self) -> None:
        """
        Carrega dinamicamente as funções ds_* do módulo train_datasets.py
        e preenche o combo de cenários.
        """
        if self.cmb_dataset is None:
            return

        import importlib
        import inspect

        try:
            module = importlib.import_module("train_datasets")
        except ModuleNotFoundError:
            self.cmb_dataset.clear()
            self.cmb_dataset.addItem("(train_datasets.py não encontrado)", "")
            self._dataset_funcs = {}
            self._log(
                "[DATASET] train_datasets.py não encontrado na raiz do projeto."
            )
            return
        except Exception as e:
            self.cmb_dataset.clear()
            self.cmb_dataset.addItem("(erro ao carregar train_datasets.py)", "")
            self._dataset_funcs = {}
            self._log(f"[DATASET] Erro ao importar train_datasets.py: {e}")
            return

        funcs: List[Tuple[str, Callable[[], Sequence[Sample]]]] = []
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("ds_"):
                funcs.append((name, obj))

        self._dataset_funcs = {name: f for name, f in funcs}
        self.cmb_dataset.clear()

        if not funcs:
            self.cmb_dataset.addItem("(nenhum ds_* encontrado)", "")
            self._log(
                "[DATASET] Nenhuma função ds_* encontrado em train_datasets.py."
            )
            return

        for name, func in funcs:
            doc = (func.__doc__ or "").strip().splitlines()
            desc = doc[0].strip() if doc else ""
            label = f"{name} - {desc}" if desc else name
            self.cmb_dataset.addItem(label, userData=name)

        self._log(
            "[DATASET] Cenários disponíveis: "
            + ", ".join(self._dataset_funcs.keys())
        )

    def _get_selected_dataset_func(self) -> Tuple[str, Callable[[], Sequence[Sample]]]:
        if not self._dataset_funcs or self.cmb_dataset is None:
            raise RuntimeError("Nenhum cenário ds_* carregado.")
        idx = self.cmb_dataset.currentIndex()
        if idx < 0:
            raise RuntimeError("Nenhum cenário selecionado.")
        name = self.cmb_dataset.itemData(idx)
        if not name:
            raise RuntimeError("Seleção de cenário inválida.")
        func = self._dataset_funcs.get(name)
        if func is None:
            raise RuntimeError(f"Cenário '{name}' não encontrado internamente.")
        return name, func

    # ==================================================================
    # Handlers de botões
    # ==================================================================
    def _ensure_active_network(self) -> Optional[str]:
        """
        Apenas garante que existe uma IA ativa.
        NÃO recarrega mais os parâmetros de LR do JSON aqui, para permitir
        que o usuário ajuste manualmente antes de iniciar o treino.
        """
        name = AIContext.get_name()
        if not name:
            QMessageBox.warning(
                self,
                "Nenhuma IA ativa",
                "Selecione uma IA na toolbar antes de iniciar o treinamento.",
            )
            return None
        return name

    def _on_load_data_clicked(self) -> None:
        """
        1) Sincroniza parâmetros de learning rate com a IA ativa (se houver).
        2) Carrega o cenário de dados selecionado.
        """
        ia_name = AIContext.get_name()
        if ia_name:
            self._load_lr_from_json(ia_name)
        else:
            self._log("[LR] Nenhuma IA ativa ao carregar dados; usando parâmetros atuais de LR.")

        try:
            ds_name, func = self._get_selected_dataset_func()
        except Exception as e:
            QMessageBox.warning(self, "Cenário de dados", str(e))
            self._log(f"[DATA] Erro ao selecionar cenário: {e}")
            return

        try:
            data = func()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erro ao gerar dados",
                f"Erro ao executar cenário '{ds_name}':\n{e}",
            )
            self._log(f"[DATA] Erro ao executar cenário '{ds_name}': {e}")
            return

        if not data:
            QMessageBox.warning(
                self,
                "Cenário vazio",
                f"Cenário '{ds_name}' não retornou nenhuma amostra.",
            )
            self._log(f"[DATA] Cenário '{ds_name}' retornou lista vazia.")
            self._train_data = []
            self._eval_data = []
            self._test_data = []
            return

        self._train_data = list(data)
        self._eval_data = list(data)
        self._test_data = list(data)
        self._total_train_samples = len(self._train_data)
        self._total_epochs = 0

        self._clear_histories()
        if self.progress_bar is not None:
            self.progress_bar.setValue(0)
        if self.lbl_progress is not None:
            self.lbl_progress.setText(
                f"Dados carregados: {len(self._train_data)} amostras."
            )
        if self.lbl_progress_detail is not None:
            self.lbl_progress_detail.setText(
                f"Dados processados: 0/{self._total_train_samples * max(1, self._total_epochs)} (0%)"
            )

        self._log(
            f"[DATA] Cenário '{ds_name}' gerou {len(self._train_data)} amostras "
            "(treino e avaliação)."
        )

    def _on_prepare_clicked(self) -> None:
        if not self._train_data:
            QMessageBox.information(
                self,
                "Sem dados de treino",
                "Carregue um cenário primeiro (botão 1).",
            )
            return
        self._log(
            "[INFO] Dados de treino já disponíveis. "
            "Se precisar de pré-processamento/split, implemente aqui."
        )

    def _on_train_clicked(self) -> None:
        name = self._ensure_active_network()
        if not name:
            return
        if not self._train_data:
            QMessageBox.warning(
                self,
                "Dados de treino ausentes",
                "Nenhum conjunto de treino foi definido.\n"
                "Use 'Carregar dados' primeiro.",
            )
            return
        if self._train_thread is not None:
            QMessageBox.information(
                self,
                "Treino em andamento",
                "Já existe um treinamento em execução.",
            )
            return

        epochs = self.spin_epochs.value()
        lr = self.spin_lr.value()
        lr_mult = self.spin_lr_mult.value()
        lr_min = self.spin_lr_min.value()
        shuffle = self.chk_shuffle.isChecked()
        limit = self.spin_eval_limit.value()
        eval_limit = limit if limit > 0 else None

        self._total_train_samples = len(self._train_data)
        self._total_epochs = epochs

        self._clear_histories()
        if self.progress_bar is not None:
            self.progress_bar.setValue(0)
        if self.lbl_progress is not None:
            self.lbl_progress.setText("Treinamento iniciado...")
        if self.lbl_progress_detail is not None:
            total_samples = self._total_train_samples * self._total_epochs
            self.lbl_progress_detail.setText(
                f"Dados processados: 0/{total_samples} (0%)"
            )

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Cria thread
        thread = QThread(self)

        # Modo geracional
        if self.chk_train_generations is not None and self.chk_train_generations.isChecked():
            if self.spin_generations is None or self.spin_population is None:
                self._log("[ERRO] Widgets de geração não inicializados.")
                return

            mutation_cfg = MutationConfig(
                allow_add_layers=bool(self.chk_allow_add_layers.isChecked()) if self.chk_allow_add_layers else True,
                allow_remove_layers=bool(self.chk_allow_remove_layers.isChecked()) if self.chk_allow_remove_layers else True,
                max_layer_delta=int(self.spin_layer_delta.value()) if self.spin_layer_delta else 1,
                allow_add_neurons=bool(self.chk_allow_add_neurons.isChecked()) if self.chk_allow_add_neurons else True,
                allow_remove_neurons=bool(self.chk_allow_remove_neurons.isChecked()) if self.chk_allow_remove_neurons else True,
                max_neuron_delta=int(self.spin_neuron_delta.value()) if self.spin_neuron_delta else 5,
            )

            out_idx = int(self.spin_output_index.value()) if self.spin_output_index else 0
            update_every_n = int(self.spin_data_update.value()) if self.spin_data_update else 1000
            bs = 64 if (_TORCH_AVAILABLE and torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else 16  # type: ignore[union-attr]

            worker = GenerationalWorker(
                parent_name=name,
                train_data=self._train_data,
                epochs_per_individual=epochs,
                learning_rate=lr,
                lr_mult=lr_mult,
                lr_min=lr_min,
                shuffle=shuffle,
                eval_limit=eval_limit,
                generations=int(self.spin_generations.value()),
                population=int(self.spin_population.value()),
                mutation_cfg=mutation_cfg,
                update_every_n=update_every_n,
                output_index=out_idx,
                batch_size=bs,
            )
            worker.moveToThread(thread)

            worker.log.connect(self._log)
            worker.error.connect(self._on_worker_error)
            worker.finished.connect(self._on_train_finished)
            worker.epoch_metrics.connect(self._on_epoch_metrics)
            worker.progress.connect(self._on_gen_progress)
            worker.tops_updated.connect(self._on_gen_tops)

            thread.started.connect(worker.run)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)

            self._train_thread = thread
            self._train_worker = worker
            thread.start()
            return

        # Modo normal (1 indivíduo)
        worker = TrainWorker(
            network_name=name,
            train_data=self._train_data,
            test_data=self._test_data,
            epochs=epochs,
            learning_rate=lr,
            lr_mult=lr_mult,
            lr_min=lr_min,
            eval_limit=eval_limit,
            shuffle=shuffle,
            progress_update_n=int(self.spin_data_update.value()) if self.spin_data_update else 1000,
            output_index=int(self.spin_output_index.value()) if self.spin_output_index else 0,
        )

        worker.moveToThread(thread)
        worker.log.connect(self._log)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(self._on_train_finished)
        worker.epoch_metrics.connect(self._on_epoch_metrics)
        worker.progress_epochs.connect(self._on_train_progress)
        worker.stopped.connect(self._on_train_stopped)

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._train_thread = thread
        self._train_worker = worker
        thread.start()

    def _on_evolve_clicked(self) -> None:
        """Substitui a IA original pela melhor da última geração (se a acc for maior)."""
        parent = AIContext.get_name()
        if not parent:
            self._log("[ERRO] Nenhuma IA ativa para evoluir.")
            return

        if not self._best_candidate:
            self._log("[INFO] Nenhuma melhor candidata disponível ainda. Treine gerações primeiro.")
            return

        best_name = str(self._best_candidate.get("name") or "")
        best_acc = self._best_candidate.get("acc")
        best_acc = float(best_acc) if best_acc is not None else -1.0

        # acc atual (da história, se existir)
        cur_acc = -1.0
        h = self._net_hist.get(parent, {})
        if h and h.get("acc"):
            try:
                cur_acc = float(h.get("acc")[-1])
            except Exception:
                cur_acc = -1.0

        if best_acc <= cur_acc:
            self._log(f"[INFO] Melhor candidata ({best_name}) não supera a original. acc_best={best_acc:.4f} <= acc_orig={cur_acc:.4f}")
            return

        try:
            p_folder, p_json, p_sanit = build_paths(parent)
            gen_dir = p_folder / "geracao"
            b_json = gen_dir / f"{best_name}.json"
            if not b_json.exists():
                # fallback: talvez best_name já seja um caminho/padrão diferente
                b_json = gen_dir / f"{Path(best_name).stem}.json"
            if not b_json.exists():
                raise FileNotFoundError(f"JSON da candidata não encontrado: {b_json}")

            parent_data = load_json(p_json)
            best_data = load_json(b_json)
            if not isinstance(parent_data, dict) or not isinstance(best_data, dict):
                raise ValueError("Manifesto inválido ao evoluir.")

            # substitui estrutura inteira
            parent_data["structure"] = best_data.get("structure", parent_data.get("structure"))

            # atualiza stats
            parent_data["stats"] = best_data.get("stats", parent_data.get("stats"))

            # version++ (se existir)
            ident = parent_data.get("identification") or {}
            if isinstance(ident, dict):
                try:
                    ident["version"] = int(ident.get("version") or 0) + 1
                except Exception:
                    ident["version"] = 1
                parent_data["identification"] = ident

            save_json(p_json, parent_data)

            self._log(f"[OK] Evolução aplicada: original '{parent}' substituída por '{best_name}' (acc {best_acc:.4f} > {cur_acc:.4f}).")
            # força re-avaliação e atualização dos gráficos
            self._on_eval_clicked()

        except Exception as e:
            self._log(f"[ERRO] Falha ao evoluir: {e}")

    def _on_gen_progress(self, name: str, processed: int, total: int, loss: float) -> None:
        # Atualização rápida (intra-época) baseada no campo "Qnt. de dados"
        if self.lbl_progress is not None:
            self.lbl_progress.setText(f"[GEN] Treinando: {name}")
        if self.lbl_progress_detail is not None:
            perc = (processed / total) * 100 if total > 0 else 0.0
            self.lbl_progress_detail.setText(f"[GEN] Dados: {processed}/{total} ({perc:.1f}%) | loss={loss:.4g}")

    def _on_gen_tops(self, tops: list) -> None:
        # tops: [{name, acc, loss, neurons_total}, ...]
        self._last_generation_tops = list(tops or [])
        self._best_candidate = self._last_generation_tops[0] if self._last_generation_tops else None

        # Atualiza o rótulo principal do progresso com o Top 3
        if self.lbl_progress is not None and self._last_generation_tops:
            parts = []
            for t in self._last_generation_tops[:3]:
                nm = t.get("name")
                acc = t.get("acc", 0.0)
                loss = t.get("loss", 0.0)
                neu = t.get("neurons_total", 0)
                parts.append(f"{nm} | acc={acc:.4f} loss={loss:.4g} neu={neu}")
            self.lbl_progress.setText("Top geração: " + " || ".join(parts))

        # Atualiza gráficos imediatamente (se habilitado)
        self._update_plots()

    def _on_stop_clicked(self) -> None:
        if self._train_worker is not None:
            self._train_worker.request_stop()
            self._log("[INFO] Solicitação de parada enviada ao worker de treino.")

    def _on_eval_clicked(self) -> None:
        name = self._ensure_active_network()
        if not name:
            return
        if not self._eval_data:
            QMessageBox.warning(
                self,
                "Dados de avaliação ausentes",
                "Nenhum conjunto de avaliação foi definido.",
            )
            return
        if self._eval_thread is not None:
            QMessageBox.information(
                self,
                "Avaliação em andamento",
                "Já existe uma avaliação em execução.",
            )
            return

        limit = self.spin_eval_limit.value()
        eval_limit = limit if limit > 0 else None

        worker = EvalWorker(
            network_name=name,
            data=self._eval_data,
            limit=eval_limit,
        )
        thread = QThread(self)
        worker.moveToThread(thread)

        worker.log.connect(self._log)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(self._on_eval_finished)

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._eval_thread = thread
        self._eval_worker = worker
        thread.start()

    # ==================================================================
    # Callbacks dos workers
    # ==================================================================
    def _on_train_progress(self, current_epoch: int, total_epochs: int) -> None:
        if total_epochs <= 0 or self.progress_bar is None or self.lbl_progress is None:
            return
        perc = int(current_epoch * 100 / total_epochs)
        perc = max(0, min(100, perc))
        self.progress_bar.setValue(perc)
        self.lbl_progress.setText(
            f"Progresso (épocas): {current_epoch}/{total_epochs}"
        )

    def _on_epoch_metrics(self, metrics: EpochMetrics) -> None:
        name = metrics.network_name

        epoch_num = int(metrics.epoch_index) + 1

        x_list = self._net_epoch_x.setdefault(name, [])
        x_list.append(epoch_num)

        hist = self._net_hist.setdefault(
            name,
            {
                "loss_final": [],
                "loss_avg": [],
                "lr": [],
                "perf": [],
                "acc": [],
                "out_min": [],
                "out_max": [],
                "class_counts_last": [],
                "class_hits_last": [],
            },
        )

        hist["loss_final"].append(float(metrics.loss_final))
        hist["loss_avg"].append(float(metrics.loss_avg))
        hist["lr"].append(float(metrics.learning_rate))
        hist["perf"].append(float(metrics.samples_per_sec))
        hist["acc"].append(float(metrics.accuracy))
        hist["out_min"].append(float(metrics.out_min))
        hist["out_max"].append(float(metrics.out_max))
        hist["class_counts_last"] = list(metrics.class_counts or [])
        hist["class_hits_last"] = list(metrics.class_hits or [])

        # Atualiza info de dados processados (mantém lógica atual: baseado no treino da rede original)
        if self.lbl_progress_detail is not None and self._total_train_samples > 0 and name == AIContext.get_name():
            processed_epochs = epoch_num
            total_epochs = metrics.total_epochs or self._total_epochs
            total_epochs = max(1, int(total_epochs))
            total_samples = self._total_train_samples * total_epochs
            processed_samples = self._total_train_samples * processed_epochs
            processed_samples = min(processed_samples, total_samples)
            perc = (processed_samples / total_samples) * 100 if total_samples > 0 else 0.0
            self.lbl_progress_detail.setText(
                f"Dados processados: {processed_samples}/{total_samples} ({perc:.1f}%)"
            )

        # Atualiza métricas atuais (para a rede original)
        if self.lbl_metrics is not None and name == AIContext.get_name():
            self.lbl_metrics.setText(
                f"Acurácia: {metrics.accuracy:.4f}\n"
                f"Loss: {metrics.loss_final:.4g} (média: {metrics.loss_avg:.4g})\n"
                f"Perf: {metrics.samples_per_sec:.1f} samp/s\n"
                f"LR: {metrics.learning_rate:g}\n"
                f"Neurônios: {metrics.neurons_total}"
            )

        # Atualiza o campo de Learning Rate na GUI a cada época (somente original)
        if name == AIContext.get_name():
            try:
                self.spin_lr.blockSignals(True)
                self.spin_lr.setValue(metrics.learning_rate)
            finally:
                self.spin_lr.blockSignals(False)

        self._maybe_update_plots(metrics)

    def _on_train_finished(self) -> None:
        self._log("[TRAIN] Treinamento concluído.")
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._train_thread = None
        self._train_worker = None

        # Recarrega LR a partir do JSON (structure) para refletir o valor final
        name = AIContext.get_name()
        if name:
            self._load_lr_from_json(name)

    def _on_train_stopped(self, msg: str) -> None:
        self._log(msg)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._train_thread = None
        self._train_worker = None

    def _on_eval_finished(self, results: List[EvalResult]) -> None:
        if not results:
            self._log("[EVAL] Nenhum resultado retornado.")
            return
        er = results[0]
        avg_loss = er.avg_loss if er.avg_loss is not None else float("nan")
        avg_time_ms = (er.avg_time_per_sample * 1000.0) if er.avg_time_per_sample is not None else float("nan")
        self._log(
            f"[EVAL] Rede '{er.name}': "
            f"acc={er.accuracy:.4f}, "
            f"loss_médio={avg_loss:.6g}, "
            f"tempo_médio/amostra={avg_time_ms:.3f} ms"
        )

    def _on_worker_error(self, msg: str) -> None:
        self._log(f"[ERRO] {msg}")
        QMessageBox.critical(self, "Erro na execução", msg)

    # ==================================================================
    # Gráficos
    # ==================================================================
    def _clear_histories(self) -> None:
        self._net_hist.clear()
        self._net_epoch_x.clear()

        # Limpa curvas existentes
        for d in (self._curves_loss_epoch, self._curves_lr, self._curves_perf, self._curves_acc):
            for _, item in list(d.items()):
                try:
                    item.setData([], [])
                except Exception:
                    pass
            d.clear()

        # Limpa distribuição
        for _, item in list(self._dist_items.items()):
            try:
                self.plot_dist.removeItem(item)
            except Exception:
                pass
        self._dist_items.clear()

        if self.lbl_progress is not None:
            self.lbl_progress.setText("Aguardando início do treino...")
        if self.lbl_progress_detail is not None:
            self.lbl_progress_detail.setText("Dados processados: 0/0 (0%)")
        if self.lbl_metrics is not None:
            self.lbl_metrics.setText(
                "Acurácia: --\nLoss: --\nPerf: --\nLR: --\nNeurônios: --"
            )
        if self.progress_bar is not None:
            self.progress_bar.setValue(0)

        self._update_plots()

    def _moving_avg(self, values: List[float], window: int) -> List[float]:
        if not values or window <= 1:
            return values[:]
        ma: List[float] = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            chunk = values[start : i + 1]
            ma.append(sum(chunk) / len(chunk))
        return ma

    def _maybe_update_plots(self, metrics: EpochMetrics) -> None:
        """
        Controla a frequência de atualização dos gráficos em tempo real.
        """
        mode = "visual"
        if hasattr(self, "cmb_update_mode") and self.cmb_update_mode is not None:
            data = self.cmb_update_mode.currentData()
            if isinstance(data, str):
                mode = data

        epoch_num = metrics.epoch_index + 1
        total_epochs = metrics.total_epochs or self._total_epochs or 1

        # Modo performance: atualiza poucas vezes (a cada ~10% + final)
        if mode == "perf":
            step = max(1, total_epochs // 10)
            if epoch_num == total_epochs or epoch_num % step == 0 or epoch_num == 1:
                self._update_plots()
            return

        # Modo visual: tenta respeitar "N dados" aproximando por épocas
        update_n = 1
        if hasattr(self, "spin_update_samples") and self.spin_update_samples is not None:
            update_n = max(1, self.spin_update_samples.value())

        total_samples = max(1, self._total_train_samples)
        epochs_step = max(1, math.ceil(update_n / total_samples))

        if epoch_num == 1 or epoch_num == total_epochs or epoch_num % epochs_step == 0:
            self._update_plots()

    def _apply_plot_enable_policy(self) -> None:
        """Se qtd redes = 0 e original desmarcado, desliga atualização."""
        if self.spin_show_top is None or self.chk_show_original is None:
            return
        show_top = int(self.spin_show_top.value())
        show_orig = bool(self.chk_show_original.isChecked())
        self._plots_enabled = (show_top > 0) or show_orig
        # Quando desliga, limpa o plot min/max para evitar custo de redraw
        if not self._plots_enabled:
            self._update_plots(force_clear=True)

    def _get_display_networks(self) -> List[str]:
        """Lista de redes a serem exibidas: original (opcional) + Top-N da última geração."""
        out: List[str] = []
        if self.chk_show_original is None or self.spin_show_top is None:
            return out

        if self.chk_show_original.isChecked():
            base = AIContext.get_name()
            if base:
                out.append(base)

        top_n = int(self.spin_show_top.value())
        if top_n > 0:
            for item in (self._last_generation_tops or [])[:top_n]:
                nm = str(item.get("name") or "")
                if nm and nm not in out:
                    out.append(nm)
        return out

    def _palette(self) -> List[Tuple[int, int, int]]:
        # Paleta simples e estável
        return [
            (0, 170, 255),
            (255, 170, 0),
            (0, 220, 120),
            (220, 0, 220),
            (255, 80, 80),
            (200, 200, 60),
            (160, 160, 255),
            (120, 255, 255),
        ]

    def _ensure_curve(self, cache: Dict[str, pg.PlotDataItem], plot: pg.PlotWidget, name: str, color: Tuple[int, int, int]) -> pg.PlotDataItem:
        item = cache.get(name)
        if item is None:
            item = plot.plot(pen=pg.mkPen(color=color, width=2))
            cache[name] = item
        return item

    def _update_plots(self, force_clear: bool = False) -> None:
        # Política de desempenho
        if force_clear:
            # min/max
            if self._minmax_bar is not None:
                self._minmax_bar.setOpts(x=[], y0=[], height=[])
            if self._minmax_scatter_min is not None:
                self._minmax_scatter_min.setData(x=[], y=[])
            if self._minmax_scatter_max is not None:
                self._minmax_scatter_max.setData(x=[], y=[])
            # curvas
            for d in (self._curves_loss_epoch, self._curves_lr, self._curves_perf, self._curves_acc):
                for item in d.values():
                    item.setData([], [])
            # dist
            self._dist_item.setOpts(x=[], height=[])
            return

        if not getattr(self, "_plots_enabled", True):
            return

        display = self._get_display_networks()
        if not display:
            # nada a exibir
            self._update_plots(force_clear=True)
            return

        palette = self._palette()

        # ----------------- Curvas (loss/lr/perf/acc) -----------------
        for i, nm in enumerate(display):
            x = self._net_epoch_x.get(nm, [])
            hist = self._net_hist.get(nm, {})
            color = palette[i % len(palette)]

            # Loss médio por época
            if hasattr(self, "chk_plot_loss_epoch") and self.chk_plot_loss_epoch.isChecked() and x:
                y = hist.get("loss_avg", [])
                self._ensure_curve(self._curves_loss_epoch, self.plot_loss_epoch, nm, color).setData(x, y)
            else:
                if nm in self._curves_loss_epoch:
                    self._curves_loss_epoch[nm].setData([], [])

            # LR
            if hasattr(self, "chk_plot_lr") and self.chk_plot_lr.isChecked() and x:
                y = hist.get("lr", [])
                self._ensure_curve(self._curves_lr, self.plot_lr, nm, color).setData(x, y)
            else:
                if nm in self._curves_lr:
                    self._curves_lr[nm].setData([], [])

            # Perf
            if hasattr(self, "chk_plot_perf") and self.chk_plot_perf.isChecked() and x:
                y = hist.get("perf", [])
                self._ensure_curve(self._curves_perf, self.plot_perf, nm, color).setData(x, y)
            else:
                if nm in self._curves_perf:
                    self._curves_perf[nm].setData([], [])

            # Acc
            if hasattr(self, "chk_plot_acc") and self.chk_plot_acc.isChecked() and x:
                y = hist.get("acc", [])
                self._ensure_curve(self._curves_acc, self.plot_acc, nm, color).setData(x, y)
            else:
                if nm in self._curves_acc:
                    self._curves_acc[nm].setData([], [])

        # ----------------- Min/Max (último ponto) -----------------
        if hasattr(self, "chk_plot_minmax") and self.chk_plot_minmax.isChecked():
            xs = list(range(len(display)))
            centers = []
            tops = []
            bottoms = []
            mins = []
            maxs = []
            for nm in display:
                hist = self._net_hist.get(nm, {})
                mn_list = hist.get("out_min", [])
                mx_list = hist.get("out_max", [])
                mn = float(mn_list[-1]) if mn_list else 0.0
                mx = float(mx_list[-1]) if mx_list else 0.0
                mins.append(mn)
                maxs.append(mx)
                c = (mn + mx) / 2.0
                centers.append(c)
                tops.append(max(0.0, mx - c))
                bottoms.append(max(0.0, c - mn))

            if self._minmax_bar is not None:
                heights = [max(0.0, float(ma) - float(mi)) for mi, ma in zip(mins, maxs)]
                self._minmax_bar.setOpts(x=xs, y0=mins, height=heights)
            if self._minmax_scatter_min is not None:
                self._minmax_scatter_min.setData(x=xs, y=mins)
            if self._minmax_scatter_max is not None:
                self._minmax_scatter_max.setData(x=xs, y=maxs)

            # ticks com nomes
            ax = self.plot_minmax.getAxis("bottom")
            ax.setTicks([[(i, str(nm)) for i, nm in enumerate(display)]])
        else:
            if self._minmax_bar is not None:
                self._minmax_bar.setOpts(x=[], y0=[], height=[])
            if self._minmax_scatter_min is not None:
                self._minmax_scatter_min.setData(x=[], y=[])
            if self._minmax_scatter_max is not None:
                self._minmax_scatter_max.setData(x=[], y=[])

        # ----------------- Distribuição (somente 1 rede por performance) -----------------
        # Para manter o custo baixo, mostramos a distribuição da melhor rede exibida (primeira da lista que tiver dados).
        if hasattr(self, "chk_plot_dist") and self.chk_plot_dist.isChecked():
            chosen = None
            for nm in display:
                hist = self._net_hist.get(nm, {})
                cc = hist.get("class_counts_last", [])
                if cc:
                    chosen = (nm, cc)
                    break
            if chosen:
                nm, cc = chosen
                xs = list(range(len(cc)))
                self._dist_item.setOpts(x=xs, height=cc, width=0.6)
            else:
                self._dist_item.setOpts(x=[], height=[])
        else:
            self._dist_item.setOpts(x=[], height=[])


    # ==================================================================
    # Log helper
    # ==================================================================
    def _log(self, msg: str) -> None:
        if self.txt_log is not None:
            self.txt_log.appendPlainText(str(msg))
            self.txt_log.ensureCursorVisible()
        else:
            print(str(msg))

    # ==================================================================
    # API pública: sincronização com IAContext
    # ==================================================================
    def refresh_from_context(self) -> None:
        """
        Chamado pelo MainWindow quando o contexto de IA muda.
        Aqui aproveitamos para carregar learning rate, mínimo e multiplicador
        do JSON da rede e aplicar nos campos.
        """
        name = AIContext.get_name()
        if not name:
            return
        self._load_lr_from_json(name)

    def _load_lr_from_json(self, name: str) -> None:
        """
        Lê do JSON da rede (manifesto) os campos dentro de 'structure':
          - learning_rate
          - learning_rate_min
          - learning_rate_mult

        Se não encontrar ou der erro, usa os defaults.
        """
        # Começa SEMPRE com os defaults
        self.spin_lr.setValue(DEFAULT_LR)
        self.spin_lr_mult.setValue(DEFAULT_LR_MULT)
        self.spin_lr_min.setValue(DEFAULT_LR_MIN)

        try:
            _, arquivo, _ = build_paths(name)
            if not arquivo.exists():
                self._log(f"[LR] Manifesto '{name}' não encontrado para ler LR.")
                return
            data = load_json(arquivo)
        except Exception as e:
            self._log(f"[LR] Não foi possível ler JSON da rede '{name}': {e}")
            return

        if not isinstance(data, dict):
            return

        struct = data.get("structure") or {}
        if not isinstance(struct, dict):
            return

        lr = struct.get("learning_rate")
        lr_min = struct.get("learning_rate_min")
        lr_mult = struct.get("learning_rate_mult")

        if isinstance(lr, (int, float)) and lr > 0:
            self.spin_lr.setValue(float(lr))
        if isinstance(lr_min, (int, float)) and lr_min >= 0:
            self.spin_lr_min.setValue(float(lr_min))
        if isinstance(lr_mult, (int, float)) and lr_mult > 0:
            self.spin_lr_mult.setValue(float(lr_mult))
