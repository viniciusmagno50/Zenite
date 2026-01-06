from __future__ import annotations

from typing import List, Sequence, Tuple, Optional, Dict, Callable, Any

from PySide6.QtCore import Qt, QThread
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

from ia_context import AIContext
from ia_training import Sample
from json_lib import build_paths, load_json
from pages.training.config import load_training_defaults
from pages.training import adapters as tr_adapters
from ia_generational import MutationConfig
from pages.training.workers import EpochMetrics, TrainWorker, GenerationalWorker, EvalWorker


_TD = load_training_defaults()
DEFAULT_EPOCHS = _TD.epochs_default
DEFAULT_LR = _TD.lr_default
DEFAULT_LR_MULT = _TD.lr_mult_default
DEFAULT_LR_MIN = _TD.lr_min_default
DEFAULT_DATA_UPDATE = _TD.data_update_default
DEFAULT_EVAL_LIMIT = _TD.eval_limit_default


class WdTrainingPane(QWidget):
    UPDATE_EPOCH = 0
    UPDATE_BY_DATA = 1

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # >>> garante tema legível do pyqtgraph (sem depender do QSS)
        pg.setConfigOptions(antialias=True, foreground=(230, 230, 230))

        self._train_thread: Optional[QThread] = None
        self._train_worker: Optional[TrainWorker] = None
        self._eval_thread: Optional[QThread] = None
        self._eval_worker: Optional[EvalWorker] = None
        self._gen_thread: Optional[QThread] = None
        self._gen_worker: Optional[GenerationalWorker] = None

        self._train_data: Optional[List[Sample]] = None
        self._test_data: Optional[List[Sample]] = None

        self._dataset_funcs: Dict[str, Callable[[], Sequence[Sample]]] = {}
        self.cmb_dataset: Optional[QComboBox] = None

        self._last_generation_tops: List[Dict[str, Any]] = []

        self._net_hist: Dict[str, Dict[str, Any]] = {}
        self._net_epoch_x: Dict[str, List[float]] = {}

        self._curves_loss_epoch: Dict[str, pg.PlotDataItem] = {}
        self._curves_lr: Dict[str, pg.PlotDataItem] = {}
        self._curves_perf: Dict[str, pg.PlotDataItem] = {}
        self._curves_acc: Dict[str, pg.PlotDataItem] = {}

        self._dist_item: Optional[pg.BarGraphItem] = None
        self._hist_item: Optional[pg.BarGraphItem] = None

        self.cmb_plot_update: Optional[QComboBox] = None

        # >>> por dados (loss)
        self._live_loss_x: Dict[str, List[float]] = {}
        self._live_loss_y_raw: Dict[str, List[float]] = {}
        self._live_loss_y_ema: Dict[str, List[float]] = {}
        self._live_loss_ema_last: Dict[str, float] = {}

        # brushes/pens visíveis
        self._pen_curve = pg.mkPen(color=(230, 230, 230), width=2)
        self._pen_bar = pg.mkPen(color=(230, 230, 230), width=1)
        self._brush_bar = pg.mkBrush(230, 230, 230)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(2)

        grp_flow = self._build_flow_group()
        grp_params = self._build_params_group()
        grp_graph_opts = self._build_graph_options_group()
        grp_gen_cfg = self._build_generation_config_group()
        grp_mut_cfg = self._build_mutation_config_group()
        grp_gen_view = self._build_generation_view_group()

        for grp in (grp_flow, grp_params, grp_graph_opts, grp_gen_cfg, grp_mut_cfg, grp_gen_view):
            grp.setMinimumWidth(180)
            grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            row1.addWidget(grp, 1)

        root.addLayout(row1)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(2)
        self.splitter.setChildrenCollapsible(False)

        row2 = QHBoxLayout()
        row2.setSpacing(2)

        self.plot_minmax = pg.PlotWidget(title="Histograma (softmax)")
        self.plot_minmax.showGrid(x=True, y=True)
        # >>> IMPORTANTE: autoRange X e Y (senão as barras ficam fora)
        self.plot_minmax.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot_minmax.setLabel("left", "Bin (0..99)")
        self.plot_minmax.setLabel("bottom", "Contagem")

        self._hist_item = pg.BarGraphItem(
            x0=[], x1=[], y0=[], y1=[],
            pen=self._pen_bar,
            brush=self._brush_bar
        )
        self.plot_minmax.addItem(self._hist_item)

        self.plot_loss_epoch = pg.PlotWidget(title="Loss médio por época")
        self.plot_loss_epoch.showGrid(x=True, y=True)
        self.plot_loss_epoch.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        self.plot_lr = pg.PlotWidget(title="Learning rate")
        self.plot_lr.showGrid(x=True, y=True)
        self.plot_lr.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        self.plot_perf = pg.PlotWidget(title="Performance (samples/s)")
        self.plot_perf.showGrid(x=True, y=True)
        self.plot_perf.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        for w in (self.plot_minmax, self.plot_loss_epoch, self.plot_lr, self.plot_perf):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            row2.addWidget(w, 1)

        row2_container = QWidget()
        row2_container.setLayout(row2)
        row2_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        row2_container.setMinimumHeight(0)

        row3 = QHBoxLayout()
        row3.setSpacing(2)

        self.plot_dist = pg.PlotWidget(title="Distribuição de saídas (predições)")
        self.plot_dist.showGrid(x=True, y=True)
        # >>> autoRange X e Y para barras
        self.plot_dist.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

        self._dist_item = pg.BarGraphItem(
            x=[], height=[], width=1.0,
            pen=self._pen_bar,
            brush=self._brush_bar
        )
        self.plot_dist.addItem(self._dist_item)

        self.plot_acc = pg.PlotWidget(title="Acurácia por época")
        self.plot_acc.showGrid(x=True, y=True)
        self.plot_acc.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        prog_group = QGroupBox("Progresso de treinamento")
        prog_layout = QVBoxLayout(prog_group)
        prog_layout.setSpacing(4)
        prog_layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_progress = QLabel("Aguardando início do treino...")
        self.lbl_progress.setWordWrap(True)

        self.lbl_progress_detail = QLabel("Dados processados: 0/0 (0%)")
        self.lbl_progress_detail.setWordWrap(True)

        self.lbl_metrics = QLabel("Acurácia: --\nLoss: --\nPerf: --\nLR: --\nNeurônios: --")
        self.lbl_metrics.setWordWrap(True)

        # >>> garante texto visível mesmo com QSS agressivo
        self.lbl_progress.setStyleSheet("color: #E6E6E6;")
        self.lbl_progress_detail.setStyleSheet("color: #E6E6E6;")
        self.lbl_metrics.setStyleSheet("color: #E6E6E6;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(16)

        prog_layout.addWidget(self.lbl_progress)
        prog_layout.addWidget(self.lbl_progress_detail)
        prog_layout.addWidget(self.lbl_metrics)
        prog_layout.addWidget(self.progress_bar)

        log_group = QGroupBox("Log de treinamento")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 6, 6, 6)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setLineWrapMode(QPlainTextEdit.NoWrap)
        log_layout.addWidget(self.txt_log)

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

        self._reload_dataset_functions()
        self._clear_histories()

    # ---------------- UI groups ----------------

    def _build_flow_group(self) -> QGroupBox:
        grp = QGroupBox("Fluxo de dados e treinamento")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)

        lbl_ds = QLabel("Dataset:")
        self.cmb_dataset = QComboBox()
        self.cmb_dataset.setMinimumWidth(170)
        self.cmb_dataset.setFixedHeight(24)

        layout.addWidget(lbl_ds)
        layout.addWidget(self.cmb_dataset)

        self.btn_load_data = QPushButton("1) Carregar dados")
        self.btn_prepare = QPushButton("2) Preparar treino")
        self.btn_train = QPushButton("3) Treinar rede")
        self.btn_eval = QPushButton("4) Avaliar rede")
        self.btn_stop = QPushButton("Parar")

        for btn in [self.btn_load_data, self.btn_prepare, self.btn_train, self.btn_eval, self.btn_stop]:
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
        layout.setContentsMargins(6, 6, 6, 8)

        def compact_spinbox(spin: QSpinBox | QDoubleSpinBox, is_double: bool = False):
            spin.setMaximumWidth(110)
            if is_double:
                spin.setStyleSheet("QDoubleSpinBox { padding: 0 3px; }")
            else:
                spin.setStyleSheet("QSpinBox { padding: 0 3px; }")

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(DEFAULT_EPOCHS)
        compact_spinbox(self.spin_epochs)

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
        compact_spinbox(self.spin_lr_mult, is_double=True)

        self.spin_lr_min = QDoubleSpinBox()
        self.spin_lr_min.setDecimals(7)
        self.spin_lr_min.setRange(0.0, 1.0)
        self.spin_lr_min.setSingleStep(1e-6)
        self.spin_lr_min.setValue(DEFAULT_LR_MIN)
        compact_spinbox(self.spin_lr_min, is_double=True)

        self.spin_data_update = QSpinBox()
        self.spin_data_update.setRange(1, 10_000_000)
        self.spin_data_update.setValue(DEFAULT_DATA_UPDATE)
        compact_spinbox(self.spin_data_update)

        self.spin_eval_limit = QSpinBox()
        self.spin_eval_limit.setRange(0, 1_000_000)
        self.spin_eval_limit.setValue(DEFAULT_EVAL_LIMIT)
        compact_spinbox(self.spin_eval_limit)

        layout.addRow("Épocas:", self.spin_epochs)
        layout.addRow("Learning rate:", self.spin_lr)
        layout.addRow("Fator LR/época:", self.spin_lr_mult)
        layout.addRow("LR mínimo:", self.spin_lr_min)
        layout.addRow("Qnt. de dados:", self.spin_data_update)
        layout.addRow("Limite p/ avaliação:", self.spin_eval_limit)

        self.chk_shuffle = QCheckBox("Embaralhar dados a cada época")
        self.chk_shuffle.setChecked(False)
        layout.addRow(self.chk_shuffle)

        return grp

    def _build_graph_options_group(self) -> QGroupBox:
        grp = QGroupBox("Gráficos em tempo real")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)

        self.chk_plot_minmax = QCheckBox("Min/Max / Histograma")
        self.chk_plot_minmax.setChecked(True)

        self.chk_plot_loss_epoch = QCheckBox("Loss médio por época")
        self.chk_plot_loss_epoch.setChecked(True)

        self.chk_plot_lr = QCheckBox("Learning rate")
        self.chk_plot_lr.setChecked(True)

        self.chk_plot_perf = QCheckBox("Performance (samples/s)")
        self.chk_plot_perf.setChecked(True)

        self.chk_plot_dist = QCheckBox("Distribuição de saídas")
        self.chk_plot_dist.setChecked(True)

        self.chk_plot_acc = QCheckBox("Acurácia por época")
        self.chk_plot_acc.setChecked(True)

        for chk in [
            self.chk_plot_minmax,
            self.chk_plot_loss_epoch,
            self.chk_plot_lr,
            self.chk_plot_perf,
            self.chk_plot_dist,
            self.chk_plot_acc,
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

        form = QFormLayout()
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(2)

        self.spin_generations = QSpinBox()
        self.spin_generations.setRange(1, 10000)
        self.spin_generations.setValue(3)
        self.spin_generations.setMaximumWidth(110)

        self.spin_population = QSpinBox()
        self.spin_population.setRange(1, 10000)
        self.spin_population.setValue(10)
        self.spin_population.setMaximumWidth(110)

        form.addRow("Gerações:", self.spin_generations)
        form.addRow("População:", self.spin_population)

        layout.addWidget(self.chk_train_generations)
        layout.addLayout(form)
        layout.addStretch(1)
        return grp

    def _build_mutation_config_group(self) -> QGroupBox:
        grp = QGroupBox("Configuração de mutação")
        layout = QFormLayout(grp)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignTop)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(2)
        grp.setContentsMargins(6, 6, 6, 6)

        self.chk_allow_add_layers = QCheckBox("Permitir adicionar camadas")
        self.chk_allow_remove_layers = QCheckBox("Permitir remover camadas")
        self.chk_allow_add_neurons = QCheckBox("Permitir adicionar neurônios")
        self.chk_allow_remove_neurons = QCheckBox("Permitir remover neurônios")

        for chk in (
            self.chk_allow_add_layers,
            self.chk_allow_remove_layers,
            self.chk_allow_add_neurons,
            self.chk_allow_remove_neurons,
        ):
            chk.setChecked(True)

        self.spin_layer_delta = QSpinBox()
        self.spin_layer_delta.setRange(0, 20)
        self.spin_layer_delta.setValue(1)
        self.spin_layer_delta.setMaximumWidth(110)

        self.spin_neuron_delta = QSpinBox()
        self.spin_neuron_delta.setRange(0, 1000)
        self.spin_neuron_delta.setValue(5)
        self.spin_neuron_delta.setMaximumWidth(110)

        layout.addRow(self.chk_allow_add_layers)
        layout.addRow(self.chk_allow_remove_layers)
        layout.addRow(self.chk_allow_add_neurons)
        layout.addRow(self.chk_allow_remove_neurons)
        layout.addRow("Δ camadas:", self.spin_layer_delta)
        layout.addRow("Δ neurônios:", self.spin_neuron_delta)

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
        self.spin_output_index.setRange(0, 10000)
        self.spin_output_index.setValue(0)
        self.spin_output_index.setMaximumWidth(110)

        self.cmb_plot_update = QComboBox()
        self.cmb_plot_update.addItem("A cada época", self.UPDATE_EPOCH)
        self.cmb_plot_update.addItem("A cada Qnt. de dados", self.UPDATE_BY_DATA)
        self.cmb_plot_update.setCurrentIndex(0)
        self.cmb_plot_update.setMaximumWidth(160)

        layout.addRow("Qtd redes no gráfico:", self.spin_show_top)
        layout.addRow("", self.chk_show_original)
        layout.addRow("Saída p/ min/max:", self.spin_output_index)
        layout.addRow("Atualizar gráficos:", self.cmb_plot_update)

        return grp

    # ---------------- datasets ----------------

    def _reload_dataset_functions(self) -> None:
        if self.cmb_dataset is None:
            return
        funcs = tr_adapters.get_dataset_functions(module_name="train_datasets", log=self._log)
        self._dataset_funcs = funcs

        self.cmb_dataset.clear()
        if not funcs:
            self.cmb_dataset.addItem("(nenhum dataset encontrado)", "")
            return
        for name in sorted(funcs.keys()):
            self.cmb_dataset.addItem(name, name)

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

    # ---------------- actions ----------------

    def _ensure_active_network(self) -> Optional[str]:
        name = AIContext.get_name()
        if not name:
            QMessageBox.warning(self, "IA não selecionada", "Selecione ou crie uma IA antes de continuar.")
            return None
        return str(name)

    def _load_lr_from_json(self, ia_name: str) -> None:
        try:
            _, manifest_path, _ = build_paths(ia_name)
            manifest = load_json(manifest_path) or {}
            trn = manifest.get("training", {}) or {}
            lr = trn.get("learning_rate", None)
            lr_mult = trn.get("lr_mult", None)
            lr_min = trn.get("lr_min", None)

            if lr is not None:
                self.spin_lr.blockSignals(True)
                self.spin_lr.setValue(float(lr))
                self.spin_lr.blockSignals(False)

            if lr_mult is not None:
                self.spin_lr_mult.blockSignals(True)
                self.spin_lr_mult.setValue(float(lr_mult))
                self.spin_lr_mult.blockSignals(False)

            if lr_min is not None:
                self.spin_lr_min.blockSignals(True)
                self.spin_lr_min.setValue(float(lr_min))
                self.spin_lr_min.blockSignals(False)

            self._log("[LR] Parâmetros carregados do JSON.")
        except Exception as e:
            self._log(f"[LR] Falha ao carregar LR do JSON: {e}")

    def _on_load_data_clicked(self) -> None:
        ia_name = AIContext.get_name()
        if ia_name:
            self._load_lr_from_json(ia_name)

        try:
            ds_name, func = self._get_selected_dataset_func()
            data = list(func())
        except Exception as e:
            QMessageBox.critical(self, "Erro ao carregar dados", str(e))
            self._log(f"[DATA] Erro: {e}")
            return

        if not data:
            QMessageBox.warning(self, "Cenário vazio", f"Cenário '{ds_name}' não retornou nenhuma amostra.")
            self._log(f"[DATA] Cenário '{ds_name}' vazio.")
            return

        self._train_data = data
        self._test_data = list(data)
        self._log(f"[OK] Dataset '{ds_name}' carregado: {len(data)} amostras.")

    def _on_prepare_clicked(self) -> None:
        if not self._train_data:
            QMessageBox.warning(self, "Treino", "Carregue os dados primeiro (1).")
            return
        self._clear_histories()
        self._log("[OK] Treino preparado.")

    def _plot_update_mode(self) -> int:
        if self.cmb_plot_update is None:
            return self.UPDATE_EPOCH
        try:
            return int(self.cmb_plot_update.currentData())
        except Exception:
            return self.UPDATE_EPOCH

    def _lock_epoch_x_axis(self, epochs_total: int) -> None:
        for w in (self.plot_loss_epoch, self.plot_lr, self.plot_perf, self.plot_acc):
            try:
                w.enableAutoRange(x=False, y=True)
                w.setXRange(1, max(1, int(epochs_total)), padding=0.0)
            except Exception:
                pass

    def _on_train_clicked(self) -> None:
        name = self._ensure_active_network()
        if not name:
            return
        if not self._train_data:
            QMessageBox.warning(self, "Treino", "Carregue os dados primeiro (1).")
            return

        self._clear_histories()

        epochs = int(self.spin_epochs.value())
        lr = float(self.spin_lr.value())
        lr_mult = float(self.spin_lr_mult.value())
        lr_min = float(self.spin_lr_min.value())
        shuffle = bool(self.chk_shuffle.isChecked())
        update_every = int(self.spin_data_update.value())
        eval_limit = int(self.spin_eval_limit.value())
        output_index = int(self.spin_output_index.value())

        self._lock_epoch_x_axis(epochs)

        # reset séries
        self._live_loss_x[name] = []
        self._live_loss_y_raw[name] = []
        self._live_loss_y_ema[name] = []
        self._live_loss_ema_last.pop(name, None)

        self._log(f"[TRAIN] Iniciando treino: {name} | epochs={epochs}, lr={lr}")

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)

        worker = TrainWorker(
            network_name=name,
            train_data=self._train_data,
            test_data=self._test_data if self._test_data else self._train_data,
            epochs=epochs,
            learning_rate=lr,
            lr_mult=lr_mult,
            lr_min=lr_min,
            eval_limit=eval_limit,
            shuffle=shuffle,
            progress_update_n=update_every,
            output_index=output_index,
            batch_size=16,
        )

        thread = QThread(self)
        worker.moveToThread(thread)

        worker.log.connect(self._log)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(self._on_train_finished)
        worker.epoch_metrics.connect(self._on_epoch_metrics)
        worker.progress_epochs.connect(self._on_train_progress)
        worker.stopped.connect(self._on_train_stopped)
        worker.batch_progress.connect(self._on_batch_progress)

        thread.started.connect(worker.run)

        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._train_thread = thread
        self._train_worker = worker
        thread.start()

    def _on_stop_clicked(self) -> None:
        if self._train_worker is not None:
            self._train_worker.stop()
            self._log("[INFO] Solicitação de parada enviada.")

    def _on_eval_clicked(self) -> None:
        self._log("[INFO] Avaliação manual não está conectada neste arquivo (use o fluxo de treino).")

    # ---------------- updates ----------------

    def _on_train_progress(self, current_epoch: int, total_epochs: int) -> None:
        if total_epochs <= 0:
            return
        perc = int(current_epoch * 100 / total_epochs)
        self.progress_bar.setValue(max(0, min(100, perc)))
        self.lbl_progress.setText(f"Progresso (épocas): {current_epoch}/{total_epochs}")

    def _on_batch_progress(self, net_name: str, step: int, total_steps: int, epoch: int, total_epochs: int, loss: float) -> None:
        if total_steps > 0:
            p = int((step * 100) / total_steps)
            self.lbl_progress_detail.setText(f"Dados processados: {step}/{total_steps} ({p}%)")

        # x contínuo em epochs
        if total_epochs > 0 and total_steps > 0:
            steps_per_epoch = float(total_steps) / float(total_epochs)
            x_frac = (float(step) / steps_per_epoch) + 1.0
        else:
            x_frac = float(epoch) + 1.0

        lx = self._live_loss_x.setdefault(net_name, [])
        ly_raw = self._live_loss_y_raw.setdefault(net_name, [])
        ly_ema = self._live_loss_y_ema.setdefault(net_name, [])

        # EMA para remover “picos por época”
        alpha = 0.10
        last = self._live_loss_ema_last.get(net_name, None)
        if last is None:
            ema = float(loss)
        else:
            ema = (alpha * float(loss)) + ((1.0 - alpha) * float(last))
        self._live_loss_ema_last[net_name] = ema

        if not lx or abs(lx[-1] - x_frac) > 1e-9:
            lx.append(x_frac)
            ly_raw.append(float(loss))
            ly_ema.append(float(ema))
        else:
            ly_raw[-1] = float(loss)
            ly_ema[-1] = float(ema)

        if self._plot_update_mode() == self.UPDATE_BY_DATA:
            self._update_plots()

    def _on_epoch_metrics(self, metrics: EpochMetrics) -> None:
        name = metrics.network_name
        epoch_num = int(metrics.epoch_index) + 1

        x_list = self._net_epoch_x.setdefault(name, [])
        if not x_list or x_list[-1] != float(epoch_num):
            x_list.append(float(epoch_num))

        hist = self._net_hist.setdefault(
            name,
            {"loss_avg": [], "lr": [], "perf": [], "acc": [], "counts": [], "hist100": []},
        )

        def _push(key: str, value: Any):
            arr = hist.setdefault(key, [])
            if len(arr) < len(x_list):
                arr.append(value)
            else:
                arr[-1] = value

        _push("loss_avg", float(metrics.loss_avg))
        _push("lr", float(metrics.learning_rate))
        _push("perf", float(metrics.samples_per_sec))
        _push("acc", float(metrics.accuracy))
        _push("counts", list(metrics.class_counts) if metrics.class_counts is not None else [])
        _push("hist100", list(getattr(metrics, "conf_hist_100", []) or [0] * 100))

        # >>> mostra acurácia SEMPRE no progresso (legível)
        self.lbl_metrics.setText(
            f"Acurácia: {metrics.accuracy:.4f}\n"
            f"Loss: {metrics.loss_final:.6g} (média: {metrics.loss_avg:.6g})\n"
            f"Perf: {metrics.samples_per_sec:.1f} samp/s\n"
            f"LR: {metrics.learning_rate:g}\n"
            f"Neurônios: {metrics.neurons_total}"
        )

        self._update_plots()

    def _on_train_finished(self) -> None:
        self._log("[TRAIN] Treinamento concluído.")
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._train_thread = None
        self._train_worker = None

    def _on_train_stopped(self) -> None:
        self._log("[INFO] Treino interrompido.")
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._train_thread = None
        self._train_worker = None

    def _on_worker_error(self, msg: str) -> None:
        self._log(f"[ERRO] {msg}")
        try:
            QMessageBox.critical(self, "Erro", msg)
        except Exception:
            pass

    # ---------------- plots ----------------

    def _ensure_curve(self, cache: Dict[str, pg.PlotDataItem], plot: pg.PlotWidget, name: str) -> pg.PlotDataItem:
        item = cache.get(name)
        if item is None:
            item = plot.plot(pen=self._pen_curve)
            cache[name] = item
        return item

    def _update_plot_dist(self, counts: List[int]) -> None:
        if self._dist_item is None:
            return
        n = len(counts)
        x = list(range(n))
        h = [int(v) for v in counts]
        self._dist_item.setOpts(x=x, height=h, width=1.0, pen=self._pen_bar, brush=self._brush_bar)
        # garante range
        self.plot_dist.getPlotItem().enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

    def _update_plot_hist100_horizontal(self, hist100: List[int]) -> None:
        if self._hist_item is None:
            return

        if not hist100:
            self._hist_item.setOpts(x0=[], x1=[], y0=[], y1=[], pen=self._pen_bar, brush=self._brush_bar)
            self.plot_minmax.getPlotItem().enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            return

        y0 = list(range(0, 100))
        y1 = [i + 1 for i in y0]
        x0 = [0.0] * 100
        x1 = [float(v) for v in hist100[:100]]
        if len(x1) < 100:
            x1.extend([0.0] * (100 - len(x1)))

        self._hist_item.setOpts(x0=x0, x1=x1, y0=y0, y1=y1, pen=self._pen_bar, brush=self._brush_bar)
        self.plot_minmax.getPlotItem().enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

    def _update_plots(self) -> None:
        base = AIContext.get_name()
        if not base:
            return

        mode = self._plot_update_mode()
        hist = self._net_hist.get(base, {})
        x_epoch = self._net_epoch_x.get(base, [])

        # LOSS
        if self.chk_plot_loss_epoch.isChecked():
            if mode == self.UPDATE_BY_DATA:
                lx = self._live_loss_x.get(base, [])
                # >>> usa EMA para remover picos por época
                ly = self._live_loss_y_ema.get(base, [])
                self._ensure_curve(self._curves_loss_epoch, self.plot_loss_epoch, base).setData(lx, ly)
            else:
                y = hist.get("loss_avg", [])
                self._ensure_curve(self._curves_loss_epoch, self.plot_loss_epoch, base).setData(x_epoch, y)
        else:
            if base in self._curves_loss_epoch:
                self._curves_loss_epoch[base].setData([], [])

        # LR / PERF / ACC por época
        if self.chk_plot_lr.isChecked() and x_epoch:
            self._ensure_curve(self._curves_lr, self.plot_lr, base).setData(x_epoch, hist.get("lr", []))
        else:
            if base in self._curves_lr:
                self._curves_lr[base].setData([], [])

        if self.chk_plot_perf.isChecked() and x_epoch:
            self._ensure_curve(self._curves_perf, self.plot_perf, base).setData(x_epoch, hist.get("perf", []))
        else:
            if base in self._curves_perf:
                self._curves_perf[base].setData([], [])

        if self.chk_plot_acc.isChecked() and x_epoch:
            self._ensure_curve(self._curves_acc, self.plot_acc, base).setData(x_epoch, hist.get("acc", []))
        else:
            if base in self._curves_acc:
                self._curves_acc[base].setData([], [])

        # Distribuição
        if self.chk_plot_dist.isChecked():
            counts_list = hist.get("counts", [])
            last_counts = counts_list[-1] if counts_list else []
            self._update_plot_dist(last_counts if isinstance(last_counts, list) else [])
        else:
            self._update_plot_dist([])

        # Histograma
        if self.chk_plot_minmax.isChecked():
            h100_list = hist.get("hist100", [])
            last_h = h100_list[-1] if h100_list else []
            self._update_plot_hist100_horizontal(last_h if isinstance(last_h, list) else [])
        else:
            self._update_plot_hist100_horizontal([])

    def _clear_histories(self) -> None:
        self._net_hist.clear()
        self._net_epoch_x.clear()
        self._last_generation_tops = []

        self._live_loss_x.clear()
        self._live_loss_y_raw.clear()
        self._live_loss_y_ema.clear()
        self._live_loss_ema_last.clear()

        self.progress_bar.setValue(0)
        self.lbl_progress.setText("Aguardando início do treino...")
        self.lbl_progress_detail.setText("Dados processados: 0/0 (0%)")
        self.lbl_metrics.setText("Acurácia: --\nLoss: --\nPerf: --\nLR: --\nNeurônios: --")

        for d in (self._curves_loss_epoch, self._curves_lr, self._curves_perf, self._curves_acc):
            for item in d.values():
                try:
                    item.setData([], [])
                except Exception:
                    pass

        self._update_plot_dist([])
        self._update_plot_hist100_horizontal([])

    def _log(self, msg: str) -> None:
        self.txt_log.appendPlainText(str(msg))
        self.txt_log.ensureCursorVisible()
