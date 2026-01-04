from __future__ import annotations

from typing import List, Sequence, Tuple, Optional, Dict, Callable, Any
from datetime import datetime

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
from ia_engine import build_engine  # mantido (mesmo se não usado aqui, evita quebra se você usar depois)
from ia_generational import MutationConfig
from ia_training import Sample
from json_lib import build_paths, load_json
from pages.training.config import load_training_defaults
from pages.training import adapters as tr_adapters

# Workers/métricas agora no lugar certo
from pages.training.workers import EpochMetrics, TrainWorker, GenerationalWorker, EvalWorker


# Defaults (carregados de config.ini em [training], se existir)
_TD = load_training_defaults()
DEFAULT_EPOCHS = _TD.epochs_default
DEFAULT_LR = _TD.lr_default
DEFAULT_LR_MULT = _TD.lr_mult_default
DEFAULT_LR_MIN = _TD.lr_min_default
DEFAULT_DATA_UPDATE = _TD.data_update_default
DEFAULT_EVAL_LIMIT = _TD.eval_limit_default


# ----------------------------------------------------------------------
# Patch defensivo: pyqtgraph.ErrorBarItem às vezes assume numpy arrays e
# quebra quando recebe listas (TypeError: 'list' - 'list').
# Isso pode derrubar o processo (0xC0000409) em alguns ambientes.
# ----------------------------------------------------------------------
def _patch_pyqtgraph_errorbaritem() -> None:
    try:
        from pyqtgraph.graphicsItems.ErrorBarItem import ErrorBarItem  # type: ignore
    except Exception:
        return

    if getattr(ErrorBarItem, "_zenite_patched", False):
        return

    orig_setData = ErrorBarItem.setData

    def setData(self, **kwds):  # type: ignore
        for key in ("x", "y", "top", "bottom", "left", "right"):
            if key in kwds and kwds[key] is not None and not isinstance(kwds[key], (list, tuple)):
                try:
                    kwds[key] = list(kwds[key])
                except Exception:
                    pass
        return orig_setData(self, **kwds)

    ErrorBarItem.setData = setData  # type: ignore
    ErrorBarItem._zenite_patched = True  # type: ignore


_patch_pyqtgraph_errorbaritem()


class WdTrainingPane(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._train_thread: Optional[QThread] = None
        self._train_worker: Optional[TrainWorker] = None
        self._eval_thread: Optional[QThread] = None
        self._eval_worker: Optional[EvalWorker] = None
        self._gen_thread: Optional[QThread] = None
        self._gen_worker: Optional[GenerationalWorker] = None

        self._train_data: Optional[List[Sample]] = None
        self._eval_data: Optional[List[Sample]] = None
        self._test_data: Optional[List[Sample]] = None

        self._dataset_funcs: Dict[str, Callable[[], Sequence[Sample]]] = {}
        self.cmb_dataset: Optional[QComboBox] = None

        self._total_train_samples: int = 0
        self._total_epochs: int = 0

        self._last_generation_tops: List[Dict[str, Any]] = []

        self._net_hist: Dict[str, Dict[str, Any]] = {}
        self._net_epoch_x: Dict[str, List[int]] = {}

        self._curves_loss_epoch: Dict[str, pg.PlotDataItem] = {}
        self._curves_lr: Dict[str, pg.PlotDataItem] = {}
        self._curves_perf: Dict[str, pg.PlotDataItem] = {}
        self._curves_acc: Dict[str, pg.PlotDataItem] = {}

        self._dist_item: Optional[pg.BarGraphItem] = None

        self._minmax_bar: Optional[pg.BarGraphItem] = None
        self._minmax_scatter_min: Optional[pg.ScatterPlotItem] = None
        self._minmax_scatter_max: Optional[pg.ScatterPlotItem] = None

        self._plots_enabled: bool = True

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

        self.plot_minmax = pg.PlotWidget(title="Min/Max da saída selecionada (prob)")
        self.plot_minmax.showGrid(x=True, y=True)
        self.plot_minmax.enableAutoRange("y", True)
        self.plot_minmax.setLabel("left", "Probabilidade")
        self.plot_minmax.setLabel("bottom", "Rede")

        self._minmax_bar = pg.BarGraphItem(x=[], height=[], width=0.6, y0=[])
        self.plot_minmax.addItem(self._minmax_bar)
        self._minmax_scatter_min = pg.ScatterPlotItem(x=[], y=[], size=8, symbol="t")
        self._minmax_scatter_max = pg.ScatterPlotItem(x=[], y=[], size=8, symbol="t1")
        self.plot_minmax.addItem(self._minmax_scatter_min)
        self.plot_minmax.addItem(self._minmax_scatter_max)

        self.plot_loss_epoch = pg.PlotWidget(title="Loss médio por época")
        self.plot_loss_epoch.showGrid(x=True, y=True)
        self.plot_loss_epoch.enableAutoRange("y", True)

        self.plot_lr = pg.PlotWidget(title="Learning rate")
        self.plot_lr.showGrid(x=True, y=True)
        self.plot_lr.enableAutoRange("y", True)

        self.plot_perf = pg.PlotWidget(title="Performance (samples/s)")
        self.plot_perf.showGrid(x=True, y=True)
        self.plot_perf.enableAutoRange("y", True)

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
        self.plot_dist.enableAutoRange("y", True)
        self._dist_item = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.plot_dist.addItem(self._dist_item)

        self.plot_acc = pg.PlotWidget(title="Acurácia por época")
        self.plot_acc.showGrid(x=True, y=True)
        self.plot_acc.enableAutoRange("y", True)

        prog_group = QGroupBox("Progresso de treinamento")
        prog_layout = QVBoxLayout(prog_group)
        prog_layout.setSpacing(4)
        prog_layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_progress = QLabel("Aguardando início do treino...")
        self.lbl_progress.setWordWrap(True)
        self.lbl_progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.lbl_progress_detail = QLabel("Dados processados: 0/0 (0%)")
        self.lbl_progress_detail.setWordWrap(True)
        self.lbl_progress_detail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.lbl_metrics = QLabel("Acurácia: --\nLoss: --\nPerf: --\nLR: --\nNeurônios: --")
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

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
        self.txt_log.setMinimumHeight(0)
        log_layout.addWidget(self.txt_log)

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

        self._reload_dataset_functions()
        self._clear_histories()

    def showEvent(self, event):
        super().showEvent(event)
        self._adjust_splitter()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adjust_splitter()

    def _adjust_splitter(self):
        if not hasattr(self, "splitter") or self.splitter is None:
            return
        h = self.splitter.height()
        if h <= 0:
            return
        half = h // 2
        self.splitter.setSizes([half, h - half])

    def _build_flow_group(self) -> QGroupBox:
        grp = QGroupBox("Fluxo de dados e treinamento")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)

        lbl_ds = QLabel("Dataset:")
        self.cmb_dataset = QComboBox()
        self.cmb_dataset.setMinimumWidth(170)
        self.cmb_dataset.setMaximumWidth(9999)

        layout.addWidget(lbl_ds)
        layout.addWidget(self.cmb_dataset)

        self.btn_load_data = QPushButton("1) Carregar dados")
        self.btn_prepare = QPushButton("2) Preparar treino")
        self.btn_train = QPushButton("3) Treinar rede")
        self.btn_eval = QPushButton("4) Avaliar rede")
        self.btn_stop = QPushButton("Parar")
        self.btn_evolve = QPushButton("Evoluir")

        for btn in [self.btn_load_data, self.btn_prepare, self.btn_train, self.btn_eval, self.btn_stop, self.btn_evolve]:
            btn.setFixedHeight(26)
            layout.addWidget(btn)

        self.btn_stop.setEnabled(False)
        layout.addStretch(1)

        self.btn_load_data.clicked.connect(self._on_load_data_clicked)
        self.btn_prepare.clicked.connect(self._on_prepare_clicked)
        self.btn_train.clicked.connect(self._on_train_clicked)
        self.btn_eval.clicked.connect(self._on_eval_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_evolve.clicked.connect(self._on_evolve_clicked)

        return grp

    def _build_params_group(self) -> QGroupBox:
        grp = QGroupBox("Parâmetros de treinamento")
        layout = QFormLayout(grp)
        layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.setFormAlignment(Qt.AlignTop)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(2)
        grp.setContentsMargins(6, 6, 6, 6)

        def compact_spinbox(spin: QSpinBox | QDoubleSpinBox, is_double: bool = False):
            spin.setMaximumWidth(110)
            if is_double:
                spin.setStyleSheet("QDoubleSpinBox { padding: 0 3px; }")
            else:
                spin.setStyleSheet("QSpinBox { padding: 0 3px; }")

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(DEFAULT_EPOCHS)
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

        self.spin_data_update = QSpinBox()
        self.spin_data_update.setRange(1, 10_000_000)
        self.spin_data_update.setValue(DEFAULT_DATA_UPDATE)
        self.spin_data_update.setToolTip(
            "Atualiza progresso/log a cada N amostras processadas (sem esperar a época acabar)."
        )
        compact_spinbox(self.spin_data_update, is_double=False)

        self.spin_eval_limit = QSpinBox()
        self.spin_eval_limit.setRange(0, 1_000_000)
        self.spin_eval_limit.setValue(DEFAULT_EVAL_LIMIT)
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

        for chk in (self.chk_allow_add_layers, self.chk_allow_remove_layers, self.chk_allow_add_neurons, self.chk_allow_remove_neurons):
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

        layout.addRow("Qtd redes no gráfico:", self.spin_show_top)
        layout.addRow("", self.chk_show_original)
        layout.addRow("Saída p/ min/max:", self.spin_output_index)

        self.spin_show_top.valueChanged.connect(self._apply_plot_enable_policy)
        self.chk_show_original.stateChanged.connect(self._apply_plot_enable_policy)

        return grp

    # ==================================================================
    # Suporte a train_datasets.py
    # ==================================================================
    def _reload_dataset_functions(self) -> None:
        """
        Carrega dinamicamente as funções ds_* do módulo train_datasets.py e preenche o combo.
        Esta lógica fica centralizada no adapter para evitar reestruturações futuras.
        """
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

    # ==================================================================
    # Handlers de botões
    # ==================================================================
    def _ensure_active_network(self) -> Optional[str]:
        name = AIContext.get_name()
        if not name:
            QMessageBox.warning(
                self,
                "IA não selecionada",
                "Selecione ou crie uma IA antes de continuar.",
            )
            return None
        return str(name)

    def _load_lr_from_json(self, ia_name: str) -> None:
        try:
            _, manifest_path, _, _ = build_paths(ia_name)
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
        else:
            self._log("[LR] Nenhuma IA ativa ao carregar dados; usando parâmetros atuais de LR.")

        try:
            ds_name, func = self._get_selected_dataset_func()
        except Exception as e:
            QMessageBox.warning(self, "Cenário de dados", str(e))
            self._log(f"[DATA] Erro ao selecionar cenário: {e}")
            return

        try:
            data = list(func())
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
            self._log(f"[DATA] Cenário '{ds_name}' vazio.")
            return

        self._train_data = data
        self._eval_data = list(data)
        self._test_data = list(data)
        self._total_train_samples = len(data)

        self._log(f"[OK] Dataset '{ds_name}' carregado: {len(data)} amostras.")

    def _on_prepare_clicked(self) -> None:
        if not self._train_data:
            QMessageBox.warning(self, "Treino", "Carregue os dados primeiro (1).")
            return
        self._clear_histories()
        self._log("[OK] Treino preparado.")

    def _on_train_clicked(self) -> None:
        name = self._ensure_active_network()
        if not name:
            return
        if not self._train_data:
            QMessageBox.warning(self, "Treino", "Carregue os dados primeiro (1).")
            return

        epochs = int(self.spin_epochs.value())
        lr = float(self.spin_lr.value())
        lr_mult = float(self.spin_lr_mult.value())
        lr_min = float(self.spin_lr_min.value())
        shuffle = bool(self.chk_shuffle.isChecked())
        update_every = int(self.spin_data_update.value())
        eval_limit = int(self.spin_eval_limit.value())
        output_index = int(self.spin_output_index.value())

        self._total_epochs = epochs
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

        if self._gen_worker is not None:
            self._gen_worker.stop()
            self._log("[INFO] Solicitação de parada (gerações) enviada.")

    def _on_eval_clicked(self) -> None:
        self._log("[INFO] Avaliação manual não está conectada neste arquivo (use o fluxo de treino).")

    def _on_evolve_clicked(self) -> None:
        name = self._ensure_active_network()
        if not name:
            return
        if not self._train_data:
            QMessageBox.warning(self, "Evolução", "Carregue os dados primeiro (1).")
            return

        generations = int(self.spin_generations.value())
        population = int(self.spin_population.value())
        epochs = int(self.spin_epochs.value())
        lr = float(self.spin_lr.value())
        lr_mult = float(self.spin_lr_mult.value())
        lr_min = float(self.spin_lr_min.value())
        shuffle = bool(self.chk_shuffle.isChecked())
        update_every = int(self.spin_data_update.value())
        eval_limit = int(self.spin_eval_limit.value())
        output_index = int(self.spin_output_index.value())

        mut_cfg = MutationConfig(
            allow_add_layers=bool(self.chk_allow_add_layers.isChecked()),
            allow_remove_layers=bool(self.chk_allow_remove_layers.isChecked()),
            allow_add_neurons=bool(self.chk_allow_add_neurons.isChecked()),
            allow_remove_neurons=bool(self.chk_allow_remove_neurons.isChecked()),
            max_layer_delta=int(self.spin_layer_delta.value()),
            max_neuron_delta=int(self.spin_neuron_delta.value()),
        )

        self._log(f"[EVOLVE] Iniciando: {name} | gen={generations}, pop={population}")

        self.btn_stop.setEnabled(True)

        worker = GenerationalWorker(
            parent_name=name,
            train_data=self._train_data,
            epochs_per_individual=epochs,
            learning_rate=lr,
            lr_mult=lr_mult,
            lr_min=lr_min,
            shuffle=shuffle,
            eval_limit=eval_limit,
            generations=generations,
            population=population,
            mutation_cfg=mut_cfg,
            update_every_n=update_every,
            output_index=output_index,
            batch_size=16,
        )

        thread = QThread(self)
        worker.moveToThread(thread)

        worker.log.connect(self._log)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(self._on_gen_finished)
        worker.epoch_metrics.connect(self._on_epoch_metrics)
        worker.tops_updated.connect(self._on_tops_updated)
        worker.progress.connect(self._on_gen_progress)

        thread.started.connect(worker.run)

        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._gen_thread = thread
        self._gen_worker = worker
        thread.start()

    def _on_train_progress(self, current_epoch: int, total_epochs: int) -> None:
        if total_epochs <= 0:
            return
        perc = int(current_epoch * 100 / total_epochs)
        perc = max(0, min(100, perc))
        self.progress_bar.setValue(perc)
        self.lbl_progress.setText(f"Progresso (épocas): {current_epoch}/{total_epochs}")

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

        if self.lbl_metrics is not None and name == AIContext.get_name():
            self.lbl_metrics.setText(
                f"Acurácia: {metrics.accuracy:.4f}\n"
                f"Loss: {metrics.loss_final:.6g} (média: {metrics.loss_avg:.6g})\n"
                f"Perf: {metrics.samples_per_sec:.1f} samp/s\n"
                f"LR: {metrics.learning_rate:g}\n"
                f"Neurônios: {metrics.neurons_total}"
            )

        if name == AIContext.get_name() and self.spin_lr is not None:
            try:
                self.spin_lr.blockSignals(True)
                self.spin_lr.setValue(metrics.learning_rate)
            finally:
                self.spin_lr.blockSignals(False)

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

    def _on_gen_progress(self, name: str, processed: int, total: int, loss: float) -> None:
        if total <= 0:
            return
        perc = int(processed * 100 / total)
        perc = max(0, min(100, perc))
        self.progress_bar.setValue(perc)
        self.lbl_progress.setText(f"Gerações: {name}")
        self.lbl_progress_detail.setText(f"Dados processados: {processed}/{total} ({perc}%) | loss={loss:.6g}")

    def _on_tops_updated(self, tops: List[Dict[str, Any]]) -> None:
        self._last_generation_tops = list(tops or [])
        self._update_plots()

    def _on_gen_finished(self) -> None:
        self._log("[EVOLVE] Concluído.")
        self.btn_stop.setEnabled(False)
        self._gen_thread = None
        self._gen_worker = None

    def _on_worker_error(self, msg: str) -> None:
        self._log(f"[ERRO] {msg}")
        try:
            QMessageBox.critical(self, "Erro", msg)
        except Exception:
            pass

    def _clear_histories(self) -> None:
        self._net_hist.clear()
        self._net_epoch_x.clear()
        self._last_generation_tops = []
        self.progress_bar.setValue(0)
        self.lbl_progress.setText("Aguardando início do treino...")
        self.lbl_progress_detail.setText("Dados processados: 0/0 (0%)")
        self.lbl_metrics.setText("Acurácia: --\nLoss: --\nPerf: --\nLR: --\nNeurônios: --")

        if self._minmax_bar is not None:
            self._minmax_bar.setOpts(x=[], y0=[], height=[])
        if self._minmax_scatter_min is not None:
            self._minmax_scatter_min.setData(x=[], y=[])
        if self._minmax_scatter_max is not None:
            self._minmax_scatter_max.setData(x=[], y=[])

        if self._dist_item is not None:
            self._dist_item.setOpts(x=[], height=[])

        for d in (self._curves_loss_epoch, self._curves_lr, self._curves_perf, self._curves_acc):
            for item in d.values():
                try:
                    item.setData([], [])
                except Exception:
                    pass

    def _apply_plot_enable_policy(self) -> None:
        show_top = int(self.spin_show_top.value())
        show_orig = bool(self.chk_show_original.isChecked())
        self._plots_enabled = (show_top > 0) or show_orig

    def _get_display_networks(self) -> List[str]:
        out: List[str] = []
        if self.chk_show_original.isChecked():
            base = AIContext.get_name() or ""
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
        return [
            (0, 180, 255),
            (255, 180, 0),
            (0, 220, 140),
            (255, 80, 80),
            (180, 0, 255),
            (180, 180, 180),
        ]

    def _ensure_curve(
        self,
        cache: Dict[str, pg.PlotDataItem],
        plot: pg.PlotWidget,
        name: str,
        color: Tuple[int, int, int],
    ) -> pg.PlotDataItem:
        item = cache.get(name)
        if item is None:
            item = plot.plot(pen=pg.mkPen(color=color, width=2))
            cache[name] = item
        return item

    def _update_plots(self) -> None:
        self._apply_plot_enable_policy()
        if not self._plots_enabled:
            return

        display = self._get_display_networks()
        if not display:
            return

        palette = self._palette()

        for i, nm in enumerate(display):
            x = self._net_epoch_x.get(nm, [])
            hist = self._net_hist.get(nm, {})
            color = palette[i % len(palette)]

            if self.chk_plot_loss_epoch.isChecked() and x:
                y = hist.get("loss_avg", [])
                self._ensure_curve(self._curves_loss_epoch, self.plot_loss_epoch, nm, color).setData(x, y)
            else:
                if nm in self._curves_loss_epoch:
                    self._curves_loss_epoch[nm].setData([], [])

            if self.chk_plot_lr.isChecked() and x:
                y = hist.get("lr", [])
                self._ensure_curve(self._curves_lr, self.plot_lr, nm, color).setData(x, y)
            else:
                if nm in self._curves_lr:
                    self._curves_lr[nm].setData([], [])

            if self.chk_plot_perf.isChecked() and x:
                y = hist.get("perf", [])
                self._ensure_curve(self._curves_perf, self.plot_perf, nm, color).setData(x, y)
            else:
                if nm in self._curves_perf:
                    self._curves_perf[nm].setData([], [])

            if self.chk_plot_acc.isChecked() and x:
                y = hist.get("acc", [])
                self._ensure_curve(self._curves_acc, self.plot_acc, nm, color).setData(x, y)
            else:
                if nm in self._curves_acc:
                    self._curves_acc[nm].setData([], [])

        if self.chk_plot_minmax.isChecked():
            xs = list(range(len(display)))
            mins: List[float] = []
            maxs: List[float] = []

            for nm in display:
                hist = self._net_hist.get(nm, {})
                mn_list = hist.get("out_min", [])
                mx_list = hist.get("out_max", [])
                mn = float(mn_list[-1]) if mn_list else 0.0
                mx = float(mx_list[-1]) if mx_list else 0.0
                mins.append(mn)
                maxs.append(mx)

            if self._minmax_bar is not None:
                heights = [max(0.0, float(ma) - float(mi)) for mi, ma in zip(mins, maxs)]
                self._minmax_bar.setOpts(x=xs, y0=mins, height=heights)

            if self._minmax_scatter_min is not None:
                self._minmax_scatter_min.setData(x=xs, y=mins)
            if self._minmax_scatter_max is not None:
                self._minmax_scatter_max.setData(x=xs, y=maxs)

            try:
                ax = self.plot_minmax.getAxis("bottom")
                ax.setTicks([[(i, str(nm)) for i, nm in enumerate(display)]])
            except Exception:
                pass
        else:
            if self._minmax_bar is not None:
                self._minmax_bar.setOpts(x=[], y0=[], height=[])
            if self._minmax_scatter_min is not None:
                self._minmax_scatter_min.setData(x=[], y=[])
            if self._minmax_scatter_max is not None:
                self._minmax_scatter_max.setData(x=[], y=[])

        if self.chk_plot_dist.isChecked():
            chosen = None
            for nm in display:
                hist = self._net_hist.get(nm, {})
                cc = hist.get("class_counts_last", [])
                if cc:
                    chosen = cc
                    break
            if chosen and self._dist_item is not None:
                xs = list(range(len(chosen)))
                self._dist_item.setOpts(x=xs, height=chosen, width=0.6)
            else:
                if self._dist_item is not None:
                    self._dist_item.setOpts(x=[], height=[])
        else:
            if self._dist_item is not None:
                self._dist_item.setOpts(x=[], height=[])

    def _log(self, msg: str) -> None:
        self.txt_log.appendPlainText(str(msg))
        self.txt_log.ensureCursorVisible()
