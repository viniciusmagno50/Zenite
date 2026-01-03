from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable, Sequence, Tuple
from datetime import datetime
import importlib
import inspect

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDoubleValidator, QDesktopServices
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSplitter,
    QFrame,
    QLineEdit,
    QPushButton,
    QComboBox,
    QSpinBox,
    QMessageBox,
)

# ✅ IMPORT NOVO (SEM biuld_structure.py)
from pages.individual.network_view.network_view import NetworkView

from ia_manifest import IAManifest
from ia_context import AIContext
from json_lib import build_paths, load_json, save_json


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _fill_grid_left_aligned(grid: QGridLayout, data: Dict[str, Any]) -> None:
    """
    Preenche QGridLayout com:
      - chave em negrito, alinhada à esquerda
      - valor ao lado, alinhado à esquerda
    """
    while grid.count():
        item = grid.takeAt(0)
        w = item.widget()
        if w:
            w.setParent(None)
            w.deleteLater()

    if not data:
        k = QLabel("<b>—</b>")
        v = QLabel("(vazio)")
        for lb in (k, v):
            lb.setTextFormat(Qt.RichText)
            lb.setStyleSheet("color: #D8D8D8;")
            lb.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(k, 0, 0)
        grid.addWidget(v, 0, 1)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        return

    row = 0
    for key, val in data.items():
        k = QLabel(f"<b>{key}:</b>")
        k.setTextFormat(Qt.RichText)
        k.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        k.setStyleSheet("color: #D8D8D8;")

        if isinstance(val, dict):
            text = ", ".join(f"{sk}={sv}" for sk, sv in list(val.items())[:6])
            if len(val) > 6:
                text += ", ..."
        elif isinstance(val, list):
            text = ", ".join(str(x) for x in val[:6])
            if len(val) > 6:
                text += ", ..."
        else:
            text = str(val)

        v = QLabel(text)
        v.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        v.setStyleSheet("color: #D8D8D8;")

        grid.addWidget(k, row, 0)
        grid.addWidget(v, row, 1)
        row += 1

    grid.setColumnStretch(0, 0)
    grid.setColumnStretch(1, 1)


def _payload_to_matrix_structure(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"layers": [], "weights": [], "biases": []}

    s = payload.get("structure", {}) or {}
    inp = int(s.get("input_size") or 0)

    neurons = s.get("neurons") or []
    if isinstance(neurons, int):
        neurons = [neurons]
    if not isinstance(neurons, list):
        neurons = []
    neurons = [int(x) for x in neurons] if neurons else []

    if inp <= 0 or not neurons:
        return {"layers": [], "weights": [], "biases": []}

    layers = [inp] + neurons
    weights_raw = s.get("weights") or []

    weights: List[List[List[float]]] = []
    biases: List[List[float]] = []

    prev = inp
    for li, n_out in enumerate(neurons):
        W_layer: List[List[float]] = []
        b_layer: List[float] = []

        src_layer = weights_raw[li] if isinstance(weights_raw, list) and li < len(weights_raw) else []

        for ni in range(int(n_out)):
            row = src_layer[ni] if isinstance(src_layer, list) and ni < len(src_layer) else None

            if isinstance(row, list) and len(row) >= 1:
                try:
                    b = float(row[0])
                except Exception:
                    b = 0.0
                w_raw = row[1:]
                try:
                    w = [float(x) for x in w_raw]
                except Exception:
                    w = []
            else:
                b = 0.0
                w = []

            if len(w) < prev:
                w += [0.0] * (prev - len(w))
            elif len(w) > prev:
                w = w[:prev]

            b_layer.append(b)
            W_layer.append(w)

        weights.append(W_layer)
        biases.append(b_layer)
        prev = int(n_out)

    return {"layers": layers, "weights": weights, "biases": biases}


# ----------------------------------------------------------------------
class WdCreateAnalisePane(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, side_width: int = 340):
        super().__init__(parent)
        self.setObjectName("ws_create_analise")
        self._side_width = int(side_width)

        self._current_view: Optional[NetworkView] = None
        self._manifest_path = None
        self._manifest_data: Optional[Dict[str, Any]] = None

        self._test_input_fields: List[QLineEdit] = []
        self._current_input_size: int = 0
        self._current_output_size: int = 0

        self._input_history: List[List[float]] = []
        self.cmb_history: Optional[QComboBox] = None

        self._dataset_funcs: Dict[str, Callable[..., Sequence[Any]]] = {}
        self.cmb_dataset: Optional[QComboBox] = None
        self.spin_batch_limit: Optional[QSpinBox] = None

        self._poll_timer: Optional[QTimer] = None
        self._last_ai_name: Optional[str] = None

        # saída em HTML (título em negrito)
        self._manual_html: str = "<b>Status:</b> (não executado)"
        self._batch_html: str = "<b>Status:</b> (não executado)"

        self.lbl_diag: Optional[QLabel] = None
        self.lbl_output_all: Optional[QLabel] = None

        self._build_ui()
        self._reload_dataset_functions()
        self._setup_ai_context_polling()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        # ========== ESQUERDA ==========
        self.side = QFrame()
        self.side.setMinimumWidth(self._side_width)
        self.side.setMaximumWidth(self._side_width)

        side_layout = QVBoxLayout(self.side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll_content = QWidget()
        self.side_blocks = QVBoxLayout(scroll_content)
        self.side_blocks.setContentsMargins(6, 6, 6, 6)
        self.side_blocks.setSpacing(8)

        self.box_ident, self.grid_ident = self._build_info_box("Identificação")
        self.box_struct, self.grid_struct = self._build_info_box("Estrutura")
        self.box_mut, self.grid_mut = self._build_info_box("Mutação")
        self.box_stats, self.grid_stats = self._build_info_box("Estatísticas")
        self.box_funcs, self.grid_funcs = self._build_info_box("Funções")

        for box in (self.box_ident, self.box_struct, self.box_mut, self.box_stats, self.box_funcs):
            box.setAlignment(Qt.AlignLeft)

        self.side_blocks.addWidget(self.box_ident)
        self.side_blocks.addWidget(self.box_struct)
        self.side_blocks.addWidget(self.box_mut)
        self.side_blocks.addWidget(self.box_stats)
        self.side_blocks.addWidget(self.box_funcs)
        self.side_blocks.addStretch(1)

        scroll.setWidget(scroll_content)
        side_layout.addWidget(scroll)

        # ========== DIREITA ==========
        self.workspace = QFrame()
        work_v = QVBoxLayout(self.workspace)
        work_v.setContentsMargins(10, 10, 10, 10)
        work_v.setSpacing(10)

        self.test_top = self._build_test_top()

        self.view_container = QFrame()
        self.view_container_layout = QVBoxLayout(self.view_container)
        self.view_container_layout.setContentsMargins(0, 0, 0, 0)

        placeholder = QLabel("Nenhuma IA carregada.\nSelecione ou crie uma IA na toolbar.")
        placeholder.setAlignment(Qt.AlignCenter)
        self.view_container_layout.addWidget(placeholder, 1)

        work_v.addWidget(self.test_top, 0)
        work_v.addWidget(self.view_container, 1)

        splitter.addWidget(self.side)
        splitter.addWidget(self.workspace)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, 1)

        _fill_grid_left_aligned(self.grid_ident, {})
        _fill_grid_left_aligned(self.grid_struct, {})
        _fill_grid_left_aligned(self.grid_mut, {})
        _fill_grid_left_aligned(self.grid_stats, {})
        _fill_grid_left_aligned(self.grid_funcs, {})

    def _build_info_box(self, title: str) -> Tuple[QGroupBox, QGridLayout]:
        box = QGroupBox(title)
        layout = QGridLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)
        return box, layout

    # ------------------------------------------------------------------
    # Topo: 3 colunas MESMO TAMANHO
    # ------------------------------------------------------------------
    def _build_test_top(self) -> QGroupBox:
        group = QGroupBox("Testes e Diagnóstico")
        group.setAlignment(Qt.AlignLeft)

        layout = QGridLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        # COL 1: manual
        g_manual = QGroupBox("Teste manual")
        g_manual.setAlignment(Qt.AlignLeft)
        lm = QVBoxLayout(g_manual)
        lm.setContentsMargins(8, 8, 8, 8)
        lm.setSpacing(6)

        self.lbl_manual_info = QLabel("Carregue uma IA para habilitar o teste.")
        self.lbl_manual_info.setStyleSheet("color: #D8D8D8;")
        self.lbl_manual_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lm.addWidget(self.lbl_manual_info)

        # diag: engine e device (sem manifest), um abaixo do outro
        self.lbl_diag = QLabel("Engine: --\nDevice: --")
        self.lbl_diag.setStyleSheet("color: #D8D8D8;")
        self.lbl_diag.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lm.addWidget(self.lbl_diag)

        # Inputs em grid (máximo 4 por linha)
        self.test_inputs_frame = QFrame()
        self.test_inputs_grid = QGridLayout(self.test_inputs_frame)
        self.test_inputs_grid.setContentsMargins(0, 0, 0, 0)
        self.test_inputs_grid.setHorizontalSpacing(6)
        self.test_inputs_grid.setVerticalSpacing(6)
        lm.addWidget(self.test_inputs_frame)

        row_actions = QHBoxLayout()
        row_actions.setSpacing(6)

        btn_run = QPushButton("Executar")
        btn_run.clicked.connect(self._on_run_forward)
        row_actions.addWidget(btn_run)

        btn_zero = QPushButton("Zerar")
        btn_zero.clicked.connect(self._on_fill_zeros)
        row_actions.addWidget(btn_zero)

        btn_example = QPushButton("Exemplo")
        btn_example.clicked.connect(self._on_fill_example)
        row_actions.addWidget(btn_example)

        btn_save = QPushButton("Salvar caso")
        btn_save.clicked.connect(self._on_save_case)
        row_actions.addWidget(btn_save)

        self.cmb_history = QComboBox()
        self.cmb_history.setMinimumWidth(180)
        self.cmb_history.addItem("(histórico vazio)", userData=None)
        self.cmb_history.currentIndexChanged.connect(self._on_history_selected)
        row_actions.addWidget(self.cmb_history, 1)

        lm.addLayout(row_actions)
        lm.addStretch(1)

        # COL 2: lote / arquivos
        g_batch = QGroupBox("Lote / Arquivos")
        g_batch.setAlignment(Qt.AlignLeft)
        lb = QVBoxLayout(g_batch)
        lb.setContentsMargins(8, 8, 8, 8)
        lb.setSpacing(6)

        lbl_batch_title = QLabel("Avaliação em lote (train_datasets.py ds_*):")
        lbl_batch_title.setStyleSheet("color: #D8D8D8;")
        lbl_batch_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lb.addWidget(lbl_batch_title)

        # dataset (igual estava)
        row_ds = QHBoxLayout()
        row_ds.setSpacing(6)
        self.cmb_dataset = QComboBox()
        self.cmb_dataset.setMinimumWidth(200)
        row_ds.addWidget(self.cmb_dataset, 1)
        lb.addLayout(row_ds)

        # limite e rodar lote: UM ABAIXO DO OUTRO
        self.spin_batch_limit = QSpinBox()
        self.spin_batch_limit.setRange(0, 1_000_000)
        self.spin_batch_limit.setValue(2000)
        self.spin_batch_limit.setToolTip("0 = usar default do cenário.")
        lb.addWidget(QLabel("Limite:"))
        lb.addWidget(self.spin_batch_limit)

        btn_batch = QPushButton("Rodar lote")
        btn_batch.clicked.connect(self._on_run_batch)
        lb.addWidget(btn_batch)

        # arquivos
        btn_open = QPushButton("Abrir pasta")
        btn_open.clicked.connect(self._on_open_manifest_folder)
        lb.addWidget(btn_open)

        btn_snapshot = QPushButton("Exportar snapshot")
        btn_snapshot.clicked.connect(self._on_export_snapshot)
        lb.addWidget(btn_snapshot)

        lb.addStretch(1)

        # COL 3: saída (HTML com títulos em negrito)
        g_out = QGroupBox("Saída (manual + lote)")
        g_out.setAlignment(Qt.AlignLeft)
        lo = QVBoxLayout(g_out)
        lo.setContentsMargins(8, 8, 8, 8)
        lo.setSpacing(6)

        self.lbl_output_all = QLabel()
        self.lbl_output_all.setStyleSheet("color: #D8D8D8;")
        self.lbl_output_all.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_output_all.setWordWrap(True)
        self.lbl_output_all.setTextFormat(Qt.RichText)
        lo.addWidget(self.lbl_output_all, 1)

        self._refresh_output_panel()

        # grid 3 colunas iguais
        layout.addWidget(g_manual, 0, 0)
        layout.addWidget(g_batch, 0, 1)
        layout.addWidget(g_out, 0, 2)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        return group

    # ------------------------------------------------------------------
    def _setup_ai_context_polling(self) -> None:
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(400)
        self._poll_timer.timeout.connect(self._poll_ai_context)
        self._poll_timer.start()

    def _poll_ai_context(self) -> None:
        current_name = AIContext.get_name()
        if current_name != self._last_ai_name:
            self._last_ai_name = current_name
            if current_name:
                self._load_manifest_from_name(current_name)
            else:
                self._clear_view()

    def _load_manifest_from_name(self, name: str) -> None:
        try:
            _, arquivo, _ = build_paths(name)
            if not arquivo.exists():
                raise FileNotFoundError(f"Manifesto não encontrado: {arquivo}")

            payload = load_json(arquivo)
            if not isinstance(payload, dict):
                raise ValueError("JSON inválido (não é dict).")

            self._manifest_path = arquivo
            self._manifest_data = payload
            self.load_from_dict(payload)

        except Exception as e:
            self._clear_view()
            QMessageBox.critical(self, "Erro ao carregar IA", str(e))

    def _clear_view(self) -> None:
        self._manifest_path = None
        self._manifest_data = None
        self._current_view = None

        _fill_grid_left_aligned(self.grid_ident, {})
        _fill_grid_left_aligned(self.grid_struct, {})
        _fill_grid_left_aligned(self.grid_mut, {})
        _fill_grid_left_aligned(self.grid_stats, {})
        _fill_grid_left_aligned(self.grid_funcs, {})

        placeholder = QLabel("Nenhuma IA carregada.\nSelecione ou crie uma IA na toolbar.")
        placeholder.setAlignment(Qt.AlignCenter)
        self.set_workspace_widget(placeholder)

        self._rebuild_test_inputs(0, 0)

        if self.lbl_diag is not None:
            self.lbl_diag.setText("Engine: --\nDevice: --")

        self._manual_html = "<b>Status:</b> (não executado)"
        self._batch_html = "<b>Status:</b> (não executado)"
        self._refresh_output_panel()

    # ------------------------------------------------------------------
    def _ensure_engine(self):
        engine = AIContext.get_instance()
        if engine is not None:
            return engine

        name = AIContext.get_name()
        if not name:
            return None

        try:
            _, arquivo, _ = build_paths(name)
            if not arquivo.exists():
                return None
            data = load_json(arquivo)
            manifest = IAManifest.from_dict(data)

            from ia_engine import NeuralNetworkEngine

            engine = NeuralNetworkEngine.from_manifest(manifest)
            AIContext.set_instance(engine)
            return engine
        except Exception:
            return None

    def _engine_runtime_dict(self, engine: Any) -> Dict[str, str]:
        etype = type(engine).__name__
        device = getattr(engine, "device", None)
        device_str = "CPU" if device is None else str(device)
        return {"engine_type": etype, "device": device_str}

    # ------------------------------------------------------------------
    def _rebuild_test_inputs(self, input_size: int, output_size: int) -> None:
        self._current_input_size = max(0, int(input_size))
        self._current_output_size = max(0, int(output_size))

        # limpa grid
        while self.test_inputs_grid.count():
            it = self.test_inputs_grid.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        self._test_input_fields.clear()

        if self._current_input_size <= 0:
            self.lbl_manual_info.setText("Estrutura inválida (input_size <= 0).")
            return

        self.lbl_manual_info.setText(f"Informe {self._current_input_size} valores de entrada (floats).")

        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)

        # máximo 4 por linha
        cols = 4
        for i in range(self._current_input_size):
            edit = QLineEdit()
            edit.setPlaceholderText(f"x{i}")
            edit.setValidator(validator)
            edit.setMaximumWidth(90)
            edit.returnPressed.connect(self._on_run_forward)

            r = i // cols
            c = i % cols
            self.test_inputs_grid.addWidget(edit, r, c)
            self._test_input_fields.append(edit)

    def _read_inputs(self) -> Optional[List[float]]:
        vals: List[float] = []
        for edit in self._test_input_fields:
            t = (edit.text() or "").strip()
            if not t:
                return None
            try:
                vals.append(float(t))
            except Exception:
                return None
        return vals if len(vals) == self._current_input_size else None

    def _on_fill_zeros(self) -> None:
        for e in self._test_input_fields:
            e.setText("0")
        if self._test_input_fields:
            self._test_input_fields[0].setFocus()

    def _on_fill_example(self) -> None:
        ex = []
        for i in range(self._current_input_size):
            ex.append(1.0 if i % 2 == 0 else -1.0)
        for e, v in zip(self._test_input_fields, ex):
            e.setText(str(v))
        if self._test_input_fields:
            self._test_input_fields[0].setFocus()

    def _on_save_case(self) -> None:
        vals = self._read_inputs()
        if vals is None:
            QMessageBox.information(self, "Salvar caso", "Preencha entradas válidas antes de salvar.")
            return
        self._add_history_case(vals, auto_select=True)

    def _add_history_case(self, vals: List[float], auto_select: bool) -> None:
        self._input_history.append(list(vals))
        if self.cmb_history is None:
            return

        if self.cmb_history.count() == 1 and self.cmb_history.itemData(0) is None:
            self.cmb_history.clear()

        label = f"{len(self._input_history)}: " + ", ".join(f"{v:g}" for v in vals[:4])
        if len(vals) > 4:
            label += ", ..."
        self.cmb_history.addItem(label, userData=len(self._input_history) - 1)

        if auto_select:
            self.cmb_history.setCurrentIndex(self.cmb_history.count() - 1)

    def _on_history_selected(self, idx: int) -> None:
        if self.cmb_history is None or idx < 0:
            return
        ref = self.cmb_history.itemData(idx)
        if ref is None:
            return
        try:
            vals = self._input_history[int(ref)]
        except Exception:
            return
        for e, v in zip(self._test_input_fields, vals):
            e.setText(str(v))

    # ------------------------------------------------------------------
    def _refresh_output_panel(self) -> None:
        if self.lbl_output_all is None:
            return
        self.lbl_output_all.setText(
            f"<b>Manual</b><br>{self._manual_html}<br><br>"
            f"<b>Lote</b><br>{self._batch_html}"
        )

    def _on_run_forward(self) -> None:
        if self._current_input_size <= 0:
            self._manual_html = "<b>Status:</b> Estrutura inválida (input_size <= 0)."
            self._refresh_output_panel()
            return

        engine = self._ensure_engine()
        if engine is None:
            self._manual_html = "<b>Status:</b> Engine não disponível."
            self._refresh_output_panel()
            return

        vals = self._read_inputs()
        if vals is None:
            self._manual_html = "<b>Status:</b> Entradas inválidas (preencha todos os campos)."
            self._refresh_output_panel()
            return

        try:
            out = engine.forward(vals)
            probs = [float(x) for x in out]
        except Exception as e:
            self._manual_html = f"<b>Status:</b> Erro no forward: {e}"
            self._refresh_output_panel()
            return

        if not probs:
            self._manual_html = "<b>Status:</b> Saída vazia."
            self._refresh_output_panel()
            return

        s = sum(probs)
        pred = max(range(len(probs)), key=lambda i: probs[i])
        top2 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:2]

        probs_str = "[" + ", ".join(f"{p:.6g}" for p in probs) + "]"
        top2_str = ", ".join(f"{i}:{probs[i]*100:.1f}%" for i in top2)

        warn = ""
        if abs(s - 1.0) > 1e-3:
            warn = f" <span style='color:#ffcc66'>(WARN soma={s:.6g})</span>"

        self._manual_html = (
            f"<b>Saída:</b> {probs_str}<br>"
            f"<b>Predição:</b> classe {pred} ({probs[pred]*100:.1f}%)<br>"
            f"<b>Top-2:</b> {top2_str}<br>"
            f"<b>Soma(prob):</b> {s:.6g}{warn}"
        )

        # atualiza diag (engine/device)
        if self.lbl_diag is not None:
            info = self._engine_runtime_dict(engine)
            self.lbl_diag.setText(f"Engine: {info['engine_type']}\nDevice: {info['device']}")

        self._add_history_case(vals, auto_select=False)
        self._refresh_output_panel()

    # ------------------------------------------------------------------
    def _on_open_manifest_folder(self) -> None:
        if self._manifest_path is None:
            QMessageBox.information(self, "Manifesto", "Nenhum manifesto carregado.")
            return
        folder = self._manifest_path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _on_export_snapshot(self) -> None:
        if self._manifest_path is None or self._manifest_data is None:
            QMessageBox.information(self, "Snapshot", "Nenhum manifesto carregado.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = self._manifest_path.stem
        out_path = self._manifest_path.parent / f"{stem}_snapshot_{ts}.json"

        try:
            save_json(out_path, self._manifest_data)
            QMessageBox.information(self, "Snapshot", f"Snapshot exportado:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Snapshot", f"Falha ao exportar snapshot:\n{e}")

    # ------------------------------------------------------------------
    def _reload_dataset_functions(self) -> None:
        if self.cmb_dataset is None:
            return

        try:
            module = importlib.import_module("train_datasets")
        except Exception:
            self.cmb_dataset.clear()
            self.cmb_dataset.addItem("(train_datasets.py não encontrado)", userData=None)
            self._dataset_funcs = {}
            return

        funcs: List[Tuple[str, Callable[..., Sequence[Any]]]] = []
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("ds_"):
                funcs.append((name, obj))

        self._dataset_funcs = {name: f for name, f in funcs}
        self.cmb_dataset.clear()

        if not funcs:
            self.cmb_dataset.addItem("(nenhum ds_* encontrado)", userData=None)
            return

        for name, func in funcs:
            doc = (func.__doc__ or "").strip().splitlines()
            desc = doc[0].strip() if doc else ""
            label = f"{name} - {desc}" if desc else name
            self.cmb_dataset.addItem(label, userData=name)

    def _get_selected_dataset_func(self) -> Optional[Tuple[str, Callable[..., Sequence[Any]]]]:
        if self.cmb_dataset is None:
            return None
        idx = self.cmb_dataset.currentIndex()
        if idx < 0:
            return None
        name = self.cmb_dataset.itemData(idx)
        if not name:
            return None
        func = self._dataset_funcs.get(name)
        return (name, func) if func is not None else None

    def _on_run_batch(self) -> None:
        engine = self._ensure_engine()
        if engine is None:
            self._batch_html = "<b>Status:</b> Engine não disponível."
            self._refresh_output_panel()
            return

        sel = self._get_selected_dataset_func()
        if sel is None:
            self._batch_html = "<b>Status:</b> Nenhum cenário ds_* disponível."
            self._refresh_output_panel()
            return

        ds_name, func = sel
        limit = self.spin_batch_limit.value() if self.spin_batch_limit is not None else 0

        try:
            sig = inspect.signature(func)
            kwargs = {}
            if limit > 0 and "n_samples" in sig.parameters:
                kwargs["n_samples"] = int(limit)
            data = func(**kwargs)  # type: ignore[misc]
        except Exception as e:
            self._batch_html = f"<b>Status:</b> Erro ao executar {ds_name}: {e}"
            self._refresh_output_panel()
            return

        if not data:
            self._batch_html = f"<b>Status:</b> Cenário {ds_name} retornou vazio."
            self._refresh_output_panel()
            return

        total = 0
        correct = 0
        n_classes = None
        pred_counts: List[int] = []
        hit_counts: List[int] = []

        tp = tn = fp = fn = 0

        for sample in data:
            if not isinstance(sample, (list, tuple)) or len(sample) != 2:
                continue
            x, target = sample
            try:
                probs = engine.forward(x)
                probs = [float(p) for p in probs]
            except Exception:
                continue

            if not probs:
                continue

            if n_classes is None:
                n_classes = len(probs)
                pred_counts = [0] * n_classes
                hit_counts = [0] * n_classes

            pred = max(range(len(probs)), key=lambda i: probs[i])
            try:
                y = int(target)
            except Exception:
                y = int(getattr(target, "value", 0))

            pred_counts[pred] += 1
            total += 1

            if pred == y:
                correct += 1
                if 0 <= pred < len(hit_counts):
                    hit_counts[pred] += 1

            if n_classes == 2:
                if pred == 1 and y == 1:
                    tp += 1
                elif pred == 0 and y == 0:
                    tn += 1
                elif pred == 1 and y == 0:
                    fp += 1
                elif pred == 0 and y == 1:
                    fn += 1

        if total <= 0:
            self._batch_html = "<b>Status:</b> Nenhuma amostra válida foi avaliada."
            self._refresh_output_panel()
            return

        acc = correct / total
        dist = ", ".join(f"{i}:{c}" for i, c in enumerate(pred_counts)) if pred_counts else "--"
        hits = ", ".join(f"{i}:{c}" for i, c in enumerate(hit_counts)) if hit_counts else "--"

        msg = (
            f"<b>Cenário:</b> {ds_name}<br>"
            f"<b>Amostras:</b> {total}<br>"
            f"<b>Acc:</b> {acc:.4f}<br>"
            f"<b>Distribuição preds:</b> [{dist}]<br>"
            f"<b>Acertos por classe:</b> [{hits}]"
        )

        if n_classes == 2:
            msg += f"<br><b>Matriz confusão (binária):</b> TP={tp}, TN={tn}, FP={fp}, FN={fn}"

        self._batch_html = msg
        self._refresh_output_panel()

    # ------------------------------------------------------------------
    def load_from_dict(self, payload: Dict[str, Any]) -> None:
        try:
            manifest = IAManifest.from_dict(payload)
            data = manifest.to_dict()
        except Exception as e:
            lab = QLabel(f"Manifesto inválido: {e}")
            lab.setAlignment(Qt.AlignCenter)
            self.set_workspace_widget(lab)
            return

        ident = data.get("identification") if isinstance(data.get("identification"), dict) else {}
        struct = data.get("structure") if isinstance(data.get("structure"), dict) else {}
        mut = data.get("mutation") if isinstance(data.get("mutation"), dict) else {}
        stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
        funcs = data.get("functions") if isinstance(data.get("functions"), dict) else {}

        # NÃO mostrar weights na esquerda
        struct_display = dict(struct)
        struct_display.pop("weights", None)

        _fill_grid_left_aligned(self.grid_ident, ident)
        _fill_grid_left_aligned(self.grid_struct, struct_display)
        _fill_grid_left_aligned(self.grid_mut, mut)
        _fill_grid_left_aligned(self.grid_stats, stats)

        funcs_display = dict(funcs)
        funcs_display["weights_status"] = type(struct.get("weights", None)).__name__
        _fill_grid_left_aligned(self.grid_funcs, funcs_display)

        input_size = int(struct.get("input_size") or 0)
        output_size = int(struct.get("output_size") or 0)
        self._rebuild_test_inputs(input_size, output_size)

        try:
            net = _payload_to_matrix_structure(data)
            view = NetworkView(net)
            self._current_view = view
            self.set_workspace_widget(view)
        except Exception as e:
            lab = QLabel(f"Falha ao montar estrutura: {e}")
            lab.setAlignment(Qt.AlignCenter)
            self.set_workspace_widget(lab)

    def set_workspace_widget(self, widget: QWidget) -> None:
        while self.view_container_layout.count():
            it = self.view_container_layout.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self.view_container_layout.addWidget(widget, 1)


WdCreateAnalise = WdCreateAnalisePane
