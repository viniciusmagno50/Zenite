# pages/individual/toolbar/tb_ind_toolbar.py
from __future__ import annotations

import math
import os
import random
import shutil
import stat
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolButton,
    QSizePolicy,
    QDialog,
    QComboBox,
    QLabel,
    QMessageBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QCheckBox,
)

from json_lib import BASE_DIR, load_json, save_json
from ia_context import AIContext
from ia_manifest import IAManifest
from ia_engine import NeuralNetworkEngine

# ✅ único ajuste: novo caminho do dialog
from pages.individual.dialogs.create_ind_dialog import CreateIndDialog


# ============================================================
# Estilo robusto do QComboBox da toolbar (Windows/Qt)
# ============================================================

def _apply_toolbar_combo_style(combo: QComboBox) -> None:
    """
    Aplica um estilo *local* ao QComboBox usado na toolbar do menu Individual.

    Motivo:
      - Em alguns temas/ordens de QSS, o QComboBox pode parecer mais escuro do que o esperado.
      - Aplicar um styleSheet no próprio widget garante consistência visual (Windows/Qt).

    Resultado:
      - Caixa do combo (fechado) mais clara.
      - Texto sempre branco.
      - Popup com fundo escuro e seleção visível.
    """
    try:
        # Combo fechado (mais claro que o fundo da toolbar)
        combo.setStyleSheet(
            """
            QComboBox {
                background-color: #4A4A4A;
                color: #FFFFFF;
                border: 1px solid #3A3A3A;
                border-bottom: 1px solid #282828;
                border-radius: 4px;
                padding: 6px 8px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #404040;
                color: #FFFFFF;
                selection-background-color: #646464;
                selection-color: #FFFFFF;
                outline: none;
            }
            """
        )

        # Popup (view) — reforço no Windows
        view = combo.view()
        view.setStyleSheet(
            """
            QListView {
                background-color: #404040;
                color: #FFFFFF;
                selection-background-color: #646464;
                selection-color: #FFFFFF;
                outline: none;
            }
            QListView::item {
                background: transparent;
                color: #FFFFFF;
            }
            QListView::item:selected {
                background-color: #646464;
                color: #FFFFFF;
            }
            """
        )
    except Exception:
        # Se falhar, não quebra o app
        pass


# ============================================================
# Helpers de inicialização de pesos / shapes
# ============================================================

def _xavier_uniform(fan_in: int, fan_out: int) -> float:
    fan_in = max(1, int(fan_in))
    fan_out = max(1, int(fan_out))
    a = math.sqrt(6.0 / float(fan_in + fan_out))
    return float(random.uniform(-a, a))


def _new_neuron_row(prev_size: int, fan_out: int) -> List[float]:
    """Row de neurônio: [bias] + weights(prev_size)"""
    prev_size = max(1, int(prev_size))
    bias = _xavier_uniform(prev_size, fan_out)
    w = [_xavier_uniform(prev_size, fan_out) for _ in range(prev_size)]
    return [bias] + w


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return list(x)


def _fix_manifest_shapes(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Schema usado pelo engine:
      structure.weights = [layer][neuron][bias, w1..wN]
    """
    if not isinstance(data, dict):
        return data

    s = data.get("structure") or {}
    if not isinstance(s, dict):
        s = {}

    input_size = max(1, int(s.get("input_size") or 1))

    neurons = s.get("neurons") or []
    if isinstance(neurons, int):
        neurons = [neurons]
    neurons = [max(1, int(n)) for n in _as_list(neurons)]
    if not neurons:
        neurons = [1]

    layers = len(neurons)
    output_size = max(1, int(neurons[-1]))

    activation = s.get("activation") or []
    activation = _as_list(activation)
    if len(activation) < layers:
        activation += [None] * (layers - len(activation))
    elif len(activation) > layers:
        activation = activation[:layers]

    weights = s.get("weights") or []
    weights = _as_list(weights)
    while len(weights) < layers:
        weights.append([])
    if len(weights) > layers:
        weights = weights[:layers]

    prev = input_size
    for li in range(layers):
        n_out = neurons[li]
        layer_rows = _as_list(weights[li])

        while len(layer_rows) < n_out:
            layer_rows.append(_new_neuron_row(prev, n_out))
        if len(layer_rows) > n_out:
            layer_rows = layer_rows[:n_out]

        exp_len = 1 + prev
        for ri in range(n_out):
            row = _as_list(layer_rows[ri])
            try:
                row = [float(v) for v in row]
            except Exception:
                row = []

            if len(row) < exp_len:
                if len(row) == 0:
                    row = _new_neuron_row(prev, n_out)
                else:
                    bias = float(row[0]) if len(row) >= 1 else _xavier_uniform(prev, n_out)
                    w = [float(x) for x in row[1:]]
                    while len(w) < prev:
                        w.append(_xavier_uniform(prev, n_out))
                    row = [bias] + w[:prev]
            elif len(row) > exp_len:
                row = row[:exp_len]

            layer_rows[ri] = row

        weights[li] = layer_rows
        prev = n_out

    s["input_size"] = input_size
    s["output_size"] = output_size
    s["layers"] = layers
    s["neurons"] = neurons
    s["activation"] = activation
    s["weights"] = weights

    data["structure"] = s
    return data


# ============================================================
# Diálogo simples de edição (identificação)
# ============================================================

@dataclass
class _EditResult:
    ok: bool
    identification: Dict[str, Any]


class EditIAInfoDialog(QDialog):
    def __init__(self, parent: QWidget, identification: Dict[str, Any]):
        super().__init__(parent)
        self.setWindowTitle("Editar IA (identificação)")
        self._initial = dict(identification or {})
        self._result = _EditResult(ok=False, identification=dict(self._initial))
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.ed_name = QLineEdit(str(self._initial.get("name") or ""))
        self.ed_generation = QSpinBox()
        self.ed_generation.setRange(0, 1_000_000)
        self.ed_generation.setValue(int(self._initial.get("generation") or 0))

        self.ed_version = QSpinBox()
        self.ed_version.setRange(0, 1_000_000)
        self.ed_version.setValue(int(self._initial.get("version") or 0))

        self.chk_learn = QCheckBox("learn (habilitar aprendizado)")
        self.chk_learn.setChecked(bool(self._initial.get("learn", True)))

        self.ed_desc = QLineEdit(str(self._initial.get("description") or ""))

        form.addRow("name:", self.ed_name)
        form.addRow("generation:", self.ed_generation)
        form.addRow("version:", self.ed_version)
        form.addRow("", self.chk_learn)
        form.addRow("description:", self.ed_desc)

        root.addLayout(form)

        row_btn = QHBoxLayout()
        row_btn.setSpacing(8)

        btn_cancel = QToolButton()
        btn_cancel.setText("Cancelar")
        btn_cancel.clicked.connect(self.reject)

        btn_ok = QToolButton()
        btn_ok.setText("Salvar")
        btn_ok.clicked.connect(self._on_save)

        row_btn.addWidget(btn_ok)
        row_btn.addWidget(btn_cancel)
        row_btn.addStretch(1)
        root.addLayout(row_btn)

    def _on_save(self):
        ident = dict(self._initial)
        ident["name"] = (self.ed_name.text() or "").strip() or ident.get("name")
        ident["generation"] = int(self.ed_generation.value())
        ident["version"] = int(self.ed_version.value())
        ident["learn"] = bool(self.chk_learn.isChecked())
        desc = (self.ed_desc.text() or "").strip()
        ident["description"] = desc if desc else None

        self._result = _EditResult(ok=True, identification=ident)
        self.accept()

    def get_result(self) -> _EditResult:
        return self._result


# ============================================================
# Toolbar principal
# ============================================================

class TbIndToolbar(QWidget):
    action_requested = Signal(str)
    ia_loaded = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("tb_ind_toolbar")

        self._build_ui()
        self._refresh_list(select_name=None)

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Linha 1: seleção + CRUD
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        lbl = QLabel("Selecionar IA:")
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        row1.addWidget(lbl)

        self.cmb_ias = QComboBox(self)
        self.cmb_ias.setObjectName("tb_ind_combo_existentes")
        _apply_toolbar_combo_style(self.cmb_ias)  # ✅ força mais claro
        self.cmb_ias.currentIndexChanged.connect(self._on_select_ia)
        row1.addWidget(self.cmb_ias, 1)

        self.btn_create = QToolButton(self)
        self.btn_create.setText("Criar")
        self.btn_create.clicked.connect(self._on_create)
        row1.addWidget(self.btn_create)

        self.btn_edit = QToolButton(self)
        self.btn_edit.setText("Editar")
        self.btn_edit.clicked.connect(self._on_edit)
        row1.addWidget(self.btn_edit)

        self.btn_delete = QToolButton(self)
        self.btn_delete.setText("Excluir")
        self.btn_delete.clicked.connect(self._on_delete)
        row1.addWidget(self.btn_delete)

        row1.addStretch(1)
        root.addLayout(row1)

        # Linha 2: topologia
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        self.btn_add_layer = QToolButton(self)
        self.btn_add_layer.setText("+ Camada")
        self.btn_add_layer.clicked.connect(self._on_add_layer)
        row2.addWidget(self.btn_add_layer)

        self.btn_remove_layer = QToolButton(self)
        self.btn_remove_layer.setText("- Camada")
        self.btn_remove_layer.clicked.connect(self._on_remove_layer)
        row2.addWidget(self.btn_remove_layer)

        self.cmb_neuron_target = QComboBox(self)
        self.cmb_neuron_target.setObjectName("tb_ind_combo_neuron_target")
        _apply_toolbar_combo_style(self.cmb_neuron_target)  # ✅ força mais claro
        row2.addWidget(self.cmb_neuron_target, 1)

        self.btn_add_neuron = QToolButton(self)
        self.btn_add_neuron.setText("+ Neurônio")
        self.btn_add_neuron.clicked.connect(lambda: self._change_neuron(+1))
        row2.addWidget(self.btn_add_neuron)

        self.btn_remove_neuron = QToolButton(self)
        self.btn_remove_neuron.setText("- Neurônio")
        self.btn_remove_neuron.clicked.connect(lambda: self._change_neuron(-1))
        row2.addWidget(self.btn_remove_neuron)

        row2.addStretch(1)
        root.addLayout(row2)

        self._set_controls_enabled(False)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.btn_edit.setEnabled(enabled)
        self.btn_delete.setEnabled(enabled)
        self.btn_add_layer.setEnabled(enabled)
        self.btn_remove_layer.setEnabled(enabled)
        self.cmb_neuron_target.setEnabled(enabled)
        self.btn_add_neuron.setEnabled(enabled)
        self.btn_remove_neuron.setEnabled(enabled)

    # ---------------- manifest IO ----------------
    def _list_valid_names(self) -> List[str]:
        out: List[str] = []
        base = BASE_DIR
        if base.exists() and base.is_dir():
            for sub in base.iterdir():
                if not sub.is_dir():
                    continue
                name = sub.name
                fp = sub / f"{name}.json"
                if fp.exists():
                    out.append(name)
        return sorted(out)

    def _manifest_path(self, name: str) -> Path:
        return BASE_DIR / name / f"{name}.json"

    def _load_manifest(self, name: str) -> Dict[str, Any]:
        fp = self._manifest_path(name)
        data = load_json(fp)
        if not isinstance(data, dict):
            raise ValueError("Manifesto inválido (não é dict).")
        return _fix_manifest_shapes(data)

    def _save_manifest(self, name: str, data: Dict[str, Any]) -> None:
        fp = self._manifest_path(name)
        save_json(fp, _fix_manifest_shapes(data))

    # ---------------- refresh / emit ----------------
    def _refresh_list(self, select_name: Optional[str]) -> None:
        names = self._list_valid_names()

        self.cmb_ias.blockSignals(True)
        self.cmb_ias.clear()
        self.cmb_ias.addItem("Selecionar IA", "")
        for n in names:
            self.cmb_ias.addItem(n, n)
        self.cmb_ias.blockSignals(False)

        if select_name and select_name in names:
            idx = self.cmb_ias.findData(select_name)
            self.cmb_ias.setCurrentIndex(idx)
        else:
            self.cmb_ias.setCurrentIndex(0)
            AIContext.clear()
            self._set_controls_enabled(False)
            self.cmb_neuron_target.clear()

    def _emit_loaded(self, name: str, data: Dict[str, Any]) -> None:
        AIContext.set_active_name(name)
        AIContext.set_instance(None)

        manifest = IAManifest.from_dict(data)
        engine = NeuralNetworkEngine.from_manifest(manifest)
        AIContext.set_instance(engine)

        self._rebuild_neuron_targets(data)
        self.ia_loaded.emit(manifest.to_dict())

    def _rebuild_neuron_targets(self, data: Dict[str, Any]) -> None:
        self.cmb_neuron_target.blockSignals(True)
        self.cmb_neuron_target.clear()

        s = data.get("structure") or {}
        inp = int(s.get("input_size") or 1)
        neurons = s.get("neurons") or []
        if isinstance(neurons, int):
            neurons = [neurons]
        neurons = [max(1, int(x)) for x in _as_list(neurons)]
        if not neurons:
            neurons = [1]

        self.cmb_neuron_target.addItem(f"Entrada (input_size={inp})", "input|-1")

        if len(neurons) >= 2:
            for i in range(len(neurons) - 1):
                self.cmb_neuron_target.addItem(f"Hidden {i+1} (n={neurons[i]})", f"layer|{i}")

        self.cmb_neuron_target.addItem(f"Saída (output_size={neurons[-1]})", f"layer|{len(neurons)-1}")

        self.cmb_neuron_target.blockSignals(False)

    def _current_name(self) -> Optional[str]:
        d = self.cmb_ias.currentData()
        return d if isinstance(d, str) and d else None

    # ---------------- slots ----------------
    def _on_select_ia(self, index: int) -> None:
        name = self._current_name()
        if not name:
            AIContext.clear()
            self._set_controls_enabled(False)
            self.cmb_neuron_target.clear()
            return

        try:
            data = self._load_manifest(name)
        except Exception as e:
            QMessageBox.critical(self, "Erro ao carregar IA", str(e))
            AIContext.clear()
            self._set_controls_enabled(False)
            self.cmb_neuron_target.clear()
            return

        self._set_controls_enabled(True)
        self._emit_loaded(name, data)

    def _on_create(self) -> None:
        dlg = CreateIndDialog(self)
        if dlg.exec() == QDialog.Accepted and getattr(dlg, "last_created_name", None):
            name = dlg.last_created_name
            self._refresh_list(select_name=name)
            idx = self.cmb_ias.findData(name)
            if idx >= 0:
                self.cmb_ias.setCurrentIndex(idx)
            self.action_requested.emit("create_ok")

    def _on_edit(self) -> None:
        name = self._current_name()
        if not name:
            return
        try:
            data = self._load_manifest(name)
        except Exception as e:
            QMessageBox.critical(self, "Editar IA", f"Falha ao carregar manifesto:\n{e}")
            return

        ident = dict(data.get("identification") or {})
        dlg = EditIAInfoDialog(self, ident)
        if dlg.exec() != QDialog.Accepted:
            return

        res = dlg.get_result()
        if not res.ok:
            return

        data["identification"] = dict(res.identification)
        try:
            self._save_manifest(name, data)
        except Exception as e:
            QMessageBox.critical(self, "Editar IA", f"Falha ao salvar manifesto:\n{e}")
            return

        self._refresh_list(select_name=name)
        idx = self.cmb_ias.findData(name)
        if idx >= 0:
            self.cmb_ias.setCurrentIndex(idx)
        self.action_requested.emit("edit_ok")

    def _on_delete(self) -> None:
        name = self._current_name()
        if not name:
            return

        folder = BASE_DIR / name
        if not folder.exists():
            return

        resp = QMessageBox.question(
            self,
            "Excluir IA",
            f"Excluir a IA '{name}'?\n\nIsso removerá a pasta inteira:\n{folder}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

        AIContext.clear()

        def _onerror(func, path, exc_info):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

        last_err: Optional[Exception] = None
        for _ in range(4):
            try:
                shutil.rmtree(folder, onerror=_onerror)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.15)

        if folder.exists():
            try:
                renamed = folder.with_name(f"{name}__DEL__{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                folder.rename(renamed)
                folder = renamed
                for _ in range(4):
                    try:
                        shutil.rmtree(folder, onerror=_onerror)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(0.15)
            except Exception as e:
                last_err = e

        if last_err is not None:
            QMessageBox.critical(
                self,
                "Excluir IA",
                "Falha ao excluir a pasta da IA.\n\n"
                f"Pasta: {folder}\n\nErro:\n{last_err}"
            )
            return

        self._refresh_list(select_name=None)
        self.ia_loaded.emit({})
        self.action_requested.emit("delete_ok")

    # -------- Camadas --------
    def _on_add_layer(self) -> None:
        name = self._current_name()
        if not name:
            return
        try:
            data = self._load_manifest(name)
        except Exception as e:
            QMessageBox.critical(self, "Adicionar camada", str(e))
            return

        s = data["structure"]
        inp = int(s["input_size"])
        neurons = list(s["neurons"])
        weights = list(s["weights"])

        out = neurons[-1]
        prev_hidden = neurons[-2] if len(neurons) >= 2 else inp
        new_hidden = max(1, min(256, int(round(math.sqrt(prev_hidden * out))) or 1))

        insert_idx = max(0, len(neurons) - 1)
        neurons.insert(insert_idx, new_hidden)
        weights.insert(insert_idx, [])

        data["structure"]["neurons"] = neurons
        data["structure"]["weights"] = weights
        data = _fix_manifest_shapes(data)

        s = data["structure"]
        weights = s["weights"]
        prev_size = inp if insert_idx == 0 else neurons[insert_idx - 1]
        fan_out = neurons[insert_idx]
        weights[insert_idx] = [_new_neuron_row(prev_size, fan_out) for _ in range(fan_out)]
        s["weights"] = weights
        data["structure"] = s
        data = _fix_manifest_shapes(data)

        try:
            self._save_manifest(name, data)
        except Exception as e:
            QMessageBox.critical(self, "Adicionar camada", f"Falha ao salvar:\n{e}")
            return

        self._refresh_list(select_name=name)
        idx = self.cmb_ias.findData(name)
        if idx >= 0:
            self.cmb_ias.setCurrentIndex(idx)
        self.action_requested.emit("add_layer_ok")

    def _on_remove_layer(self) -> None:
        name = self._current_name()
        if not name:
            return
        try:
            data = self._load_manifest(name)
        except Exception as e:
            QMessageBox.critical(self, "Remover camada", str(e))
            return

        s = data["structure"]
        neurons = list(s["neurons"])
        weights = list(s["weights"])

        if len(neurons) <= 1:
            QMessageBox.information(self, "Remover camada", "A rede já está no mínimo (sem hidden).")
            return

        remove_idx = len(neurons) - 2
        neurons.pop(remove_idx)
        if remove_idx < len(weights):
            weights.pop(remove_idx)

        data["structure"]["neurons"] = neurons
        data["structure"]["weights"] = weights
        data = _fix_manifest_shapes(data)

        try:
            self._save_manifest(name, data)
        except Exception as e:
            QMessageBox.critical(self, "Remover camada", f"Falha ao salvar:\n{e}")
            return

        self._refresh_list(select_name=name)
        idx = self.cmb_ias.findData(name)
        if idx >= 0:
            self.cmb_ias.setCurrentIndex(idx)
        self.action_requested.emit("remove_layer_ok")

    # -------- Neurônios --------
    def _get_target(self) -> Optional[Tuple[str, int]]:
        d = self.cmb_neuron_target.currentData()
        if d is None:
            return None
        if not isinstance(d, str):
            try:
                d = str(d)
            except Exception:
                return None
        if "|" not in d:
            return None

        kind, idx_s = d.split("|", 1)
        kind = kind.strip().lower()
        try:
            idx = int(idx_s.strip())
        except Exception:
            return None

        if kind not in ("input", "layer"):
            return None
        return kind, idx

    def _change_neuron(self, delta: int) -> None:
        name = self._current_name()
        if not name:
            return

        selected_target_data = self.cmb_neuron_target.currentData()

        tgt = self._get_target()
        if not tgt:
            QMessageBox.warning(self, "Neurônios", "Seleção inválida (target).")
            return

        kind, layer_idx = tgt

        try:
            data = self._load_manifest(name)
        except Exception as e:
            QMessageBox.critical(self, "Neurônios", str(e))
            return

        data = _fix_manifest_shapes(data)
        s = data["structure"]
        inp = int(s["input_size"])
        neurons = list(s["neurons"])
        weights = list(s["weights"])

        if kind == "input":
            new_inp = inp + delta
            if new_inp < 1:
                return

            if weights and len(neurons) >= 1:
                fan_out = neurons[0]
                for r in range(len(weights[0])):
                    row = _as_list(weights[0][r])
                    if not row:
                        row = _new_neuron_row(new_inp, fan_out)

                    bias = float(row[0])
                    w = [float(x) for x in row[1:]]

                    if delta > 0:
                        w.append(_xavier_uniform(new_inp, fan_out))
                    else:
                        if w:
                            w.pop()

                    weights[0][r] = [bias] + w

            s["input_size"] = new_inp
            s["weights"] = weights
            data["structure"] = s
            data = _fix_manifest_shapes(data)

        elif kind == "layer":
            if layer_idx < 0 or layer_idx >= len(neurons):
                QMessageBox.warning(self, "Neurônios", "Camada alvo inválida.")
                return

            old_n = neurons[layer_idx]
            new_n = old_n + delta
            if new_n < 1:
                return

            prev_size = inp if layer_idx == 0 else neurons[layer_idx - 1]
            neurons[layer_idx] = new_n

            while len(weights) < len(neurons):
                weights.append([])

            if delta > 0:
                weights[layer_idx].append(_new_neuron_row(prev_size, new_n))
            else:
                if weights[layer_idx]:
                    weights[layer_idx].pop()

            next_idx = layer_idx + 1
            if next_idx < len(weights):
                for r in range(len(weights[next_idx])):
                    row = _as_list(weights[next_idx][r])
                    if not row:
                        continue
                    bias = float(row[0])
                    w = [float(x) for x in row[1:]]

                    if delta > 0:
                        w.append(_xavier_uniform(new_n, len(weights[next_idx])))
                    else:
                        if w:
                            w.pop()

                    weights[next_idx][r] = [bias] + w

            s["neurons"] = neurons
            s["weights"] = weights
            data["structure"] = s
            data = _fix_manifest_shapes(data)
        else:
            return

        try:
            self._save_manifest(name, data)
        except Exception as e:
            QMessageBox.critical(self, "Neurônios", f"Falha ao salvar:\n{e}")
            return

        self._refresh_list(select_name=name)
        idx = self.cmb_ias.findData(name)
        if idx >= 0:
            self.cmb_ias.setCurrentIndex(idx)

        if selected_target_data is not None:
            try:
                tidx = self.cmb_neuron_target.findData(selected_target_data)
                if tidx >= 0:
                    self.cmb_neuron_target.setCurrentIndex(tidx)
            except Exception:
                pass

        self.action_requested.emit("neuron_changed_ok")
