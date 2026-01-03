# pages/individual/dialogs/create_ind_dialog.py
from __future__ import annotations

import math
import random
import re
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QTextEdit,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QSpinBox,
)

from json_lib import build_paths, ensure_dir, save_json
from ia_manifest import IAManifest, Structure


class CreateIndDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Criar IA Individual")
        self.setModal(True)

        self.last_created_name: Optional[str] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        row = 0

        # Nome
        lbl_nome = QLabel("Nome da IA:")
        self.edit_nome = QLineEdit()
        self.edit_nome.setPlaceholderText("Ex.: minha_rede_teste")
        self.edit_nome.setClearButtonEnabled(True)
        self.edit_nome.setFocus()
        grid.addWidget(lbl_nome, row, 0)
        grid.addWidget(self.edit_nome, row, 1, 1, 2)
        row += 1

        # Geração
        lbl_ger = QLabel("Geração:")
        self.edit_ger = QLineEdit("1")
        self.edit_ger.setValidator(QIntValidator(0, 1_000_000, self))
        self.edit_ger.setMaximumWidth(120)
        grid.addWidget(lbl_ger, row, 0)
        grid.addWidget(self.edit_ger, row, 1)
        row += 1

        # Versão
        lbl_ver = QLabel("Versão:")
        self.edit_ver = QLineEdit("1")
        self.edit_ver.setValidator(QIntValidator(0, 1_000_000, self))
        self.edit_ver.setMaximumWidth(120)
        grid.addWidget(lbl_ver, row, 0)
        grid.addWidget(self.edit_ver, row, 1)
        row += 1

        # Learn
        lbl_learn = QLabel("Modo:")
        self.cmb_learn = QComboBox()
        self.cmb_learn.addItem("Treinável (learn=True)", True)
        self.cmb_learn.addItem("Somente inferência (learn=False)", False)
        grid.addWidget(lbl_learn, row, 0)
        grid.addWidget(self.cmb_learn, row, 1, 1, 2)
        row += 1

        # Inputs
        lbl_in = QLabel("Entradas (input_size):")
        self.spin_in = QSpinBox()
        self.spin_in.setRange(1, 4096)
        self.spin_in.setValue(1)
        grid.addWidget(lbl_in, row, 0)
        grid.addWidget(self.spin_in, row, 1)
        row += 1

        # Outputs
        lbl_out = QLabel("Saídas (output_size):")
        self.spin_out = QSpinBox()
        self.spin_out.setRange(1, 1024)
        self.spin_out.setValue(2)
        grid.addWidget(lbl_out, row, 0)
        grid.addWidget(self.spin_out, row, 1)
        row += 1

        # Topologia simples
        lbl_topo = QLabel("Topologia:")
        self.cmb_topo = QComboBox()
        self.cmb_topo.addItem("Sem hidden (apenas saída)", "no_hidden")
        self.cmb_topo.addItem("1 hidden (heurística)", "one_hidden")
        grid.addWidget(lbl_topo, row, 0)
        grid.addWidget(self.cmb_topo, row, 1, 1, 2)
        row += 1

        # Descrição
        lbl_desc = QLabel("Descrição:")
        self.txt_desc = QTextEdit()
        self.txt_desc.setPlaceholderText("Descrição opcional…")
        self.txt_desc.setFixedHeight(70)
        grid.addWidget(lbl_desc, row, 0, Qt.AlignTop)
        grid.addWidget(self.txt_desc, row, 1, 1, 2)
        row += 1

        root.addLayout(grid)

        # Botões
        row_btn = QHBoxLayout()
        row_btn.addStretch(1)

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        row_btn.addWidget(btn_cancel)

        btn_ok = QPushButton("Criar")
        btn_ok.clicked.connect(self._on_create)
        row_btn.addWidget(btn_ok)

        root.addLayout(row_btn)

    def _sanitize(self, name: str) -> str:
        name = (name or "").strip()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
        name = name.strip("_-")
        return name

    def _xavier_uniform(self, fan_in: int, fan_out: int) -> float:
        fan_in = max(1, int(fan_in))
        fan_out = max(1, int(fan_out))
        a = math.sqrt(6.0 / float(fan_in + fan_out))
        return float(random.uniform(-a, a))

    def _init_layer(self, fan_in: int, fan_out: int) -> List[List[float]]:
        layer = []
        for _ in range(fan_out):
            bias = self._xavier_uniform(fan_in, fan_out)
            w = [self._xavier_uniform(fan_in, fan_out) for _ in range(fan_in)]
            layer.append([bias] + w)
        return layer

    def _on_create(self) -> None:
        nome = self._sanitize(self.edit_nome.text())
        if not nome:
            QMessageBox.warning(self, "Criar IA", "Informe um nome válido.")
            return

        gen = int(self.edit_ger.text() or 1)
        ver = int(self.edit_ver.text() or 1)
        learn = bool(self.cmb_learn.currentData())

        input_size = int(self.spin_in.value())
        output_size = int(self.spin_out.value())
        topo = self.cmb_topo.currentData()

        neurons: List[int] = []
        if topo == "no_hidden":
            neurons = [output_size]
        else:
            hidden = max(2, min(256, int(round(math.sqrt(input_size * output_size)))))
            neurons = [hidden, output_size]

        layers = len(neurons)
        activation = ["relu"] * max(0, layers - 1) + ["softmax"]

        weights: List[List[List[float]]] = []
        prev = input_size
        for n_out in neurons:
            weights.append(self._init_layer(prev, int(n_out)))
            prev = int(n_out)

        structure = Structure(
            input_size=input_size,
            neurons=neurons,
            activation=activation,
            weights=weights,
        )

        payload = IAManifest(
            schema_version=1,
            identification={
                "name": f"{nome}.json",
                "generation": gen,
                "version": ver,
                "description": (self.txt_desc.toPlainText() or "").strip() or None,
                "learn": learn,
            },
            structure=structure,
            mutation={
                "rate": 0.1,
                "chance": 0.5,
            },
            stats={
                "accuracy": None,
                "loss": None,
                "last_train_time": None,
            },
            functions={},
        ).to_dict()

        try:
            pasta, arquivo, _ = build_paths(nome)
            ensure_dir(pasta)
            save_json(arquivo, payload)
        except Exception as e:
            QMessageBox.critical(self, "Criar IA", f"Falha ao salvar manifesto:\n{e}")
            return

        self.last_created_name = nome

        msg = QMessageBox(self)
        msg.setWindowTitle("Criar IA")
        msg.setText(
            f"Manifesto criado com sucesso!\n\n"
            f"Pasta: {pasta}\nArquivo: {arquivo.name}\n\n"
            f"Resumo estrutura:\n"
            f"• Entradas: {payload['structure']['input_size']}  |  "
            f"Saídas: {payload['structure']['output_size']}\n"
            f"• Camadas: {payload['structure']['layers']}  |  "
            f"Neurônios por camada: {payload['structure']['neurons']}\n"
            f"• Ativações: {payload['structure']['activation']}"
        )
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

        self.accept()
