# left_toolbar.py
from __future__ import annotations

from pathlib import Path
from functools import partial
from typing import Callable, List, Tuple, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QWidget

from appconfig import ensure_config, get_style_path
from ui_functions import UIFunctions


class LeftToolbar(QFrame):
    """
    Barra lateral esquerda (menu).
    Emite specs no formato "modulo:Classe" para o module_loader.

    Sinais:
      - request_toolbar(spec: str)
      - request_data(spec: str)
    """
    request_toolbar = Signal(str)
    request_data = Signal(str)

    def __init__(self, app, cfg, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app = app
        self.cfg = cfg

        self.setObjectName("left_toolbar")
        self._montar_ui()

    # -------------------------------
    # Montagem de interface
    # -------------------------------
    def _montar_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Título (opcional)
        titulo = QLabel("Menu")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setObjectName("left_toolbar_title")
        root.addWidget(titulo)

        # ===== Seção: Rede Individual =====
        self.container_individual = self._criar_container_n2([
            ("Criação e Análise",     partial(self._abrir_ind_criacao_analise)),
            ("Treinamento",           partial(self._abrir_ind_treinamento)),
            ("Estrutura",             partial(self._abrir_ind_estrutura)),
            ("Aplicação em RealTime", partial(self._abrir_ind_realtime)),
        ])
        btn_rede_individual = UIFunctions.create_button(
            "Rede Individual",
            nivel=1,
            on_click=partial(self._toggle_container, self.container_individual)
        )
        root.addWidget(btn_rede_individual)
        root.addWidget(self.container_individual)

        # ===== Seção: Rede Composta =====
        self.container_composta = self._criar_container_n2([
            ("Criação e Análise",     partial(self._abrir_comp_criacao_analise)),
            ("Treinamento",           partial(self._abrir_comp_treinamento)),
            ("Estrutura",             partial(self._abrir_comp_estrutura)),
            ("Aplicação em RealTime", partial(self._abrir_comp_realtime)),
        ])
        btn_rede_composta = UIFunctions.create_button(
            "Rede Composta",
            nivel=1,
            on_click=partial(self._toggle_container, self.container_composta)
        )
        root.addWidget(btn_rede_composta)
        root.addWidget(self.container_composta)

        # ===== Seção: Configuração =====
        btn_config = UIFunctions.create_button(
            "Configuração",
            nivel=1,
            on_click=self._abrir_configuracao
        )
        root.addWidget(btn_config)

        btn_reload = UIFunctions.create_button(
            "Recarregar Estilos",
            nivel=1,
            tooltip="Reaplica o QSS",
            on_click=self._recarregar_estilos
        )
        root.addWidget(btn_reload)

        root.addStretch(1)

    def _criar_container_n2(self, itens: List[Tuple[str, Callable]]) -> QWidget:
        container = QFrame(self)
        container.setObjectName("left_toolbar_container_n2")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        for texto, callback in itens:
            btn = UIFunctions.create_button(texto, nivel=2, on_click=callback)
            layout.addWidget(btn)

        container.setVisible(False)
        return container

    # -------------------------------
    # Ações / Handlers
    # -------------------------------
    def _toggle_container(self, container: QWidget):
        container.setVisible(not container.isVisible())

    def _recarregar_estilos(self):
        try:
            ensure_config()
            style_path = get_style_path(self.cfg)
            qss = Path(style_path).read_text(encoding="utf-8")
            self.app.setStyleSheet(qss)
        except Exception as e:
            print(f"[WARN] Falha ao recarregar estilos: {e}")

    # -------------------------------
    # Helper de emissão (spec completa)
    # -------------------------------
    def _emit_view(self, toolbar_spec: str, data_spec: str) -> None:
        self.request_toolbar.emit(toolbar_spec)
        self.request_data.emit(data_spec)

    # -------------------------------
    # Navegação: Rede Individual
    # -------------------------------
    def _abrir_ind_criacao_analise(self):
        self._emit_view(
            "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
            "pages.wd_create_analise:WdCreateAnalisePane"
        )

    def _abrir_ind_treinamento(self):
        self._emit_view(
            "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
            "pages.wd_training:WdTrainingPane"
        )

    def _abrir_ind_estrutura(self):
        self._emit_view(
            "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
            "pages.wd_create_analise:WdCreateAnalisePane"
        )

    def _abrir_ind_realtime(self):
        self._emit_view(
            "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
            "pages.wd_create_analise:WdCreateAnalisePane"
        )

    # -------------------------------
    # Navegação: Rede Composta (placeholders)
    # -------------------------------
    def _abrir_comp_criacao_analise(self):
        self._emit_view(
            "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
            "pages.wd_create_analise:WdCreateAnalisePane"
        )

    def _abrir_comp_treinamento(self):
        self._emit_view(
            "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
            "pages.wd_training:WdTrainingPane"
        )

    def _abrir_comp_estrutura(self):
        self._emit_view(
            "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
            "pages.wd_create_analise:WdCreateAnalisePane"
        )

    def _abrir_comp_realtime(self):
        self._emit_view(
            "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
            "pages.wd_create_analise:WdCreateAnalisePane"
        )

    # -------------------------------
    # Configuração
    # -------------------------------
    def _abrir_configuracao(self):
        self._emit_view(
            "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar",
            "pages.wd_create_analise:WdCreateAnalisePane"
        )
