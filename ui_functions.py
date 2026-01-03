# ui_functions.py
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QPushButton


class UIFunctions:
    """
    Funções auxiliares de UI.

    Regras importantes:
      - NÃO aplicar stylesheet diretamente no botão (para não sobrescrever o QSS global).
      - Usar objectName = "nivel1" | "nivel2" | "nivel3" para o QSS estilizar.
    """

    @staticmethod
    def create_button(
        text: str,
        *,
        nivel: int = 1,
        on_click: Optional[Callable[[], None]] = None,
        tooltip: Optional[str] = None,
        checkable: bool = False,
        checked: bool = False,
    ) -> QPushButton:
        btn = QPushButton(text)

        # QSS do projeto usa "#nivel1/#nivel2/#nivel3"
        nivel = int(nivel) if nivel in (1, 2, 3) else 1
        btn.setObjectName(f"nivel{nivel}")

        btn.setCursor(QCursor(Qt.PointingHandCursor))
        btn.setCheckable(bool(checkable))
        if checkable:
            btn.setChecked(bool(checked))

        if tooltip:
            btn.setToolTip(str(tooltip))

        if on_click is not None:
            btn.clicked.connect(on_click)

        return btn

    @staticmethod
    def create_left_button(
        text: str,
        on_click: Optional[Callable[[], None]] = None,
        *,
        nivel: int = 1,
        tooltip: Optional[str] = None,
    ) -> QPushButton:
        """
        Compatibilidade com versões antigas do projeto que chamavam create_left_button().
        """
        return UIFunctions.create_button(
            text,
            nivel=nivel,
            on_click=on_click,
            tooltip=tooltip,
        )
