# main_windows.py
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame

from module_loader import load_class


class MainWindows(QWidget):
    """
    Host do lado direito:
      - Área superior: Toolbar (ex: TbIndToolbar)
      - Área inferior: Page/Data (ex: WdCreateAnalisePane / WdTrainingPane)

    Carregamento via spec "module:Class".
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("main_windows")

        self._toolbar_container = QFrame(self)
        self._toolbar_container.setObjectName("right_toolbar_container")
        self._toolbar_container.setFrameShape(QFrame.NoFrame)

        self._data_container = QFrame(self)
        self._data_container.setObjectName("right_data_container")
        self._data_container.setFrameShape(QFrame.NoFrame)

        self._toolbar_layout = QVBoxLayout(self._toolbar_container)
        self._toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self._toolbar_layout.setSpacing(0)

        self._data_layout = QVBoxLayout(self._data_container)
        self._data_layout.setContentsMargins(0, 0, 0, 0)
        self._data_layout.setSpacing(0)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        root.addWidget(self._toolbar_container, 0)
        root.addWidget(self._data_container, 1)

        self._toolbar_widget: Optional[QWidget] = None
        self._data_widget: Optional[QWidget] = None

    # -------------------------
    # API pública (novo padrão)
    # -------------------------
    def load_toolbar(self, spec: str) -> None:
        self._toolbar_widget = self._replace_widget(
            layout=self._toolbar_layout,
            current=self._toolbar_widget,
            spec=spec,
            container=self._toolbar_container,
        )

    def load_data(self, spec: str) -> None:
        self._data_widget = self._replace_widget(
            layout=self._data_layout,
            current=self._data_widget,
            spec=spec,
            container=self._data_container,
        )

    def toolbar_widget(self) -> Optional[QWidget]:
        return self._toolbar_widget

    def data_widget(self) -> Optional[QWidget]:
        return self._data_widget

    # -------------------------
    # Interno
    # -------------------------
    def _replace_widget(self, layout, current: Optional[QWidget], spec: str, container: QWidget) -> QWidget:
        if current is not None:
            current.setParent(None)
            current.deleteLater()

        WidgetCls = load_class(spec)

        # construtor padrão: (parent=None) ou (parent, ...)
        # Vamos passar o container como parent para manter ownership no Qt.
        try:
            w = WidgetCls(container)
        except TypeError:
            # fallback: construtor sem args
            w = WidgetCls()

        w.setParent(container)
        layout.addWidget(w, 1, alignment=Qt.AlignmentFlag.AlignTop)
        return w
