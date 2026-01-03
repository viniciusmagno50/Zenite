# main.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QSplitter, QWidget

from appconfig import ensure_config, get_style_path, get_startup_specs
from left_toolbar import LeftToolbar
from main_windows import MainWindows

from ia_context import AIContext
from ia_engine import build_engine


class MainWindow(QMainWindow):
    def __init__(self, app: QApplication, cfg, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.app = app
        self.cfg = cfg

        self.setWindowTitle("DataAnalise")
        self.setMinimumSize(1200, 720)

        # Split principal: esquerda (menu) / direita (conteúdo)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        self.left = LeftToolbar(app, cfg, splitter)
        self.right = MainWindows(splitter)

        splitter.addWidget(self.left)
        splitter.addWidget(self.right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 920])

        self.setCentralWidget(splitter)

        # Conexões de navegação
        self.left.request_toolbar.connect(self.right.load_toolbar)
        self.left.request_data.connect(self.right.load_data)

        # Startup
        toolbar_spec, data_spec = get_startup_specs(cfg)
        self.right.load_toolbar(toolbar_spec)
        self.right.load_data(data_spec)


def apply_styles(app: QApplication, style_path_str: str) -> None:
    style_path = Path(style_path_str)

    if style_path.exists() and style_path.is_file():
        app.setStyleSheet(style_path.read_text(encoding="utf-8"))
    else:
        app.setStyleSheet("")


def main() -> None:
    cfg = ensure_config()

    app = QApplication(sys.argv)

    # Factory do contexto global de IA
    AIContext.set_factory(build_engine)

    # Estilos
    apply_styles(app, get_style_path(cfg))

    win = MainWindow(app, cfg)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
