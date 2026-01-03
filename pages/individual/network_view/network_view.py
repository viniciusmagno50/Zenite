from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPen, QColor, QBrush, QFont, QPainter
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsSimpleTextItem,
    QGraphicsItem,
)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def _color_for_weight(w: float) -> QColor:
    """
    Peso -> cor:
      negativo: vermelho
      positivo: verde
      magnitude -> alpha
    """
    mag = _clamp(abs(w), 0.0, 2.0) / 2.0  # 0..1
    alpha = int(60 + mag * 170)  # 60..230
    if w >= 0:
        return QColor(80, 220, 120, alpha)
    return QColor(240, 90, 90, alpha)


class InfoBubble(QGraphicsSimpleTextItem):
    def __init__(self, text: str, parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(text, parent)
        self.setBrush(QBrush(QColor(230, 230, 230)))
        self.setZValue(10)
        font = QFont()
        font.setPointSize(9)
        self.setFont(font)


class NeuronItem(QGraphicsEllipseItem):
    """
    Neurônio com pos() = centro real (via setPos), essencial para desenhar conexões.
    """
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        bias: Optional[float] = None,
        color: Optional[QColor] = None,
    ) -> None:
        super().__init__()
        self.bias = bias
        self.radius = radius

        self.setRect(-radius, -radius, radius * 2, radius * 2)
        self.setPos(QPointF(x, y))

        base_col = color or QColor(80, 120, 240, 190)
        self.setBrush(QBrush(base_col))
        self.setPen(QPen(QColor(30, 30, 30, 180), 1.2))

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        self._bubble: Optional[InfoBubble] = None

    def hoverEnterEvent(self, event) -> None:
        txt = "Neuron"
        if self.bias is not None:
            txt += f"\nBias: {self.bias:.6g}"

        self._bubble = InfoBubble(txt)

        p = self.pos()
        self._bubble.setPos(p.x() + self.radius + 8, p.y() - self.radius)

        if self.scene() is not None:
            self.scene().addItem(self._bubble)

        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._bubble is not None and self.scene() is not None:
            self.scene().removeItem(self._bubble)
        self._bubble = None
        super().hoverLeaveEvent(event)


class NetworkView(QGraphicsView):
    """
    Compatível com o layout antigo:
      - construtor aceita `net: dict` (layers/weights/biases)
      - método set_network(net)
    """

    def __init__(self, net: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # ✅ sem fundo preto: deixa transparente e herda o fundo do app/QSS
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0, 0)))
        self.setStyleSheet("QGraphicsView { background: transparent; border: none; }")
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.setObjectName("network_view")
        self.setRenderHint(QPainter.Antialiasing, True)

        self._net: Optional[Dict[str, Any]] = None
        if net is not None:
            self.set_network(net)
        else:
            self._normalize_scene_rect()

    def clear(self) -> None:
        self._scene.clear()
        self._net = None
        self._normalize_scene_rect()

    def set_network(self, net: Dict[str, Any]) -> None:
        self._net = net
        self._rebuild()

    # -------------------------
    def _rebuild(self) -> None:
        self._scene.clear()

        if not isinstance(self._net, dict):
            self._normalize_scene_rect()
            return

        layers = self._net.get("layers") or []
        weights = self._net.get("weights") or []
        biases = self._net.get("biases") or []

        if not isinstance(layers, list) or len(layers) < 2:
            self._normalize_scene_rect()
            return

        # Normaliza tamanhos
        layer_sizes: List[int] = []
        for n in layers:
            try:
                layer_sizes.append(max(1, int(n)))
            except Exception:
                layer_sizes.append(1)

        n_layers = len(layer_sizes)
        max_neurons = max(layer_sizes) if layer_sizes else 1

        # -------------------------
        # Visual "compacto e organizado"
        # - espaçamento vertical SEMPRE igual
        # - centralização horizontal e vertical
        # -------------------------
        radius = float(_clamp(18 - (max_neurons - 6) * 1.0, 8, 18))
        y_gap = max(44.0, radius * 4.0)   # passo vertical fixo (igual para todas)
        x_gap = max(150.0, radius * 8.0)  # passo horizontal fixo

        # eixo X centrado
        x0 = -((n_layers - 1) * x_gap) / 2.0
        xs = [x0 + i * x_gap for i in range(n_layers)]

        # eixo Y centrado para max_neurons
        y0 = -((max_neurons - 1) * y_gap) / 2.0
        y_all = [y0 + j * y_gap for j in range(max_neurons)]

        # -------------------------
        # Cria neurônios, centralizando camadas menores dentro da grade vertical
        # -------------------------
        node_items: List[List[NeuronItem]] = [[] for _ in range(n_layers)]

        for li, size in enumerate(layer_sizes):
            # pega um "slice" central de y_all com tamanho = size
            start = (max_neurons - size) // 2
            ys = [y_all[start + j] for j in range(size)]

            for ni, y in enumerate(ys):
                if li == 0:
                    color = QColor(120, 200, 255, 190)  # Entrada
                    b = None
                else:
                    color = QColor(80, 220, 140, 190) if li < n_layers - 1 else QColor(250, 190, 90, 190)
                    b = None
                    try:
                        if (li - 1) < len(biases) and isinstance(biases[li - 1], list) and ni < len(biases[li - 1]):
                            b = _safe_float(biases[li - 1][ni], 0.0)
                    except Exception:
                        b = None

                it = NeuronItem(xs[li], y, radius=radius, bias=b, color=color)
                it.setZValue(2)
                self._scene.addItem(it)
                node_items[li].append(it)

        # -------------------------
        # Conexões (linhas)
        # weights[li] = matriz (n_out x n_in) ligando layer li -> li+1
        # -------------------------
        for li in range(n_layers - 1):
            W = weights[li] if isinstance(weights, list) and li < len(weights) else None
            if not isinstance(W, list):
                continue

            src_layer = node_items[li]
            dst_layer = node_items[li + 1]

            for dst_i, dst_item in enumerate(dst_layer):
                row = W[dst_i] if dst_i < len(W) and isinstance(W[dst_i], list) else []
                p_dst = dst_item.pos()

                for src_i, src_item in enumerate(src_layer):
                    p_src = src_item.pos()

                    w = _safe_float(row[src_i], 0.0) if src_i < len(row) else 0.0
                    col = _color_for_weight(w)
                    thick = 1.0 + _clamp(abs(w), 0.0, 2.0) * 1.2

                    line = QGraphicsLineItem(p_src.x(), p_src.y(), p_dst.x(), p_dst.y())
                    line.setPen(QPen(col, thick))
                    line.setZValue(0)
                    self._scene.addItem(line)

        # -------------------------
        # Rótulos abaixo de cada camada
        # -------------------------
        y_bottom = y_all[-1] if y_all else 0.0
        y_label = y_bottom + radius + 26.0

        label_font = QFont()
        label_font.setPointSize(10)
        label_font.setBold(True)

        for li in range(n_layers):
            if li == 0:
                label = "Entrada"
            elif li == n_layers - 1:
                label = "Saída"
            else:
                label = f"Camada {li}"

            t = QGraphicsSimpleTextItem(label)
            t.setFont(label_font)
            t.setBrush(QBrush(QColor(220, 220, 220, 200)))
            t.setZValue(3)

            # centraliza o texto no X da camada
            br = t.boundingRect()
            t.setPos(xs[li] - br.width() / 2.0, y_label)
            self._scene.addItem(t)

        # Ajusta área e zoom automaticamente
        self._normalize_scene_rect()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _normalize_scene_rect(self) -> None:
        items_rect: QRectF = self._scene.itemsBoundingRect()
        if not items_rect.isValid() or items_rect.isEmpty():
            items_rect = QRectF(-320, -240, 640, 480)
        self._scene.setSceneRect(items_rect.adjusted(-80, -80, +80, +80))

    def showEvent(self, event):
        super().showEvent(event)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
