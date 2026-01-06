from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import math

import pyqtgraph as pg
from PySide6.QtWidgets import QCheckBox, QSpinBox


class TrainingPlotManager:
    """
    Centraliza o estado e as atualizações dos gráficos do wd_training.

    Importante:
    - NÃO cria widgets de UI; apenas recebe referências já criadas.
    - Não altera layout do wd_training.
    """

    def __init__(
        self,
        *,
        plot_minmax: pg.PlotWidget,
        plot_loss_epoch: pg.PlotWidget,
        plot_lr: pg.PlotWidget,
        plot_perf: pg.PlotWidget,
        plot_dist: pg.PlotWidget,
        plot_acc: pg.PlotWidget,
        chk_plot_minmax: Optional[QCheckBox] = None,
        chk_plot_loss_epoch: Optional[QCheckBox] = None,
        chk_plot_lr: Optional[QCheckBox] = None,
        chk_plot_perf: Optional[QCheckBox] = None,
        chk_plot_dist: Optional[QCheckBox] = None,
        chk_plot_acc: Optional[QCheckBox] = None,
        chk_show_original: Optional[QCheckBox] = None,
        spin_show_top: Optional[QSpinBox] = None,
        get_active_name: Optional[Callable[[], str]] = None,
        get_generation_tops: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    ) -> None:
        self.plot_minmax = plot_minmax
        self.plot_loss_epoch = plot_loss_epoch
        self.plot_lr = plot_lr
        self.plot_perf = plot_perf
        self.plot_dist = plot_dist
        self.plot_acc = plot_acc

        self.chk_plot_minmax = chk_plot_minmax
        self.chk_plot_loss_epoch = chk_plot_loss_epoch
        self.chk_plot_lr = chk_plot_lr
        self.chk_plot_perf = chk_plot_perf
        self.chk_plot_dist = chk_plot_dist
        self.chk_plot_acc = chk_plot_acc

        self.chk_show_original = chk_show_original
        self.spin_show_top = spin_show_top

        self.get_active_name = get_active_name or (lambda: "")
        self.get_generation_tops = get_generation_tops or (lambda: [])

        # Estado (alimentado pelo wd_training)
        self.net_hist: Dict[str, Dict[str, Any]] = {}
        self.net_epoch_x: Dict[str, List[int]] = {}

        # Itens (lazy)
        self.minmax_bar: Optional[pg.BarGraphItem] = None
        self.minmax_scatter_min: Optional[pg.ScatterPlotItem] = None
        self.minmax_scatter_max: Optional[pg.ScatterPlotItem] = None

        self.loss_curves: Dict[str, pg.PlotDataItem] = {}
        self.lr_curves: Dict[str, pg.PlotDataItem] = {}
        self.perf_curves: Dict[str, pg.PlotDataItem] = {}
        self.acc_curves: Dict[str, pg.PlotDataItem] = {}

        self.dist_bars_pred: Optional[pg.BarGraphItem] = None
        self.dist_bars_hit: Optional[pg.BarGraphItem] = None

        self._setup_plots()

    def _setup_plots(self) -> None:
        # Min/max
        self.plot_minmax.showGrid(x=True, y=True)
        self.plot_minmax.setLabel("left", "prob")
        self.plot_minmax.setLabel("bottom", "classe")

        self.minmax_bar = pg.BarGraphItem(x=[], height=[], width=0.7, y0=[])
        self.plot_minmax.addItem(self.minmax_bar)

        self.minmax_scatter_min = pg.ScatterPlotItem()
        self.minmax_scatter_max = pg.ScatterPlotItem()
        self.plot_minmax.addItem(self.minmax_scatter_min)
        self.plot_minmax.addItem(self.minmax_scatter_max)

        # Outros plots (mantém estrutura atual do seu projeto)
        for p in [self.plot_loss_epoch, self.plot_lr, self.plot_perf, self.plot_acc]:
            p.showGrid(x=True, y=True)

        self.plot_dist.showGrid(x=True, y=True)

    def _palette(self) -> List[Tuple[int, int, int]]:
        return [
            (200, 200, 200),
            (120, 220, 120),
            (120, 180, 255),
            (255, 180, 120),
            (220, 120, 220),
            (255, 80, 80),
        ]

    def clear(self) -> None:
        self.net_hist.clear()
        self.net_epoch_x.clear()

        self.loss_curves.clear()
        self.lr_curves.clear()
        self.perf_curves.clear()
        self.acc_curves.clear()

        self.plot_minmax.clear()
        self.plot_loss_epoch.clear()
        self.plot_lr.clear()
        self.plot_perf.clear()
        self.plot_dist.clear()
        self.plot_acc.clear()

        # re-setup itens
        self._setup_plots()

    def record(self, *, net_epoch_x: Dict[str, List[int]], net_hist: Dict[str, Dict[str, Any]]) -> None:
        self.net_epoch_x = net_epoch_x
        self.net_hist = net_hist

    def get_display_networks(self) -> List[str]:
        out: List[str] = []
        if self.spin_show_top is None or self.chk_show_original is None:
            return out

        if self.chk_show_original.isChecked():
            base = self.get_active_name() or ""
            if base:
                out.append(base)

        top_n = int(self.spin_show_top.value())
        if top_n > 0:
            for item in (self.get_generation_tops() or [])[:top_n]:
                nm = str(item.get("name") or "")
                if nm and nm not in out:
                    out.append(nm)
        return out

    def update(self) -> None:
        display = self.get_display_networks()
        colors = self._palette()

        # ----------------- Loss por época (média) -----------------
        if self.chk_plot_loss_epoch is not None and self.chk_plot_loss_epoch.isChecked():
            for idx, nm in enumerate(display):
                hist = self.net_hist.get(nm, {})
                xs = self.net_epoch_x.get(nm, [])
                ys = list(hist.get("loss_avg", []) or [])
                if not xs or not ys:
                    continue
                pen = pg.mkPen(colors[idx % len(colors)], width=2)
                curve = self.loss_curves.get(nm)
                if curve is None:
                    curve = self.plot_loss_epoch.plot(xs, ys, pen=pen, name=nm)
                    self.loss_curves[nm] = curve
                else:
                    curve.setData(xs, ys)

        # ----------------- LR -----------------
        if self.chk_plot_lr is not None and self.chk_plot_lr.isChecked():
            for idx, nm in enumerate(display):
                hist = self.net_hist.get(nm, {})
                xs = self.net_epoch_x.get(nm, [])
                ys = list(hist.get("lr", []) or [])
                if not xs or not ys:
                    continue
                pen = pg.mkPen(colors[idx % len(colors)], width=2)
                curve = self.lr_curves.get(nm)
                if curve is None:
                    curve = self.plot_lr.plot(xs, ys, pen=pen, name=nm)
                    self.lr_curves[nm] = curve
                else:
                    curve.setData(xs, ys)

        # ----------------- Performance -----------------
        if self.chk_plot_perf is not None and self.chk_plot_perf.isChecked():
            for idx, nm in enumerate(display):
                hist = self.net_hist.get(nm, {})
                xs = self.net_epoch_x.get(nm, [])
                ys = list(hist.get("perf", []) or [])
                if not xs or not ys:
                    continue
                pen = pg.mkPen(colors[idx % len(colors)], width=2)
                curve = self.perf_curves.get(nm)
                if curve is None:
                    curve = self.plot_perf.plot(xs, ys, pen=pen, name=nm)
                    self.perf_curves[nm] = curve
                else:
                    curve.setData(xs, ys)

        # ----------------- Acurácia -----------------
        if self.chk_plot_acc is not None and self.chk_plot_acc.isChecked():
            for idx, nm in enumerate(display):
                hist = self.net_hist.get(nm, {})
                xs = self.net_epoch_x.get(nm, [])
                ys = list(hist.get("acc", []) or [])
                if not xs or not ys:
                    continue
                pen = pg.mkPen(colors[idx % len(colors)], width=2)
                curve = self.acc_curves.get(nm)
                if curve is None:
                    curve = self.plot_acc.plot(xs, ys, pen=pen, name=nm)
                    self.acc_curves[nm] = curve
                else:
                    curve.setData(xs, ys)

        # ----------------- Min/Max -----------------
        if self.chk_plot_minmax is not None and self.chk_plot_minmax.isChecked():
            # Novo contrato: barras representam as SAÍDAS (classes) da rede ativa,
            # e acumulam min/max de softmax APENAS quando a rede acertou a classe.
            base = display[0] if display else ""
            hist0 = self.net_hist.get(base, {}) if base else {}

            cls_min = list(hist0.get("class_prob_min") or [])
            cls_max = list(hist0.get("class_prob_max") or [])

            if cls_min and cls_max and len(cls_min) == len(cls_max):
                xs = list(range(len(cls_min)))
                mins = [float(v) for v in cls_min]
                maxs = [float(v) for v in cls_max]

                if self.minmax_bar is not None:
                    heights = [max(0.0, ma - mi) for mi, ma in zip(mins, maxs)]
                    self.minmax_bar.setOpts(x=xs, y0=mins, height=heights)

                if self.minmax_scatter_min is not None:
                    self.minmax_scatter_min.setData(x=xs, y=mins)
                if self.minmax_scatter_max is not None:
                    self.minmax_scatter_max.setData(x=xs, y=maxs)

                try:
                    ax = self.plot_minmax.getAxis("bottom")
                    ax.setTicks([[(i, str(i)) for i in xs]])
                except Exception:
                    pass

                try:
                    self.plot_minmax.setTitle(f"Min/Max (softmax) — {base}")
                except Exception:
                    pass
            else:
                # Fallback (legado): min/max por rede (quando não há stats por classe).
                xs = list(range(len(display)))
                mins: List[float] = []
                maxs: List[float] = []
                for nm in display:
                    hist = self.net_hist.get(nm, {})
                    mn_list = hist.get("out_min", [])
                    mx_list = hist.get("out_max", [])
                    mn = float(mn_list[-1]) if mn_list else 0.0
                    mx = float(mx_list[-1]) if mx_list else 0.0
                    mins.append(mn)
                    maxs.append(mx)

                if self.minmax_bar is not None:
                    heights = [max(0.0, float(ma) - float(mi)) for mi, ma in zip(mins, maxs)]
                    self.minmax_bar.setOpts(x=xs, y0=mins, height=heights)

                if self.minmax_scatter_min is not None:
                    self.minmax_scatter_min.setData(x=xs, y=mins)
                if self.minmax_scatter_max is not None:
                    self.minmax_scatter_max.setData(x=xs, y=maxs)

                try:
                    ax = self.plot_minmax.getAxis("bottom")
                    ax.setTicks([[(i, str(nm)) for i, nm in enumerate(display)]])
                except Exception:
                    pass

        # ----------------- Distribuição -----------------
        if self.chk_plot_dist is not None and self.chk_plot_dist.isChecked():
            base = display[0] if display else ""
            hist = self.net_hist.get(base, {}) if base else {}
            counts = list(hist.get("class_counts_last") or [])
            hits = list(hist.get("class_hits_last") or [])

            n = max(len(counts), len(hits))
            if n > 0:
                counts = (counts + [0] * n)[:n]
                hits = (hits + [0] * n)[:n]
                xs = list(range(n))

                self.plot_dist.clear()
                bar_pred = pg.BarGraphItem(x=xs, height=counts, width=0.4)
                bar_hit = pg.BarGraphItem(x=[x + 0.45 for x in xs], height=hits, width=0.4)
                self.plot_dist.addItem(bar_pred)
                self.plot_dist.addItem(bar_hit)

                try:
                    ax = self.plot_dist.getAxis("bottom")
                    ax.setTicks([[(i, str(i)) for i in xs]])
                except Exception:
                    pass
