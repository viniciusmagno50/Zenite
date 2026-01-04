from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import math

import pyqtgraph as pg
from PySide6.QtWidgets import QCheckBox, QSpinBox


class TrainingPlotManager:
    """
    Centraliza o estado e as atualizações dos gráficos do wd_training.

    Responsabilidades:
    - Decidir quais redes aparecem (original + Top-N)
    - Aplicar política de habilitação (desempenho)
    - Atualizar/limpar os PlotWidgets/Itens do pyqtgraph

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
        dist_item: pg.BarGraphItem,
        minmax_bar: Optional[pg.BarGraphItem],
        minmax_scatter_min: Optional[pg.ScatterPlotItem],
        minmax_scatter_max: Optional[pg.ScatterPlotItem],
        chk_plot_minmax: Optional[QCheckBox],
        chk_plot_loss_epoch: Optional[QCheckBox],
        chk_plot_lr: Optional[QCheckBox],
        chk_plot_perf: Optional[QCheckBox],
        chk_plot_dist: Optional[QCheckBox],
        chk_plot_acc: Optional[QCheckBox],
        spin_show_top: Optional[QSpinBox],
        chk_show_original: Optional[QCheckBox],
        get_active_name: Callable[[], str],
        get_generation_tops: Callable[[], List[Dict[str, Any]]],
        net_epoch_x: Optional[Dict[str, List[int]]] = None,
        net_hist: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        # widgets
        self.plot_minmax = plot_minmax
        self.plot_loss_epoch = plot_loss_epoch
        self.plot_lr = plot_lr
        self.plot_perf = plot_perf
        self.plot_dist = plot_dist
        self.plot_acc = plot_acc

        # itens
        self.dist_item = dist_item
        self.minmax_bar = minmax_bar
        self.minmax_scatter_min = minmax_scatter_min
        self.minmax_scatter_max = minmax_scatter_max

        # controles
        self.chk_plot_minmax = chk_plot_minmax
        self.chk_plot_loss_epoch = chk_plot_loss_epoch
        self.chk_plot_lr = chk_plot_lr
        self.chk_plot_perf = chk_plot_perf
        self.chk_plot_dist = chk_plot_dist
        self.chk_plot_acc = chk_plot_acc

        self.spin_show_top = spin_show_top
        self.chk_show_original = chk_show_original

        # callbacks de contexto
        self.get_active_name = get_active_name
        self.get_generation_tops = get_generation_tops

        # estado/históricos (compartilhados com wd_training)
        self.net_epoch_x: Dict[str, List[int]] = net_epoch_x if net_epoch_x is not None else {}
        self.net_hist: Dict[str, Dict[str, Any]] = net_hist if net_hist is not None else {}

        # caches de curvas (por rede)
        self._curves_loss_epoch: Dict[str, pg.PlotDataItem] = {}
        self._curves_lr: Dict[str, pg.PlotDataItem] = {}
        self._curves_perf: Dict[str, pg.PlotDataItem] = {}
        self._curves_acc: Dict[str, pg.PlotDataItem] = {}

        self._plots_enabled: bool = True

    @property
    def plots_enabled(self) -> bool:
        return self._plots_enabled

    # ------------------------------------------------------------------
    # Política de habilitação (desempenho)
    # ------------------------------------------------------------------
    def apply_policy(self) -> bool:
        """Retorna True se gráficos devem atualizar; se False, limpa e desliga."""
        if self.spin_show_top is None or self.chk_show_original is None:
            self._plots_enabled = True
            return self._plots_enabled

        show_top = int(self.spin_show_top.value())
        show_orig = bool(self.chk_show_original.isChecked())
        self._plots_enabled = (show_top > 0) or show_orig

        if not self._plots_enabled:
            self.update(force_clear=True)

        return self._plots_enabled

    # ------------------------------------------------------------------
    # Redes exibidas
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Registro de métricas (opcional; use se quiser mover essa responsabilidade)
    # ------------------------------------------------------------------
    def record(self, metrics: Any) -> None:
        """
        Registra métricas no dicionário compartilhado net_hist/net_epoch_x.
        Espera atributos:
        - network_name, epoch_index
        - loss_final, loss_avg, learning_rate, samples_per_sec, accuracy
        - out_min, out_max
        - class_counts, class_hits
        """
        name = str(getattr(metrics, "network_name", "") or "")
        if not name:
            return

        epoch_num = int(getattr(metrics, "epoch_index", 0)) + 1
        self.net_epoch_x.setdefault(name, []).append(epoch_num)

        hist = self.net_hist.setdefault(
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

        hist["loss_final"].append(float(getattr(metrics, "loss_final", 0.0)))
        hist["loss_avg"].append(float(getattr(metrics, "loss_avg", 0.0)))
        hist["lr"].append(float(getattr(metrics, "learning_rate", 0.0)))
        hist["perf"].append(float(getattr(metrics, "samples_per_sec", 0.0)))
        hist["acc"].append(float(getattr(metrics, "accuracy", 0.0)))
        hist["out_min"].append(float(getattr(metrics, "out_min", 0.0)))
        hist["out_max"].append(float(getattr(metrics, "out_max", 0.0)))

        cc = getattr(metrics, "class_counts", None) or []
        ch = getattr(metrics, "class_hits", None) or []
        hist["class_counts_last"] = list(cc)
        hist["class_hits_last"] = list(ch)

    # ------------------------------------------------------------------
    # Frequência de atualização (opcional; idem ao comportamento antigo)
    # ------------------------------------------------------------------
    def maybe_update(
        self,
        *,
        metrics: Any,
        total_train_samples: int,
        total_epochs: int,
        cmb_update_mode: Any = None,
        spin_update_samples: Any = None,
    ) -> None:
        mode = "visual"
        if cmb_update_mode is not None:
            try:
                data = cmb_update_mode.currentData()
                if isinstance(data, str):
                    mode = data
            except Exception:
                pass

        epoch_num = int(getattr(metrics, "epoch_index", 0)) + 1
        total_epochs_eff = int(getattr(metrics, "total_epochs", 0) or total_epochs or 1)
        total_epochs_eff = max(1, total_epochs_eff)

        if mode == "perf":
            step = max(1, total_epochs_eff // 10)
            if epoch_num == total_epochs_eff or epoch_num % step == 0 or epoch_num == 1:
                self.update()
            return

        update_n = 1
        if spin_update_samples is not None:
            try:
                update_n = max(1, int(spin_update_samples.value()))
            except Exception:
                update_n = 1

        total_samples = max(1, int(total_train_samples or 1))
        epochs_step = max(1, math.ceil(update_n / total_samples))

        if epoch_num == 1 or epoch_num == total_epochs_eff or epoch_num % epochs_step == 0:
            self.update()

    # ------------------------------------------------------------------
    # Atualização de plots
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.net_hist.clear()
        self.net_epoch_x.clear()
        self.update(force_clear=True)

    def _palette(self) -> List[Tuple[int, int, int]]:
        # Paleta estável (igual ao layout/original do wd_training)
        return [
            (0, 170, 255),
            (255, 170, 0),
            (0, 220, 120),
            (220, 0, 220),
            (255, 80, 80),
            (200, 200, 60),
            (160, 160, 255),
            (120, 255, 255),
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

    def update(self, *, force_clear: bool = False) -> None:
        if force_clear:
            # min/max
            if self.minmax_bar is not None:
                self.minmax_bar.setOpts(x=[], y0=[], height=[])
            if self.minmax_scatter_min is not None:
                self.minmax_scatter_min.setData(x=[], y=[])
            if self.minmax_scatter_max is not None:
                self.minmax_scatter_max.setData(x=[], y=[])

            # curvas
            for d in (self._curves_loss_epoch, self._curves_lr, self._curves_perf, self._curves_acc):
                for item in d.values():
                    try:
                        item.setData([], [])
                    except Exception:
                        pass

            # dist
            try:
                self.dist_item.setOpts(x=[], height=[])
            except Exception:
                pass
            return

        # Política (desempenho)
        if self.spin_show_top is not None and self.chk_show_original is not None:
            show_top = int(self.spin_show_top.value())
            show_orig = bool(self.chk_show_original.isChecked())
            self._plots_enabled = (show_top > 0) or show_orig

        if not self._plots_enabled:
            return

        display = self.get_display_networks()
        colors = self._palette()

        # ----------------- Loss por época (média) -----------------
        if self.chk_plot_loss_epoch is not None and self.chk_plot_loss_epoch.isChecked():
            for idx, nm in enumerate(display):
                hist = self.net_hist.get(nm, {})
                xs = self.net_epoch_x.get(nm, [])
                ys = hist.get("loss_avg", [])
                curve = self._ensure_curve(self._curves_loss_epoch, self.plot_loss_epoch, nm, colors[idx % len(colors)])
                curve.setData(xs, ys)
        else:
            for item in self._curves_loss_epoch.values():
                item.setData([], [])

        # ----------------- LR -----------------
        if self.chk_plot_lr is not None and self.chk_plot_lr.isChecked():
            for idx, nm in enumerate(display):
                hist = self.net_hist.get(nm, {})
                xs = self.net_epoch_x.get(nm, [])
                ys = hist.get("lr", [])
                curve = self._ensure_curve(self._curves_lr, self.plot_lr, nm, colors[idx % len(colors)])
                curve.setData(xs, ys)
        else:
            for item in self._curves_lr.values():
                item.setData([], [])

        # ----------------- Performance (samples/s) -----------------
        if self.chk_plot_perf is not None and self.chk_plot_perf.isChecked():
            for idx, nm in enumerate(display):
                hist = self.net_hist.get(nm, {})
                xs = self.net_epoch_x.get(nm, [])
                ys = hist.get("perf", [])
                curve = self._ensure_curve(self._curves_perf, self.plot_perf, nm, colors[idx % len(colors)])
                curve.setData(xs, ys)
        else:
            for item in self._curves_perf.values():
                item.setData([], [])

        # ----------------- Accuracy -----------------
        if self.chk_plot_acc is not None and self.chk_plot_acc.isChecked():
            for idx, nm in enumerate(display):
                hist = self.net_hist.get(nm, {})
                xs = self.net_epoch_x.get(nm, [])
                ys = hist.get("acc", [])
                curve = self._ensure_curve(self._curves_acc, self.plot_acc, nm, colors[idx % len(colors)])
                curve.setData(xs, ys)
        else:
            for item in self._curves_acc.values():
                item.setData([], [])

        # ----------------- Min/Max -----------------
        if self.chk_plot_minmax is not None and self.chk_plot_minmax.isChecked():
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
        else:
            if self.minmax_bar is not None:
                self.minmax_bar.setOpts(x=[], y0=[], height=[])
            if self.minmax_scatter_min is not None:
                self.minmax_scatter_min.setData(x=[], y=[])
            if self.minmax_scatter_max is not None:
                self.minmax_scatter_max.setData(x=[], y=[])

        # ----------------- Distribuição -----------------
        if self.chk_plot_dist is not None and self.chk_plot_dist.isChecked():
            chosen = None
            for nm in display:
                hist = self.net_hist.get(nm, {})
                cc = hist.get("class_counts_last", [])
                if cc:
                    chosen = (nm, cc)
                    break
            if chosen:
                _, cc = chosen
                xs = list(range(len(cc)))
                self.dist_item.setOpts(x=xs, height=cc, width=0.6)
            else:
                self.dist_item.setOpts(x=[], height=[])
        else:
            self.dist_item.setOpts(x=[], height=[])
