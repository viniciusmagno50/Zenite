from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TrainingState:
    """
    Estado do treinamento (fonte única de verdade para o wd_training).

    Objetivo:
    - Não depender de UI.
    - Guardar dataset atual, históricos, tops de geração e melhor candidato.
    - Permitir reset claro e previsível.

    Observação:
    - A UI pode ler esses campos, mas NÃO deve escrever diretamente em históricos;
      ideal é o Controller/PlotManager registrarem.
    """

    # --- datasets (carregados dinamicamente de train_datasets.py)
    dataset_funcs: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    # --- dados em memória
    train_data: List[Any] = field(default_factory=list)
    eval_data: List[Any] = field(default_factory=list)
    test_data: List[Any] = field(default_factory=list)
    total_train_samples: int = 0

    # --- progresso do treino
    total_epochs: int = 0

    # --- históricos por rede (para plots)
    # net_epoch_x[name] = [1,2,3...]
    net_epoch_x: Dict[str, List[int]] = field(default_factory=dict)

    # net_hist[name] = dict com listas: loss_final/loss_avg/lr/perf/acc/out_min/out_max...
    net_hist: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- gerações / tops
    generation_tops: List[Dict[str, Any]] = field(default_factory=list)  # [{name, acc/accuracy, loss, ...}, ...]
    best_candidate: Optional[Dict[str, Any]] = None

    # --- logs / debugging (opcional)
    last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Reset / limpeza
    # ------------------------------------------------------------------
    def reset_data(self) -> None:
        self.train_data = []
        self.eval_data = []
        self.test_data = []
        self.total_train_samples = 0

    def reset_histories(self) -> None:
        self.net_epoch_x.clear()
        self.net_hist.clear()

    def reset_generations(self) -> None:
        self.generation_tops = []
        self.best_candidate = None

    def reset_all(self) -> None:
        self.reset_data()
        self.reset_histories()
        self.reset_generations()
        self.total_epochs = 0
        self.last_error = None

    # ------------------------------------------------------------------
    # Tops / melhor candidato
    # ------------------------------------------------------------------
    @staticmethod
    def _get_acc(item: Dict[str, Any]) -> float:
        v = item.get("accuracy", None)
        if v is None:
            v = item.get("acc", None)
        try:
            return float(v) if v is not None else -1.0
        except Exception:
            return -1.0

    @staticmethod
    def _get_loss(item: Dict[str, Any]) -> float:
        # suporta vários nomes
        for k in ("loss", "final_loss", "loss_final", "avg_loss", "loss_avg"):
            if k in item and item[k] is not None:
                try:
                    return float(item[k])
                except Exception:
                    pass
        return 1e18

    def set_generation_tops(self, tops: List[Dict[str, Any]]) -> None:
        self.generation_tops = list(tops or [])
        self.best_candidate = self._pick_best(self.generation_tops)

    def _pick_best(self, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not items:
            return None

        # melhor = maior acc, desempate = menor loss
        best = None
        best_key = None
        for it in items:
            acc = self._get_acc(it)
            loss = self._get_loss(it)
            key = (acc, -loss)  # loss menor => -loss maior
            if best is None or key > best_key:
                best = it
                best_key = key
        return best
