from __future__ import annotations

from typing import Callable, Optional, Any, Dict, List
from pathlib import Path

# Compatibilidade Qt (PyQt6 ou PySide6)
try:  # PyQt6
    from PyQt6.QtCore import QThread, QObject, pyqtSlot as Slot
    from PyQt6.QtWidgets import QMessageBox
except Exception:  # PySide6
    from PySide6.QtCore import QThread, QObject, Slot
    from PySide6.QtWidgets import QMessageBox

from ia_context import AIContext
from json_lib import build_paths, load_json

from pages.training.state import TrainingState
from pages.training.workers import TrainWorker, GenerationalWorker, EvalWorker
from ia_generational import MutationConfig


class TrainingController(QObject):
    """
    Controller do treino (caminho oficial).

    IMPORTANTE (correção do crash 0xC0000005):
    - Este controller PRECISA ser QObject para que conexões de signals vindos
      de workers em outra thread sejam executadas no thread da UI (QueuedConnection).
    - Se for classe Python "pura", os callbacks podem rodar dentro da thread do worker,
      causando crash nativo ao mexer em UI/pyqtgraph.
    """

    def __init__(self, pane: Any, state: TrainingState):
        super().__init__(pane)  # parent = widget (thread da UI)
        self.pane = pane
        self.state = state
        self.plot_manager = None

    # ------------------------------------------------------------------
    # Injeções opcionais
    # ------------------------------------------------------------------
    def set_plot_manager(self, plot_manager: Any) -> None:
        self.plot_manager = plot_manager

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def reload_dataset_functions(self) -> None:
        if getattr(self.pane, "cmb_dataset", None) is None:
            return

        import importlib
        import inspect

        try:
            module = importlib.import_module("train_datasets")
        except ModuleNotFoundError:
            self.pane.cmb_dataset.clear()
            self.pane.cmb_dataset.addItem("(train_datasets.py não encontrado)", "")
            self.state.dataset_funcs = {}
            self._log("[DATASET] train_datasets.py não encontrado.")
            return
        except Exception as e:
            self.pane.cmb_dataset.clear()
            self.pane.cmb_dataset.addItem("(erro ao importar train_datasets.py)", "")
            self.state.dataset_funcs = {}
            self._log(f"[DATASET] Erro ao importar train_datasets.py: {e}")
            return

        funcs = []
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("ds_"):
                funcs.append((name, obj))

        funcs.sort(key=lambda t: t[0])
        self.state.dataset_funcs = {name: f for name, f in funcs}

        self.pane.cmb_dataset.clear()

        if not funcs:
            self.pane.cmb_dataset.addItem("(nenhum ds_* encontrado)", "")
            self._log("[DATASET] Nenhuma função ds_* encontrada em train_datasets.py.")
            return

        for name, func in funcs:
            doc = (func.__doc__ or "").strip().splitlines()
            desc = doc[0].strip() if doc else ""
            label = f"{name} - {desc}" if desc else name
            self.pane.cmb_dataset.addItem(label, userData=name)

        self._log("[DATASET] Cenários disponíveis: " + ", ".join(self.state.dataset_funcs.keys()))

    def get_selected_dataset_func(self) -> tuple[str, Callable]:
        if not self.state.dataset_funcs or getattr(self.pane, "cmb_dataset", None) is None:
            raise RuntimeError("Nenhum cenário ds_* carregado.")

        idx = self.pane.cmb_dataset.currentIndex()
        if idx < 0:
            raise RuntimeError("Nenhum cenário selecionado.")

        name = self.pane.cmb_dataset.itemData(idx)
        if not name:
            raise RuntimeError("Seleção de cenário inválida.")

        func = self.state.dataset_funcs.get(name)
        if func is None:
            raise RuntimeError(f"Cenário '{name}' não encontrado internamente.")

        return str(name), func

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if hasattr(self.pane, "_log"):
            self.pane._log(msg)
        else:
            print(msg)

    def ensure_active_network(self) -> Optional[str]:
        name = AIContext.get_name()
        if not name:
            QMessageBox.warning(
                self.pane,
                "IA não selecionada",
                "Selecione ou crie uma IA antes de continuar.",
            )
            return None
        return str(name)

    def load_lr_from_json(self, ia_name: str) -> None:
        try:
            _, manifest_path, _ = build_paths(ia_name)  # 3 retornos
            manifest = load_json(manifest_path) or {}
            trn = manifest.get("training", {}) or {}

            lr = trn.get("learning_rate", None)
            lr_mult = trn.get("lr_mult", None)
            lr_min = trn.get("lr_min", None)

            if lr is not None and getattr(self.pane, "spin_lr", None) is not None:
                self.pane.spin_lr.blockSignals(True)
                self.pane.spin_lr.setValue(float(lr))
                self.pane.spin_lr.blockSignals(False)

            if lr_mult is not None and getattr(self.pane, "spin_lr_mult", None) is not None:
                self.pane.spin_lr_mult.blockSignals(True)
                self.pane.spin_lr_mult.setValue(float(lr_mult))
                self.pane.spin_lr_mult.blockSignals(False)

            if lr_min is not None and getattr(self.pane, "spin_lr_min", None) is not None:
                self.pane.spin_lr_min.blockSignals(True)
                self.pane.spin_lr_min.setValue(float(lr_min))
                self.pane.spin_lr_min.blockSignals(False)

            self._log("[LR] Parâmetros carregados do JSON.")
        except Exception as e:
            self._log(f"[LR] Falha ao carregar LR do JSON: {e}")

    # ------------------------------------------------------------------
    # Ações (botões)
    # ------------------------------------------------------------------
    @Slot()
    def on_load_data_clicked(self) -> None:
        ia_name = AIContext.get_name()
        if ia_name:
            self.load_lr_from_json(ia_name)

        try:
            ds_name, func = self.get_selected_dataset_func()
        except Exception as e:
            QMessageBox.warning(self.pane, "Cenário de dados", str(e))
            self._log(f"[DATA] Erro ao selecionar cenário: {e}")
            return

        try:
            data = list(func())
        except Exception as e:
            QMessageBox.critical(self.pane, "Erro ao gerar dados", f"Erro ao executar cenário '{ds_name}':\n{e}")
            self._log(f"[DATA] Erro ao executar cenário '{ds_name}': {e}")
            return

        if not data:
            QMessageBox.warning(self.pane, "Cenário vazio", f"Cenário '{ds_name}' não retornou nenhuma amostra.")
            self._log(f"[DATA] Cenário '{ds_name}' vazio.")
            return

        self.state.train_data = data
        self.state.eval_data = list(data)
        self.state.test_data = list(data)
        self.state.total_train_samples = len(data)

        # compat com wd_training legado
        setattr(self.pane, "_train_data", self.state.train_data)
        setattr(self.pane, "_eval_data", self.state.eval_data)
        setattr(self.pane, "_test_data", self.state.test_data)
        setattr(self.pane, "_total_train_samples", self.state.total_train_samples)

        self._log(f"[OK] Dataset '{ds_name}' carregado: {len(data)} amostras.")

    @Slot()
    def on_prepare_clicked(self) -> None:
        if not self.state.train_data:
            QMessageBox.warning(self.pane, "Treino", "Carregue os dados primeiro (1).")
            return

        self.state.reset_histories()
        self.state.reset_generations()

        if hasattr(self.pane, "_clear_histories"):
            self.pane._clear_histories()

        if self.plot_manager is not None:
            try:
                self.plot_manager.clear()
            except Exception:
                pass

        self._log("[OK] Treino preparado.")

    @Slot()
    def on_stop_clicked(self) -> None:
        w = getattr(self.pane, "_train_worker", None)
        if w is not None:
            try:
                w.stop()
            except Exception:
                pass
            self._log("[INFO] Solicitação de parada enviada.")

    @Slot()
    def on_eval_clicked(self) -> None:
        name = self.ensure_active_network()
        if not name:
            return

        data = self.state.eval_data or self.state.train_data
        if not data:
            QMessageBox.warning(self.pane, "Avaliação", "Carregue os dados primeiro (1).")
            return

        if getattr(self.pane, "_eval_thread", None) is not None:
            QMessageBox.information(self.pane, "Avaliação em andamento", "Já existe uma avaliação em execução.")
            return

        limit = int(self.pane.spin_eval_limit.value()) if getattr(self.pane, "spin_eval_limit", None) else 0

        worker = EvalWorker(network_name=name, data=data, limit=limit)
        thread = QThread(self.pane)
        worker.moveToThread(thread)

        worker.log.connect(self._log)
        if hasattr(self.pane, "_on_worker_error"):
            worker.error.connect(self.pane._on_worker_error)
        else:
            worker.error.connect(lambda e: self._log(f"[ERRO] {e}"))

        if hasattr(self.pane, "_on_eval_finished"):
            worker.finished.connect(self.pane._on_eval_finished)
        else:
            worker.finished.connect(lambda _: self._log("[EVAL] Finalizado."))

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.pane._eval_thread = thread
        self.pane._eval_worker = worker
        thread.start()

    # -------------------- Handlers internos (SLOTS) --------------------
    @Slot(object)
    def _handle_epoch_metrics(self, metrics: Any) -> None:
        # Agora ESTE SLOT roda no thread da UI (por ser QObject)
        if self.plot_manager is not None:
            try:
                self.plot_manager.record(metrics)
            except Exception:
                pass

        if hasattr(self.pane, "_on_epoch_metrics"):
            self.pane._on_epoch_metrics(metrics)

        if self.plot_manager is not None:
            try:
                total_samples = int(self.state.total_train_samples or 1)
                total_epochs = int(self.state.total_epochs or getattr(metrics, "total_epochs", 1) or 1)
                self.plot_manager.maybe_update(
                    metrics=metrics,
                    total_train_samples=total_samples,
                    total_epochs=total_epochs,
                    cmb_update_mode=getattr(self.pane, "cmb_update_mode", None),
                    spin_update_samples=getattr(self.pane, "spin_data_update", None),
                )
            except Exception:
                pass

    @Slot(list)
    def _handle_gen_tops(self, tops: List[Dict[str, Any]]) -> None:
        self.state.set_generation_tops(tops or [])
        if hasattr(self.pane, "_on_gen_tops"):
            self.pane._on_gen_tops(tops)

    @Slot(str, int, int, float)
    def _handle_gen_progress(self, name: str, step: int, total: int, loss: float) -> None:
        if hasattr(self.pane, "_on_gen_progress"):
            self.pane._on_gen_progress(name, step, total, loss)

    @Slot(int, int)
    def _handle_train_progress(self, ep: int, total: int) -> None:
        if hasattr(self.pane, "_on_train_progress"):
            self.pane._on_train_progress(ep, total)

    @Slot()
    def _handle_train_finished(self) -> None:
        if hasattr(self.pane, "_on_train_finished"):
            self.pane._on_train_finished()

    @Slot()
    def _handle_train_stopped(self) -> None:
        if hasattr(self.pane, "_on_train_stopped"):
            self.pane._on_train_stopped()

    # -------------------- START TREINO --------------------
    @Slot()
    def on_train_clicked(self) -> None:
        name = self.ensure_active_network()
        if not name:
            return

        if not self.state.train_data:
            QMessageBox.warning(
                self.pane,
                "Dados de treino ausentes",
                "Nenhum conjunto de treino foi definido.\nUse 'Carregar dados' primeiro.",
            )
            return

        if getattr(self.pane, "_train_thread", None) is not None:
            QMessageBox.information(self.pane, "Treino em andamento", "Já existe um treino em execução.")
            return

        epochs = int(self.pane.spin_epochs.value()) if getattr(self.pane, "spin_epochs", None) else 1
        lr = float(self.pane.spin_lr.value()) if getattr(self.pane, "spin_lr", None) else 0.01
        lr_mult = float(self.pane.spin_lr_mult.value()) if getattr(self.pane, "spin_lr_mult", None) else 1.0
        lr_min = float(self.pane.spin_lr_min.value()) if getattr(self.pane, "spin_lr_min", None) else 0.0
        shuffle = bool(self.pane.chk_shuffle.isChecked()) if getattr(self.pane, "chk_shuffle", None) else True
        update_every = int(self.pane.spin_data_update.value()) if getattr(self.pane, "spin_data_update", None) else 1000
        eval_limit = int(self.pane.spin_eval_limit.value()) if getattr(self.pane, "spin_eval_limit", None) else 0
        out_idx = int(self.pane.spin_output_index.value()) if getattr(self.pane, "spin_output_index", None) else 0

        self.state.total_epochs = epochs
        setattr(self.pane, "_total_epochs", epochs)

        self._log(f"[TRAIN] Iniciando treino: {name} | epochs={epochs}, lr={lr}")

        if getattr(self.pane, "btn_train", None):
            self.pane.btn_train.setEnabled(False)
        if getattr(self.pane, "btn_stop", None):
            self.pane.btn_stop.setEnabled(True)

        thread = QThread(self.pane)

        # ------------- Geração -------------
        if getattr(self.pane, "chk_train_generations", None) is not None and self.pane.chk_train_generations.isChecked():
            mutation_cfg = MutationConfig(
                allow_add_layers=bool(self.pane.chk_allow_add_layers.isChecked()) if getattr(self.pane, "chk_allow_add_layers", None) else True,
                allow_remove_layers=bool(self.pane.chk_allow_remove_layers.isChecked()) if getattr(self.pane, "chk_allow_remove_layers", None) else True,
                max_layer_delta=int(self.pane.spin_layer_delta.value()) if getattr(self.pane, "spin_layer_delta", None) else 1,
                allow_add_neurons=bool(self.pane.chk_allow_add_neurons.isChecked()) if getattr(self.pane, "chk_allow_add_neurons", None) else True,
                allow_remove_neurons=bool(self.pane.chk_allow_remove_neurons.isChecked()) if getattr(self.pane, "chk_allow_remove_neurons", None) else True,
                max_neuron_delta=int(self.pane.spin_neuron_delta.value()) if getattr(self.pane, "spin_neuron_delta", None) else 5,
            )

            worker = GenerationalWorker(
                parent_name=name,
                train_data=self.state.train_data,
                epochs_per_individual=epochs,
                learning_rate=lr,
                lr_mult=lr_mult,
                lr_min=lr_min,
                shuffle=shuffle,
                eval_limit=eval_limit,
                generations=int(self.pane.spin_generations.value()) if getattr(self.pane, "spin_generations", None) else 1,
                population=int(self.pane.spin_population.value()) if getattr(self.pane, "spin_population", None) else 1,
                mutation_cfg=mutation_cfg,
                update_every_n=update_every,
                output_index=out_idx,
                batch_size=16,
            )
            worker.moveToThread(thread)

            worker.log.connect(self._log)
            if hasattr(self.pane, "_on_worker_error"):
                worker.error.connect(self.pane._on_worker_error)
            else:
                worker.error.connect(lambda e: self._log(f"[ERRO] {e}"))

            worker.finished.connect(self._handle_train_finished)
            worker.epoch_metrics.connect(self._handle_epoch_metrics)
            worker.progress.connect(self._handle_gen_progress)
            worker.tops_updated.connect(self._handle_gen_tops)

            thread.started.connect(worker.run)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)

            self.pane._train_thread = thread
            self.pane._train_worker = worker
            thread.start()
            return

        # ------------- Treino normal -------------
        worker = TrainWorker(
            network_name=name,
            train_data=self.state.train_data,
            test_data=self.state.test_data if self.state.test_data else self.state.train_data,
            epochs=epochs,
            learning_rate=lr,
            lr_mult=lr_mult,
            lr_min=lr_min,
            eval_limit=eval_limit,
            shuffle=shuffle,
            progress_update_n=update_every,
            output_index=out_idx,
            batch_size=16,
        )
        worker.moveToThread(thread)

        worker.log.connect(self._log)
        if hasattr(self.pane, "_on_worker_error"):
            worker.error.connect(self.pane._on_worker_error)
        else:
            worker.error.connect(lambda e: self._log(f"[ERRO] {e}"))

        worker.finished.connect(self._handle_train_finished)
        worker.epoch_metrics.connect(self._handle_epoch_metrics)
        worker.progress_epochs.connect(self._handle_train_progress)
        worker.stopped.connect(self._handle_train_stopped)

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.pane._train_thread = thread
        self.pane._train_worker = worker
        thread.start()

    # ------------------------------------------------------------------
    # Evolução: aplicar melhor candidato (geracao/*) sobre o parent
    # ------------------------------------------------------------------
    def _find_generation_manifest(self, parent_name: str, child_name: str) -> Optional[Path]:
        try:
            parent_folder, _, _ = build_paths(parent_name)
            cand = parent_folder / "geracao" / f"{child_name}.json"
            if cand.exists():
                return cand
        except Exception:
            pass

        try:
            _, manifest_path, _ = build_paths(child_name)
            if manifest_path.exists():
                return manifest_path
        except Exception:
            pass

        return None

    @Slot()
    def on_evolve_clicked(self) -> None:
        parent = AIContext.get_name()
        if not parent:
            self._log("[ERRO] Nenhuma IA ativa para evoluir.")
            return

        if not self.state.best_candidate:
            self._log("[INFO] Nenhuma melhor candidata disponível ainda. Treine gerações primeiro.")
            return

        best_name = str(self.state.best_candidate.get("name") or "")
        if not best_name:
            self._log("[ERRO] Melhor candidata sem 'name'.")
            return

        best_acc = self.state.best_candidate.get("accuracy", None)
        if best_acc is None:
            best_acc = self.state.best_candidate.get("acc", None)
        try:
            best_acc_f = float(best_acc) if best_acc is not None else -1.0
        except Exception:
            best_acc_f = -1.0

        cur_acc_f = -1.0
        h = self.state.net_hist.get(str(parent), {})
        if h and h.get("acc"):
            try:
                cur_acc_f = float(h.get("acc")[-1])
            except Exception:
                cur_acc_f = -1.0

        if best_acc_f <= cur_acc_f:
            self._log(
                f"[INFO] Melhor candidata ({best_name}) não supera a original. "
                f"acc_best={best_acc_f:.4f} <= acc_orig={cur_acc_f:.4f}"
            )
            return

        try:
            parent_folder, parent_manifest_path, _ = build_paths(str(parent))

            child_manifest_path = self._find_generation_manifest(str(parent), best_name)
            if child_manifest_path is None or not child_manifest_path.exists():
                self._log(f"[ERRO] Manifest do candidato não encontrado: {best_name}")
                return

            # backup
            backup_dir = parent_folder / "backup_before_evolve"
            backup_dir.mkdir(parents=True, exist_ok=True)

            import shutil
            if parent_manifest_path.exists():
                shutil.copy2(parent_manifest_path, backup_dir / parent_manifest_path.name)

            shutil.copy2(child_manifest_path, parent_manifest_path)

            self._log(f"[OK] Evolução aplicada: '{parent}' substituída por '{best_name}' (manifest).")

        except Exception as e:
            self._log(f"[ERRO] Falha ao aplicar evolução: {e}")
