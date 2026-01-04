from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMessageBox

from ia_context import AIContext
from json_lib import build_paths, load_json

from pages.training.state import TrainingState
from pages.training.workers import TrainWorker, GenerationalWorker, EvalWorker
from ia_generational import MutationConfig


class TrainingController:
    """
    Lógica de negócio/fluxo (datasets, threads, treino, evolução).
    A UI (wd_training) chama este controller, mantendo o layout intacto.
    """

    def __init__(self, pane, state: TrainingState):
        self.pane = pane
        self.state = state

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def reload_dataset_functions(self) -> None:
        if self.pane.cmb_dataset is None:
            return

        import importlib
        import inspect

        try:
            module = importlib.import_module("train_datasets")
        except ModuleNotFoundError:
            self.pane.cmb_dataset.clear()
            self.pane.cmb_dataset.addItem("(train_datasets.py não encontrado)", "")
            self.state.dataset_funcs = {}
            self.pane._log("[DATASET] train_datasets.py não encontrado.")
            return
        except Exception as e:
            self.pane.cmb_dataset.clear()
            self.pane.cmb_dataset.addItem("(erro ao importar train_datasets.py)", "")
            self.state.dataset_funcs = {}
            self.pane._log(f"[DATASET] Erro ao importar train_datasets.py: {e}")
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
            self.pane._log("[DATASET] Nenhuma função ds_* encontrado em train_datasets.py.")
            return

        for name, func in funcs:
            doc = (func.__doc__ or "").strip().splitlines()
            desc = doc[0].strip() if doc else ""
            label = f"{name} - {desc}" if desc else name
            self.pane.cmb_dataset.addItem(label, userData=name)

        self.pane._log("[DATASET] Cenários disponíveis: " + ", ".join(self.state.dataset_funcs.keys()))

    def get_selected_dataset_func(self) -> tuple[str, Callable]:
        if not self.state.dataset_funcs or self.pane.cmb_dataset is None:
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
            _, manifest_path, _, _ = build_paths(ia_name)
            manifest = load_json(manifest_path) or {}
            trn = manifest.get("training", {}) or {}
            lr = trn.get("learning_rate", None)
            lr_mult = trn.get("lr_mult", None)
            lr_min = trn.get("lr_min", None)

            if lr is not None and self.pane.spin_lr is not None:
                self.pane.spin_lr.blockSignals(True)
                self.pane.spin_lr.setValue(float(lr))
                self.pane.spin_lr.blockSignals(False)

            if lr_mult is not None and self.pane.spin_lr_mult is not None:
                self.pane.spin_lr_mult.blockSignals(True)
                self.pane.spin_lr_mult.setValue(float(lr_mult))
                self.pane.spin_lr_mult.blockSignals(False)

            if lr_min is not None and self.pane.spin_lr_min is not None:
                self.pane.spin_lr_min.blockSignals(True)
                self.pane.spin_lr_min.setValue(float(lr_min))
                self.pane.spin_lr_min.blockSignals(False)

            self.pane._log("[LR] Parâmetros carregados do JSON.")
        except Exception as e:
            self.pane._log(f"[LR] Falha ao carregar LR do JSON: {e}")

    # ------------------------------------------------------------------
    # Ações (botões)
    # ------------------------------------------------------------------
    def on_load_data_clicked(self) -> None:
        ia_name = AIContext.get_name()
        if ia_name:
            self.load_lr_from_json(ia_name)

        try:
            ds_name, func = self.get_selected_dataset_func()
        except Exception as e:
            QMessageBox.warning(self.pane, "Cenário de dados", str(e))
            self.pane._log(f"[DATA] Erro ao selecionar cenário: {e}")
            return

        try:
            data = list(func())
        except Exception as e:
            QMessageBox.critical(self.pane, "Erro ao gerar dados", f"Erro ao executar cenário '{ds_name}':\n{e}")
            self.pane._log(f"[DATA] Erro ao executar cenário '{ds_name}': {e}")
            return

        if not data:
            QMessageBox.warning(self.pane, "Cenário vazio", f"Cenário '{ds_name}' não retornou nenhuma amostra.")
            self.pane._log(f"[DATA] Cenário '{ds_name}' vazio.")
            return

        self.state.train_data = data
        self.state.eval_data = list(data)
        self.state.test_data = list(data)
        self.state.total_train_samples = len(data)

        # mantém compatibilidade (caso UI ainda leia isso em algum ponto)
        self.pane._train_data = self.state.train_data
        self.pane._eval_data = self.state.eval_data
        self.pane._test_data = self.state.test_data
        self.pane._total_train_samples = self.state.total_train_samples

        self.pane._log(f"[OK] Dataset '{ds_name}' carregado: {len(data)} amostras.")

    def on_prepare_clicked(self) -> None:
        if not self.state.train_data:
            QMessageBox.warning(self.pane, "Treino", "Carregue os dados primeiro (1).")
            return
        self.pane._clear_histories()
        self.pane._log("[OK] Treino preparado.")

    def on_stop_clicked(self) -> None:
        if getattr(self.pane, "_train_worker", None) is not None:
            self.pane._train_worker.stop()
            self.pane._log("[INFO] Solicitação de parada enviada.")

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

        limit = int(self.pane.spin_eval_limit.value()) if self.pane.spin_eval_limit else 0

        worker = EvalWorker(network_name=name, data=data, limit=limit)
        thread = QThread(self.pane)
        worker.moveToThread(thread)

        worker.log.connect(self.pane._log)
        worker.error.connect(self.pane._on_worker_error)
        worker.finished.connect(self.pane._on_eval_finished)

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.pane._eval_thread = thread
        self.pane._eval_worker = worker
        thread.start()

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

        epochs = int(self.pane.spin_epochs.value()) if self.pane.spin_epochs else 1
        lr = float(self.pane.spin_lr.value()) if self.pane.spin_lr else 0.01
        lr_mult = float(self.pane.spin_lr_mult.value()) if self.pane.spin_lr_mult else 1.0
        lr_min = float(self.pane.spin_lr_min.value()) if self.pane.spin_lr_min else 0.0
        shuffle = bool(self.pane.chk_shuffle.isChecked()) if self.pane.chk_shuffle else True
        update_every = int(self.pane.spin_data_update.value()) if self.pane.spin_data_update else 1000
        eval_limit = int(self.pane.spin_eval_limit.value()) if self.pane.spin_eval_limit else 0
        out_idx = int(self.pane.spin_output_index.value()) if self.pane.spin_output_index else 0

        self.state.total_epochs = epochs
        self.pane._total_epochs = epochs

        self.pane._log(f"[TRAIN] Iniciando treino: {name} | epochs={epochs}, lr={lr}")

        if self.pane.btn_train:
            self.pane.btn_train.setEnabled(False)
        if self.pane.btn_stop:
            self.pane.btn_stop.setEnabled(True)

        thread = QThread(self.pane)

        if self.pane.chk_train_generations is not None and self.pane.chk_train_generations.isChecked():
            mutation_cfg = MutationConfig(
                allow_add_layers=bool(self.pane.chk_allow_add_layers.isChecked()) if self.pane.chk_allow_add_layers else True,
                allow_remove_layers=bool(self.pane.chk_allow_remove_layers.isChecked()) if self.pane.chk_allow_remove_layers else True,
                max_layer_delta=int(self.pane.spin_layer_delta.value()) if self.pane.spin_layer_delta else 1,
                allow_add_neurons=bool(self.pane.chk_allow_add_neurons.isChecked()) if self.pane.chk_allow_add_neurons else True,
                allow_remove_neurons=bool(self.pane.chk_allow_remove_neurons.isChecked()) if self.pane.chk_allow_remove_neurons else True,
                max_neuron_delta=int(self.pane.spin_neuron_delta.value()) if self.pane.spin_neuron_delta else 5,
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
                generations=int(self.pane.spin_generations.value()) if self.pane.spin_generations else 1,
                population=int(self.pane.spin_population.value()) if self.pane.spin_population else 1,
                mutation_cfg=mutation_cfg,
                update_every_n=update_every,
                output_index=out_idx,
                batch_size=16,
            )
            worker.moveToThread(thread)

            worker.log.connect(self.pane._log)
            worker.error.connect(self.pane._on_worker_error)
            worker.finished.connect(self.pane._on_train_finished)
            worker.epoch_metrics.connect(self.pane._on_epoch_metrics)
            worker.progress.connect(self.pane._on_gen_progress)
            worker.tops_updated.connect(self.pane._on_gen_tops)

            thread.started.connect(worker.run)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)

            self.pane._train_thread = thread
            self.pane._train_worker = worker
            thread.start()
            return

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

        worker.log.connect(self.pane._log)
        worker.error.connect(self.pane._on_worker_error)
        worker.finished.connect(self.pane._on_train_finished)
        worker.epoch_metrics.connect(self.pane._on_epoch_metrics)
        worker.progress_epochs.connect(self.pane._on_train_progress)
        worker.stopped.connect(self.pane._on_train_stopped)

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self.pane._train_thread = thread
        self.pane._train_worker = worker
        thread.start()

    def on_evolve_clicked(self) -> None:
        parent = AIContext.get_name()
        if not parent:
            self.pane._log("[ERRO] Nenhuma IA ativa para evoluir.")
            return

        if not self.state.best_candidate:
            self.pane._log("[INFO] Nenhuma melhor candidata disponível ainda. Treine gerações primeiro.")
            return

        best_name = str(self.state.best_candidate.get("name") or "")
        best_acc = self.state.best_candidate.get("acc")
        best_acc = float(best_acc) if best_acc is not None else -1.0

        cur_acc = -1.0
        h = self.state.net_hist.get(parent, {})
        if h and h.get("acc"):
            try:
                cur_acc = float(h.get("acc")[-1])
            except Exception:
                cur_acc = -1.0

        if best_acc <= cur_acc:
            self.pane._log(
                f"[INFO] Melhor candidata ({best_name}) não supera a original. "
                f"acc_best={best_acc:.4f} <= acc_orig={cur_acc:.4f}"
            )
            return

        try:
            p_folder, _, _ = build_paths(parent)
            src_folder, _, _ = build_paths(best_name)

            backup_dir = p_folder / "backup_before_evolve"
            backup_dir.mkdir(parents=True, exist_ok=True)

            import shutil
            for fn in ("manifest.json", "weights.json", "structure.json"):
                src = src_folder / fn
                dst = p_folder / fn
                if dst.exists():
                    shutil.copy2(dst, backup_dir / fn)
                if src.exists():
                    shutil.copy2(src, dst)

            self.pane._log(f"[OK] Evolução aplicada: '{parent}' substituída por '{best_name}'.")
        except Exception as e:
            self.pane._log(f"[ERRO] Falha ao aplicar evolução: {e}")
