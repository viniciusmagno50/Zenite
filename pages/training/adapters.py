from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Any, Tuple
import importlib
import inspect


# =============================================================================
# Parte existente (mantida): AIContext + datasets
# =============================================================================
def get_active_ai_name(AIContext: Any) -> str:
    """
    Obtém o nome ativo da IA de forma resiliente a mudanças de API.
    """
    if hasattr(AIContext, "get_name") and callable(getattr(AIContext, "get_name")):
        try:
            return str(AIContext.get_name() or "")
        except Exception:
            pass

    if hasattr(AIContext, "active_name") and callable(getattr(AIContext, "active_name")):
        try:
            return str(AIContext.active_name() or "")
        except Exception:
            pass

    return str(getattr(AIContext, "_active_name", "") or "")


def get_dataset_functions(
    module_name: str = "train_datasets",
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Callable[[], Sequence[Any]]]:
    """
    Retorna { 'ds_xxx': func } para todas as funções do módulo que começam com ds_.
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        if log:
            log(f"[DATASET] Erro ao importar {module_name}.py: {e}")
        return {}

    funcs: Dict[str, Callable[[], Sequence[Any]]] = {}
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("ds_"):
            funcs[name] = obj

    if log:
        log(f"[DATASET] {len(funcs)} dataset(s) carregado(s).")

    return funcs


def load_dataset(
    dataset_name: str,
    dataset_funcs: Dict[str, Callable[[], Sequence[Any]]],
) -> List[Any]:
    """
    Executa o dataset selecionado e retorna lista.
    """
    fn = dataset_funcs.get(dataset_name)
    if fn is None:
        raise KeyError(f"Dataset '{dataset_name}' não encontrado.")
    return list(fn())


def torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())  # type: ignore[attr-defined]
    except Exception:
        return False


# =============================================================================
# NOVO: Camada de compatibilidade para treino/avaliação (assinaturas instáveis)
# =============================================================================
def _import_attr(module_name: str, attr_name: str) -> Any:
    mod = importlib.import_module(module_name)
    return getattr(mod, attr_name)


def _get_signature(fn: Callable[..., Any]) -> Optional[inspect.Signature]:
    try:
        return inspect.signature(fn)
    except Exception:
        return None


def _has_varkw(sig: Optional[inspect.Signature]) -> bool:
    if sig is None:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _filter_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove kwargs que não existem na assinatura, exceto se o backend aceita **kwargs.
    """
    sig = _get_signature(fn)
    if _has_varkw(sig):
        return dict(kwargs)
    if sig is None:
        return dict(kwargs)

    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _pick_first_present(sig: Optional[inspect.Signature], *names: str) -> Optional[str]:
    if sig is None:
        return None
    params = sig.parameters
    for n in names:
        if n in params:
            return n
    return None


def _set_if_target_exists(sig: Optional[inspect.Signature], out: Dict[str, Any], target: str, value: Any) -> None:
    if sig is None:
        out[target] = value
        return
    if _has_varkw(sig) or target in sig.parameters:
        out[target] = value


def _wrap_progress_callback(progress_cb: Optional[Callable[..., Any]]) -> Optional[Callable[..., Any]]:
    """
    Normaliza callbacks para suportar variações de assinatura no backend.

    O seu worker usa progress_callback(name, step, total_steps, epoch, total_epochs, loss) -> bool
    Mas alguns backends podem chamar:
      - cb(step, total, loss)
      - cb(epoch, total_epochs, loss)
      - cb(metrics_dict)
      - cb(name, epoch, total_epochs, loss)
    Então criamos um wrapper que aceita *args/**kwargs e tenta adaptar.
    """
    if progress_cb is None:
        return None

    def _cb(*args: Any, **kwargs: Any) -> Any:
        try:
            return progress_cb(*args, **kwargs)
        except TypeError:
            # Tenta converter chamadas comuns para a assinatura do worker.
            try:
                name = kwargs.get("name", "")
                step = kwargs.get("step", kwargs.get("step_counter", 0))
                total_steps = kwargs.get("total", kwargs.get("total_steps", 0))
                epoch = kwargs.get("epoch", 0)
                total_epochs = kwargs.get("total_epochs", 0)
                loss = kwargs.get("loss", kwargs.get("loss_val", 0.0))
                return progress_cb(name, step, total_steps, epoch, total_epochs, loss)
            except Exception:
                # Último fallback: não interromper treino
                return True
        except Exception:
            # Não derrubar o treino por erro de callback
            return True

    return _cb


def _wrap_on_epoch_metrics(on_epoch_metrics: Optional[Callable[..., Any]]) -> Optional[Callable[..., Any]]:
    """
    Normaliza callback de época para cobrir variações:
      - on_epoch(epoch_metrics_dict)
      - on_epoch_metrics(name, epoch, total_epochs, loss, acc, lr, ...)
    Aqui apenas repassamos e não derrubamos o treino em caso de incompatibilidade.
    """
    if on_epoch_metrics is None:
        return None

    def _cb(*args: Any, **kwargs: Any) -> None:
        try:
            on_epoch_metrics(*args, **kwargs)
        except Exception:
            # Não derrubar treino
            return

    return _cb


# -----------------------------------------------------------------------------
# API pública do adapter: treino/avaliação/engine (blindada)
# -----------------------------------------------------------------------------
def safe_build_engine(network_name: str) -> Any:
    """
    Retorna engine plugável (se existir).
    - Se não existir ia_engine.build_engine, retorna None.
    """
    try:
        fn = _import_attr("ia_engine", "build_engine")
    except Exception:
        return None
    try:
        return fn(network_name)
    except Exception:
        return None


def safe_train_networks(
    *,
    network_names: Sequence[str],
    data: Sequence[Any],
    learning_rate: float,
    n_epochs: int,
    shuffle: bool = False,
    verbose: bool = False,
    progress_callback: Optional[Callable[..., Any]] = None,
    on_epoch_metrics: Optional[Callable[..., Any]] = None,
    engine: Any = None,
    batch_size: Optional[int] = None,
    **extra_kwargs: Any,
) -> Any:
    """
    Chama ia_training.train_networks com compatibilidade de assinatura.
    - Mapeia nomes de parâmetros (network_names/names/networks, data/dataset/samples, etc.)
    - Filtra kwargs não suportados
    - Normaliza callbacks (progress_callback / on_epoch / on_epoch_metrics)
    - engine=... só é passado se o backend aceitar (ou **kwargs).
    """
    fn = _import_attr("ia_training", "train_networks")
    sig = _get_signature(fn)

    out: Dict[str, Any] = {}

    # network_names pode ser chamado de names/networks/models...
    key_names = _pick_first_present(sig, "network_names", "names", "networks", "models")
    if key_names is None:
        key_names = "network_names"
    _set_if_target_exists(sig, out, key_names, list(network_names))

    # data pode ser data/dataset/samples/train_data
    key_data = _pick_first_present(sig, "data", "dataset", "samples", "train_data")
    if key_data is None:
        key_data = "data"
    _set_if_target_exists(sig, out, key_data, list(data))

    # learning rate pode ser learning_rate/lr
    key_lr = _pick_first_present(sig, "learning_rate", "lr")
    if key_lr is None:
        key_lr = "learning_rate"
    _set_if_target_exists(sig, out, key_lr, float(learning_rate))

    # epochs pode ser n_epochs/epochs
    key_epochs = _pick_first_present(sig, "n_epochs", "epochs")
    if key_epochs is None:
        key_epochs = "n_epochs"
    _set_if_target_exists(sig, out, key_epochs, int(n_epochs))

    # shuffle/verbose
    _set_if_target_exists(sig, out, "shuffle", bool(shuffle))
    _set_if_target_exists(sig, out, "verbose", bool(verbose))

    # batch_size (se existir)
    if batch_size is not None:
        _set_if_target_exists(sig, out, "batch_size", int(batch_size))

    # callbacks: alguns backends usam progress_callback, progress, on_progress, callback...
    pcb = _wrap_progress_callback(progress_callback)
    if pcb is not None:
        key_pcb = _pick_first_present(sig, "progress_callback", "progress", "on_progress", "callback")
        if key_pcb is None:
            key_pcb = "progress_callback"
        _set_if_target_exists(sig, out, key_pcb, pcb)

    # on_epoch / on_epoch_metrics (assinaturas podem variar)
    oem = _wrap_on_epoch_metrics(on_epoch_metrics)
    if oem is not None:
        # Prioridade: on_epoch_metrics, depois on_epoch
        key_epoch = _pick_first_present(sig, "on_epoch_metrics", "on_epoch", "epoch_callback")
        if key_epoch is None:
            key_epoch = "on_epoch_metrics"
        _set_if_target_exists(sig, out, key_epoch, oem)

    # engine plugável (só se suportado)
    if engine is not None:
        _set_if_target_exists(sig, out, "engine", engine)

    # extras (somente se suportado)
    out.update(extra_kwargs)
    out = _filter_kwargs(fn, out)

    return fn(**out)


def safe_evaluate_networks(
    *,
    network_names: Sequence[str],
    data: Sequence[Any],
    limit: Optional[int] = None,
    verbose: bool = False,
    engine: Any = None,
    **extra_kwargs: Any,
) -> Any:
    """
    Chama ia_training.evaluate_networks com compatibilidade de assinatura.
    """
    fn = _import_attr("ia_training", "evaluate_networks")
    sig = _get_signature(fn)

    out: Dict[str, Any] = {}

    key_names = _pick_first_present(sig, "network_names", "names", "networks", "models")
    if key_names is None:
        key_names = "network_names"
    _set_if_target_exists(sig, out, key_names, list(network_names))

    key_data = _pick_first_present(sig, "data", "dataset", "samples", "eval_data", "test_data")
    if key_data is None:
        key_data = "data"
    _set_if_target_exists(sig, out, key_data, list(data))

    if limit is not None:
        _set_if_target_exists(sig, out, "limit", int(limit))

    _set_if_target_exists(sig, out, "verbose", bool(verbose))

    if engine is not None:
        _set_if_target_exists(sig, out, "engine", engine)

    out.update(extra_kwargs)
    out = _filter_kwargs(fn, out)

    return fn(**out)


def extract_losses(train_result: Any) -> Tuple[float, float]:
    """
    Helper para UI/Workers: suporta nomes diferentes de loss no TrainResult.
    Preferência:
      - final_loss/avg_loss
      - loss_final/loss_avg
      - loss/avg_loss
    """
    if train_result is None:
        return 0.0, 0.0

    def _get(*names: str, default: float = 0.0) -> float:
        for n in names:
            if hasattr(train_result, n):
                try:
                    v = getattr(train_result, n)
                    if v is None:
                        continue
                    return float(v)
                except Exception:
                    continue
        return float(default)

    final_loss = _get("final_loss", "loss_final", "loss", default=0.0)
    avg_loss = _get("avg_loss", "loss_avg", default=final_loss)
    return float(final_loss), float(avg_loss)


def extract_accuracy(eval_result: Any) -> float:
    """
    Helper para UI/Workers: extrai accuracy de EvalResult com fallback.
    """
    if eval_result is None:
        return 0.0
    try:
        v = getattr(eval_result, "accuracy", None)
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0
