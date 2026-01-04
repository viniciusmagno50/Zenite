from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Any
import importlib
import inspect


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
