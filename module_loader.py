# module_loader.py
from __future__ import annotations

import importlib
from typing import Any, Type


class ModuleSpecError(RuntimeError):
    pass


def load_class(spec: str) -> Type[Any]:
    """
    spec: "module.path:ClassName"
      ex: "pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar"

    Compatibilidade:
      - Se receber "tb_ind:TbIndToolbar" e não existir tb_ind como módulo,
        tenta automaticamente "pages.individual.toolbar.tb_ind_toolbar".
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ModuleSpecError(
            f"Spec inválida '{spec}'. Use o formato 'modulo:Classe' (ex: 'pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar')."
        )

    module_path, class_name = spec.split(":", 1)
    module_path = module_path.strip()
    class_name = class_name.strip()

    if not module_path or not class_name:
        raise ModuleSpecError(
            f"Spec inválida '{spec}'. Use o formato 'modulo:Classe' (ex: 'pages.individual.toolbar.tb_ind_toolbar:TbIndToolbar')."
        )

    last_error: Exception | None = None

    candidates = [module_path]

    # fallback automático: "tb_ind" -> "pages.individual.toolbar.tb_ind_toolbar"
    if "." not in module_path:
        candidates.append(f"pages.{module_path}")

    mod = None
    for cand in candidates:
        try:
            mod = importlib.import_module(cand)
            module_path = cand
            break
        except Exception as e:
            last_error = e

    if mod is None:
        raise ModuleSpecError(
            f"Falha ao importar módulo '{candidates[0]}' (spec '{spec}'). "
            f"Tentativas: {candidates}. Erro: {last_error}"
        ) from last_error

    try:
        cls = getattr(mod, class_name)
    except Exception as e:
        raise ModuleSpecError(
            f"Classe '{class_name}' não encontrada no módulo '{module_path}' (spec '{spec}')."
        ) from e

    return cls
