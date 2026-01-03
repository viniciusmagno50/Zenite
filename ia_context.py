# ia_context.py
from __future__ import annotations

from typing import Any, Callable, Optional

__all__ = ["AIContext"]


class AIContext:
    """
    Contexto global da IA ativa.

    Requisitos do seu projeto (já vi em wd_training.py):
      - AIContext.get_token() deve existir (controle de troca de IA durante treino)
      - AIContext.get_name() / set_active_name()
      - get_instance() e set_instance() (para engine atual)
      - set_factory() para criação lazy da engine (opcional)
      - clear()

    IMPORTANTE:
      - NÃO importar nenhum módulo do projeto aqui (evita import circular).
    """

    _active_name: Optional[str] = None
    _active_instance: Any = None
    _factory: Optional[Callable[[str], Any]] = None

    # Token monotônico: incrementa quando o contexto muda
    _token: int = 0

    # ---------------------------------------------------------
    # Interno
    # ---------------------------------------------------------
    @classmethod
    def _bump(cls) -> None:
        cls._token += 1

    # ---------------------------------------------------------
    # Token (usado por wd_training.py)
    # ---------------------------------------------------------
    @classmethod
    def get_token(cls) -> int:
        return int(cls._token)

    # ---------------------------------------------------------
    # Nome da IA ativa
    # ---------------------------------------------------------
    @classmethod
    def get_name(cls) -> Optional[str]:
        return cls._active_name

    @classmethod
    def set_active_name(cls, name: Optional[str]) -> None:
        """
        Define a IA ativa.
        - Invalida instância para evitar usar engine antiga
        - Incrementa token para permitir que workers detectem mudança
        """
        if name == cls._active_name:
            return
        cls._active_name = name
        cls._active_instance = None
        cls._bump()

    # ---------------------------------------------------------
    # Instância (engine)
    # ---------------------------------------------------------
    @classmethod
    def get_instance(cls) -> Any:
        """
        Retorna instância atual se existir.
        Não cria automaticamente.
        """
        return cls._active_instance

    @classmethod
    def ensure_instance(cls) -> Any:
        """
        Se não houver instância, tenta criar via factory usando o nome ativo.
        """
        if cls._active_instance is not None:
            return cls._active_instance
        if cls._factory is None or not cls._active_name:
            return None
        cls._active_instance = cls._factory(cls._active_name)
        return cls._active_instance

    @classmethod
    def set_instance(cls, instance: Any) -> None:
        """
        Define instância atual. Incrementa token pois muda o contexto efetivo.
        """
        cls._active_instance = instance
        cls._bump()

    # ---------------------------------------------------------
    # Factory
    # ---------------------------------------------------------
    @classmethod
    def set_factory(cls, factory: Optional[Callable[[str], Any]]) -> None:
        """
        Registra uma função que cria a engine dado o nome.
        Trocar a factory invalida a instância atual.
        """
        cls._factory = factory
        cls._active_instance = None
        cls._bump()

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------
    @classmethod
    def clear(cls) -> None:
        cls._active_name = None
        cls._active_instance = None
        cls._bump()
