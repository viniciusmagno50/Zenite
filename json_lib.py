# json_lib.py
# Biblioteca de utilidades para persistência em JSON e sanitização de nomes.
#
# Regras de diretórios:
#   - IA "raiz":       dados/{nome}/{nome}.json
#   - IA geracional:   dados/{parent}/geracao/{parent}_{geracao}_{id}.json
#
# Observação:
#   build_paths() SEMPRE retorna (pasta, arquivo_json, nome_sanitizado).
#   Isso mantém compatibilidade com o restante do projeto.

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Tuple, Optional

# Raiz do projeto (mesma pasta onde está json_lib.py)
ROOT = Path(__file__).resolve().parent

# Diretório base para dados de IAs individuais
BASE_DIR = ROOT / "dados"


def ensure_dir(pasta: Path) -> None:
    """Garante a existência da pasta (pais incluídos)."""
    pasta.mkdir(parents=True, exist_ok=True)


def sanitize_name(texto: str) -> str:
    """
    Normaliza nome para uso em pasta/arquivo.

    Regras:
      - mantém letras/números, '-' e '_'
      - espaços viram '_'
      - remove demais caracteres
      - remove '_' duplicados nas bordas
    """
    permitido: list[str] = []
    for ch in texto.strip():
        if ch.isalnum() or ch in ("-", "_"):
            permitido.append(ch)
        elif ch.isspace():
            permitido.append("_")

    nome = "".join(permitido)
    while "__" in nome:
        nome = nome.replace("__", "_")
    return nome.strip("_")


def _try_parse_generation_name(nome_sanitizado: str) -> Optional[Tuple[str, int, int]]:
    """
    Detecta padrão geracional: {parent}_{geracao}_{id}
    Ex.: PETR4_3_12  -> ("PETR4", 3, 12)

    Para evitar falso positivo, só considera geracional se existir a pasta do parent em dados/.
    """
    parts = nome_sanitizado.split("_")
    if len(parts) < 3:
        return None
    gen_s = parts[-2]
    ind_s = parts[-1]
    if not (gen_s.isdigit() and ind_s.isdigit()):
        return None
    parent = "_".join(parts[:-2]).strip("_")
    if not parent:
        return None

    parent_dir = BASE_DIR / parent
    if not parent_dir.exists():
        # se o parent não existe, tratamos como IA raiz (compatibilidade)
        return None

    return parent, int(gen_s), int(ind_s)


def build_paths(nome_bruto: str) -> Tuple[Path, Path, str]:
    """
    A partir de 'nome_bruto', gera:
      - pasta:  BASE_DIR / {nome_sanitizado}                 (IA raiz)
               ou BASE_DIR / {parent} / 'geracao'            (IA geracional)
      - arquivo: pasta / {nome_sanitizado}.json
      - nome_sanitizado

    IMPORTANTE:
      - Não cria arquivos, apenas retorna Paths.
      - Para IA geracional, o nome do arquivo já vem completo no nome_sanitizado
        (ex.: PETR4_2_7.json dentro de dados/PETR4/geracao/)
    """
    base = (nome_bruto or "").replace(".json", "")
    nome = sanitize_name(base)

    gen = _try_parse_generation_name(nome)
    if gen is not None:
        parent, g, i = gen
        pasta = BASE_DIR / parent / "geracao"
        arquivo = pasta / f"{nome}.json"
        return pasta, arquivo, nome

    pasta = BASE_DIR / nome
    arquivo = pasta / f"{nome}.json"
    return pasta, arquivo, nome


# ------------------------
# Lock + escrita atômica
# ------------------------
def _lock_path(json_path: Path) -> Path:
    return json_path.with_suffix(json_path.suffix + ".lock")


def acquire_lock(json_path: Path, timeout_s: float = 2.5, poll_s: float = 0.05) -> Path:
    """
    Lock simples por arquivo: cria um .lock com O_EXCL.
    Evita que uma thread leia enquanto outra escreve (ou vice-versa).
    """
    lock = _lock_path(json_path)
    start = time.time()

    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            finally:
                os.close(fd)
            return lock
        except FileExistsError:
            if (time.time() - start) >= timeout_s:
                raise TimeoutError(f"Timeout aguardando lock do manifesto: {json_path}")
            time.sleep(poll_s)


def release_lock(lock_path: Path) -> None:
    """Libera lock. Ignora se já tiver sido removido."""
    try:
        lock_path.unlink(missing_ok=True)  # py>=3.8
    except Exception:
        pass


def save_json(caminho: Path, dados: Any) -> None:
    """
    Salva 'dados' em JSON no caminho informado, com indentação e UTF-8.

    Correção:
      - lock de escrita
      - escrita em arquivo temporário
      - replace atômico
    """
    ensure_dir(caminho.parent)
    lock = acquire_lock(caminho)
    try:
        tmp = caminho.with_suffix(caminho.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp), str(caminho))  # atômico no mesmo filesystem
    finally:
        release_lock(lock)


def load_json(caminho: Path) -> Any:
    """
    Carrega JSON do caminho informado e retorna o objeto.

    Correção:
      - tenta adquirir lock curto (para evitar ler no meio da escrita)
      - retries rápidos se pegar JSON truncado por qualquer motivo externo
    """
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            lock = acquire_lock(caminho, timeout_s=1.0)
            try:
                with caminho.open("r", encoding="utf-8") as f:
                    return json.load(f)
            finally:
                release_lock(lock)
        except Exception as e:
            last_err = e
            time.sleep(0.05 * (attempt + 1))
    if last_err:
        raise last_err
    raise RuntimeError("Falha inesperada ao carregar JSON.")


# ------------------------
# Validação de manifesto
# ------------------------
def is_new_manifest_schema(data: Any) -> bool:
    """Heurística rápida para distinguir schema novo (dados/) do legado (network/)."""
    if not isinstance(data, dict):
        return False
    return (
        "schema_version" in data
        and isinstance(data.get("identification"), dict)
        and isinstance(data.get("structure"), dict)
    )


def assert_manifest_schema(data: Any) -> None:
    """
    Valida o mínimo necessário para o schema novo.
    Lança ValueError se for legado/inválido.
    """
    if not is_new_manifest_schema(data):
        raise ValueError(
            "Manifesto inválido ou em schema legado. Esperado: {schema_version, identification, structure}."
        )

    ident = data.get("identification") or {}
    struct = data.get("structure") or {}

    if not ident.get("name"):
        raise ValueError("Manifesto sem identification.name")

    if "input_size" not in struct or "output_size" not in struct:
        raise ValueError("Manifesto sem structure.input_size/output_size")
