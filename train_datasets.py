from __future__ import annotations

import random
from typing import List, Optional, Dict, Tuple

from ia_training import Sample


# ======================================================================================
# Dataset simples (mantido)
# ======================================================================================

def ds_greater_less(
    n_samples: int = 10_000,
    low: float = -100.0,
    high: float = 100.0,
) -> List[Sample]:
    data: List[Sample] = []
    for _ in range(n_samples):
        a = random.uniform(low, high)
        b = random.uniform(low, high)
        target = 0 if a > b else 1
        data.append(([a, b], target))
    return data


# ======================================================================================
# MT5 helpers (robustos)
# ======================================================================================

def _import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "MetaTrader5 não está disponível.\n"
            "Instale com:\n"
            "  pip install MetaTrader5\n\n"
            f"Detalhe: {e}"
        )
    return mt5


def _timeframe_from_str(mt5, timeframe: str):
    tf = (timeframe or "").upper().strip()
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5,
        "M6": mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H2": mt5.TIMEFRAME_H2,
        "H3": mt5.TIMEFRAME_H3,
        "H4": mt5.TIMEFRAME_H4,
        "H6": mt5.TIMEFRAME_H6,
        "H8": mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    if tf not in mapping:
        raise ValueError(f"timeframe inválido: {timeframe!r}. Ex.: 'M15', 'H1', 'D1'.")
    return mapping[tf]


def _normalize_login(login: Optional[object]) -> Optional[int]:
    """
    Converte login para int, ou retorna None se não for válido.
    Evita: invalid 'login' argument
    """
    if login is None:
        return None
    if isinstance(login, bool):
        return None
    if isinstance(login, int):
        return login if login > 0 else None
    if isinstance(login, float):
        li = int(login)
        return li if li > 0 else None
    if isinstance(login, str):
        s = login.strip()
        if not s:
            return None
        try:
            li = int(s)
            return li if li > 0 else None
        except Exception:
            return None
    return None


def _mt5_initialize(
    mt5,
    *,
    terminal_path: Optional[str] = None,
    login: Optional[object] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
) -> None:
    """
    Inicialização robusta:
    - Se login vier inválido, NÃO passa login (usa terminal já logado)
    """
    kwargs = {}
    if terminal_path:
        kwargs["path"] = terminal_path

    login_int = _normalize_login(login)
    if login_int is not None:
        kwargs["login"] = login_int
        if password is not None:
            kwargs["password"] = str(password)
        if server is not None:
            kwargs["server"] = str(server)

    ok = mt5.initialize(**kwargs) if kwargs else mt5.initialize()
    if not ok:
        code, msg = mt5.last_error()
        raise RuntimeError(f"Falha ao inicializar MT5: {code} - {msg}")


def _ensure_symbol(mt5, symbol: str) -> str:
    sym = (symbol or "").strip()
    if not sym:
        raise ValueError("Símbolo vazio.")

    info = mt5.symbol_info(sym)
    if info is None:
        alts = mt5.symbols_get(sym + "*") or []
        if not alts and sym.upper().startswith("PETR4"):
            alts = mt5.symbols_get("PETR4*") or []
        alt_names = [s.name for s in alts[:10]]
        raise RuntimeError(
            f"Símbolo '{sym}' não encontrado no MT5.\n"
            f"Alternativas possíveis (até 10): {alt_names}"
        )

    if not info.visible:
        mt5.symbol_select(sym, True)

    return sym


def _field_exists(dtype_names: Optional[Tuple[str, ...]], field: str) -> bool:
    return bool(dtype_names) and field in dtype_names


def _get_rate_field(r, dtype_names: Optional[Tuple[str, ...]], field: str, default: float = 0.0) -> float:
    """
    Lê campo de um numpy structured record (numpy.void) com segurança.
    """
    if _field_exists(dtype_names, field):
        try:
            return float(r[field])
        except Exception:
            return float(default)
    return float(default)


def _fetch_ohlc_from_mt5(
    symbol: str = "PETR4",
    timeframe: str = "M15",
    n_bars: int = 5000,
    *,
    skip_current_bar: bool = True,
    terminal_path: Optional[str] = None,
    login: Optional[object] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
) -> List[Dict[str, float]]:
    """
    Retorna candles em ordem cronológica (antigo -> recente).
    """
    mt5 = _import_mt5()
    _mt5_initialize(mt5, terminal_path=terminal_path, login=login, password=password, server=server)

    sym = _ensure_symbol(mt5, symbol)
    tf = _timeframe_from_str(mt5, timeframe)

    start_pos = 1 if skip_current_bar else 0
    rates = mt5.copy_rates_from_pos(sym, tf, start_pos, int(n_bars))

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"MT5 não retornou dados para {sym} {timeframe}. "
            "Verifique se há histórico e se o símbolo está correto no seu broker."
        )

    dtype_names = getattr(rates.dtype, "names", None)

    # decide qual campo de volume usar
    has_tick = _field_exists(dtype_names, "tick_volume")
    has_real = _field_exists(dtype_names, "real_volume")

    out: List[Dict[str, float]] = []
    for r in rates:
        o = _get_rate_field(r, dtype_names, "open")
        h = _get_rate_field(r, dtype_names, "high")
        l = _get_rate_field(r, dtype_names, "low")
        c = _get_rate_field(r, dtype_names, "close")

        if has_tick:
            vol = _get_rate_field(r, dtype_names, "tick_volume", 0.0)
        elif has_real:
            vol = _get_rate_field(r, dtype_names, "real_volume", 0.0)
        else:
            vol = 0.0

        out.append({"open": o, "high": h, "low": l, "close": c, "volume": vol})

    # MT5 geralmente retorna do mais recente -> mais antigo
    out.reverse()
    return out


# ======================================================================================
# Features + classificadores
# ======================================================================================

_EPS = 1e-12


def _candle_features(o: float, h: float, l: float, c: float) -> List[float]:
    rng = max(_EPS, h - l)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l

    body_ratio = body / rng
    upper_ratio = max(0.0, upper) / rng
    lower_ratio = max(0.0, lower) / rng

    if c > o:
        direction = 1.0
    elif c < o:
        direction = -1.0
    else:
        direction = 0.0

    close_pos = (c - l) / rng  # 0..1

    # clamp
    body_ratio = min(max(body_ratio, 0.0), 1.0)
    upper_ratio = min(max(upper_ratio, 0.0), 1.0)
    lower_ratio = min(max(lower_ratio, 0.0), 1.0)
    close_pos = min(max(close_pos, 0.0), 1.0)

    return [body_ratio, upper_ratio, lower_ratio, direction, close_pos]


# ---------- Candle atual: 10 classes ----------
CANDLE_CURRENT_LABELS = {
    0: "OTHER",
    1: "DOJI",
    2: "SPINNING_TOP",
    3: "MARUBOZU_BULL",
    4: "MARUBOZU_BEAR",
    5: "HAMMER",
    6: "INVERTED_HAMMER",
    7: "SHOOTING_STAR",
    8: "LONG_BODY_BULL",
    9: "LONG_BODY_BEAR",
}


def _classify_candle_current(o: float, h: float, l: float, c: float) -> int:
    body_ratio, upper_ratio, lower_ratio, direction, close_pos = _candle_features(o, h, l, c)

    if direction > 0 and body_ratio >= 0.85 and upper_ratio <= 0.05 and lower_ratio <= 0.05:
        return 3
    if direction < 0 and body_ratio >= 0.85 and upper_ratio <= 0.05 and lower_ratio <= 0.05:
        return 4

    if body_ratio <= 0.10:
        return 1

    if 0.10 < body_ratio <= 0.30 and upper_ratio >= 0.25 and lower_ratio >= 0.25:
        return 2

    if body_ratio <= 0.30 and lower_ratio >= 0.55 and upper_ratio <= 0.20 and close_pos >= 0.60:
        return 5
    if body_ratio <= 0.30 and upper_ratio >= 0.55 and lower_ratio <= 0.20 and close_pos <= 0.40:
        return 6

    if body_ratio <= 0.25 and upper_ratio >= 0.60 and close_pos <= 0.35:
        return 7

    if direction > 0 and body_ratio >= 0.60:
        return 8
    if direction < 0 and body_ratio >= 0.60:
        return 9

    return 0


# ---------- Últimos 3 candles: 12 classes ----------
CANDLE_LAST3_LABELS = {
    0: "OTHER",
    1: "THREE_WHITE_SOLDIERS",
    2: "THREE_BLACK_CROWS",
    3: "MORNING_STAR",
    4: "EVENING_STAR",
    5: "BULLISH_ENGULFING",
    6: "BEARISH_ENGULFING",
    7: "INSIDE_CHAIN",
    8: "OUTSIDE_BAR",
    9: "REVERSAL_UP_SIMPLE",
    10: "REVERSAL_DOWN_SIMPLE",
    11: "VOLATILITY_EXPANSION",
}


def _is_bull(o: float, c: float) -> bool:
    return c > o


def _is_bear(o: float, c: float) -> bool:
    return c < o


def _mid(o: float, c: float) -> float:
    return (o + c) / 2.0


def _engulfing_bull(o1, c1, o2, c2) -> bool:
    if not _is_bull(o2, c2):
        return False
    body1_low = min(o1, c1)
    body1_high = max(o1, c1)
    body2_low = min(o2, c2)
    body2_high = max(o2, c2)
    return body2_low <= body1_low and body2_high >= body1_high


def _engulfing_bear(o1, c1, o2, c2) -> bool:
    if not _is_bear(o2, c2):
        return False
    body1_low = min(o1, c1)
    body1_high = max(o1, c1)
    body2_low = min(o2, c2)
    body2_high = max(o2, c2)
    return body2_low <= body1_low and body2_high >= body1_high


def _inside_bar(h1, l1, h2, l2) -> bool:
    return h2 <= h1 and l2 >= l1


def _outside_bar(h1, l1, h2, l2) -> bool:
    return h2 >= h1 and l2 <= l1


def _classify_last3_pattern(c1: Dict[str, float], c2: Dict[str, float], c3: Dict[str, float]) -> int:
    o1, h1, l1, cl1 = c1["open"], c1["high"], c1["low"], c1["close"]
    o2, h2, l2, cl2 = c2["open"], c2["high"], c2["low"], c2["close"]
    o3, h3, l3, cl3 = c3["open"], c3["high"], c3["low"], c3["close"]

    f1 = _candle_features(o1, h1, l1, cl1)
    f2 = _candle_features(o2, h2, l2, cl2)
    f3 = _candle_features(o3, h3, l3, cl3)

    body1, body2, body3 = f1[0], f2[0], f3[0]

    rng1 = max(_EPS, h1 - l1)
    rng2 = max(_EPS, h2 - l2)
    rng3 = max(_EPS, h3 - l3)

    # Engulfing (2 últimos)
    if _engulfing_bull(o2, cl2, o3, cl3):
        return 5
    if _engulfing_bear(o2, cl2, o3, cl3):
        return 6

    # Inside chain
    if _inside_bar(h1, l1, h2, l2) and _inside_bar(h2, l2, h3, l3):
        return 7

    # Outside bar (c3 vs c2)
    if _outside_bar(h2, l2, h3, l3):
        return 8

    # Three soldiers / crows
    if _is_bull(o1, cl1) and _is_bull(o2, cl2) and _is_bull(o3, cl3):
        if cl1 < cl2 < cl3 and body1 >= 0.50 and body2 >= 0.50 and body3 >= 0.50:
            return 1

    if _is_bear(o1, cl1) and _is_bear(o2, cl2) and _is_bear(o3, cl3):
        if cl1 > cl2 > cl3 and body1 >= 0.50 and body2 >= 0.50 and body3 >= 0.50:
            return 2

    # Morning/Evening star (heurística)
    if _is_bear(o1, cl1) and body1 >= 0.50 and body2 <= 0.20 and _is_bull(o3, cl3) and body3 >= 0.45:
        if cl3 > _mid(o1, cl1):
            return 3

    if _is_bull(o1, cl1) and body1 >= 0.50 and body2 <= 0.20 and _is_bear(o3, cl3) and body3 >= 0.45:
        if cl3 < _mid(o1, cl1):
            return 4

    # Reversal simples
    if _is_bear(o1, cl1) and _is_bull(o3, cl3) and (cl3 > cl2):
        return 9
    if _is_bull(o1, cl1) and _is_bear(o3, cl3) and (cl3 < cl2):
        return 10

    # Expansão de volatilidade
    if (rng2 > rng1 * 1.20) and (rng3 > rng2 * 1.20):
        return 11

    return 0


# ======================================================================================
# Datasets MT5 (PETR4 por padrão)
# ======================================================================================

def ds_mt5_candle_current(
    symbol: str = "PETR4",
    timeframe: str = "M15",
    n_bars: int = 5000,
    *,
    skip_current_bar: bool = True,
    terminal_path: Optional[str] = None,
    login: Optional[object] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
) -> List[Sample]:
    """
    Candle atual (10 classes)
    X: 5 features
    y: 0..9
    """
    candles = _fetch_ohlc_from_mt5(
        symbol=symbol,
        timeframe=timeframe,
        n_bars=n_bars,
        skip_current_bar=skip_current_bar,
        terminal_path=terminal_path,
        login=login,
        password=password,
        server=server,
    )

    data: List[Sample] = []
    for c in candles:
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        x = _candle_features(o, h, l, cl)
        y = _classify_candle_current(o, h, l, cl)
        data.append((x, int(y)))

    return data


def ds_mt5_candle_last3(
    symbol: str = "PETR4",
    timeframe: str = "M15",
    n_bars: int = 5000,
    *,
    skip_current_bar: bool = True,
    terminal_path: Optional[str] = None,
    login: Optional[object] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
) -> List[Sample]:
    """
    Últimos 3 candles (12 classes)
    X: 15 features (3 * 5)
    y: 0..11
    """
    candles = _fetch_ohlc_from_mt5(
        symbol=symbol,
        timeframe=timeframe,
        n_bars=n_bars,
        skip_current_bar=skip_current_bar,
        terminal_path=terminal_path,
        login=login,
        password=password,
        server=server,
    )

    if len(candles) < 3:
        return []

    data: List[Sample] = []
    for i in range(2, len(candles)):
        c1 = candles[i - 2]
        c2 = candles[i - 1]
        c3 = candles[i]

        x = (
            _candle_features(c1["open"], c1["high"], c1["low"], c1["close"])
            + _candle_features(c2["open"], c2["high"], c2["low"], c2["close"])
            + _candle_features(c3["open"], c3["high"], c3["low"], c3["close"])
        )
        y = _classify_last3_pattern(c1, c2, c3)
        data.append((x, int(y)))

    return data
# ======================================================================================
# Helpers para UI (WdTraining refatorado)
# - Mantém compatibilidade com o padrão ds_*
# ======================================================================================

def list_dataset_functions() -> List[str]:
    """
    Retorna a lista de nomes de funções ds_* disponíveis neste módulo.
    """
    import inspect
    module = globals()
    names: List[str] = []
    for name, obj in module.items():
        if name.startswith("ds_") and callable(obj):
            names.append(name)
    names.sort()
    return names


def load_dataset_by_name(ds_name: str):
    """
    Executa um cenário ds_* pelo nome e retorna (train, eval).

    Compatibilidade:
    - Se a função retornar apenas List[Sample], usa o mesmo conjunto para train/eval.
    - Se retornar (train, eval), respeita.
    """
    if not ds_name:
        raise ValueError("ds_name vazio.")

    fn = globals().get(ds_name)
    if fn is None or not callable(fn):
        raise ValueError(f"Dataset '{ds_name}' não encontrado em train_datasets.py")

    data = fn()  # usa defaults do cenário

    # Permite retornar (train, eval)
    if isinstance(data, tuple) and len(data) == 2:
        train, eval_ = data
        return list(train), list(eval_)

    # Caso padrão: um único dataset -> usa em ambos
    return list(data), list(data)
