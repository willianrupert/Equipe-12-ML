"""Carregamento, limpeza e checagem de qualidade dos dados de crédito da Praso (KAN-1).

A limpeza só faz o que não perde informação e não exige escolha de modelagem:
tipos, datas, remoção de registros inválidos e ordenação. As colunas devolvidas
são as mesmas do CSV original. Representar faixas, criar flags ou preencher
nulos é engenharia de features (src/features_aplicacao.py e
src/features_comportamentais.py).

Uso:
    from src.dados import carregar_dados_limpos
    clientes, pedidos = carregar_dados_limpos()
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESSED = RAIZ / "data" / "processed"

ARQ_CLIENTES = "credito_aplicacao_clientes_final.csv"
ARQ_PEDIDOS = "credito_comportamental_pedidos_final.csv"

# Colunas que vêm como 0.0/1.0 (ou contagens) em float, sem nulos: viram int sem perda.
INTEIROS_CLIENTES = [
    "inadimplente",
    "serasa_contagem_negativacoes",
    "serasa_contagem_protestos",
    "serasa_socio_tem_negativacao",
]

# Colunas que a Praso transformou em faixas, ex: "(150, 250]".
INTERVALOS_CLIENTES = [
    "capital_social",
    "idade_cnpj",
    "ifood_contagem_avaliacoes",
    "google_maps_avaliacao",
    "google_maps_contagem_avaliacoes",
]
INTERVALOS_PEDIDOS = ["valor", "atraso"]

_RE_INTERVALO = re.compile(r"^\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]$")


# --- Utilitárias (usadas pela EDA e pelas features; a limpeza não as usa) ---

def parse_intervalo(valor: object) -> tuple[float, float]:
    """Converte "(a, b]" em (a, b). Retorna (nan, nan) para nulos."""
    if pd.isna(valor):
        return np.nan, np.nan
    m = _RE_INTERVALO.match(str(valor).strip())
    if not m:
        raise ValueError(f"Intervalo em formato inesperado: {valor!r}")
    return float(m.group(1)), float(m.group(2))


def intervalo_para_numero(serie: pd.Series, razao_aberto: float = 100) -> pd.Series:
    """Ponto médio da faixa.

    A Praso fecha a última faixa com um teto artificial, ex: "(60, 10000000]".
    Quando o teto é >= razao_aberto vezes o limite inferior, a faixa é tratada
    como aberta e o valor vira o limite inferior.
    """
    limites = serie.map(parse_intervalo)
    inf = limites.str[0]
    sup = limites.str[1]
    meio = (inf + sup) / 2
    aberto = (inf > 0) & (sup >= razao_aberto * inf)
    return meio.mask(aberto, inf)


def intervalo_para_ordinal(serie: pd.Series) -> pd.Series:
    """Código ordinal (0, 1, 2, ...) da faixa, ordenado pelo limite inferior."""
    inf = serie.map(parse_intervalo).str[0]
    ordem = {v: i for i, v in enumerate(sorted(inf.dropna().unique()))}
    return inf.map(ordem).astype("Int64")


# --- Carregamento e validação ---

def carregar_brutos() -> tuple[pd.DataFrame, pd.DataFrame]:
    clientes = pd.read_csv(DIR_RAW / ARQ_CLIENTES)
    pedidos = pd.read_csv(DIR_RAW / ARQ_PEDIDOS)
    return clientes, pedidos


def relatorio_qualidade(df: pd.DataFrame) -> pd.DataFrame:
    """Tipo, nulos e cardinalidade de cada coluna."""
    return pd.DataFrame(
        {
            "tipo": df.dtypes.astype(str),
            "nulos": df.isna().sum(),
            "pct_nulos": (df.isna().mean() * 100).round(1),
            "valores_distintos": df.nunique(),
        }
    )


def checar_chaves(clientes: pd.DataFrame, pedidos: pd.DataFrame) -> dict:
    return {
        "clientes_duplicados": int(clientes["id_cliente"].duplicated().sum()),
        "pedidos_duplicados": int(pedidos["id_pedido"].duplicated().sum()),
        "pedidos_sem_cliente": int((~pedidos["id_cliente"].isin(clientes["id_cliente"])).sum()),
        "clientes_com_pedidos": int(clientes["id_cliente"].isin(pedidos["id_cliente"]).sum()),
    }


def checar_formatos(clientes: pd.DataFrame, pedidos: pd.DataFrame) -> dict:
    """Quantos valores não nulos fogem do formato esperado, por coluna (0 = tudo certo)."""
    falhas = {}
    for df, colunas in [(clientes, INTERVALOS_CLIENTES), (pedidos, INTERVALOS_PEDIDOS)]:
        for col in colunas:
            s = df[col].dropna().astype(str).str.strip()
            falhas[col] = int((~s.str.match(_RE_INTERVALO)).sum())
    cnae = clientes["cnae_codigo"].dropna().astype(str)
    falhas["cnae_codigo"] = int((~cnae.str.match(r"^\d{2}\.\d{2}-\d-\d{2}$")).sum())
    datas = pd.to_datetime(pedidos["data_entrega"], errors="coerce")
    falhas["data_entrega"] = int((datas.isna() & pedidos["data_entrega"].notna()).sum())
    return falhas


# --- Limpeza ---

def limpar_clientes(clientes: pd.DataFrame) -> pd.DataFrame:
    """Só converte tipos sem perda. Faixas continuam como texto e nulos como vieram."""
    df = clientes.copy()
    for col in INTEIROS_CLIENTES:
        df[col] = df[col].astype(int)
    return df


def pedidos_invalidos(pedidos: pd.DataFrame) -> pd.Series:
    """Pedidos sem valor ou sem atraso: não dá para imputar com segurança a partir de faixas."""
    return pedidos[["valor", "atraso"]].isna().any(axis=1)


def limpar_pedidos(pedidos: pd.DataFrame) -> pd.DataFrame:
    """Remove pedidos inválidos, converte a data e ordena por cliente e data."""
    df = pedidos.loc[~pedidos_invalidos(pedidos)].copy()
    df["data_entrega"] = pd.to_datetime(df["data_entrega"])
    return df.sort_values(["id_cliente", "data_entrega"]).reset_index(drop=True)


def carregar_dados_limpos(salvar: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    clientes, pedidos = carregar_brutos()
    clientes, pedidos = limpar_clientes(clientes), limpar_pedidos(pedidos)
    if salvar:
        DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
        clientes.to_csv(DIR_PROCESSED / "clientes_limpo.csv", index=False)
        pedidos.to_csv(DIR_PROCESSED / "pedidos_limpo.csv", index=False)
    return clientes, pedidos
