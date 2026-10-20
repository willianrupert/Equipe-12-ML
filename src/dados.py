"""Carregamento, limpeza e checagem de qualidade dos dados de crédito da Praso (KAN-1).

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

# A Praso fecha o último intervalo com um teto artificial, ex: "(60, 10000000]" ou
# "(10000000, 100000000000]". Nesses casos o ponto médio não faz sentido, então usamos
# o limite inferior. Consideramos aberto quando o teto é >= 100x o limite inferior.
RAZAO_INTERVALO_ABERTO = 100

# Colunas que a Praso transformou em intervalos, ex: "(150, 250]".
INTERVALOS_CLIENTES = [
    "capital_social",
    "idade_cnpj",
    "ifood_contagem_avaliacoes",
    "google_maps_avaliacao",
    "google_maps_contagem_avaliacoes",
]
INTERVALOS_PEDIDOS = ["valor", "atraso"]

_RE_INTERVALO = re.compile(r"^\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]$")


def parse_intervalo(valor: object) -> tuple[float, float]:
    """Converte "(a, b]" em (a, b). Retorna (nan, nan) para nulos."""
    if pd.isna(valor):
        return np.nan, np.nan
    m = _RE_INTERVALO.match(str(valor).strip())
    if not m:
        raise ValueError(f"Intervalo em formato inesperado: {valor!r}")
    return float(m.group(1)), float(m.group(2))


def intervalo_para_numero(serie: pd.Series) -> pd.Series:
    """Ponto médio do intervalo; limite inferior quando o intervalo é aberto."""
    limites = serie.map(parse_intervalo)
    inf = limites.str[0]
    sup = limites.str[1]
    meio = (inf + sup) / 2
    aberto = (inf > 0) & (sup >= RAZAO_INTERVALO_ABERTO * inf)
    return meio.mask(aberto, inf)


def intervalo_para_ordinal(serie: pd.Series) -> pd.Series:
    """Código ordinal (0, 1, 2, ...) do intervalo, ordenado pelo limite inferior."""
    inf = serie.map(parse_intervalo).str[0]
    ordem = {v: i for i, v in enumerate(sorted(inf.dropna().unique()))}
    return inf.map(ordem).astype("Int64")


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


def limpar_clientes(clientes: pd.DataFrame) -> pd.DataFrame:
    df = clientes.copy()

    df["inadimplente"] = df["inadimplente"].astype(int)
    for col in ["serasa_contagem_negativacoes", "serasa_contagem_protestos", "serasa_socio_tem_negativacao"]:
        df[col] = df[col].astype(int)

    # Intervalos -> número (ponto médio) e ordinal; a coluna original é mantida.
    for col in INTERVALOS_CLIENTES:
        df[f"{col}_num"] = intervalo_para_numero(df[col])
        df[f"{col}_ord"] = intervalo_para_ordinal(df[col])

    # Faixa de preço do iFood ("$" a "$$$$$") -> 1 a 5.
    df["ifood_faixa_preco_num"] = df["ifood_faixa_preco"].str.len().astype("Int64")

    # Nulo nas colunas de plataformas = estabelecimento não está na plataforma.
    df["tem_ifood"] = df[["ifood_contagem_avaliacoes", "ifood_faixa_preco"]].notna().any(axis=1).astype(int)
    df["tem_google_maps"] = df["google_maps_contagem_avaliacoes"].notna().astype(int)
    df["tem_credores_serasa"] = df["serasa_credores"].notna().astype(int)
    # google_maps_tem_website só tem valor 1 ou nulo; nulo vira 0 (sem site conhecido).
    df["google_maps_tem_website"] = df["google_maps_tem_website"].fillna(0).astype(int)

    return df


def limpar_pedidos(pedidos: pd.DataFrame) -> pd.DataFrame:
    df = pedidos.copy()
    # Há 2 pedidos com nulo (um sem valor, outro sem atraso): sem como imputar com
    # segurança a partir de intervalos, são removidos.
    df = df.dropna(subset=["valor", "atraso"])

    df["data_entrega"] = pd.to_datetime(df["data_entrega"])
    df["valor_num"] = intervalo_para_numero(df["valor"])
    df["valor_ord"] = intervalo_para_ordinal(df["valor"])

    # "(-100, 0]" significa que não houve atraso: o ponto médio (-50) não faz sentido.
    atraso_num = intervalo_para_numero(df["atraso"])
    df["atraso_num"] = atraso_num.where(atraso_num > 0, 0.0)
    df["atraso_ord"] = intervalo_para_ordinal(df["atraso"])
    df["atrasou"] = (df["atraso_num"] > 0).astype(int)

    return df.sort_values(["id_cliente", "data_entrega"]).reset_index(drop=True)


def checar_chaves(clientes: pd.DataFrame, pedidos: pd.DataFrame) -> dict:
    return {
        "clientes_duplicados": int(clientes["id_cliente"].duplicated().sum()),
        "pedidos_duplicados": int(pedidos["id_pedido"].duplicated().sum()),
        "pedidos_sem_cliente": int((~pedidos["id_cliente"].isin(clientes["id_cliente"])).sum()),
        "clientes_com_pedidos": int(clientes["id_cliente"].isin(pedidos["id_cliente"]).sum()),
    }


def carregar_dados_limpos(salvar: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    clientes, pedidos = carregar_brutos()
    clientes, pedidos = limpar_clientes(clientes), limpar_pedidos(pedidos)
    if salvar:
        DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
        clientes.to_csv(DIR_PROCESSED / "clientes_limpo.csv", index=False)
        pedidos.to_csv(DIR_PROCESSED / "pedidos_limpo.csv", index=False)
    return clientes, pedidos
