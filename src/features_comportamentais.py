"""Features do histórico de pedidos (KAN-7, Pessoa D).

Ponto de partida: o código abaixo saiu de src/dados.py na revisão da KAN-1 e
mantém as decisões tomadas lá. Cabe à KAN-7 revisar e decidir se mantém:
- valor e atraso (faixas) -> ponto médio (_num) e ordinal (_ord);
- atraso "(-100, 0]" significa sem atraso, então atraso_num = 0 (o ponto
  médio seria -50);
- flag atrasou (atraso_num > 0).

A limpeza não remove pedidos: 1 pedido sem valor e 1 sem atraso continuam na
base, com nulo nas colunas derivadas. Decidir se entram nas contagens é papel
desta etapa.

Atenção ao vazamento temporal: as agregações por cliente devem usar só pedidos
anteriores à data de referência. Hoje os dados não têm data de cadastro nem
data em que a inadimplência foi observada (pergunta em aberto da KAN-1).
"""
from __future__ import annotations

import pandas as pd

from src.dados import limites


def ponto_medio(serie: pd.Series, razao_aberto: float = 100) -> pd.Series:
    """Ponto médio da faixa. Decisão desta etapa, não da limpeza.

    A Praso fecha a última faixa com um teto artificial, ex: "(60, 10000000]".
    Quando o teto é >= razao_aberto vezes o limite inferior, a faixa é tratada
    como aberta e o valor vira o limite inferior.
    """
    lim = limites(serie)
    aberto = (lim["inf"] > 0) & (lim["sup"] >= razao_aberto * lim["inf"])
    return ((lim["inf"] + lim["sup"]) / 2).mask(aberto, lim["inf"])


def ordinal(serie: pd.Series) -> pd.Series:
    """Código ordinal estável da faixa (categórica da limpeza); nulo continua nulo."""
    return serie.cat.codes.astype("Int64").mask(serie.isna())


def converter_pedidos(pedidos: pd.DataFrame, razao_aberto: float = 100) -> pd.DataFrame:
    """Cria valor_num/_ord, atraso_num/_ord e atrasou em cada pedido (sem agregar)."""
    df = pedidos.copy()
    df["valor_num"] = ponto_medio(df["valor"], razao_aberto)
    df["valor_ord"] = ordinal(df["valor"])

    atraso_num = ponto_medio(df["atraso"], razao_aberto)
    df["atraso_num"] = atraso_num.where(atraso_num.isna() | (atraso_num > 0), 0.0)
    df["atraso_ord"] = ordinal(df["atraso"])
    df["atrasou"] = (df["atraso_num"] > 0).astype("Int64").mask(df["atraso_num"].isna())
    return df
