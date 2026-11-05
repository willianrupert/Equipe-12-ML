"""Features do histórico de pedidos (KAN-7), para o modelo comportamental (KAN-8).

Duas etapas, as duas determinísticas (não aprendem nada dos dados):
1. converter_pedidos: representa as faixas de cada pedido como número, sem agregar.
   - valor e atraso -> ponto médio da faixa (_num) e posição da faixa (_ord);
   - a última faixa é aberta ("(2000, 10000000]", "(60, 10000000]"): vale o limite inferior;
   - atraso "(-100, 0]" significa sem atraso: atraso_num = 0 (o ponto médio seria -50);
   - atrasou = atraso_num > 0; atrasou_2d = atraso_num > 2, porque quase metade dos
     pedidos cai em "(0, 2]" e um atraso de 1 a 2 dias pode ser só logística.
2. agregar_pedidos: uma linha por cliente, indexada por id_cliente (DICIONARIO).

Pedidos com nulo: os 2 pedidos sem valor ou sem atraso ficam. A data deles vale, então
contam em n_pedidos, nos intervalos e no span; nas estatísticas da coluna que falta,
o nulo é ignorado.

Vazamento temporal: os dados não têm data de referência nem a data em que a
inadimplência foi observada, e quase todo cliente tem exatamente os últimos 3
pedidos. O atraso do último pedido separa o alvo muito mais do que o dos
anteriores (notebook 04, seção 4), o que sugere que ele pode ser o pedido em que a
inadimplência aconteceu. Por isso agregar_pedidos aceita excluir_ultimo=True
(versão conservadora), e a KAN-8 deve comparar as duas versões. Recência absoluta
(dias até uma data fixa) não é calculada: ela mediria a época, não o comportamento.

Nada aqui usa o alvo nem estatísticas do conjunto de clientes, então as features
podem ser calculadas na base inteira antes do split. Clientes sem pedidos (ou sem
intervalo) ficam com nulo: imputar é decisão do modelo, dentro do Pipeline.

Uso:
    from src.dados import carregar_dados_limpos
    from src import features_comportamentais as fc
    clientes, pedidos = carregar_dados_limpos()
    feats = fc.agregar_pedidos(fc.converter_pedidos(pedidos), excluir_ultimo=False)
    base = fc.juntar_clientes(clientes, feats)  # left join + tem_pedidos
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dados import limites

TETO_ABERTO = 1_000_000  # limite superior a partir do qual a faixa é tratada como aberta

DICIONARIO = {
    "n_pedidos": "Número de pedidos do cliente (quase sempre 3: não entra no modelo).",
    "n_dias_com_pedido": "Número de dias distintos com pedido (vários pedidos no mesmo dia contam como 1).",
    "valor_media": "Média do valor dos pedidos (ponto médio da faixa, R$).",
    "valor_mediana": "Mediana do valor dos pedidos (ponto médio da faixa, R$).",
    "valor_min": "Menor valor de pedido (ponto médio da faixa, R$).",
    "valor_max": "Maior valor de pedido (ponto médio da faixa, R$).",
    "valor_ord_max": "Maior faixa de valor (0 = (0, 500], 3 = acima de 2.000).",
    "atraso_media": "Média do atraso em dias (sem atraso = 0; ponto médio da faixa).",
    "atraso_mediana": "Mediana do atraso em dias (sem atraso = 0; ponto médio da faixa).",
    "atraso_max": "Maior atraso em dias (ponto médio da faixa; faixa aberta = limite inferior).",
    "atraso_ord_max": "Maior faixa de atraso (0 = sem atraso, 8 = mais de 60 dias).",
    "pct_atraso": "Proporção de pedidos com qualquer atraso (0 a 1).",
    "pct_atraso_2d": "Proporção de pedidos com atraso acima de 2 dias (0 a 1).",
    "atraso_ultimo": "Atraso do pedido mais recente (dias). Suspeita de vazamento: ver docstring do módulo.",
    "atraso_tendencia": "Atraso do pedido mais recente menos o do mais antigo (dias); nulo com 1 pedido.",
    "intervalo_medio_dias": "Média de dias entre pedidos consecutivos; nulo com 1 pedido.",
    "span_dias": "Dias entre o primeiro e o último pedido.",
}
FEATURES = [f for f in DICIONARIO if f != "n_pedidos"]


def _ponto_medio(faixa: pd.Series) -> pd.Series:
    """Ponto médio de cada faixa; faixa aberta (limite superior >= TETO_ABERTO) vale o limite inferior."""
    lim = limites(faixa)
    return ((lim["inf"] + lim["sup"]) / 2).where(lim["sup"] < TETO_ABERTO, lim["inf"])


def _posicao(faixa: pd.Series) -> pd.Series:
    """Código da faixa (categórica ordenada da limpeza), em float para o nulo ficar nulo em qualquer subconjunto."""
    return faixa.cat.codes.astype(float).where(faixa.notna())


def converter_pedidos(pedidos: pd.DataFrame) -> pd.DataFrame:
    """Cria valor_num/_ord, atraso_num/_ord, atrasou e atrasou_2d em cada pedido (sem agregar). Nulo fica nulo."""
    df = pedidos.copy()
    df["valor_num"] = _ponto_medio(df["valor"])
    df["valor_ord"] = _posicao(df["valor"])

    atraso = _ponto_medio(df["atraso"])
    df["atraso_num"] = atraso.clip(lower=0)  # "(-100, 0]" = sem atraso
    df["atraso_ord"] = _posicao(df["atraso"])
    df["atrasou"] = (df["atraso_num"] > 0).astype(float).where(df["atraso_num"].notna())
    df["atrasou_2d"] = (df["atraso_num"] > 2).astype(float).where(df["atraso_num"].notna())
    return df


def ordenar_pedidos(pedidos: pd.DataFrame) -> pd.DataFrame:
    """Ordem cronológica por cliente; id_pedido desempata pedidos do mesmo dia."""
    return pedidos.sort_values(["id_cliente", "data_entrega", "id_pedido"]).reset_index(drop=True)


def remover_ultimo_pedido(pedidos: pd.DataFrame) -> pd.DataFrame:
    """Tira o pedido mais recente de cada cliente (clientes com 1 pedido somem)."""
    df = ordenar_pedidos(pedidos)
    ultimo = df.groupby("id_cliente").cumcount(ascending=False) == 0
    return df.loc[~ultimo].reset_index(drop=True)


def agregar_pedidos(pedidos: pd.DataFrame, excluir_ultimo: bool = False) -> pd.DataFrame:
    """Uma linha por cliente com as features de DICIONARIO, indexada por id_cliente.

    pedidos: saída de converter_pedidos. Com excluir_ultimo=True, agrega só os
    pedidos anteriores ao mais recente.
    """
    df = remover_ultimo_pedido(pedidos) if excluir_ultimo else ordenar_pedidos(pedidos)
    df["intervalo_dias"] = df.groupby("id_cliente")["data_entrega"].diff().dt.days
    g = df.groupby("id_cliente")

    feats = g.agg(
        n_pedidos=("id_pedido", "size"),
        n_dias_com_pedido=("data_entrega", "nunique"),
        valor_media=("valor_num", "mean"),
        valor_mediana=("valor_num", "median"),
        valor_min=("valor_num", "min"),
        valor_max=("valor_num", "max"),
        valor_ord_max=("valor_ord", "max"),
        atraso_media=("atraso_num", "mean"),
        atraso_mediana=("atraso_num", "median"),
        atraso_max=("atraso_num", "max"),
        atraso_ord_max=("atraso_ord", "max"),
        pct_atraso=("atrasou", "mean"),
        pct_atraso_2d=("atrasou_2d", "mean"),
        intervalo_medio_dias=("intervalo_dias", "mean"),
    )
    # nth, e não first/last: first/last pulam nulos e pegariam outro pedido
    feats["atraso_ultimo"] = g["atraso_num"].nth(-1).set_axis(g.size().index)
    primeiro = g["atraso_num"].nth(0).set_axis(g.size().index)
    feats["atraso_tendencia"] = (feats["atraso_ultimo"] - primeiro).where(feats["n_pedidos"] > 1)
    feats["span_dias"] = (g["data_entrega"].max() - g["data_entrega"].min()).dt.days
    return feats[list(DICIONARIO)]


def juntar_clientes(clientes: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Junta as features aos clientes (left join), na ordem de `clientes`. Sem pedidos -> nulos e tem_pedidos = 0."""
    df = clientes.merge(feats, left_on="id_cliente", right_index=True, how="left")
    df["tem_pedidos"] = df["id_cliente"].isin(feats.index).astype(int)
    return df
