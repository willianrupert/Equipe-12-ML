"""Features do histórico de pedidos (KAN-7).

Duas etapas:
1. converter_pedidos: representa as faixas de cada pedido como número (sem agregar).
   Decisões herdadas da KAN-1 e mantidas aqui:
   - valor e atraso (faixas) -> ponto médio (_num) e ordinal (_ord);
   - atraso "(-100, 0]" significa sem atraso, então atraso_num = 0 (o ponto
     médio seria -50);
   - atrasou = atraso_num > 0; atrasou_2d = atraso_num > 2, porque 46% dos
     pedidos caem em "(0, 2]" e um atraso de 1-2 dias pode ser só logística.
2. agregar_pedidos: uma linha por cliente, indexada por id_cliente.

Vazamento temporal: os dados não têm data de referência nem a data em que a
inadimplência foi observada, e quase todo cliente tem exatamente os últimos 3
pedidos. O atraso do último pedido separa muito mais o alvo do que o dos
anteriores, o que sugere que ele pode ser o pedido em que a inadimplência
aconteceu. Por isso agregar_pedidos aceita excluir_ultimo=True (versão
conservadora), e a KAN-8 deve comparar as duas versões. Recência absoluta
(dias até uma data fixa) não é calculada: ela mediria a época, não o
comportamento.

Nada aqui usa o alvo nem depende de estatísticas do conjunto inteiro, então as
features podem ser calculadas antes do split. Nulos (clientes sem pedidos ou
sem intervalo) ficam nulos: imputar é decisão do modelo, dentro do Pipeline.
"""
from __future__ import annotations

import pandas as pd

from src.dados import intervalo_para_numero, intervalo_para_ordinal

DICIONARIO = {
    "n_pedidos": "Número de pedidos válidos do cliente (quase sempre 3).",
    "valor_media": "Média do valor dos pedidos (ponto médio da faixa, R$).",
    "valor_min": "Menor valor de pedido (ponto médio da faixa, R$).",
    "valor_max": "Maior valor de pedido (ponto médio da faixa, R$).",
    "valor_ord_max": "Maior faixa de valor (0 = (0, 500], 3 = acima de 2000).",
    "atraso_media": "Média do atraso em dias (sem atraso = 0; ponto médio da faixa).",
    "atraso_max": "Maior atraso em dias (ponto médio da faixa; faixa aberta = limite inferior).",
    "atraso_ord_max": "Maior faixa de atraso (0 = sem atraso, 8 = mais de 60 dias).",
    "pct_atraso": "Proporção de pedidos com qualquer atraso (0 a 1).",
    "pct_atraso_2d": "Proporção de pedidos com atraso acima de 2 dias (0 a 1).",
    "atraso_ultimo": "Atraso do pedido mais recente (dias). Suspeita de vazamento: ver docstring.",
    "atraso_tendencia": "Atraso do pedido mais recente menos o do mais antigo (dias); nulo com 1 pedido.",
    "intervalo_medio_dias": "Média de dias entre pedidos consecutivos; nulo com 1 pedido.",
    "span_dias": "Dias entre o primeiro e o último pedido.",
}


def converter_pedidos(pedidos: pd.DataFrame, razao_aberto: float = 100) -> pd.DataFrame:
    """Cria valor_num/_ord, atraso_num/_ord, atrasou e atrasou_2d em cada pedido (sem agregar)."""
    df = pedidos.copy()
    df["valor_num"] = intervalo_para_numero(df["valor"], razao_aberto)
    df["valor_ord"] = intervalo_para_ordinal(df["valor"])

    atraso_num = intervalo_para_numero(df["atraso"], razao_aberto)
    df["atraso_num"] = atraso_num.where(atraso_num > 0, 0.0)
    df["atraso_ord"] = intervalo_para_ordinal(df["atraso"])
    df["atrasou"] = (df["atraso_num"] > 0).astype(int)
    df["atrasou_2d"] = (df["atraso_num"] > 2).astype(int)
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
    """Uma linha por cliente com as features de DICIONARIO.

    pedidos: saída de converter_pedidos. Com excluir_ultimo=True, agrega só os
    pedidos anteriores ao mais recente.
    """
    df = remover_ultimo_pedido(pedidos) if excluir_ultimo else ordenar_pedidos(pedidos)
    df["intervalo_dias"] = df.groupby("id_cliente")["data_entrega"].diff().dt.days
    g = df.groupby("id_cliente")

    feats = g.agg(
        n_pedidos=("id_pedido", "size"),
        valor_media=("valor_num", "mean"),
        valor_min=("valor_num", "min"),
        valor_max=("valor_num", "max"),
        valor_ord_max=("valor_ord", "max"),
        atraso_media=("atraso_num", "mean"),
        atraso_max=("atraso_num", "max"),
        atraso_ord_max=("atraso_ord", "max"),
        pct_atraso=("atrasou", "mean"),
        pct_atraso_2d=("atrasou_2d", "mean"),
        atraso_ultimo=("atraso_num", "last"),
        intervalo_medio_dias=("intervalo_dias", "mean"),
    )
    primeiro = g["atraso_num"].first()
    feats["atraso_tendencia"] = (feats["atraso_ultimo"] - primeiro).where(feats["n_pedidos"] > 1)
    feats["span_dias"] = (g["data_entrega"].max() - g["data_entrega"].min()).dt.days
    return feats[list(DICIONARIO)]


def juntar_clientes(clientes: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Junta as features aos clientes (left join). Sem pedidos -> features nulas e tem_pedidos = 0."""
    df = clientes.merge(feats, left_on="id_cliente", right_index=True, how="left")
    df["tem_pedidos"] = df["id_cliente"].isin(feats.index).astype(int)
    return df
