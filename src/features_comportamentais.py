"""Features do histórico de pedidos (KAN-7, Pessoa D).

Ponto de partida: o código abaixo saiu de src/dados.py na revisão da KAN-1 e
mantém as decisões tomadas lá. Cabe à KAN-7 revisar e decidir se mantém:
- valor e atraso (faixas) -> ponto médio (_num) e ordinal (_ord);
- atraso "(-100, 0]" significa sem atraso, então atraso_num = 0 (o ponto
  médio seria -50);
- flag atrasou (atraso_num > 0).

Atenção ao vazamento temporal: as agregações por cliente devem usar só pedidos
anteriores à data de referência. Hoje os dados não têm data de cadastro nem
data em que a inadimplência foi observada (pergunta em aberto da KAN-1).
"""
from __future__ import annotations

import pandas as pd

from src.dados import intervalo_para_numero, intervalo_para_ordinal


def converter_pedidos(pedidos: pd.DataFrame, razao_aberto: float = 100) -> pd.DataFrame:
    """Cria valor_num/_ord, atraso_num/_ord e atrasou em cada pedido (sem agregar)."""
    df = pedidos.copy()
    df["valor_num"] = intervalo_para_numero(df["valor"], razao_aberto)
    df["valor_ord"] = intervalo_para_ordinal(df["valor"])

    atraso_num = intervalo_para_numero(df["atraso"], razao_aberto)
    df["atraso_num"] = atraso_num.where(atraso_num > 0, 0.0)
    df["atraso_ord"] = intervalo_para_ordinal(df["atraso"])
    df["atrasou"] = (df["atraso_num"] > 0).astype(int)
    return df
