"""Features do modelo de aplicação (KAN-5, Pessoa B).

Ponto de partida: o código abaixo saiu de src/dados.py na revisão da KAN-1 e
mantém as decisões tomadas lá. Cabe à KAN-5 revisar cada uma e decidir se
mantém, troca ou testa alternativas:
- faixas -> ponto médio (_num) e ordinal (_ord); o limiar de faixa aberta
  (razao_aberto=100) também é uma escolha;
- ifood_faixa_preco "$".."$$$$$" -> 1..5 (assume distância igual entre faixas;
  tratar como categoria é igualmente válido);
- flags de presença (tem_ifood, tem_google_maps, tem_credores_serasa);
- google_maps_tem_website: o nulo junta "não está no Maps" e "está, mas não tem
  site". Nos dados só aparecem 1 ou nulo.

Use só variáveis disponíveis antes do cadastro. Transformações que dependem dos
dados (agrupar categorias raras, imputar, target encoding) devem ficar dentro do
Pipeline e ser ajustadas só no treino.
"""
from __future__ import annotations

import pandas as pd

from src.dados import INTERVALOS_CLIENTES, intervalo_para_numero, intervalo_para_ordinal


def adicionar_faixas_numericas(clientes: pd.DataFrame, razao_aberto: float = 100) -> pd.DataFrame:
    """Para cada coluna em faixa, cria <col>_num (ponto médio) e <col>_ord (ordinal)."""
    df = clientes.copy()
    for col in INTERVALOS_CLIENTES:
        df[f"{col}_num"] = intervalo_para_numero(df[col], razao_aberto)
        df[f"{col}_ord"] = intervalo_para_ordinal(df[col])
    return df


def adicionar_faixa_preco_ifood(clientes: pd.DataFrame) -> pd.DataFrame:
    """Faixa de preço do iFood ("$" a "$$$$$") -> 1 a 5; nulo continua nulo."""
    df = clientes.copy()
    df["ifood_faixa_preco_num"] = df["ifood_faixa_preco"].str.len().astype("Int64")
    return df


def adicionar_flags_presenca(clientes: pd.DataFrame) -> pd.DataFrame:
    """Nulo nas colunas de plataformas = estabelecimento não está na plataforma."""
    df = clientes.copy()
    df["tem_ifood"] = df[["ifood_contagem_avaliacoes", "ifood_faixa_preco"]].notna().any(axis=1).astype(int)
    df["tem_google_maps"] = df["google_maps_contagem_avaliacoes"].notna().astype(int)
    df["tem_credores_serasa"] = df["serasa_credores"].notna().astype(int)
    return df


def preencher_website(clientes: pd.DataFrame) -> pd.DataFrame:
    """google_maps_tem_website: nulo -> 0 (sem site conhecido). Perde a distinção descrita no topo."""
    df = clientes.copy()
    df["google_maps_tem_website"] = df["google_maps_tem_website"].fillna(0).astype(int)
    return df


def features_herdadas_kan1(clientes: pd.DataFrame) -> pd.DataFrame:
    """Aplica tudo o que a KAN-1 fazia antes da revisão (para comparação)."""
    df = adicionar_faixas_numericas(clientes)
    df = adicionar_faixa_preco_ifood(df)
    df = adicionar_flags_presenca(df)
    return preencher_website(df)
