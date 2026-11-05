"""Split treino/teste e folds da validação cruzada, compartilhados pelos modelos de aplicação (KAN-6) e comportamental (KAN-8).

80/20, estratificado por inadimplente × tem_pedidos: os 664 clientes com pedidos
ficam na mesma proporção no treino e no teste, para a KAN-8 comparar os dois
modelos nos mesmos clientes. "Tem pedidos" só entra na estratificação, nunca como
feature do modelo de aplicação.

O treino é avaliado por validação cruzada estratificada repetida (5 folds × 5
repetições), com a mesma estratificação. O teste é usado uma vez só, na avaliação final.

O split é salvo em data/processed/ (fora do git) e recriado igual pela seed.

Uso:
    from src import split
    s = split.carregar_split()          # id_cliente -> "treino" | "teste"
    treino = clientes[clientes["id_cliente"].map(s) == "treino"]
    cv = split.folds(treino, pedidos)   # lista de (índices de treino, índices de validação)
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.dados import DIR_PROCESSED

SEED = 42
FRAC_TESTE = 0.20
N_FOLDS, N_REPETICOES = 5, 5
ARQ_SPLIT = DIR_PROCESSED / "split.csv"


def estrato(clientes: pd.DataFrame, pedidos: pd.DataFrame) -> pd.Series:
    """Rótulo de estratificação: alvo × tem pedidos, ex: "1_com_pedidos"."""
    tem_pedidos = clientes["id_cliente"].isin(pedidos["id_cliente"]).map({True: "com_pedidos", False: "sem_pedidos"})
    return clientes["inadimplente"].astype(str) + "_" + tem_pedidos


def criar_split(clientes: pd.DataFrame, pedidos: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Série id_cliente -> "treino" | "teste"."""
    treino, teste = train_test_split(
        clientes["id_cliente"], test_size=FRAC_TESTE, stratify=estrato(clientes, pedidos), random_state=seed
    )
    conjunto = pd.concat([pd.Series("treino", index=treino.values), pd.Series("teste", index=teste.values)])
    return conjunto.rename_axis("id_cliente").rename("conjunto").sort_index()


def folds(treino: pd.DataFrame, pedidos: pd.DataFrame, seed: int = SEED) -> list[tuple]:
    """Folds da validação cruzada no treino (posições, para usar em cross_validate(cv=...)).

    5 folds × 5 repetições = 25 pares (treino, validação), estratificados como o split.
    """
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPETICOES, random_state=seed)
    return list(cv.split(treino, estrato(treino, pedidos)))


def salvar_split(conjunto: pd.Series) -> None:
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    conjunto.rename_axis("id_cliente").rename("conjunto").to_csv(ARQ_SPLIT)


def carregar_split() -> pd.Series:
    return pd.read_csv(ARQ_SPLIT, index_col="id_cliente")["conjunto"]
