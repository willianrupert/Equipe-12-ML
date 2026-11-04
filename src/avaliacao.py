"""Validação cruzada e comparação pareada de configurações (KAN-5, KAN-6, KAN-8).

As configurações comparadas usam os mesmos folds (src.split.folds), então a
comparação é pareada: fold a fold, nos mesmos clientes.

Os folds da validação cruzada repetida reaproveitam clientes, então os AUCs não
são independentes e o erro padrão simples (desvio / raiz do nº de folds) sai
pequeno demais. comparar() usa a correção de Nadeau e Bengio (2003), que leva
em conta a sobreposição entre os folds.

Uso:
    aucs = auc_por_fold(pipeline, X_treino, y_treino, folds)
    comparar(aucs_com_a_feature, aucs_sem_a_feature, folds)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate


def auc_por_fold(modelo, X, y, folds: list[tuple]) -> np.ndarray:
    """ROC-AUC de `modelo` (de preferência um Pipeline com o pré-processador) em cada fold de validação."""
    res = cross_validate(modelo, X, y, cv=folds, scoring="roc_auc", error_score="raise")
    return res["test_score"]


def comparar(auc_com: np.ndarray, auc_sem: np.ndarray, folds: list[tuple]) -> dict:
    """Quanto um item (feature, bloco, opção) acrescenta ao AUC, com erro padrão corrigido.

    ganho = média de (auc_com - auc_sem) nos mesmos folds. Regra da KAN-5: o item
    "acrescenta" se o ganho passa de 1 erro padrão corrigido, "atrapalha" se a perda
    passa de 1 erro padrão, e é "neutro" no meio.
    """
    d = np.asarray(auc_com) - np.asarray(auc_sem)
    n_treino, n_validacao = len(folds[0][0]), len(folds[0][1])
    erro_padrao = float(np.sqrt((1 / len(d) + n_validacao / n_treino) * d.var(ddof=1)))
    ganho = float(d.mean())
    veredito = "acrescenta" if ganho > erro_padrao else "atrapalha" if ganho < -erro_padrao else "neutro"
    return {"ganho": ganho, "erro padrão": erro_padrao, "ganho / erro padrão": ganho / erro_padrao, "veredito": veredito}


def auc_por_fold_e_grupo(modelo, X, y, folds: list[tuple], grupos: pd.Series) -> pd.DataFrame:
    """ROC-AUC em cada fold, na validação inteira e dentro de cada grupo (ex: com e sem registro na Serasa).

    O modelo é treinado no treino do fold, como em auc_por_fold; só a avaliação é separada por grupo.
    Devolve um DataFrame com uma linha por fold e uma coluna por grupo, mais a coluna "todos".
    """
    linhas = []
    for treino, validacao in folds:
        m = clone(modelo).fit(X.iloc[treino], y.iloc[treino])
        score = m.predict_proba(X.iloc[validacao])[:, 1]
        y_val, g_val = y.iloc[validacao].to_numpy(), grupos.iloc[validacao].to_numpy()
        linha = {"todos": roc_auc_score(y_val, score)}
        for g in pd.unique(grupos):
            linha[g] = roc_auc_score(y_val[g_val == g], score[g_val == g])
        linhas.append(linha)
    return pd.DataFrame(linhas)
