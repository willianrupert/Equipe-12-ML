"""Modelo comportamental: features de aplicação + histórico de pedidos, para os clientes recorrentes.

Junta as 10 features do modelo de aplicação (fa.FEATURES_FINAIS) com as agregações dos pedidos
(src.features_comportamentais, versão com todos os pedidos, escolhida no notebook 04). O alvo é
o mesmo; a população avaliada são os clientes com histórico de pedidos (os "recorrentes": a base
não tem data de referência para aplicar o corte de 30 dias do enunciado).

O modelo é treinado e avaliado só nos clientes com pedidos (treinar_em="com pedidos", o padrão):
é para eles que ele existe. Treinar em todos, com as features de pedidos nulas para quem não tem
histórico, foi descartado (notebook 06, seção 4). A escolha das features fica no notebook 06
(BLOCOS).

Os modelos reaproveitam os de src.modelo_aplicacao (mesmos parâmetros-base e espaços de busca).
Nas árvores, as features de pedidos passam como estão. Na logística, são padronizadas; o nulo
(atraso_tendencia e intervalo_medio_dias de quem tem 1 pedido) é imputado pela mediana do treino,
com uma coluna indicando o nulo.

Uso:
    from src import modelo_comportamental as mc
    X = mc.montar_X(clientes, pedidos)                 # aplicação + pedidos, uma linha por cliente
    res = mc.validacao_cruzada(mc.random_forest(cols), X_treino, y_treino, folds, com_pedidos)  # treina e avalia nos recorrentes
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src import features_aplicacao as fa
from src import features_comportamentais as fc
from src import modelo_aplicacao as ma
from src.dados import DIR_PROCESSED
from src.split import SEED

ARQ_SCORES = DIR_PROCESSED / "scores_comportamental.csv"

# Blocos de features de pedidos (notebook 04, seções 6 e 7). Dos pares redundantes (Spearman >= 0,99),
# fica um de cada: atraso_ord_max = atraso_max, valor_max e valor_ord_max ~ valor_media, span_dias ~
# intervalo_medio_dias. atraso_ultimo sai porque não acrescenta com o resto do bloco (ganho 0,000).
BLOCOS = {
    "atraso": ["atraso_media", "atraso_mediana", "atraso_max", "pct_atraso", "pct_atraso_2d", "atraso_tendencia"],
    "valor": ["valor_media", "valor_mediana", "valor_min"],
    "ritmo": ["n_dias_com_pedido", "intervalo_medio_dias"],
}


def montar_X(clientes: pd.DataFrame, pedidos: pd.DataFrame) -> pd.DataFrame:
    """Features de aplicação (todas de fa.criar_features) + features de pedidos (fc.FEATURES), mesmo índice de `clientes`.

    Quem não tem pedidos fica com as features de pedidos nulas. Tudo é determinístico: pode rodar antes do split.
    """
    feats = fc.agregar_pedidos(fc.converter_pedidos(pedidos))
    comport = clientes[["id_cliente"]].join(feats, on="id_cliente")[fc.FEATURES]
    return pd.concat([fa.criar_features(clientes), comport], axis=1)


# --- Modelos ---

def _pre_arvores(pedidos_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("aplicacao", fa.montar_preprocessador(fa.FEATURES_FINAIS), fa.FEATURES_FINAIS),
        ("pedidos", "passthrough", pedidos_cols),
    ], verbose_feature_names_out=False).set_output(transform="pandas")


def arvores(modelo, pedidos_cols: list[str]) -> Pipeline:
    """Pré-processador do modelo de aplicação nas 10 features dele; as de pedidos passam como estão."""
    return make_pipeline(_pre_arvores(pedidos_cols), modelo)


def random_forest(pedidos_cols: list[str], **params) -> Pipeline:
    """Random Forest com os parâmetros-base do modelo de aplicação (300 árvores, ≥ 20 clientes por folha)."""
    return arvores(clone(ma.random_forest(**params)[-1]), pedidos_cols)


def xgboost(pedidos_cols: list[str], **params) -> Pipeline:
    return arvores(clone(ma.xgboost(**params)[-1]), pedidos_cols)


def lightgbm(pedidos_cols: list[str], **params) -> Pipeline:
    return arvores(clone(ma.lightgbm(**params)[-1]), pedidos_cols)


def logistica(pedidos_cols: list[str], C: float = 1.0) -> Pipeline:
    """Regressão logística L2: pré-processador do modelo de aplicação + pedidos imputados (mediana, com indicador) e padronizados."""
    pre = ColumnTransformer([
        ("aplicacao", ma.preprocessador_logistica(fa.FEATURES_FINAIS), fa.FEATURES_FINAIS),
        ("pedidos", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), pedidos_cols),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    return make_pipeline(pre, LogisticRegression(C=C, max_iter=2000))


FABRICAS = {"random forest": random_forest, "xgboost": xgboost, "lightgbm": lightgbm}
ESPACOS = ma.ESPACOS


def aplicacao_referencia() -> Pipeline:
    """O modelo de aplicação final sem a calibração (a calibração não muda a ordem), para comparar fold a fold."""
    return ma.xgboost(**ma.PARAMS_FINAIS)


# --- Validação cruzada ---

def validacao_cruzada(modelo, X: pd.DataFrame, y: pd.Series, folds: list[tuple], com_pedidos: pd.Series,
                      treinar_em: str = "com pedidos") -> dict:
    """Ajusta `modelo` em cada fold e avalia só nos clientes com pedidos da validação.

    treinar_em: "com pedidos" (só os clientes com pedidos do treino do fold, o modelo comportamental) ou
      "todos" (todo o treino do fold: usado para o modelo de aplicação de referência e no registro da seção 4).
    Devolve auc_validacao e auc_treino por fold (o de treino também só nos clientes com pedidos) e a
    matriz pred (folds × clientes) com a probabilidade prevista na validação e NaN no resto.
    """
    if treinar_em not in ("todos", "com pedidos"):
        raise ValueError(f"treinar_em deve ser 'todos' ou 'com pedidos', não {treinar_em!r}")
    tem = np.asarray(com_pedidos, dtype=bool)
    pred = np.full((len(folds), len(X)), np.nan, dtype=np.float32)
    auc_val, auc_tr = [], []
    for i, (tr, va) in enumerate(folds):
        tr_fit = tr if treinar_em == "todos" else tr[tem[tr]]
        tr_aval, va_aval = tr[tem[tr]], va[tem[va]]
        m = clone(modelo).fit(X.iloc[tr_fit], y.iloc[tr_fit])
        pred[i, va_aval] = m.predict_proba(X.iloc[va_aval])[:, 1]
        auc_val.append(roc_auc_score(y.iloc[va_aval], pred[i, va_aval]))
        auc_tr.append(roc_auc_score(y.iloc[tr_aval], m.predict_proba(X.iloc[tr_aval])[:, 1]))
    return {"auc_validacao": np.array(auc_val), "auc_treino": np.array(auc_tr), "pred": pred}


score_fora_da_amostra = ma.score_fora_da_amostra
auc_bootstrap = ma.auc_bootstrap
diferenca_bootstrap = ma.diferenca_bootstrap
calibrado = ma.calibrado


# --- Corte, métricas de classificação e tempo de inferência ---

def corte_f1(y, score) -> float:
    """Corte que maximiza o F1 (a usar com scores fora da amostra do treino, nunca do teste)."""
    y, score = np.asarray(y), np.asarray(score)
    cortes = np.unique(np.round(score, 3))
    f1 = [f1_score(y, score >= c, zero_division=0) for c in cortes]
    return float(cortes[int(np.argmax(f1))])


def metricas_no_corte(y, score, corte: float) -> dict:
    """Precisão, recall e F1 classificando como inadimplente quem tem score >= corte."""
    prev = np.asarray(score) >= corte
    return {
        "precisão": precision_score(y, prev, zero_division=0),
        "recall": recall_score(y, prev, zero_division=0),
        "F1": f1_score(y, prev, zero_division=0),
        "% aprovados": 1 - prev.mean(),
    }


def tempo_inferencia(modelo, X: pd.DataFrame, repeticoes: int = 20) -> dict:
    """Tempo de predict_proba (mediana de `repeticoes`): por lote inteiro e por cliente, em ms."""
    modelo.predict_proba(X.iloc[:5])  # aquecimento
    tempos = []
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        modelo.predict_proba(X)
        tempos.append(time.perf_counter() - inicio)
    lote = float(np.median(tempos)) * 1000
    um = []
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        modelo.predict_proba(X.iloc[:1])
        um.append(time.perf_counter() - inicio)
    return {"ms por lote": lote, "clientes no lote": len(X), "ms por cliente (lote)": lote / len(X),
            "ms para 1 cliente": float(np.median(um)) * 1000}


# --- Scores ---

def salvar_scores(scores: pd.DataFrame) -> None:
    """Salva id_cliente, conjunto e score em data/processed/ (fora do git: são dados de clientes)."""
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    scores[["id_cliente", "conjunto", "score"]].to_csv(ARQ_SCORES, index=False)


def carregar_scores() -> pd.DataFrame:
    return pd.read_csv(ARQ_SCORES)
