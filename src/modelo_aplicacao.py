"""Modelo de aplicação (KAN-6): modelos, validação cruzada com predições fora da amostra, calibração e scores.

Usa as features e o pré-processador da KAN-5 (src.features_aplicacao). Os modelos de árvore
(árvore de decisão, Random Forest, XGBoost e LightGBM) usam fa.montar_preprocessador. A
regressão logística tem um pré-processador próprio, definido aqui: one-hot, escala e, como
opção, as interações que a EDA indicou (seção 6).

O modelo final é escolhido no notebook 05_modelo_aplicacao.ipynb, com a regra da KAN-5: um
modelo mais complexo só substitui o mais simples se ganhar mais de 1 erro padrão corrigido
(avaliacao.comparar). Os parâmetros escolhidos ficam fixados em PARAMS_FINAIS.

Scores de cada cliente:
- treino: fora da amostra, a média das 5 predições de validação (uma por repetição dos folds).
  Um cliente nunca recebe score de um modelo que o viu no ajuste;
- teste: do modelo final, ajustado no treino inteiro.
Os scores são probabilidades calibradas (sigmoide). Ficam em data/processed/ (fora do git).

Uso:
    from src import modelo_aplicacao as ma
    res = ma.validacao_cruzada(ma.modelo_final(), X_treino, y_treino, folds)
    score_treino = ma.score_fora_da_amostra(res)   # um score por cliente do treino
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import features_aplicacao as fa
from src.dados import DIR_PROCESSED
from src.split import SEED

ARQ_SCORES = DIR_PROCESSED / "scores_aplicacao.csv"

# Interações da EDA (seção 6): o peso do sócio negativado e o dos credores mudam com a idade do CNPJ.
INTERACOES = {
    "idade_x_socio": ("idade_cnpj_ord", "socio_negativado"),
    "idade_x_credores": ("idade_cnpj_ord", "n_credores"),
}


# --- Regressão logística ---

def _com_interacoes(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{nome: df[a] * df[b] for nome, (a, b) in INTERACOES.items()})


def _nomes_com_interacoes(_, nomes) -> list[str]:
    return [*nomes, *INTERACOES]


def preprocessador_logistica(colunas: list[str] | None = None, interacoes: bool = False) -> ColumnTransformer:
    """Pré-processador da regressão logística, a ajustar só no treino, dentro do Pipeline.

    Numéricas e binárias: padronizadas (média 0, desvio 1), para a regularização tratar todas igual.
    Categóricas: one-hot, com as raras (< fa.MIN_FREQ clientes no treino) numa coluna só, como nas
    árvores. Categoria que não apareceu no treino cai na coluna das raras.
    interacoes: acrescenta os produtos de INTERACOES antes da padronização.
    """
    colunas = fa.FEATURES_FINAIS if colunas is None else colunas
    num = [c for c in colunas if c in fa.NUMERICAS + fa.BINARIAS]
    cat = [c for c in colunas if c in fa.CATEGORICAS]
    if interacoes:
        faltando = {c for par in INTERACOES.values() for c in par} - set(num)
        if faltando:
            raise ValueError(f"interações precisam de {sorted(faltando)} nas colunas")
        passo_num = make_pipeline(
            FunctionTransformer(_com_interacoes, feature_names_out=_nomes_com_interacoes),
            StandardScaler(),
        )
    else:
        passo_num = StandardScaler()
    onehot = OneHotEncoder(min_frequency=fa.MIN_FREQ, handle_unknown="infrequent_if_exist", sparse_output=False)
    return ColumnTransformer(
        [("num", passo_num, num), ("cat", onehot, cat)], verbose_feature_names_out=False
    ).set_output(transform="pandas")


def logistica(C: float = 1.0, interacoes: bool = False) -> Pipeline:
    """Regressão logística com regularização L2 (C menor = mais regularização)."""
    return make_pipeline(
        preprocessador_logistica(interacoes=interacoes),
        LogisticRegression(C=C, max_iter=2000),
    )


# --- Modelos de árvore ---

def arvores(modelo) -> Pipeline:
    """Pré-processador da KAN-5 com as 10 features finais + `modelo`."""
    return make_pipeline(fa.montar_preprocessador(fa.FEATURES_FINAIS), modelo)


def arvore_decisao() -> Pipeline:
    """Linha de base da KAN-5: profundidade até 4, ≥ 30 clientes por folha."""
    return arvores(DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=SEED))


def random_forest(**params) -> Pipeline:
    """Random Forest. Sem parâmetros, é a da KAN-5 (300 árvores, ≥ 20 clientes por folha, √ das features)."""
    base = dict(n_estimators=300, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=SEED)
    return arvores(RandomForestClassifier(**(base | params)))


def xgboost(**params) -> Pipeline:
    base = dict(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED, verbosity=0)
    return arvores(XGBClassifier(**(base | params)))


def lightgbm(**params) -> Pipeline:
    # Os códigos das categóricas entram como números, como nas outras árvores: o tratamento nativo de
    # categóricas do LightGBM tende a superajustar justamente nas categorias raras.
    base = dict(n_estimators=300, num_leaves=8, learning_rate=0.05, min_child_samples=30,
                n_jobs=-1, random_state=SEED, verbose=-1)
    return arvores(LGBMClassifier(**(base | params)))


FABRICAS = {"random forest": random_forest, "xgboost": xgboost, "lightgbm": lightgbm}

# Espaços da busca aleatória (notebook 05, seção 4). Favorecem modelos regularizados: com 2.400
# clientes no treino, árvores profundas e folhas pequenas decoram o treino.
ESPACOS = {
    "random forest": {
        "min_samples_leaf": [5, 10, 20, 30, 50],
        "max_features": ["sqrt", 0.5, 0.8],
        "max_depth": [None, 4, 6, 10],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 5, 10, 20],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1, 5, 10],
    },
    "lightgbm": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [4, 8, 16, 31],
        "min_child_samples": [10, 20, 30, 50, 100],
        "subsample": [0.6, 0.8, 1.0],
        "subsample_freq": [1],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0, 1, 5, 10],
    },
}


# --- Modelo final ---

def calibrado(modelo) -> CalibratedClassifierCV:
    """Calibração sigmoide (Platt), mais estável que a isotônica com poucos clientes.

    ensemble=False: o calibrador aprende com predições fora da amostra (5 folds internos) e o
    modelo é reajustado no treino inteiro. O score calibrado é uma função crescente do score
    original, então a ordem dos clientes (e o AUC) não muda.
    """
    return CalibratedClassifierCV(
        modelo, method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), ensemble=False
    )


# Escolhido no notebook 05 (seção 5): a melhor configuração do XGBoost na busca, a única que ganha da Random
# Forest da KAN-5 por mais de 1 erro padrão nos 25 folds (+0,004) e também quando é escolhida em parte das
# repetições e avaliada nas outras (seção 5.1).
MODELO_FINAL = "xgboost"
PARAMS_FINAIS = {
    "n_estimators": 500, "learning_rate": 0.01, "max_depth": 5, "min_child_weight": 1,
    "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 10,
}


def modelo_final() -> CalibratedClassifierCV:
    """Modelo final, calibrado e ainda não ajustado."""
    return calibrado(FABRICAS[MODELO_FINAL](**PARAMS_FINAIS))


# --- Validação cruzada com predições fora da amostra ---

def validacao_cruzada(modelo, X: pd.DataFrame, y: pd.Series, folds: list[tuple]) -> dict:
    """Ajusta `modelo` no treino de cada fold e guarda o AUC e as predições de validação.

    Devolve:
    - auc_validacao, auc_treino: um AUC por fold (o de treino serve para medir overfitting);
    - pred: matriz folds × clientes com a probabilidade prevista para os clientes de validação
      de cada fold e NaN nos demais.
    """
    pred = np.full((len(folds), len(X)), np.nan, dtype=np.float32)
    auc_val, auc_tr = [], []
    for i, (tr, va) in enumerate(folds):
        m = clone(modelo).fit(X.iloc[tr], y.iloc[tr])
        pred[i, va] = m.predict_proba(X.iloc[va])[:, 1]
        auc_val.append(roc_auc_score(y.iloc[va], pred[i, va]))
        auc_tr.append(roc_auc_score(y.iloc[tr], m.predict_proba(X.iloc[tr])[:, 1]))
    return {"auc_validacao": np.array(auc_val), "auc_treino": np.array(auc_tr), "pred": pred}


def score_fora_da_amostra(res: dict) -> np.ndarray:
    """Um score por cliente do treino: a média das predições de validação (uma por repetição)."""
    return np.nanmean(res["pred"], axis=0)


# --- Avaliação no teste ---

def auc_bootstrap(y, score, n: int = 2000, seed: int = SEED) -> dict:
    """AUC com intervalo de 95% por bootstrap (reamostra os clientes com reposição)."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            boot.append(roc_auc_score(y[idx], score[idx]))
    return {"AUC": roc_auc_score(y, score), "IC inf": np.percentile(boot, 2.5), "IC sup": np.percentile(boot, 97.5)}


def diferenca_bootstrap(y, score_a, score_b, n: int = 2000, seed: int = SEED) -> dict:
    """Diferença de AUC (a − b) nos mesmos clientes, com intervalo de 95% por bootstrap pareado."""
    y, a, b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            boot.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    dif = roc_auc_score(y, a) - roc_auc_score(y, b)
    return {"diferença": dif, "IC inf": np.percentile(boot, 2.5), "IC sup": np.percentile(boot, 97.5)}


# --- Scores ---

def salvar_scores(scores: pd.DataFrame) -> None:
    """Salva id_cliente, conjunto e score em data/processed/ (fora do git: são dados de clientes)."""
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    scores[["id_cliente", "conjunto", "score"]].to_csv(ARQ_SCORES, index=False)


def carregar_scores() -> pd.DataFrame:
    return pd.read_csv(ARQ_SCORES)
