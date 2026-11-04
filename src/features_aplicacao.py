"""Features do modelo de aplicação (KAN-5).

Só usa a tabela de clientes: nada vem dos pedidos, porque no cadastro o cliente
ainda não comprou. As escolhas de cada feature estão justificadas no notebook
03_features_aplicacao.ipynb, com referência à seção da EDA (notebook 02).

Duas camadas:
- criar_features: transformações determinísticas, que não aprendem nada dos
  dados (converter faixas, criar flags, contar credores). Pode rodar na base
  inteira antes do split.
- o que aprende dos dados (agrupar categorias raras, target encoding, escala)
  fica no pré-processador, ajustado só no treino dentro do Pipeline.

Uso:
    from src.dados import carregar_dados_limpos
    from src.features_aplicacao import criar_features
    clientes, _ = carregar_dados_limpos()
    X = criar_features(clientes)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler, TargetEncoder,
)

# Região Metropolitana de Fortaleza sem a capital: 18 municípios (IPECE), grafia da base.
RMF = {
    "AQUIRAZ", "CASCAVEL", "CAUCAIA", "CHOROZINHO", "EUSEBIO", "GUAIUBA", "HORIZONTE", "ITAITINGA", "MARACANAU",
    "MARANGUAPE", "PACAJUS", "PACATUBA", "PARACURU", "PARAIPABA", "PINDORETAMA", "SAO GONCALO DO AMARANTE",
    "SAO LUIS DO CURU", "TRAIRI",
}

# Segmento do credor (serasa_credores) -> tipo de credor, pelo significado do nome (notebook 03, seção 3.4).
# A classificação não olha o alvo. Segmento fora do dicionário não recebe tipo: entra só em
# tem_credores e n_credores.
GRUPO_CREDOR = {
    # quem vende produto ao varejista
    "Alimentos e Bebidas": "mercadoria", "Distribuição": "mercadoria", "Comércio": "mercadoria",
    "Indústria": "mercadoria", "Tabaco": "mercadoria", "Representação Comercial": "mercadoria",
    "Atacadista": "mercadoria",
    # dívida de crédito
    "Bancos e Financeiro": "financeiro", "Tecnologia e Pagamentos": "financeiro", "Financeiro": "financeiro",
    "Seguros e Financeiro": "financeiro",
    # custo de operação do negócio
    "Saúde": "servicos", "Telecomunicações": "servicos", "Tecnologia": "servicos", "Energia": "servicos",
    "Comunicação": "servicos", "Logística": "servicos", "Serviços": "servicos", "Consultoria": "servicos",
    "Segurança": "servicos", "Imobiliário": "servicos", "Limpeza": "servicos", "Comunicação e Mídia": "servicos",
    "Contabilidade": "servicos", "Educação": "servicos", "Educação ou Tecnologia": "servicos",
    "Impressão e Gráficas": "servicos", "Serviços Ambientais": "servicos",
    # sem tipo (significado pouco claro): "Diversos", "Governo"
}
TIPOS_CREDOR = ["mercadoria", "financeiro", "servicos"]

NATUREZA = {
    "213-5 - Empresário (Individual)": "individual",
    "206-2 - Sociedade Empresária Limitada": "ltda",
}

# Colunas de clientes que não entram no modelo (motivo no notebook 03, seção 4).
DESCARTADAS = [
    "id_cliente",            # chave
    "uf",                    # constante: todos são do CE
    "municipio",             # substituído por regiao (V = 0,05; 23 de 36 municípios raros)
    "cnae_codigo",           # subclasse: 138 de 153 raras; substituída por cnae_classe
    "google_maps_avaliacao", # nota: só 34 valores
    "google_maps_tem_website",  # nulo exatamente quando não está no Maps: cópia de tem_gmaps
    "inadimplente",          # alvo
]

# Tipos das features, para o pré-processador.
NUMERICAS = [
    "idade_cnpj_ord", "capital_social_ord",
    "negativacoes", "protestos", "n_credores",
    "ifood_preco_ord", "ifood_avaliacoes_ord", "gmaps_avaliacoes_ord",
]
BINARIAS = [
    "socio_negativado", "tem_credores", "sem_registro_serasa", "negativado_sem_credor",
    *(f"deve_{t}" for t in TIPOS_CREDOR),
    "tem_ifood", "tem_gmaps",
]
CATEGORICAS = ["natureza", "fonte_cliente", "segmento_cliente", "cnae_divisao", "cnae_classe", "regiao"]

# Features do modelo de aplicação, escolhidas pela ablação no treino (notebook 03, seção 6). Saíram, sem
# perda de AUC: plataformas, negativacoes (redundante com n_credores), deve_* e cnae_divisao. Também saíram,
# por serem funções de outras features que a árvore monta sozinha: tem_credores (n_credores > 0) e
# sem_registro_serasa (n_credores = 0, sócio limpo e negativado_sem_credor = 0). sem_registro_serasa continua
# em criar_features para separar os grupos na análise de dependência da Serasa.
FEATURES_FINAIS = [
    "idade_cnpj_ord",
    "capital_social_ord", "natureza",
    "n_credores", "socio_negativado", "negativado_sem_credor", "protestos",
    "fonte_cliente", "segmento_cliente", "cnae_classe", "regiao",
]

# Pré-processador (ajustado só no treino)
MIN_FREQ = 30  # categoria com menos clientes no treino vira "infrequent" (o "outros" da EDA)
CONTAGENS = ["negativacoes", "protestos", "n_credores"]  # cauda longa: log na versão linear
N_KNOTS_IDADE = 5


def lista_credores(serasa_credores: pd.Series) -> pd.Series:
    """Texto "seg1, seg2" -> lista de segmentos sem espaços nem itens vazios. Nulo -> lista vazia."""
    return serasa_credores.map(
        lambda v: [x.strip() for x in v.split(",") if x.strip()] if isinstance(v, str) else []
    )


def criar_features(clientes: pd.DataFrame) -> pd.DataFrame:
    """Features determinísticas do modelo de aplicação, uma linha por cliente (mesmo índice de `clientes`).

    Espera os dados de carregar_dados_limpos(), com as faixas já como categórica ordenada.
    Códigos ordinais de faixa ausente valem -1 (ex: avaliações de quem não está na plataforma).
    """
    c = clientes
    credores = lista_credores(c["serasa_credores"])
    n_credores = credores.str.len()
    negativacoes = c["serasa_contagem_negativacoes"]
    socio = c["serasa_socio_tem_negativacao"]

    X = pd.DataFrame(index=c.index)

    # Idade e capital: posição da faixa (categorias fixadas na base inteira pela limpeza).
    X["idade_cnpj_ord"] = c["idade_cnpj"].cat.codes
    X["capital_social_ord"] = c["capital_social"].cat.codes

    # Serasa
    X["negativacoes"] = negativacoes
    X["protestos"] = c["serasa_contagem_protestos"]
    X["n_credores"] = n_credores
    X["socio_negativado"] = socio
    X["tem_credores"] = (n_credores > 0).astype(int)
    X["sem_registro_serasa"] = ((negativacoes == 0) & (n_credores == 0) & (socio == 0)).astype(int)
    X["negativado_sem_credor"] = ((negativacoes > 0) & (n_credores == 0)).astype(int)
    tipos = credores.map(lambda l: {GRUPO_CREDOR[s] for s in l if s in GRUPO_CREDOR})
    for t in TIPOS_CREDOR:
        X[f"deve_{t}"] = tipos.map(lambda g: t in g).astype(int)

    # Plataformas: está no iFood se tem faixa de preço ou contagem (177 têm só a faixa).
    X["tem_ifood"] = c[["ifood_faixa_preco", "ifood_contagem_avaliacoes"]].notna().any(axis=1).astype(int)
    X["tem_gmaps"] = c["google_maps_contagem_avaliacoes"].notna().astype(int)
    X["ifood_preco_ord"] = c["ifood_faixa_preco"].str.len().fillna(0).astype(int)  # "$" = 1 ... "$$$$$" = 5; 0 = ausente
    X["ifood_avaliacoes_ord"] = c["ifood_contagem_avaliacoes"].cat.codes
    X["gmaps_avaliacoes_ord"] = c["google_maps_contagem_avaliacoes"].cat.codes

    # Categóricas: o agrupamento de raras fica no pré-processador (depende das contagens do treino).
    X["natureza"] = c["natureza_juridica"].map(NATUREZA).fillna("outros")
    X["fonte_cliente"] = c["fonte_cliente"]
    X["segmento_cliente"] = c["segmento_cliente"]
    X["cnae_divisao"] = c["cnae_codigo"].str[:2]
    X["cnae_classe"] = c["cnae_codigo"].str[:7]  # com dígito verificador: a base tem classes que só diferem nele
    X["regiao"] = c["municipio"].map(
        lambda m: "fortaleza" if m == "FORTALEZA" else "metropolitana" if m in RMF else "interior"
    )

    return X


def _target_encoder() -> TargetEncoder:
    """TargetEncoder reprodutível. No scikit-learn >= 1.9 o embaralhamento vai no `cv` (random_state foi depreciado)."""
    if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 9):
        return TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return TargetEncoder(target_type="binary", random_state=0)


def montar_preprocessador(tipo: str, colunas: list[str] | None = None, cnae: str = "raras") -> ColumnTransformer:
    """Pré-processador a ajustar só no treino, dentro do Pipeline. Recebe a saída de criar_features.

    - "arvore": numéricas e binárias passam como estão; categóricas viram códigos, com as raras
      (< MIN_FREQ clientes no treino) num código só.
    - "linear": idade por splines (nós nos quantis, sem olhar o alvo), log nas contagens,
      padronização das ordinais e one-hot das categóricas, com as raras numa coluna só.
    Categoria que não apareceu no treino cai no grupo das raras ("linear") ou no código -1 ("arvore").

    colunas: subconjunto das features a usar (padrão: todas), para a ablação.
    cnae: "raras" (cnae_classe tratada como as outras categóricas) ou "target" (target encoding,
      com validação cruzada interna e encolhimento das classes pequenas para a média).
    """
    if cnae not in ("raras", "target"):
        raise ValueError(f"cnae deve ser 'raras' ou 'target', não {cnae!r}")
    usar = set(NUMERICAS + BINARIAS + CATEGORICAS) if colunas is None else set(colunas)
    sel = lambda cols: [c for c in cols if c in usar]
    te = ["cnae_classe"] if cnae == "target" else []
    categoricas = [c for c in sel(CATEGORICAS) if c not in te]

    if tipo == "arvore":
        transformadores = [
            ("num", "passthrough", sel(NUMERICAS + BINARIAS)),
            ("cat", OrdinalEncoder(min_frequency=MIN_FREQ, handle_unknown="use_encoded_value", unknown_value=-1),
             categoricas),
        ]
    elif tipo == "linear":
        transformadores = [
            ("idade", SplineTransformer(n_knots=N_KNOTS_IDADE, degree=3, knots="quantile"), sel(["idade_cnpj_ord"])),
            ("contagens", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()),
             sel(CONTAGENS)),
            ("ordinais", StandardScaler(), sel([c for c in NUMERICAS if c not in CONTAGENS and c != "idade_cnpj_ord"])),
            ("bin", "passthrough", sel(BINARIAS)),
            ("cat", OneHotEncoder(min_frequency=MIN_FREQ, handle_unknown="infrequent_if_exist", sparse_output=False),
             categoricas),
        ]
    else:
        raise ValueError(f"tipo deve ser 'arvore' ou 'linear', não {tipo!r}")
    transformadores.append(("cnae_te", _target_encoder(), sel(te)))
    transformadores = [t for t in transformadores if t[2]]  # sem colunas selecionadas, a etapa sai
    return ColumnTransformer(transformadores, verbose_feature_names_out=False).set_output(transform="pandas")
