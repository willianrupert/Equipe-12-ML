# Equipe 12 — Previsão de Inadimplência (Praso)

Projeto final de **CIN0144 — Aprendizagem de Máquina** (CIn/UFPE, 2026.2), Tema 8: **Praso, Desafio 1: Previsão de Inadimplência**.

O objetivo é estimar a probabilidade de um varejista ficar inadimplente com dois modelos:
- **modelo de aplicação**: usa só os dados disponíveis no cadastro;
- **modelo comportamental**: usa também o histórico de pedidos.

A métrica principal é ROC-AUC, como a Praso pede. No fim, discutimos como esses scores entrariam numa política de crédito.

**Equipe:** Luiz Miguel Freitas da Silva (lmfs3), Mirella Laura Fontinelle Martins (mlfm), Severino João da Cruz Júnior (sjcj2), Willian Neves Rupert Jones (wnrj).

## Dados e confidencialidade

Os dados são reais da Praso e estão sob termo de confidencialidade:
- os CSVs **não são versionados**: o `.gitignore` bloqueia `data/` e `*.csv`;
- os notebooks mostram **só agregados**, nunca linhas de clientes (nada de `df.head()` com dados reais);
- antes de cada commit, confira as saídas dos notebooks, porque o repositório é público.

## Estrutura

```
data/raw/          CSVs originais da Praso (não versionados)
data/processed/    dados limpos gerados pelos notebooks (não versionados)
notebooks/         um notebook por etapa, numerados na ordem de execução
reports/figures/   gráficos usados no relatório e nos slides
src/dados.py               limpeza compartilhada (KAN-1): tipos, datas, registros inválidos, utilitárias de faixas
src/features_aplicacao.py  features do modelo de aplicação (KAN-5)
src/features_comportamentais.py  features do histórico de pedidos (KAN-7)
```

## Etapas (Jira: projeto KAN)

| Etapa | Ticket | Notebook / código | Status |
|---|---|---|---|
| Setup, carregamento e qualidade dos dados | KAN-1 | `notebooks/01_setup_qualidade_dados.ipynb`, `src/dados.py` | em revisão |
| EDA vs. inadimplência | KAN-4 | `notebooks/02_eda_inadimplencia.ipynb` | em revisão |
| Features do modelo de aplicação | KAN-5 | `src/features_aplicacao.py` (ponto de partida herdado da KAN-1) | a fazer |
| Modelo de aplicação | KAN-6 | — | a fazer |
| Agregações do histórico de pedidos | KAN-7 | `notebooks/03_features_comportamentais.ipynb`, `src/features_comportamentais.py` | em andamento |
| Modelo comportamental | KAN-8 | — | a fazer |
| Política de crédito e discussão | KAN-9 | — | a fazer |
| Relatório (até 8 páginas, template Overleaf) | KAN-10 | — | a fazer |
| Slides (15 min, modelo CIn) | KAN-11 | — | a fazer |

Entrega final no Classroom: **25/11/2026** (relatório, slides e link deste repositório).

## Como executar

1. Baixe os CSVs do material "Praso - Dados" no Classroom e coloque-os em `data/raw/`:
   - `credito_aplicacao_clientes_final.csv`
   - `credito_comportamental_pedidos_final.csv`
2. Crie o ambiente (Python 3.12) e instale as dependências:
   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   ```
   Sem `uv`: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`
3. Rode os notebooks em ordem, a partir da pasta `notebooks/`:
   ```bash
   cd notebooks && ../.venv/bin/jupyter lab
   ```

Nos notebooks, os dados limpos vêm de `src.dados.carregar_dados_limpos()`. Eles têm as mesmas colunas do CSV original: as faixas continuam como texto e os nulos como vieram.

## Pipeline

- **Limpeza** (`src/dados.py`): é a única etapa compartilhada. Só faz o que não perde informação nem exige escolha de modelagem: converte tipos e datas, remove registros inválidos (mostrando quantos e por quê), valida chaves e formatos e ordena os pedidos.
- **Não fica na limpeza:** criar colunas novas (`_num`, `_ord`, `tem_*`, `atrasou`), preencher nulos, agregar pedidos ou cruzar qualquer coluna com o alvo.
- **Features:** cada uma fica com o dono do modelo (KAN-5 e KAN-7).
- **Ajuste só no treino:** transformações que dependem dos dados (agrupar categorias raras, imputar, *target encoding*) ficam dentro do `Pipeline` do scikit-learn e são ajustadas só no treino.
- **Split compartilhado:** a KAN-6 salva o split com a seed, e a KAN-8 usa o mesmo split.

## Como contribuir

- Uma branch por ticket, com o código do Jira no nome: `kan-5-features-aplicacao`.
- Commits com o ticket na mensagem: `KAN-5: cria flags de presença no iFood`.
- Abra um pull request para `main` e peça revisão de alguém da equipe.
- Novos notebooks seguem a numeração (`03_...`, `04_...`). Código reutilizado por mais de um notebook vai para `src/`.
- Figuras ficam em `reports/figures/`, com o prefixo do ticket (`kan5_...png`).
