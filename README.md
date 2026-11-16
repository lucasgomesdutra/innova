# AtendeAI — Detector Inteligente de Problemas no Atendimento

Projeto acadêmico de Inteligência Artificial desenvolvido para o **Sicredi**, capaz de analisar avaliações de clientes, classificar categorias e sentimentos, agrupar reclamações semelhantes por significado e gerar insights acionáveis para o time de atendimento.

---

## Como rodar o projeto

### Pré-requisitos

- Python 3.10 ou superior
- pip atualizado
- Git

### 1. Clone o repositório

```bash
git clone https://github.com/lucasgomesdutra/innova.git
cd innova
```

### 2. Crie e ative o ambiente virtual

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> ⚠️ O pacote `sentence-transformers` (~120MB de modelo) será baixado automaticamente na primeira vez que o agrupamento semântico for utilizado. É necessário conexão com a internet nesse momento.

### 4. Treine os modelos de IA

```bash
python src/treinamento.py
```

Isso vai gerar os arquivos `models/modelo_categorias.joblib` e `models/modelo_sentimentos.joblib`.

> Os modelos já vêm pré-treinados no repositório. Só é necessário retreinar se você modificar os dados de treinamento.

### 5. Inicie o painel web

```bash
python -m streamlit run app.py
```

Acesse no navegador: **http://localhost:8501**

---

## Funcionalidades

| Página | O que faz |
|---|---|
| 📊 Dashboard | Métricas gerais, evolução temporal, alertas e resumo da IA |
| 📋 Avaliações | Lista todas as avaliações com filtros e busca |
| 🔍 Análise com IA | Classifica um novo comentário em tempo real |
| 🚨 Problemas Identificados | Gerenciamento, agrupamento semântico e score de urgência |
| 📄 Relatórios | Gráficos completos e exportação para Excel |

---

## Arquitetura de IA

O sistema usa **dois níveis de NLP**:

### Classificação supervisionada (TF-IDF + Regressão Logística)
- Classifica cada avaliação em **9 categorias** e **3 sentimentos**
- Treinado com ~28.500 exemplos (base sintética + dataset de 27.000 registros)
- Acurácia: ~80% em categorias, ~77% em sentimentos

### Agrupamento semântico (sentence-transformers)
- Modelo: `paraphrase-multilingual-MiniLM-L12-v2` (multilíngue, 384 dimensões)
- Reconhece que *"meu Pix não caiu"*, *"transferência sumiu"* e *"dinheiro não chegou ao destino"* são o mesmo problema
- K-Means com seleção automática do número ideal de clusters (Silhouette Score)
- Cache em disco para não recalcular embeddings desnecessariamente

### Insights automáticos
- Score de urgência por categoria (volume + crescimento + proporção negativa)
- Alertas de aumento mês a mês
- Textos de insight prontos para o dashboard

---

## Categorias de classificação

| # | Categoria |
|---|-----------|
| 1 | Problemas no aplicativo |
| 2 | Problemas com Pix e transferências |
| 3 | Cobranças indevidas |
| 4 | Demora no atendimento |
| 5 | Problemas de acesso à conta |
| 6 | Problemas com cartões |
| 7 | Dificuldades com serviços bancários |
| 8 | Elogios e avaliações positivas |
| 9 | Outros |

---

## Estrutura do projeto

```
innova/
├── .streamlit/
│   └── config.toml               # Tema claro forçado
│
├── data/
│   ├── raw/
│   │   ├── atendimentos.csv          # Base sintética original
│   │   ├── atendimentos_27000.csv    # Dataset de 27.000 registros
│   │   ├── dados_reais_exemplo.csv   # Exemplos curados para treino (~378 registros)
│   │   └── gerar_base.py             # Script para regenerar a base sintética
│   └── processed/
│       ├── atendimentos_processados.csv   # Base após pré-processamento
│       ├── avaliacoes_reais.csv           # Avaliações inseridas pelo usuário
│       └── problemas_gerenciados.csv      # Gerenciamento de problemas
│
├── models/
│   ├── modelo_categorias.joblib      # Modelo treinado de categorias
│   ├── modelo_sentimentos.joblib     # Modelo treinado de sentimentos
│   ├── metricas_treinamento.json     # Métricas de avaliação dos modelos
│   └── embeddings_cache/             # Cache de embeddings semânticos
│
├── reports/                          # Gráficos e relatórios gerados
│
├── src/
│   ├── preprocessing.py              # Limpeza e normalização de textos
│   ├── treinamento.py                # Pipeline de treinamento dos modelos
│   ├── classificacao.py              # Classificação de novos comentários
│   ├── analise_sentimentos.py        # Análise e agregação de sentimentos
│   ├── agrupamento.py                # Clustering TF-IDF (com fallback semântico)
│   ├── agrupamento_semantico.py      # Clustering por embeddings semânticos
│   ├── embeddings.py                 # Geração e cache de embeddings
│   ├── insights.py                   # Score de urgência e insights automáticos
│   ├── analise_problemas.py          # Identificação de problemas recorrentes
│   └── relatorios.py                 # Geração de relatórios e gráficos
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_classificacao.py
│   └── test_agrupamento.py
│
├── app.py                            # Aplicação Streamlit (painel web)
├── requirements.txt
└── README.md
```

---

## Tecnologias utilizadas

| Biblioteca | Versão mínima | Finalidade |
|---|---|---|
| `pandas` | 2.2 | Manipulação de dados |
| `scikit-learn` | 1.5 | TF-IDF, Regressão Logística, K-Means |
| `sentence-transformers` | 3.0 | Embeddings semânticos multilíngues |
| `nltk` | 3.8 | Pré-processamento NLP em português |
| `streamlit` | 1.35 | Interface web |
| `plotly` | 5.22 | Gráficos interativos |
| `matplotlib` / `seaborn` | — | Matrizes de confusão e gráficos estáticos |
| `joblib` | — | Serialização de modelos |
| `openpyxl` | 3.1 | Exportação para Excel |

---

## Retreinar os modelos

Se você inserir novas avaliações ou modificar os dados de treinamento, retreine assim:

```bash
# Regenera a base sintética (opcional)
python data/raw/gerar_base.py

# Retreina os modelos
python src/treinamento.py

# Reinicia o app para carregar os novos modelos
python -m streamlit run app.py
```

> ⚠️ Sempre reinicie o Streamlit após retreinar — ele mantém os modelos em cache de memória.

---

## Aviso

> Este projeto utiliza dados sintéticos e exemplos curados para fins acadêmicos.
> As classificações da IA são indicativas e devem ser revisadas por profissionais.
> A qualidade dos resultados depende diretamente do volume e qualidade dos dados de treinamento.

---

## Autores

Projeto acadêmico — IBM / IA Aplicada ao Atendimento Sicredi
