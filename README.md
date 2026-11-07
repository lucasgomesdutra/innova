# Detector de Problemas no Atendimento — Sicredi

Projeto acadêmico de Inteligência Artificial para análise de atendimentos e pesquisas de satisfação de clientes da Sicredi.

---

## Objetivo

Desenvolver um agente de IA capaz de analisar grandes volumes de textos de atendimentos, identificar reclamações recorrentes, agrupar relatos semelhantes e encontrar possíveis falhas nos processos internos.

---

## Estrutura do Projeto

```
innova/
├── data/
│   ├── raw/
│   │   └── atendimentos.csv        # Base fictícia de atendimentos
│   └── processed/                  # Dados após pré-processamento
│
├── models/                         # Modelos treinados (.joblib)
│
├── reports/                        # Relatórios gerados
│
├── src/
│   ├── preprocessing.py            # Limpeza e preparação dos textos
│   ├── treinamento.py              # Treinamento dos modelos de ML
│   ├── classificacao.py            # Classificação de novos comentários
│   ├── analise_sentimentos.py      # Análise de sentimentos (positivo/negativo/neutro)
│   ├── agrupamento.py              # Clustering semântico de comentários
│   ├── analise_problemas.py        # Identificação de problemas recorrentes
│   └── relatorios.py               # Geração de relatórios e exportação
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_classificacao.py
│   └── test_agrupamento.py
│
├── app.py                          # Painel web Streamlit
├── requirements.txt
└── README.md
```

---

## Categorias de Classificação

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

## Instalação

### Pré-requisitos
- Python 3.10 ou superior
- pip atualizado

### Configuração do ambiente (Windows)

```powershell
# 1. Acesse o diretório do projeto
cd innova

# 2. Crie e ative um ambiente virtual
python -m venv venv
venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe os recursos do NLTK (execute uma vez)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('rslp')"
```

---

## Execução

### 1. Treinar os modelos

```powershell
python src/treinamento.py
```

### 2. Executar o painel web

```powershell
streamlit run app.py
```

### 3. Testar a classificação via terminal

```powershell
python src/classificacao.py
```

---

## Pipeline de Funcionamento

```
Textos de atendimentos
        ↓
  Pré-processamento (preprocessing.py)
        ↓
  Classificação de categoria (classificacao.py)
        ↓
  Análise de sentimentos (analise_sentimentos.py)
        ↓
  Agrupamento semântico (agrupamento.py)
        ↓
  Identificação de problemas recorrentes (analise_problemas.py)
        ↓
  Geração de relatórios (relatorios.py)
        ↓
  Painel de visualização (app.py)
```

---

## Tecnologias Utilizadas

| Biblioteca | Finalidade |
|------------|-----------|
| pandas | Manipulação de dados |
| scikit-learn | Classificação supervisionada, TF-IDF, K-Means |
| sentence-transformers | Embeddings semânticos |
| nltk | Pré-processamento NLP em português |
| matplotlib / plotly | Visualizações |
| streamlit | Painel web |
| joblib | Serialização de modelos |

---

## Aviso Importante

> **Este projeto utiliza exclusivamente dados fictícios criados para fins acadêmicos.**
> Nenhuma informação real de clientes é utilizada.
> As classificações da IA são indicativas e devem ser revisadas por profissionais.
> A qualidade dos resultados depende diretamente da qualidade e volume dos dados de treinamento.

---

## Limitações Conhecidas

- A base inicial é sintética, com vocabulário limitado.
- O modelo de sentimentos utiliza regras e um classificador treinado na base fictícia, não um modelo pré-treinado em português bancário.
- O agrupamento semântico depende da qualidade dos embeddings gerados.

---

## Próximas Evoluções Previstas

- [ ] Ingestão de dados reais anonimizados
- [ ] Integração com modelo de linguagem pré-treinado em português (BERTimbau)
- [ ] API REST para integração com sistemas externos
- [ ] Alertas automáticos para picos de reclamações

---

## Autores

Projeto acadêmico — Sicredi / IA Aplicada ao Atendimento
