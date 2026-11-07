"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
src/treinamento.py — Treinamento, avaliação e persistência dos modelos
=============================================================================

Responsabilidades:
  - Treinar classificador de categorias (TF-IDF + Regressão Logística)
  - Treinar classificador de sentimentos (TF-IDF + Regressão Logística)
  - Avaliar os modelos com métricas completas (accuracy, F1, matriz de confusão)
  - Salvar os modelos treinados com joblib
  - Gerar gráfico da matriz de confusão
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Não abre janela gráfica em execução headless
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Permite importar src/preprocessing.py tanto ao rodar diretamente quanto via import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import carregar_dados, preprocessar_dataframe

# =============================================================================
# CAMINHOS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMINHO_CSV_RAW       = os.path.join(BASE_DIR, "data", "raw", "atendimentos.csv")
CAMINHO_CSV_PROC      = os.path.join(BASE_DIR, "data", "processed", "atendimentos_processados.csv")
CAMINHO_MODELO_CAT    = os.path.join(BASE_DIR, "models", "modelo_categorias.joblib")
CAMINHO_MODELO_SENT   = os.path.join(BASE_DIR, "models", "modelo_sentimentos.joblib")
CAMINHO_METRICAS      = os.path.join(BASE_DIR, "models", "metricas_treinamento.json")
CAMINHO_CONF_CAT      = os.path.join(BASE_DIR, "reports", "matriz_confusao_categorias.png")
CAMINHO_CONF_SENT     = os.path.join(BASE_DIR, "reports", "matriz_confusao_sentimentos.png")

# =============================================================================
# PIPELINE DE MODELO
# =============================================================================

def criar_pipeline_classificacao(
    ngram_range=(1, 2),
    max_features=15000,
    C=5.0,
    max_iter=1000,
    class_weight="balanced",
) -> Pipeline:
    """
    Cria um pipeline TF-IDF + Regressão Logística.

    Escolhas técnicas:
      - TF-IDF com bigramas captura frases como "não funciona", "não abre"
      - max_features=15000 cobre o vocabulário da base sem overfit
      - Regressão Logística é robusta, interpretável e eficiente para texto
      - class_weight='balanced' compensa categorias com menos exemplos
      - C=5.0 permite alguma regularização sem penalizar demais
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,       # log(TF) reduz impacto de termos muito frequentes
            min_df=2,                # ignora termos que aparecem em menos de 2 docs
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=42,
        )),
    ])


# =============================================================================
# AVALIAÇÃO
# =============================================================================

def avaliar_modelo(pipeline: Pipeline, X_test, y_test, nome: str, caminho_figura: str) -> dict:
    """
    Calcula métricas completas e gera matriz de confusão.

    Returns:
        dict com accuracy, f1_macro, f1_weighted e o report completo
    """
    y_pred = pipeline.predict(X_test)

    acc        = accuracy_score(y_test, y_pred)
    f1_macro   = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report     = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*60}")
    print(f"  AVALIAÇÃO — {nome}")
    print(f"{'='*60}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 Macro   : {f1_macro:.4f}")
    print(f"  F1 Weighted: {f1_weighted:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), max(6, len(labels) * 0.9)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusão — {nome}", fontsize=13, pad=12)
    ax.set_xlabel("Predito", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(caminho_figura), exist_ok=True)
    fig.savefig(caminho_figura, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matriz de confusão salva em: {caminho_figura}")

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "report": report,
    }


def validacao_cruzada(pipeline: Pipeline, X, y, nome: str, cv: int = 5):
    """Executa k-fold cross-validation e imprime resultado."""
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"\n  [{nome}] Cross-validation F1-macro ({cv}-fold):")
    print(f"    Scores : {[round(s, 4) for s in scores]}")
    print(f"    Média  : {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# =============================================================================
# TREINAMENTO PRINCIPAL
# =============================================================================

def treinar(test_size: float = 0.20, random_state: int = 42) -> dict:
    """
    Executa o pipeline completo de treinamento:
      1. Carrega e pré-processa dados
      2. Treina modelo de categorias
      3. Treina modelo de sentimentos
      4. Avalia ambos os modelos
      5. Salva modelos e métricas

    Args:
        test_size: proporção para conjunto de teste (padrão 20%)
        random_state: semente aleatória

    Returns:
        dict com métricas dos dois modelos
    """
    print("\n" + "="*60)
    print("  INÍCIO DO TREINAMENTO")
    print("="*60)

    # -----------------------------------------------------------------
    # 1. Carregar dados
    # -----------------------------------------------------------------
    print("\n[1/5] Carregando e pré-processando dados...")
    df = carregar_dados(CAMINHO_CSV_RAW)
    df = preprocessar_dataframe(df, usar_stemming=False)

    # Salva versão processada
    os.makedirs(os.path.dirname(CAMINHO_CSV_PROC), exist_ok=True)
    df.to_csv(CAMINHO_CSV_PROC, index=False, encoding="utf-8-sig")
    print(f"  {len(df)} registros prontos para treinamento.")
    print(f"  Distribuição de categorias:\n{df['categoria'].value_counts().to_string()}")

    # -----------------------------------------------------------------
    # 2. Divisão treino / teste — estratificada por categoria
    # -----------------------------------------------------------------
    print(f"\n[2/5] Dividindo dados (treino={1-test_size:.0%} / teste={test_size:.0%})...")
    X = df["texto_limpo"]
    y_cat  = df["categoria"]
    y_sent = df["sentimento"]

    X_train, X_test, y_cat_train, y_cat_test = train_test_split(
        X, y_cat, test_size=test_size, random_state=random_state, stratify=y_cat
    )
    # Para sentimentos: usa os mesmos índices do split de categoria
    # para garantir que os textos sejam os mesmos em ambos os modelos
    y_sent_train = y_sent.iloc[X_train.index]
    y_sent_test  = y_sent.iloc[X_test.index]

    print(f"  Treino: {len(X_train)} | Teste: {len(X_test)}")

    # -----------------------------------------------------------------
    # 3. Treinar classificador de categorias
    # -----------------------------------------------------------------
    print("\n[3/5] Treinando classificador de CATEGORIAS...")
    pipe_cat = criar_pipeline_classificacao(C=5.0)
    pipe_cat.fit(X_train, y_cat_train)
    validacao_cruzada(pipe_cat, X, y_cat, nome="Categorias")
    metricas_cat = avaliar_modelo(pipe_cat, X_test, y_cat_test,
                                  nome="Categorias",
                                  caminho_figura=CAMINHO_CONF_CAT)

    # -----------------------------------------------------------------
    # 4. Treinar classificador de sentimentos
    # -----------------------------------------------------------------
    print("\n[4/5] Treinando classificador de SENTIMENTOS...")
    pipe_sent = criar_pipeline_classificacao(C=3.0, ngram_range=(1, 2))
    pipe_sent.fit(X_train, y_sent_train)
    validacao_cruzada(pipe_sent, X, y_sent, nome="Sentimentos")
    metricas_sent = avaliar_modelo(pipe_sent, X_test, y_sent_test,
                                   nome="Sentimentos",
                                   caminho_figura=CAMINHO_CONF_SENT)

    # -----------------------------------------------------------------
    # 5. Salvar modelos e métricas
    # -----------------------------------------------------------------
    print("\n[5/5] Salvando modelos...")
    os.makedirs(os.path.dirname(CAMINHO_MODELO_CAT), exist_ok=True)
    joblib.dump(pipe_cat,  CAMINHO_MODELO_CAT)
    joblib.dump(pipe_sent, CAMINHO_MODELO_SENT)
    print(f"  Modelos salvos em: {os.path.dirname(CAMINHO_MODELO_CAT)}")

    metricas = {
        "categorias": metricas_cat,
        "sentimentos": metricas_sent,
    }
    with open(CAMINHO_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print(f"  Métricas salvas em: {CAMINHO_METRICAS}")

    print("\n" + "="*60)
    print("  TREINAMENTO CONCLUÍDO COM SUCESSO")
    print("="*60)

    return metricas


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    metricas = treinar()

    print("\nRESUMO FINAL:")
    print(f"  Categorias  — Accuracy: {metricas['categorias']['accuracy']:.4f}  "
          f"F1-macro: {metricas['categorias']['f1_macro']:.4f}")
    print(f"  Sentimentos — Accuracy: {metricas['sentimentos']['accuracy']:.4f}  "
          f"F1-macro: {metricas['sentimentos']['f1_macro']:.4f}")
