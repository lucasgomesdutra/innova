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
CAMINHO_CSV_REAIS     = os.path.join(BASE_DIR, "data", "raw", "dados_reais_exemplo.csv")
CAMINHO_CSV_27K       = os.path.join(BASE_DIR, "data", "raw", "atendimentos_27000.csv")
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
    max_features=30000,
    C=5.0,
    max_iter=1000,
    class_weight="balanced",
) -> Pipeline:
    """
    Cria um pipeline TF-IDF + Regressão Logística.

    Escolhas técnicas:
      - TF-IDF com bigramas captura frases como "não funciona", "não abre"
      - max_features=30000 cobre o vocabulário ampliado do dataset de 27k
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

    # --- base original (atendimentos.csv + dados_reais_exemplo.csv) ---
    df_base = carregar_dados(CAMINHO_CSV_RAW)
    if os.path.exists(CAMINHO_CSV_REAIS):
        df_reais = carregar_dados(CAMINHO_CSV_REAIS)
        df_base = pd.concat([df_base, df_reais], ignore_index=True)
        print(f"  + {len(df_reais)} registros de dados_reais_exemplo.csv")

    # Normaliza labels da base original (remove espaços/case)
    df_base["categoria"]  = df_base["categoria"].str.strip()
    df_base["sentimento"] = df_base["sentimento"].str.strip().str.lower()

    # --- dataset grande de 27 000 registros ---
    df_27k = None
    if os.path.exists(CAMINHO_CSV_27K):
        df_27k = pd.read_csv(CAMINHO_CSV_27K, encoding="utf-8-sig")
        # garante colunas mínimas
        cols = ["texto", "categoria", "sentimento"]
        if "particao" in df_27k.columns:
            cols.append("particao")
        df_27k = df_27k[cols].copy()
        df_27k = df_27k[df_27k["texto"].notna() & (df_27k["texto"].str.strip() != "")]
        # Normaliza labels
        df_27k["categoria"]  = df_27k["categoria"].str.strip()
        df_27k["sentimento"] = df_27k["sentimento"].str.strip().str.lower()
        print(f"  + {len(df_27k)} registros de atendimentos_27000.csv")

    # --- pré-processa base original ---
    df_base = preprocessar_dataframe(df_base, usar_stemming=False)

    # Salva versão processada (apenas base original, como histórico)
    os.makedirs(os.path.dirname(CAMINHO_CSV_PROC), exist_ok=True)
    df_base.to_csv(CAMINHO_CSV_PROC, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------
    # 2. Divisão treino / teste
    # -----------------------------------------------------------------
    print(f"\n[2/5] Montando conjuntos de treino e teste...")

    if df_27k is not None and "particao" in df_27k.columns:
        # Usa a partição pré-definida do arquivo 27k
        df_27k_proc = preprocessar_dataframe(df_27k.drop(columns=["particao"]),
                                              usar_stemming=False)
        df_27k_proc["particao"] = df_27k["particao"].values

        df_train_27k = df_27k_proc[df_27k_proc["particao"] == "treino"]
        df_test_27k  = df_27k_proc[df_27k_proc["particao"] == "teste"]

        # Base original vai inteira para treino (é pequena)
        X_train = pd.concat([df_base["texto_limpo"], df_train_27k["texto_limpo"]], ignore_index=True)
        y_cat_train  = pd.concat([df_base["categoria"],   df_train_27k["categoria"]],  ignore_index=True)
        y_sent_train = pd.concat([df_base["sentimento"],  df_train_27k["sentimento"]], ignore_index=True)

        X_test       = df_test_27k["texto_limpo"].reset_index(drop=True)
        y_cat_test   = df_test_27k["categoria"].reset_index(drop=True)
        y_sent_test  = df_test_27k["sentimento"].reset_index(drop=True)

    elif df_27k is not None:
        # sem coluna 'particao': junta tudo e faz split aleatório
        df_27k_proc = preprocessar_dataframe(df_27k, usar_stemming=False)
        df_all = pd.concat([df_base, df_27k_proc], ignore_index=True)
        X_train, X_test, y_cat_train, y_cat_test = train_test_split(
            df_all["texto_limpo"], df_all["categoria"],
            test_size=test_size, random_state=random_state, stratify=df_all["categoria"]
        )
        y_sent_train = df_all["sentimento"].iloc[X_train.index]
        y_sent_test  = df_all["sentimento"].iloc[X_test.index]

    else:
        # só base original
        X = df_base["texto_limpo"]
        y_cat  = df_base["categoria"]
        y_sent = df_base["sentimento"]
        X_train, X_test, y_cat_train, y_cat_test = train_test_split(
            X, y_cat, test_size=test_size, random_state=random_state, stratify=y_cat
        )
        y_sent_train = y_sent.iloc[X_train.index]
        y_sent_test  = y_sent.iloc[X_test.index]

    total = len(X_train) + len(X_test)
    print(f"  Total: {total} registros | Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"  Distribuição de categorias (treino):\n{y_cat_train.value_counts().to_string()}")

    # -----------------------------------------------------------------
    # 3. Treinar classificador de categorias
    # -----------------------------------------------------------------
    print("\n[3/5] Treinando classificador de CATEGORIAS...")
    pipe_cat = criar_pipeline_classificacao(C=5.0)
    pipe_cat.fit(X_train, y_cat_train)
    validacao_cruzada(pipe_cat, X_train, y_cat_train, nome="Categorias")
    metricas_cat = avaliar_modelo(pipe_cat, X_test, y_cat_test,
                                  nome="Categorias",
                                  caminho_figura=CAMINHO_CONF_CAT)

    # -----------------------------------------------------------------
    # 4. Treinar classificador de sentimentos
    # -----------------------------------------------------------------
    print("\n[4/5] Treinando classificador de SENTIMENTOS...")
    pipe_sent = criar_pipeline_classificacao(C=3.0, ngram_range=(1, 2))
    pipe_sent.fit(X_train, y_sent_train)
    validacao_cruzada(pipe_sent, X_train, y_sent_train, nome="Sentimentos")
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
