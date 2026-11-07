"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
src/classificacao.py — Classificação de novos comentários
=============================================================================

Responsabilidades:
  - Carregar os modelos treinados de categorias e sentimentos
  - Classificar um ou múltiplos comentários novos
  - Retornar categoria, sentimento, probabilidades e nível de confiança
  - Sinalizar classificações com baixa confiança para revisão humana
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import limpar_texto

# =============================================================================
# CAMINHOS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMINHO_MODELO_CAT  = os.path.join(BASE_DIR, "models", "modelo_categorias.joblib")
CAMINHO_MODELO_SENT = os.path.join(BASE_DIR, "models", "modelo_sentimentos.joblib")

# Limiar de confiança abaixo do qual o resultado é marcado como "baixa confiança"
LIMIAR_CONFIANCA = 0.45

# =============================================================================
# CARREGAMENTO DOS MODELOS
# =============================================================================

_modelo_categorias  = None
_modelo_sentimentos = None


def _carregar_modelos():
    """Carrega os modelos em memória (singleton simples)."""
    global _modelo_categorias, _modelo_sentimentos

    for caminho, nome in [(CAMINHO_MODELO_CAT, "categorias"), (CAMINHO_MODELO_SENT, "sentimentos")]:
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"Modelo de {nome} não encontrado em: {caminho}\n"
                "Execute primeiro: python src/treinamento.py"
            )

    if _modelo_categorias is None:
        _modelo_categorias  = joblib.load(CAMINHO_MODELO_CAT)
    if _modelo_sentimentos is None:
        _modelo_sentimentos = joblib.load(CAMINHO_MODELO_SENT)


# =============================================================================
# CLASSIFICAÇÃO
# =============================================================================

def classificar_comentario(texto: str) -> dict:
    """
    Classifica um único comentário.

    Args:
        texto: string com o comentário do cliente

    Returns:
        dict com:
          - texto_original: str
          - texto_limpo: str
          - categoria: str
          - prob_categorias: dict {categoria: probabilidade}
          - sentimento: str
          - prob_sentimentos: dict {sentimento: probabilidade}
          - confianca_categoria: float (0-1)
          - confianca_sentimento: float (0-1)
          - baixa_confianca: bool — True se abaixo do limiar
          - requer_revisao: bool — True se baixa_confianca
    """
    _carregar_modelos()

    texto_limpo = limpar_texto(texto)

    if not texto_limpo.strip():
        return {
            "texto_original": texto,
            "texto_limpo": "",
            "categoria": "Outros",
            "prob_categorias": {},
            "sentimento": "neutro",
            "prob_sentimentos": {},
            "confianca_categoria": 0.0,
            "confianca_sentimento": 0.0,
            "baixa_confianca": True,
            "requer_revisao": True,
        }

    # --- Categorias ---
    probs_cat = _modelo_categorias.predict_proba([texto_limpo])[0]
    classes_cat = _modelo_categorias.classes_
    idx_cat = int(np.argmax(probs_cat))
    categoria = classes_cat[idx_cat]
    conf_cat = float(probs_cat[idx_cat])
    prob_cat_dict = {c: round(float(p), 4) for c, p in zip(classes_cat, probs_cat)}

    # --- Sentimentos ---
    probs_sent = _modelo_sentimentos.predict_proba([texto_limpo])[0]
    classes_sent = _modelo_sentimentos.classes_
    idx_sent = int(np.argmax(probs_sent))
    sentimento = classes_sent[idx_sent]
    conf_sent = float(probs_sent[idx_sent])
    prob_sent_dict = {c: round(float(p), 4) for c, p in zip(classes_sent, probs_sent)}

    baixa_conf = conf_cat < LIMIAR_CONFIANCA

    return {
        "texto_original": texto,
        "texto_limpo": texto_limpo,
        "categoria": categoria,
        "prob_categorias": prob_cat_dict,
        "sentimento": sentimento,
        "prob_sentimentos": prob_sent_dict,
        "confianca_categoria": round(conf_cat, 4),
        "confianca_sentimento": round(conf_sent, 4),
        "baixa_confianca": baixa_conf,
        "requer_revisao": baixa_conf,
    }


def classificar_dataframe(df: pd.DataFrame, coluna_texto: str = "texto") -> pd.DataFrame:
    """
    Classifica todos os comentários de um DataFrame.

    Args:
        df: DataFrame com coluna de texto
        coluna_texto: nome da coluna de texto (padrão: "texto")

    Returns:
        DataFrame original com colunas adicionais de classificação
    """
    _carregar_modelos()

    resultados = []
    for texto in df[coluna_texto].fillna(""):
        r = classificar_comentario(texto)
        resultados.append({
            "categoria_predita": r["categoria"],
            "confianca_categoria": r["confianca_categoria"],
            "sentimento_predito": r["sentimento"],
            "confianca_sentimento": r["confianca_sentimento"],
            "requer_revisao": r["requer_revisao"],
        })

    df_resultado = df.copy()
    df_resultado = df_resultado.join(pd.DataFrame(resultados, index=df.index))
    return df_resultado


# =============================================================================
# EXECUÇÃO DIRETA — DEMONSTRAÇÃO
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  DEMO DE CLASSIFICAÇÃO — Detector de Problemas Sicredi")
    print("=" * 65)

    EXEMPLOS = [
        # Casos claros
        "Não consigo entrar no aplicativo desde ontem.",
        "Fiz um Pix e o dinheiro sumiu da minha conta.",
        "Me cobraram por um seguro que nunca contratei.",
        "Esperei mais de uma hora na fila do atendimento.",
        "O atendimento foi excelente, muito satisfeito!",
        # Casos ambíguos / de fronteira
        "Preciso de ajuda com minha conta.",
        "Estou insatisfeito com o banco.",
        "Problema urgente.",
        # Frase muito curta / potencialmente fora das categorias
        "Ok",
        "???",
    ]

    for texto in EXEMPLOS:
        r = classificar_comentario(texto)
        revisao = " [REVISAO RECOMENDADA]" if r["requer_revisao"] else ""
        print(f"\nTexto   : {r['texto_original']}")
        print(f"Categoria : {r['categoria']}  (confiança: {r['confianca_categoria']:.2%}){revisao}")
        print(f"Sentimento: {r['sentimento']}  (confiança: {r['confianca_sentimento']:.2%})")
