"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
src/preprocessing.py — Pré-processamento e limpeza dos textos
=============================================================================

Responsabilidades:
  - Limpeza e normalização de textos em português
  - Remoção de stopwords, pontuação e caracteres especiais
  - Stemming / lematização básica (via NLTK RSLP)
  - Carregamento e validação do CSV de atendimentos
  - Exportação dos dados processados
"""

import re
import os
import unicodedata
import pandas as pd
import nltk

# Baixa recursos do NLTK se não existirem
for recurso in ["stopwords", "punkt", "rslp", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{recurso}" if recurso.startswith("punkt") else f"corpora/{recurso}" if recurso == "stopwords" else f"stemmers/{recurso}")
    except LookupError:
        nltk.download(recurso, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

# =============================================================================
# CONSTANTES
# =============================================================================

STOPWORDS_PT = set(stopwords.words("portuguese"))

# Stopwords extras específicas para o contexto bancário que não agregam valor
STOPWORDS_EXTRAS = {
    "sicredi", "banco", "conta", "cliente", "atendimento",
    "quero", "preciso", "gostaria", "favor", "obrigado",
    "boa", "bom", "dia", "tarde", "noite", "olá", "oi",
    "consegui", "consigo",
}

STOPWORDS_TODAS = STOPWORDS_PT | STOPWORDS_EXTRAS

stemmer = RSLPStemmer()

# =============================================================================
# FUNÇÕES DE LIMPEZA
# =============================================================================

def remover_acentos(texto: str) -> str:
    """Remove acentos e normaliza caracteres Unicode."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def limpar_texto(texto: str, usar_stemming: bool = False) -> str:
    """
    Pipeline de limpeza de texto:
      1. Converte para minúsculas
      2. Remove acentos
      3. Remove caracteres especiais (mantém espaços e letras)
      4. Remove stopwords
      5. (Opcional) Aplica stemming
      6. Remove tokens muito curtos (< 2 chars)

    Args:
        texto: string de entrada
        usar_stemming: se True, aplica o stemmer RSLP

    Returns:
        texto limpo como string
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""

    # 1. Minúsculas
    texto = texto.lower()

    # 2. Remove acentos
    texto = remover_acentos(texto)

    # 3. Remove caracteres que não sejam letras ou espaços
    texto = re.sub(r"[^a-z\s]", " ", texto)

    # 4. Tokeniza
    tokens = word_tokenize(texto, language="portuguese")

    # 5. Remove stopwords e tokens curtos
    tokens = [t for t in tokens if t not in STOPWORDS_TODAS and len(t) > 1]

    # 6. (Opcional) Stemming
    if usar_stemming:
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocessar_serie(serie: pd.Series, usar_stemming: bool = False) -> pd.Series:
    """Aplica limpar_texto em uma Series do pandas."""
    return serie.fillna("").apply(lambda x: limpar_texto(x, usar_stemming=usar_stemming))


# =============================================================================
# CARREGAMENTO E VALIDAÇÃO DE DADOS
# =============================================================================

def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Carrega o CSV de atendimentos e valida colunas obrigatórias.

    Args:
        caminho: caminho relativo ou absoluto do CSV

    Returns:
        DataFrame validado

    Raises:
        FileNotFoundError: se o arquivo não existir
        ValueError: se colunas obrigatórias estiverem faltando
    """
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    df = pd.read_csv(caminho, encoding="utf-8-sig")

    colunas_obrigatorias = {"texto", "categoria", "sentimento"}
    colunas_faltando = colunas_obrigatorias - set(df.columns)
    if colunas_faltando:
        raise ValueError(f"Colunas obrigatórias faltando: {colunas_faltando}")

    # Remove linhas sem texto
    antes = len(df)
    df = df[df["texto"].notna() & (df["texto"].str.strip() != "")].copy()
    removidos = antes - len(df)
    if removidos > 0:
        print(f"[preprocessing] {removidos} linhas sem texto removidas.")

    return df.reset_index(drop=True)


def preprocessar_dataframe(df: pd.DataFrame, usar_stemming: bool = False) -> pd.DataFrame:
    """
    Adiciona a coluna `texto_limpo` ao DataFrame.

    Args:
        df: DataFrame com coluna `texto`
        usar_stemming: se True, aplica stemming

    Returns:
        DataFrame com coluna `texto_limpo` adicionada
    """
    df = df.copy()
    df["texto_limpo"] = preprocessar_serie(df["texto"], usar_stemming=usar_stemming)

    # Filtra textos que ficaram vazios após limpeza
    mask_vazios = df["texto_limpo"].str.strip() == ""
    if mask_vazios.sum() > 0:
        print(f"[preprocessing] {mask_vazios.sum()} textos ficaram vazios após limpeza — removidos.")
        df = df[~mask_vazios].reset_index(drop=True)

    return df


def salvar_dados_processados(df: pd.DataFrame, caminho: str) -> None:
    """Salva o DataFrame processado em CSV."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    print(f"[preprocessing] Dados processados salvos em: {caminho}")


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_raw = os.path.join(BASE_DIR, "data", "raw", "atendimentos.csv")
    caminho_proc = os.path.join(BASE_DIR, "data", "processed", "atendimentos_processados.csv")

    print("[preprocessing] Carregando dados...")
    df = carregar_dados(caminho_raw)
    print(f"[preprocessing] {len(df)} registros carregados.")

    print("[preprocessing] Aplicando pré-processamento...")
    df_proc = preprocessar_dataframe(df, usar_stemming=False)

    salvar_dados_processados(df_proc, caminho_proc)

    print("\nExemplos de limpeza:")
    for _, row in df_proc.head(5).iterrows():
        print(f"  ORIGINAL : {row['texto']}")
        print(f"  LIMPO    : {row['texto_limpo']}")
        print()
