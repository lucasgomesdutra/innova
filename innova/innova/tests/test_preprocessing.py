"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
tests/test_preprocessing.py — Testes do módulo de pré-processamento
=============================================================================
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import limpar_texto, preprocessar_serie, preprocessar_dataframe


class TestLimparTexto:

    def test_minusculas(self):
        assert limpar_texto("TEXTO EM MAIÚSCULAS") == limpar_texto("texto em maiúsculas")

    def test_remove_pontuacao(self):
        resultado = limpar_texto("Texto, com pontuação! e outros: caracteres?")
        assert "," not in resultado
        assert "!" not in resultado
        assert "?" not in resultado

    def test_remove_acentos(self):
        resultado = limpar_texto("ação coração limão")
        assert "ã" not in resultado
        assert "ç" not in resultado

    def test_remove_stopwords(self):
        resultado = limpar_texto("eu não consigo acessar o aplicativo")
        # Stopwords como 'eu', 'o', 'não' (estrutural) devem ser removidas
        assert "eu" not in resultado.split()

    def test_texto_vazio(self):
        assert limpar_texto("") == ""

    def test_texto_none(self):
        assert limpar_texto(None) == ""

    def test_texto_apenas_stopwords(self):
        # Se o texto tem apenas stopwords, o resultado pode ser vazio ou com poucos tokens
        resultado = limpar_texto("eu e você")
        assert isinstance(resultado, str)

    def test_preserva_palavras_relevantes(self):
        resultado = limpar_texto("aplicativo lento erro acesso bloqueado")
        assert "aplicativo" in resultado or "lento" in resultado or "erro" in resultado

    def test_tokens_curtos_removidos(self):
        resultado = limpar_texto("a b c aplicativo")
        tokens = resultado.split()
        assert all(len(t) > 1 for t in tokens)


class TestPreprocessarSerie:

    def test_retorna_series(self):
        s = pd.Series(["texto um", "texto dois"])
        resultado = preprocessar_serie(s)
        assert isinstance(resultado, pd.Series)
        assert len(resultado) == 2

    def test_trata_nan(self):
        s = pd.Series(["texto", None, float("nan")])
        resultado = preprocessar_serie(s)
        assert resultado.iloc[1] == ""
        assert resultado.iloc[2] == ""


class TestPreprocessarDataframe:

    def test_adiciona_coluna_texto_limpo(self):
        df = pd.DataFrame({"texto": ["Não consigo acessar o app.", "Ótimo atendimento!"]})
        df_proc = preprocessar_dataframe(df)
        assert "texto_limpo" in df_proc.columns

    def test_nao_altera_outras_colunas(self):
        df = pd.DataFrame({
            "texto": ["texto aqui"],
            "categoria": ["Outros"],
            "sentimento": ["neutro"],
        })
        df_proc = preprocessar_dataframe(df)
        assert "categoria" in df_proc.columns
        assert "sentimento" in df_proc.columns
        assert df_proc.iloc[0]["categoria"] == "Outros"

    def test_remove_textos_vazios_apos_limpeza(self):
        df = pd.DataFrame({"texto": ["", "   ", "aplicativo erro acesso"]})
        df_proc = preprocessar_dataframe(df)
        # Os dois textos vazios devem ser removidos
        assert len(df_proc) <= 3
        assert all(df_proc["texto_limpo"].str.strip() != "")
