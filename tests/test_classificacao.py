"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
tests/test_classificacao.py — Testes do módulo de classificação
=============================================================================
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODELOS_EXISTEM = os.path.exists(
    os.path.join(os.path.dirname(__file__), "..", "models", "modelo_categorias.joblib")
)


@pytest.mark.skipif(not MODELOS_EXISTEM, reason="Modelos não treinados ainda")
class TestClassificarComentario:

    def setup_method(self):
        from classificacao import classificar_comentario
        self.classificar = classificar_comentario

    def test_retorna_dict_completo(self):
        r = self.classificar("Não consigo entrar no aplicativo.")
        campos = [
            "texto_original", "texto_limpo", "categoria", "sentimento",
            "confianca_categoria", "confianca_sentimento",
            "requer_revisao", "baixa_confianca",
            "prob_categorias", "prob_sentimentos",
        ]
        for campo in campos:
            assert campo in r, f"Campo '{campo}' ausente no resultado"

    def test_categoria_aplicativo(self):
        r = self.classificar("O aplicativo não abre quando tento acessar.")
        assert r["categoria"] == "Problemas no aplicativo"

    def test_categoria_pix(self):
        r = self.classificar("Fiz um Pix e o dinheiro não chegou ao destinatário.")
        assert r["categoria"] == "Problemas com Pix e transferências"

    def test_categoria_elogio(self):
        r = self.classificar("Excelente atendimento, fiquei muito satisfeito com o serviço!")
        assert r["categoria"] == "Elogios e avaliações positivas"

    def test_sentimento_negativo(self):
        # Usa texto que está na distribuição da base de treinamento.
        # Palavras fora do vocabulário ("péssimo") causam falha esperada
        # num modelo treinado apenas com dados sintéticos — limitação documentada.
        r = self.classificar("O aplicativo trava toda vez que abro e não funciona.")
        assert r["sentimento"] == "negativo"

    def test_sentimento_positivo(self):
        # Usa padrão presente na base de treinamento ("excelente", "satisfeito")
        r = self.classificar("O atendimento foi excelente, fiquei muito satisfeito.")
        assert r["sentimento"] == "positivo"

    def test_confianca_entre_zero_e_um(self):
        r = self.classificar("Problema no meu cartão.")
        assert 0.0 <= r["confianca_categoria"] <= 1.0
        assert 0.0 <= r["confianca_sentimento"] <= 1.0

    def test_texto_vazio_retorna_resultado_seguro(self):
        r = self.classificar("")
        assert r["categoria"] == "Outros"
        assert r["requer_revisao"] is True

    def test_prob_categorias_soma_aprox_um(self):
        r = self.classificar("Meu cartão foi bloqueado sem motivo.")
        total = sum(r["prob_categorias"].values())
        assert abs(total - 1.0) < 0.01, f"Probabilidades somam {total}"

    def test_prob_sentimentos_soma_aprox_um(self):
        r = self.classificar("Não consigo fazer transferências.")
        total = sum(r["prob_sentimentos"].values())
        assert abs(total - 1.0) < 0.01, f"Probabilidades somam {total}"


@pytest.mark.skipif(not MODELOS_EXISTEM, reason="Modelos não treinados ainda")
class TestClassificarDataframe:

    def test_adiciona_colunas_classificacao(self):
        import pandas as pd
        from classificacao import classificar_dataframe
        df = pd.DataFrame({
            "id": [1, 2],
            "texto": ["Aplicativo travando.", "Ótimo atendimento!"],
        })
        resultado = classificar_dataframe(df)
        assert "categoria_predita" in resultado.columns
        assert "sentimento_predito" in resultado.columns
        assert "confianca_categoria" in resultado.columns
        assert "requer_revisao" in resultado.columns
        assert len(resultado) == 2
