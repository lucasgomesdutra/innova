"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
tests/test_agrupamento.py — Testes do módulo de agrupamento
=============================================================================
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agrupamento import vetorizar_textos, agrupar_comentarios, termos_por_cluster, resumo_clusters


TEXTOS_TESTE = [
    "aplicativo nao abre erro acesso",
    "app travando problema login",
    "nao consigo entrar aplicativo",
    "pix nao chegou transferencia erro",
    "dinheiro sumiu pix enviado",
    "transferencia falhou banco",
    "atendimento demorado fila espera",
    "esperou muito tempo atendente",
    "demora para ser atendido agencia",
]


class TestVetorizar:

    def test_retorna_matriz_e_vectorizer(self):
        X, vec = vetorizar_textos(TEXTOS_TESTE)
        assert X.shape[0] == len(TEXTOS_TESTE)
        assert X.shape[1] > 0

    def test_aceita_series_pandas(self):
        s = pd.Series(TEXTOS_TESTE)
        X, vec = vetorizar_textos(s)
        assert X.shape[0] == len(TEXTOS_TESTE)

    def test_normalizado_entre_zero_e_um(self):
        import numpy as np
        X, _ = vetorizar_textos(TEXTOS_TESTE)
        # Norma L2 de cada linha deve ser aprox. 1 (ou 0 se vazio)
        normas = np.sqrt(np.array(X.power(2).sum(axis=1))).flatten()
        for n in normas:
            assert abs(n - 1.0) < 0.01 or n == 0.0


class TestAgrupar:

    def test_adiciona_coluna_cluster_id(self):
        df = pd.DataFrame({"texto_limpo": TEXTOS_TESTE, "texto": TEXTOS_TESTE})
        resultado, km, vec = agrupar_comentarios(df, n_clusters=3)
        assert "cluster_id" in resultado.columns
        assert len(resultado) == len(TEXTOS_TESTE)

    def test_numero_de_clusters_correto(self):
        df = pd.DataFrame({"texto_limpo": TEXTOS_TESTE, "texto": TEXTOS_TESTE})
        resultado, km, vec = agrupar_comentarios(df, n_clusters=3)
        assert resultado["cluster_id"].nunique() <= 3

    def test_cluster_id_sao_inteiros(self):
        df = pd.DataFrame({"texto_limpo": TEXTOS_TESTE, "texto": TEXTOS_TESTE})
        resultado, km, vec = agrupar_comentarios(df, n_clusters=2)
        assert resultado["cluster_id"].dtype in [int, "int32", "int64"]


class TestTermosPorCluster:

    def test_retorna_dict_com_termos(self):
        X, vec = vetorizar_textos(TEXTOS_TESTE)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=3, random_state=42, n_init=5)
        km.fit(X)
        termos = termos_por_cluster(km, vec, top_n=5)
        assert isinstance(termos, dict)
        assert len(termos) == 3
        for k, v in termos.items():
            assert isinstance(v, list)
            assert len(v) <= 5


class TestResumoClusters:

    def test_retorna_lista_de_resumos(self):
        df = pd.DataFrame({"texto_limpo": TEXTOS_TESTE, "texto": TEXTOS_TESTE})
        resultado, km, vec = agrupar_comentarios(df, n_clusters=3)
        resumos = resumo_clusters(resultado, km, vec)
        assert isinstance(resumos, list)
        assert len(resumos) > 0
        for r in resumos:
            assert "cluster_id" in r
            assert "tamanho" in r
            assert "termos_principais" in r
            assert "exemplos" in r

    def test_ordenado_por_tamanho(self):
        df = pd.DataFrame({"texto_limpo": TEXTOS_TESTE, "texto": TEXTOS_TESTE})
        resultado, km, vec = agrupar_comentarios(df, n_clusters=3)
        resumos = resumo_clusters(resultado, km, vec)
        tamanhos = [r["tamanho"] for r in resumos]
        assert tamanhos == sorted(tamanhos, reverse=True)
