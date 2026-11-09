"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
app.py — Plataforma de análise de avaliações com Streamlit
=============================================================================
Execução:
    streamlit run app.py
"""

import os, sys, json, io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from classificacao import classificar_comentario
from analise_problemas import gerar_relatorio_problemas, rankear_problemas
from analise_sentimentos import resumo_sentimentos, sentimentos_por_periodo, sentimentos_por_categoria
from relatorios import gerar_todos_relatorios

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

st.set_page_config(
    page_title="Sicredi · Análise de Atendimentos",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR               = os.path.dirname(os.path.abspath(__file__))
CAMINHO_AVALIACOES     = os.path.join(BASE_DIR, "data", "processed", "avaliacoes_reais.csv")
CAMINHO_FICTICIOS      = os.path.join(BASE_DIR, "data", "processed", "atendimentos_processados.csv")
CAMINHO_METRICAS       = os.path.join(BASE_DIR, "models", "metricas_treinamento.json")
CAMINHO_PROBLEMAS      = os.path.join(BASE_DIR, "data", "processed", "problemas_gerenciados.csv")

COLUNAS_AVALIACOES = ["id", "data", "texto", "categoria", "sentimento",
                      "confianca_categoria", "confianca_sentimento", "requer_revisao"]

VERDE    = "#16a34a"
VERMELHO = "#dc2626"
AMARELO  = "#d97706"
CINZA    = "#6b7280"
AZUL     = "#2563eb"

# CSS global — força tema claro em toda a aplicação
st.markdown("""
<style>
/* Fundo geral */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main .block-container { background:#f8fafc !important; }

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div { background:#ffffff !important; border-right:1px solid #e2e8f0; }
[data-testid="stSidebar"] * { color:#1e293b !important; }

/* Texto geral da página */
.main, .main p, .main span, .main div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span { color:#1e293b !important; }

/* Títulos */
h1,h2,h3,h4 { color:#1e293b !important; font-weight:700 !important; }
.main h1 { font-size:1.6rem !important; }

/* Caption / texto secundário */
[data-testid="stCaptionContainer"] p,
small, .stCaption { color:#64748b !important; }

/* Métricas nativas do Streamlit */
[data-testid="stMetric"] {
    background:#ffffff !important;
    border:1px solid #e2e8f0 !important;
    border-radius:10px !important;
    padding:12px 16px !important;
}
[data-testid="stMetricLabel"] p { color:#374151 !important; font-size:13px !important; }
[data-testid="stMetricValue"] { color:#1e293b !important; font-size:1.8rem !important; font-weight:700 !important; }
[data-testid="stMetricDelta"] { color:#64748b !important; }

/* Inputs, selectbox, text_area */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] { background:#ffffff !important; color:#1e293b !important; }
[data-baseweb="select"] * { color:#1e293b !important; }

/* Tabelas dataframe */
[data-testid="stDataFrame"] { background:#ffffff !important; }

/* Botões */
[data-testid="stButton"] button { color:#ffffff !important; }

/* Radio / menu lateral */
[data-testid="stRadio"] label { color:#1e293b !important; }
[data-testid="stRadio"] p { color:#1e293b !important; }

/* Expander */
[data-testid="stExpander"] summary p { color:#1e293b !important; }

/* Info / warning / success boxes */
[data-testid="stAlert"] p { color:#1e293b !important; }

/* Slider */
[data-testid="stSlider"] p { color:#1e293b !important; }

/* Separadores e espaçamento */
.block-container { padding-top:1.5rem !important; padding-bottom:2rem !important; }
hr { border-color:#e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS DE DADOS
# =============================================================================

def _garantir_arquivos():
    os.makedirs(os.path.dirname(CAMINHO_AVALIACOES), exist_ok=True)
    if not os.path.exists(CAMINHO_AVALIACOES):
        pd.DataFrame(columns=COLUNAS_AVALIACOES).to_csv(CAMINHO_AVALIACOES, index=False, encoding="utf-8-sig")
    if not os.path.exists(CAMINHO_PROBLEMAS):
        pd.DataFrame(columns=["id","categoria","prioridade","status","observacao","criado_em","atualizado_em"]
                     ).to_csv(CAMINHO_PROBLEMAS, index=False, encoding="utf-8-sig")

_garantir_arquivos()


def salvar_avaliacao(resultado: dict):
    df_ex = pd.read_csv(CAMINHO_AVALIACOES, encoding="utf-8-sig")
    novo_id = int(df_ex["id"].max() + 1) if not df_ex.empty else 1
    nova = pd.DataFrame([{
        "id": novo_id, "data": datetime.now().strftime("%Y-%m-%d"),
        "texto": resultado["texto_original"], "categoria": resultado["categoria"],
        "sentimento": resultado["sentimento"],
        "confianca_categoria": resultado["confianca_categoria"],
        "confianca_sentimento": resultado["confianca_sentimento"],
        "requer_revisao": resultado["requer_revisao"],
    }])
    pd.concat([df_ex, nova], ignore_index=True).to_csv(CAMINHO_AVALIACOES, index=False, encoding="utf-8-sig")
    carregar_dados.clear()


@st.cache_data(show_spinner=False)
def carregar_dados():
    df_reais = pd.read_csv(CAMINHO_AVALIACOES, encoding="utf-8-sig")
    partes = []
    if os.path.exists(CAMINHO_FICTICIOS):
        df_f = pd.read_csv(CAMINHO_FICTICIOS, encoding="utf-8-sig")[
            ["texto","categoria","sentimento","data"]].copy()
        df_f["id"] = range(-len(df_f), 0)
        df_f["confianca_categoria"] = 1.0
        df_f["confianca_sentimento"] = 1.0
        df_f["requer_revisao"] = False
        partes.append(df_f)
    if not df_reais.empty:
        partes.append(df_reais)
    if not partes:
        return None
    return pd.concat(partes, ignore_index=True)


@st.cache_data(show_spinner=False)
def carregar_metricas():
    if not os.path.exists(CAMINHO_METRICAS):
        return None
    with open(CAMINHO_METRICAS, encoding="utf-8") as f:
        return json.load(f)


def filtrar_por_periodo(df: pd.DataFrame, periodo: str, data_ini=None, data_fim=None) -> pd.DataFrame:
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    hoje = pd.Timestamp.now().normalize()
    if periodo == "Hoje":
        return df[df["data"] >= hoje]
    if periodo == "Últimos 7 dias":
        return df[df["data"] >= hoje - timedelta(days=7)]
    if periodo == "Últimos 30 dias":
        return df[df["data"] >= hoje - timedelta(days=30)]
    if periodo == "Últimos 90 dias":
        return df[df["data"] >= hoje - timedelta(days=90)]
    if periodo == "Personalizado" and data_ini and data_fim:
        return df[(df["data"] >= pd.Timestamp(data_ini)) & (df["data"] <= pd.Timestamp(data_fim))]
    return df  # "Todo o período"


def gerar_resumo_ia(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Sem dados suficientes para gerar um resumo."
    total = len(df)
    neg = int((df["sentimento"] == "negativo").sum())
    pos = int((df["sentimento"] == "positivo").sum())
    pct_neg = neg / total * 100 if total else 0
    ranking = rankear_problemas(df)
    linhas = []
    if ranking is not None and not ranking.empty:
        top = ranking.iloc[0]
        linhas.append(
            f"A categoria com mais reclamações é **{top['categoria']}**, "
            f"representando {top['percentual']:.1f}% das avaliações negativas."
        )
        if len(ranking) >= 2:
            seg = ranking.iloc[1]
            linhas.append(
                f"Em segundo lugar, **{seg['categoria']}** concentra {seg['percentual']:.1f}% das reclamações."
            )
    if pct_neg > 60:
        linhas.append(
            f"Atenção: {pct_neg:.1f}% das avaliações do período são negativas, "
            "o que indica um nível elevado de insatisfação. Recomenda-se revisão ampla dos processos."
        )
    elif pct_neg > 40:
        linhas.append(
            f"O índice de avaliações negativas ({pct_neg:.1f}%) está acima do esperado. "
            "Sugere-se investigar as categorias com maior volume de reclamações."
        )
    else:
        linhas.append(
            f"O índice de satisfação está relativamente equilibrado, com {pos} avaliações positivas "
            f"e {neg} negativas no período."
        )
    cats_neg = df[df["sentimento"] == "negativo"]["categoria"].value_counts()
    if not cats_neg.empty and cats_neg.iloc[0] > 3:
        linhas.append(
            f"Destaque para **{cats_neg.index[0]}**: {cats_neg.iloc[0]} ocorrências negativas identificadas."
        )
    linhas.append(
        "Os resultados acima são gerados automaticamente pelo modelo de IA e devem ser validados por profissionais."
    )
    return " ".join(linhas)


def _card_metrica(titulo, valor, subtitulo="", cor=AZUL, icone="📊"):
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #e2e8f0;border-radius:12px;
                padding:16px 18px;height:100%;'>
        <div style='font-size:22px;margin-bottom:4px;'>{icone}</div>
        <div style='font-size:26px;font-weight:700;color:{cor};line-height:1.1;'>{valor}</div>
        <div style='font-size:13px;font-weight:600;color:#374151;margin-top:4px;'>{titulo}</div>
        <div style='font-size:11px;color:#9ca3af;margin-top:2px;'>{subtitulo}</div>
    </div>""", unsafe_allow_html=True)


def _badge(texto, cor_bg, cor_txt="white"):
    return f"<span style='background:{cor_bg};color:{cor_txt};padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;'>{texto}</span>"


# =============================================================================
# SIDEBAR
# =============================================================================

# CSS da sidebar — design refinado com estado ativo/hover
st.markdown("""
<style>
/* Esconde o label do st.radio na sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] > label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] .st-emotion-cache-ue6h4q,
[data-testid="stSidebar"] [data-testid="stRadio"] + div > p {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Oculta o radio nativo e substitui pelo nosso visual */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 13px 16px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background 0.15s ease !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #f0fdf4 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p {
    color: #16a34a !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}
/* Esconde o círculo do radio */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}
/* Item selecionado */
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {
    background: #f0fdf4 !important;
    border-left: 3px solid #16a34a !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] p {
    color: #15803d !important;
    font-weight: 600 !important;
}
/* Label texto do radio */
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 15px !important;
    color: #374151 !important;
    margin: 0 !important;
    line-height: 1.3 !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # --- Logo + Nome ---
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding:18px 14px 14px;'>
        <div style='width:38px;height:38px;background:#16a34a;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;flex-shrink:0;'>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                 xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48
                         10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"
                      fill="white"/>
                <circle cx="12" cy="12" r="3" fill="white" opacity="0.3"/>
                <path d="M9 12l2 2 4-4" stroke="white" stroke-width="1.5"
                      stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div>
            <div style='font-size:16px;font-weight:800;color:#1e293b;
                        letter-spacing:-0.3px;line-height:1.2;'>AtendeAI</div>
            <div style='font-size:11px;color:#64748b;font-weight:400;'>
                Inteligência no Atendimento</div>
        </div>
    </div>
    <hr style='border:none;border-top:1px solid #e2e8f0;margin:0 0 8px;'>
    """, unsafe_allow_html=True)

    pagina = st.radio("Menu", [
        "📊 Dashboard",
        "📋 Avaliações",
        "🔍 Análise com IA",
        "🚨 Problemas Identificados",
        "📄 Relatórios",
    ], label_visibility="collapsed")

    # --- Rodapé ---
    st.markdown("""
    <div style='position:absolute;bottom:0;left:0;right:0;
                padding:14px 16px;border-top:1px solid #e2e8f0;
                background:#ffffff;'>
        <div style='font-size:11px;color:#9ca3af;font-weight:500;'>
            Sistema de Análise Inteligente</div>
        <div style='font-size:10px;color:#cbd5e1;margin-top:2px;'>Versão 1.0</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CARREGA DADOS
# =============================================================================

df_all   = carregar_dados()
metricas = carregar_metricas()

def _vazio():
    st.info("**Sem dados disponíveis.** Execute `python src/treinamento.py` ou insira avaliações em **🔍 Análise com IA**.")

# =============================================================================
# 1. DASHBOARD
# =============================================================================

if pagina == "📊 Dashboard":
    # Cabeçalho
    col_t, col_btn = st.columns([3, 1])
    with col_t:
        st.markdown("## 📊 Dashboard Inteligente")
        st.caption("Visão geral das avaliações e da satisfação dos clientes")
    with col_btn:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Atualizar", use_container_width=True):
            carregar_dados.clear()
            st.rerun()

    # Filtro de período
    col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
    with col_p1:
        periodo = st.selectbox("Período", ["Todo o período","Hoje","Últimos 7 dias",
                                           "Últimos 30 dias","Últimos 90 dias","Personalizado"],
                               label_visibility="collapsed")
    data_ini = data_fim = None
    if periodo == "Personalizado":
        with col_p2:
            data_ini = st.date_input("De", value=datetime.now().date() - timedelta(days=30),
                                     label_visibility="collapsed")
        with col_p3:
            data_fim = st.date_input("Até", value=datetime.now().date(),
                                     label_visibility="collapsed")

    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0 16px;'>",
                unsafe_allow_html=True)

    if df_all is None:
        _vazio()
    else:
        df = filtrar_por_periodo(df_all, periodo, data_ini, data_fim)
        if df.empty:
            st.warning("Nenhuma avaliação encontrada para o período selecionado.")
        else:
            total     = len(df)
            neg       = int((df["sentimento"] == "negativo").sum())
            pos       = int((df["sentimento"] == "positivo").sum())
            neu       = int((df["sentimento"] == "neutro").sum())
            pct_neg   = neg / total * 100 if total else 0
            pct_pos   = pos / total * 100 if total else 0
            sat       = round(pct_pos - pct_neg, 1)
            cor_sat   = VERDE if sat >= 0 else VERMELHO

            # Cards
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: _card_metrica("Total de Avaliações", f"{total:,}", "analisadas pela IA", AZUL, "📋")
            with c2: _card_metrica("Reclamações", f"{neg:,}", f"{pct_neg:.1f}% do total", VERMELHO, "😠")
            with c3: _card_metrica("Elogios", f"{pos:,}", f"{pct_pos:.1f}% do total", VERDE, "😊")
            with c4: _card_metrica("Neutras", f"{neu:,}", "sem sentimento claro", AMARELO, "😐")
            with c5: _card_metrica("Índice de Satisfação", f"{sat:+.1f}%", "elogios − reclamações", cor_sat, "⭐")

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

            # Linha 1: Donut + Evolução
            col_a, col_b = st.columns([1, 2])
            cores_sent = {"negativo": VERMELHO, "positivo": VERDE, "neutro": AMARELO}

            with col_a:
                st.markdown("#### 💬 Distribuição de Sentimentos")
                cont_s = df["sentimento"].value_counts().reset_index()
                cont_s.columns = ["Sentimento", "Qtd"]
                fig_d = px.pie(cont_s, values="Qtd", names="Sentimento",
                               color="Sentimento", color_discrete_map=cores_sent, hole=0.52)
                fig_d.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=10),
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    legend=dict(orientation="h", y=-0.1))
                fig_d.update_traces(textinfo="percent+value", textfont_size=12)
                st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

            with col_b:
                st.markdown("#### 📈 Evolução das Avaliações")
                df_ev = df.copy()
                df_ev["data"] = pd.to_datetime(df_ev["data"], errors="coerce")
                df_ev = df_ev.dropna(subset=["data"])
                df_ev["periodo"] = df_ev["data"].dt.to_period("M").astype(str)
                evo = (df_ev.groupby(["periodo","sentimento"]).size()
                       .unstack(fill_value=0).reset_index())
                cols_s = [c for c in evo.columns if c != "periodo"]
                evo = evo[evo[cols_s].sum(axis=1) > 0]
                evo_long = evo.melt(id_vars="periodo", var_name="sentimento", value_name="qtd")
                evo_long = evo_long[evo_long["qtd"] > 0]
                fig_ev = px.line(evo_long, x="periodo", y="qtd", color="sentimento",
                                 color_discrete_map=cores_sent, markers=True,
                                 labels={"periodo":"Período","qtd":"Avaliações","sentimento":"Sentimento"})
                fig_ev.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=40),
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     legend=dict(orientation="h",y=-0.25),
                                     xaxis=dict(tickangle=-30, type="category"))
                fig_ev.update_traces(line=dict(width=2.5), marker=dict(size=6), connectgaps=False)
                st.plotly_chart(fig_ev, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:16px 0;'>",
                        unsafe_allow_html=True)

            # Linha 2: Ranking + Resumo IA
            col_c, col_d = st.columns([1, 1])

            with col_c:
                st.markdown("#### 🚨 Principais Problemas")
                ranking = rankear_problemas(df)
                if ranking is not None and not ranking.empty:
                    rp = ranking.sort_values("total_reclamacoes", ascending=True)
                    fig_r = px.bar(rp, x="total_reclamacoes", y="categoria", orientation="h",
                                   color="total_reclamacoes",
                                   color_continuous_scale=[[0,"#fca5a5"],[1,VERMELHO]],
                                   text="total_reclamacoes",
                                   labels={"total_reclamacoes":"","categoria":""})
                    fig_r.update_layout(height=max(260, len(rp)*52),
                                        showlegend=False, coloraxis_showscale=False,
                                        margin=dict(l=0,r=60,t=10,b=10),
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        bargap=0.3,
                                        xaxis=dict(visible=False),
                                        yaxis=dict(tickfont=dict(size=11)))
                    fig_r.update_traces(textposition="outside", textfont_size=12, marker_line_width=0)
                    st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("Nenhuma reclamação no período.")

            with col_d:
                st.markdown("#### 🤖 Análise Inteligente da IA")
                resumo = gerar_resumo_ia(df)
                st.markdown(f"""
                <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                            padding:16px 18px;font-size:13px;line-height:1.7;color:#1e293b;'>
                {resumo}
                </div>""", unsafe_allow_html=True)

            st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:16px 0;'>",
                        unsafe_allow_html=True)

            # Linha 3: Alertas + Avaliações recentes
            col_e, col_f = st.columns([1, 1])

            with col_e:
                st.markdown("#### ⚠️ Pontos de Atenção")
                rel = gerar_relatorio_problemas(df)
                if rel["alertas_aumento"]:
                    for a in rel["alertas_aumento"][:4]:
                        st.markdown(f"""
                        <div style='background:#fff7ed;border-left:4px solid {AMARELO};
                                    border-radius:6px;padding:10px 14px;margin-bottom:8px;'>
                            <div style='font-size:13px;font-weight:600;color:#92400e;'>
                                ⚠ {a['categoria']}
                            </div>
                            <div style='font-size:12px;color:#78350f;'>
                                +{a['variacao_percentual']:.1f}% em {a['periodo_atual']}
                                ({a['volume_anterior']} → {a['volume_atual']} reclamações)
                            </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    for s in rel["sugestoes_investigacao"][:3]:
                        st.markdown(f"""
                        <div style='background:#eff6ff;border-left:4px solid {AZUL};
                                    border-radius:6px;padding:10px 14px;margin-bottom:8px;
                                    font-size:12px;color:#1e40af;'>
                            → {s}
                        </div>""", unsafe_allow_html=True)

            with col_f:
                st.markdown("#### 🕓 Avaliações Recentes")
                df_rec = df.sort_values("id", ascending=False).head(8)[
                    ["data","texto","categoria","sentimento"]
                ].rename(columns={"data":"Data","texto":"Comentário",
                                   "categoria":"Categoria","sentimento":"Sentimento"})
                st.dataframe(df_rec, hide_index=True, use_container_width=True, height=300)


# =============================================================================
# 2. AVALIAÇÕES
# =============================================================================

elif pagina == "📋 Avaliações":
    st.markdown("## 📋 Avaliações")
    st.caption("Histórico completo de avaliações analisadas pela IA.")
    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0 16px;'>",
                unsafe_allow_html=True)

    if df_all is None:
        _vazio()
    else:
        df_av = df_all.copy()
        df_av["data"] = pd.to_datetime(df_av["data"], errors="coerce")

        # Filtros
        cf1, cf2, cf3, cf4 = st.columns(4)
        with cf1:
            busca = st.text_input("🔍 Buscar no texto", placeholder="Digite palavra-chave...")
        with cf2:
            cats = ["Todas"] + sorted(df_av["categoria"].dropna().unique().tolist())
            cat_sel = st.selectbox("Categoria", cats)
        with cf3:
            sent_sel = st.selectbox("Sentimento", ["Todos", "negativo", "positivo", "neutro"])
        with cf4:
            ordem = st.selectbox("Ordenar por", ["Mais recentes","Mais antigas"])

        # Aplica filtros
        if busca:
            df_av = df_av[df_av["texto"].str.contains(busca, case=False, na=False)]
        if cat_sel != "Todas":
            df_av = df_av[df_av["categoria"] == cat_sel]
        if sent_sel != "Todos":
            df_av = df_av[df_av["sentimento"] == sent_sel]
        if ordem == "Mais recentes":
            df_av = df_av.sort_values("data", ascending=False)
        else:
            df_av = df_av.sort_values("data", ascending=True)

        st.markdown(f"**{len(df_av):,}** avaliações encontradas")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        cores_sent = {"negativo": VERMELHO, "positivo": VERDE, "neutro": AMARELO}

        for _, row in df_av.head(50).iterrows():
            cor = cores_sent.get(str(row.get("sentimento","")), CINZA)
            conf_cat  = row.get("confianca_categoria", "")
            conf_sent = row.get("confianca_sentimento", "")
            conf_txt = ""
            if conf_cat != "" and not pd.isna(conf_cat):
                conf_txt = f"Confiança cat.: {float(conf_cat):.0%} | sent.: {float(conf_sent):.0%}"
            data_str = str(row.get("data",""))[:10]
            badge = _badge(str(row.get("sentimento","")).capitalize(), cor)
            st.markdown(f"""
            <div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                        padding:12px 16px;margin-bottom:8px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
                    <span style='font-size:12px;color:#64748b;'>{data_str} · {row.get("categoria","")}</span>
                    {badge}
                </div>
                <div style='font-size:13px;color:#1e293b;'>{row.get("texto","")}</div>
                <div style='font-size:11px;color:#9ca3af;margin-top:4px;'>{conf_txt}</div>
            </div>""", unsafe_allow_html=True)

        if len(df_av) > 50:
            st.caption(f"Exibindo 50 de {len(df_av)} avaliações.")


# =============================================================================
# 3. ANÁLISE COM IA
# =============================================================================

elif pagina == "🔍 Análise com IA":
    st.markdown("## 🔍 Análise com IA")
    st.caption("Insira um comentário e a IA classificará automaticamente a categoria, o sentimento e o grau de confiança.")
    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0 16px;'>",
                unsafe_allow_html=True)

    texto_input = st.text_area("💬 Comentário do cliente:", height=110,
                               placeholder="Ex: Fiz um Pix e o dinheiro não chegou ao destinatário...")

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analisar = st.button("🤖 Analisar", type="primary", use_container_width=True)

    if analisar:
        if not texto_input.strip():
            st.warning("Insira um comentário para analisar.")
        else:
            with st.spinner("Analisando com IA..."):
                res = classificar_comentario(texto_input)
                salvar_avaliacao(res)

            cores_sent = {"negativo": VERMELHO, "positivo": VERDE, "neutro": AMARELO}
            cor_sent = cores_sent.get(res["sentimento"], CINZA)

            st.success("✅ Avaliação analisada e salva no dashboard.")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;'>
                    <div style='font-size:12px;color:#64748b;font-weight:600;'>CATEGORIA IDENTIFICADA</div>
                    <div style='font-size:16px;font-weight:700;color:#1e293b;margin:6px 0;'>
                        {res['categoria']}
                    </div>
                    <div style='font-size:12px;color:#64748b;'>
                        Confiança: {res['confianca_categoria']:.1%}
                    </div>
                </div>""", unsafe_allow_html=True)
                st.progress(res["confianca_categoria"])

            with c2:
                sent_label = res["sentimento"].capitalize()
                st.markdown(f"""
                <div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;'>
                    <div style='font-size:12px;color:#64748b;font-weight:600;'>SENTIMENTO DETECTADO</div>
                    <div style='font-size:16px;font-weight:700;color:{cor_sent};margin:6px 0;'>
                        {sent_label}
                    </div>
                    <div style='font-size:12px;color:#64748b;'>
                        Confiança: {res['confianca_sentimento']:.1%}
                    </div>
                </div>""", unsafe_allow_html=True)
                st.progress(res["confianca_sentimento"])

            with c3:
                rev_txt = "Sim — baixa confiança, recomenda revisão humana." if res["requer_revisao"] else "Não — classificação com boa confiança."
                rev_cor = AMARELO if res["requer_revisao"] else VERDE
                st.markdown(f"""
                <div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;'>
                    <div style='font-size:12px;color:#64748b;font-weight:600;'>REVISÃO NECESSÁRIA?</div>
                    <div style='font-size:14px;font-weight:600;color:{rev_cor};margin:6px 0;'>
                        {"⚠ " if res["requer_revisao"] else "✅ "}{rev_txt}
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            with st.expander("🔬 Probabilidades por categoria"):
                df_pc = pd.DataFrame(list(res["prob_categorias"].items()),
                                     columns=["Categoria","Prob"]).sort_values("Prob", ascending=True)
                fig_pc = px.bar(df_pc, x="Prob", y="Categoria", orientation="h",
                                color="Prob", color_continuous_scale="Blues",
                                text=df_pc["Prob"].apply(lambda x: f"{x:.1%}"))
                fig_pc.update_layout(height=380, coloraxis_showscale=False,
                                     margin=dict(l=0,r=60,t=10,b=10),
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     xaxis=dict(visible=False))
                fig_pc.update_traces(textposition="outside", marker_line_width=0)
                st.plotly_chart(fig_pc, use_container_width=True, config={"displayModeBar": False})

            with st.expander("🔬 Probabilidades por sentimento"):
                df_ps = pd.DataFrame(list(res["prob_sentimentos"].items()),
                                     columns=["Sentimento","Prob"])
                fig_ps = px.bar(df_ps, x="Sentimento", y="Prob", color="Sentimento",
                                color_discrete_map={"negativo":VERMELHO,"positivo":VERDE,"neutro":AMARELO},
                                text=df_ps["Prob"].apply(lambda x: f"{x:.1%}"))
                fig_ps.update_layout(height=280, showlegend=False,
                                     margin=dict(l=0,r=0,t=10,b=10),
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     yaxis=dict(visible=False))
                fig_ps.update_traces(textposition="outside", marker_line_width=0)
                st.plotly_chart(fig_ps, use_container_width=True, config={"displayModeBar": False})

            with st.expander("🔤 Texto pré-processado"):
                st.code(res["texto_limpo"], language=None)


# =============================================================================
# 4. PROBLEMAS RECORRENTES
# =============================================================================

# =============================================================================
# 5. PROBLEMAS IDENTIFICADOS
# =============================================================================

elif pagina == "🚨 Problemas Identificados":
    st.markdown("## 🚨 Problemas Identificados")
    st.caption("Gerencie os problemas identificados pela IA, defina prioridade e acompanhe o andamento.")
    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0 16px;'>",
                unsafe_allow_html=True)

    if df_all is None:
        _vazio()
    else:
        df_prob = pd.read_csv(CAMINHO_PROBLEMAS, encoding="utf-8-sig")
        ranking = rankear_problemas(df_all)

        # Adiciona categorias novas ao gerenciamento automaticamente
        if ranking is not None and not ranking.empty:
            for _, row in ranking.iterrows():
                cat = row["categoria"]
                if cat not in df_prob["categoria"].values:
                    novo = pd.DataFrame([{
                        "id": len(df_prob) + 1, "categoria": cat,
                        "prioridade": "Média" if row["percentual"] > 20 else "Baixa",
                        "status": "Pendente", "observacao": "",
                        "criado_em": datetime.now().strftime("%Y-%m-%d"),
                        "atualizado_em": datetime.now().strftime("%Y-%m-%d"),
                    }])
                    df_prob = pd.concat([df_prob, novo], ignore_index=True)
            df_prob.to_csv(CAMINHO_PROBLEMAS, index=False, encoding="utf-8-sig")

        # Exibe e permite editar
        rel = gerar_relatorio_problemas(df_all)
        alertas = rel["alertas_aumento"]

        if alertas:
            st.markdown("#### ⚡ Alertas Ativos")
            for a in alertas[:5]:
                st.markdown(f"""
                <div style='background:#fff7ed;border-left:4px solid {AMARELO};border-radius:6px;
                            padding:10px 14px;margin-bottom:8px;'>
                    <b>{a['categoria']}</b> — aumento de {a['variacao_percentual']:.1f}% em {a['periodo_atual']}
                </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown("#### 📋 Gerenciamento de Problemas")
        cores_prior = {"Alta": VERMELHO, "Média": AMARELO, "Baixa": VERDE}
        cores_status = {"Pendente": CINZA, "Em análise": AZUL, "Resolvido": VERDE}

        df_prob_edit = df_prob.copy()
        for i, row in df_prob_edit.iterrows():
            rec_count = 0
            if ranking is not None:
                match = ranking[ranking["categoria"] == row["categoria"]]
                if not match.empty:
                    rec_count = int(match.iloc[0]["total_reclamacoes"])

            cp, cs, cobs, cbtn = st.columns([1.5, 1.5, 3, 0.8])
            with cp:
                nova_prior = st.selectbox("Prioridade", ["Alta","Média","Baixa"],
                                          index=["Alta","Média","Baixa"].index(str(row["prioridade"])),
                                          key=f"prior_{i}", label_visibility="collapsed")
            with cs:
                novo_status = st.selectbox("Status", ["Pendente","Em análise","Resolvido"],
                                           index=["Pendente","Em análise","Resolvido"].index(str(row["status"])),
                                           key=f"status_{i}", label_visibility="collapsed")
            with cobs:
                nova_obs = st.text_input("Observação", value=str(row["observacao"]) if pd.notna(row["observacao"]) else "",
                                         key=f"obs_{i}", label_visibility="collapsed",
                                         placeholder=f"{row['categoria']} · {rec_count} reclamações")
            with cbtn:
                if st.button("💾", key=f"save_{i}", use_container_width=True):
                    df_prob.at[i, "prioridade"]    = nova_prior
                    df_prob.at[i, "status"]        = novo_status
                    df_prob.at[i, "observacao"]    = nova_obs
                    df_prob.at[i, "atualizado_em"] = datetime.now().strftime("%Y-%m-%d")
                    df_prob.to_csv(CAMINHO_PROBLEMAS, index=False, encoding="utf-8-sig")
                    st.success("Salvo!", icon="✅")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("#### 💬 Exemplos por Categoria")
        exemplos = {k: v for k, v in rel["exemplos_por_categoria"].items() if v}
        for cat, exs in exemplos.items():
            with st.expander(f"📂 {cat} ({len(exs)} exemplo(s))"):
                for ex in exs:
                    st.markdown(f"• {ex}")

        st.caption("⚠️ Resultados gerados por IA. Indicativos — revisar antes de qualquer decisão.")


# =============================================================================
# 6. RELATÓRIOS
# =============================================================================

elif pagina == "📄 Relatórios":
    st.markdown("## 📄 Relatórios")
    st.caption("Gere e exporte relatórios completos das avaliações analisadas.")
    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0 16px;'>",
                unsafe_allow_html=True)

    if df_all is None:
        _vazio()
    else:
        col_per, col_gen = st.columns([3, 1])
        with col_per:
            periodo_rel = st.selectbox("Período do relatório",
                                       ["Todo o período","Últimos 30 dias","Últimos 90 dias"],
                                       label_visibility="collapsed")
        with col_gen:
            gerar_btn = st.button("📊 Gerar Relatório", type="primary", use_container_width=True)

        if gerar_btn:
            df_r = filtrar_por_periodo(df_all, periodo_rel)

            if df_r.empty:
                st.warning("Nenhum dado no período.")
            else:
                rel = gerar_relatorio_problemas(df_r)
                total_r = len(df_r)
                neg_r   = int((df_r["sentimento"] == "negativo").sum())
                pos_r   = int((df_r["sentimento"] == "positivo").sum())
                neu_r   = int((df_r["sentimento"] == "neutro").sum())

                st.markdown(f"### Relatório · {periodo_rel}")
                st.markdown(f"**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
                st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", f"{total_r:,}")
                c2.metric("Negativas", f"{neg_r:,}", f"{neg_r/total_r*100:.1f}%")
                c3.metric("Positivas", f"{pos_r:,}", f"{pos_r/total_r*100:.1f}%")
                c4.metric("Neutras", f"{neu_r:,}")

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                col_g1, col_g2 = st.columns(2)
                cores_sent = {"negativo":VERMELHO,"positivo":VERDE,"neutro":AMARELO}

                with col_g1:
                    cont_s = df_r["sentimento"].value_counts().reset_index()
                    cont_s.columns = ["Sentimento","Qtd"]
                    fig_pie = px.pie(cont_s, values="Qtd", names="Sentimento",
                                     color="Sentimento", color_discrete_map=cores_sent, hole=0.45)
                    fig_pie.update_layout(height=260, margin=dict(l=0,r=0,t=20,b=0),
                                          paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

                with col_g2:
                    ranking_r = rankear_problemas(df_r)
                    if ranking_r is not None and not ranking_r.empty:
                        fig_rb = px.bar(ranking_r.sort_values("total_reclamacoes", ascending=True),
                                        x="total_reclamacoes", y="categoria", orientation="h",
                                        color_discrete_sequence=[VERMELHO], text="total_reclamacoes")
                        fig_rb.update_layout(height=260, showlegend=False,
                                             margin=dict(l=0,r=60,t=20,b=0),
                                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                             xaxis=dict(visible=False))
                        fig_rb.update_traces(textposition="outside", marker_line_width=0)
                        st.plotly_chart(fig_rb, use_container_width=True, config={"displayModeBar": False})

                # Resumo IA
                st.markdown("#### 🤖 Resumo da IA")
                st.markdown(f"""
                <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                            padding:14px 16px;font-size:13px;line-height:1.7;'>
                {gerar_resumo_ia(df_r)}
                </div>""", unsafe_allow_html=True)

                # Exportação Excel
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                st.markdown("#### 📥 Exportar")
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df_r[["data","texto","categoria","sentimento","confianca_categoria","confianca_sentimento"]
                         ].rename(columns={"data":"Data","texto":"Comentário","categoria":"Categoria",
                                           "sentimento":"Sentimento","confianca_categoria":"Conf. Categoria",
                                           "confianca_sentimento":"Conf. Sentimento"}
                         ).to_excel(writer, index=False, sheet_name="Avaliações")
                    if ranking_r is not None and not ranking_r.empty:
                        ranking_r.to_excel(writer, index=False, sheet_name="Ranking Problemas")
                buf.seek(0)
                st.download_button("⬇️ Baixar Excel", data=buf,
                                   file_name=f"relatorio_sicredi_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)


# =============================================================================
# 7. MÉTRICAS DO MODELO
# =============================================================================

