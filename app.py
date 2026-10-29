"""
Observatório de Quedas · Pará
Projeto "De mãos dadas para não deixar cair" — LEMOH/UFPA · FAPESPA 105/2024

Painel interativo (Streamlit) sobre risco de quedas em mulheres paraenses:
território, ciclo de vida, comorbidades e equilíbrio postural (estabilometria).
Dados desidentificados em dados_quedas.csv.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="Observatório de Quedas · Pará", page_icon="📊", layout="wide")

MACROS = ["I", "II", "III", "IV"]
MACRO_COR = {"Macro I": "#2a78d6", "Macro II": "#eb6834", "Macro III": "#1baf7a", "Macro IV": "#eda100"}
HEAT = [[0, "#fdf1ea"], [1, "#c2410c"]]

COORD = {
    "Augusto Corrêa": (-1.0219, -46.6347), "Capanema": (-1.1969, -47.1806), "Bragança": (-1.0536, -46.7656),
    "Tracuateua": (-1.0767, -46.9031), "Cachoeira do Piriá": (-1.7597, -46.5458), "Castanhal": (-1.2939, -47.9264),
    "Belém": (-1.4558, -48.4902), "Ananindeua": (-1.3656, -48.3722), "Marituba": (-1.3550, -48.3419),
    "Oeiras do Pará": (-2.0033, -49.8547), "Altamira": (-3.2033, -52.2064), "Vitória do Xingu": (-2.8792, -52.0089),
    "Juruti": (-2.1522, -56.0922), "Óbidos": (-1.9178, -55.5181), "Oriximiná": (-1.7656, -55.8661),
    "Santarém": (-2.4431, -54.7083), "Itaituba": (-4.2761, -55.9836), "Itupiranga": (-5.1347, -49.3269),
    "Tucuruí": (-3.7661, -49.6725), "Nova Ipixuna": (-4.9172, -49.0822), "Marabá": (-5.3686, -49.1178),
    "Parauapebas": (-6.0675, -49.9022),
}

BINLBL = {
    "queda": "Queda (12 meses)", "lesao": "Queda com lesão", "recorrente": "Quedas recorrentes (≥2)",
    "dequil": "Dificuldade de equilíbrio", "vertigem": "Vertigem recente", "auxilio": "Usa auxílio p/ caminhar",
    "medequil": "Medicação que altera equilíbrio", "obst": "Obstáculos em casa", "orient": "Recebeu orientação s/ quedas",
    "dvis": "Doença visual", "dsis": "Doença sistêmica crônica", "dneu": "Doença neurológica", "hipert": "Hipertensão",
    "diabetes": "Diabetes", "osteoart": "Doença osteoarticular", "fibro": "Fibromialgia", "cardio": "Doença cardíaca",
    "respir": "Doença respiratória/alérgica", "tireoide": "Tireoide", "depans": "Depressão/ansiedade",
    "colest": "Colesterol alto", "catarata": "Catarata", "glaucoma": "Glaucoma", "refrativo": "Erro refrativo",
    "avc": "AVC/AIT", "cores": "Dificuldade p/ cores", "famcego": "Familiar cego/perda visual",
    "malaria": "Malária (histórico)", "dengue": "Dengue/arbovirose", "med": "Usa medicação", "fumo": "Fuma",
    "alcool": "Consome álcool", "agrotox": "Exposição a agrotóxico", "quimicos": "Outros agentes químicos",
    "ativ": "Atividade física", "suplem": "Suplementação",
}


# ---------------------------------------------------------------- dados
@st.cache_data
def carregar():
    d = pd.read_csv("dados_quedas.csv")
    d["macro_lbl"] = "Macro " + d["macro"].astype(str)
    d["faixa"] = pd.cut(d.idade, [0, 29.99, 39.99, 49.99, 59.99, 69.99, 200],
                        labels=["18–29", "30–39", "40–49", "50–59", "60–69", "70+"])
    d["idoso"] = np.where(d.idade.isna(), np.nan, (d.idade >= 60).astype(float))
    d["esc_g"] = pd.cut(d.esc, [-1, 1, 3, 4, 6, 7], labels=["Até fund. incompleto", "Fund. completo / médio inc.",
                                                        "Médio completo", "Superior", "Pós-graduação"])
    d["renda_g"] = pd.cut(d.renda, [-1, 0.999, 1.999, 2.999, 1e9], labels=["< 1 SM", "1–2 SM", "2–3 SM", "≥ 3 SM"])
    d["imc_g"] = pd.cut(d.imc, [0, 24.99, 29.99, 34.99, 100], labels=["< 25", "25–29,9", "30–34,9", "≥ 35"])
    d["cint_g"] = np.where(d.cintura.isna(), None, np.where(d.cintura > 88, "> 88 cm", "≤ 88 cm"))
    d["pant_g"] = np.where(d.panturrilha.isna(), None, np.where(d.panturrilha < 31, "< 31 cm", "≥ 31 cm"))
    pa = np.where(d.pas.isna(), None, np.where((d.pas >= 140) | (d.pad >= 90), "Hipertensiva (≥140/90)",
                  np.where((d.pas >= 130) | (d.pad >= 80), "Elevada (≥130/80)", "Normal")))
    d["pa_g"] = pa
    d["pahas"] = np.where(d.pas.isna(), np.nan, ((d.pas >= 140) | (d.pad >= 90)).astype(float))
    d["sono_g"] = pd.cut(d.sono, [0, 5.99, 8, 24], labels=["< 6 h", "6–8 h", "> 8 h"])
    d["sonoc"] = np.where(d.sono.isna(), np.nan, (d.sono < 6).astype(float))
    d["multimed"] = np.where(d.nmed.isna(), np.nan, (d.nmed >= 2).astype(float))
    d["nmed_g"] = d.nmed.map(lambda v: None if pd.isna(v) else ("3+" if v >= 3 else str(int(v))))
    com = ["hipert", "diabetes", "osteoart", "cardio", "respir", "tireoide", "depans", "dneu", "dvis"]
    d["ncom"] = d[com].fillna(0).sum(axis=1)
    d["ncom_g"] = d.ncom.map(lambda v: "3+" if v >= 3 else str(int(v)))
    d["obes"] = np.where(d.imc.isna(), np.nan, (d.imc >= 30).astype(float))
    d["cint88"] = np.where(d.cintura.isna(), np.nan, (d.cintura > 88).astype(float))
    d["pant31"] = np.where(d.panturrilha.isna(), np.nan, (d.panturrilha < 31).astype(float))
    d["r1"] = np.where(d.renda.isna(), np.nan, (d.renda < 1).astype(float))
    d["baixaesc"] = np.where(d.esc.isna(), np.nan, (d.esc <= 1).astype(float))
    p75 = d.area_oa.quantile(0.75)
    p25 = d.area_oa.quantile(0.25)
    d["hiosc"] = np.where(d.area_oa.isna(), np.nan, (d.area_oa >= p75).astype(float))
    d["oscil_g"] = np.where(d.area_oa.isna(), None, np.where(d.area_oa >= p75, "Quartil superior",
                            np.where(d.area_oa <= p25, "Quartil inferior", "Intermediária")))
    f = lambda s: s.fillna(0)
    d["score"] = (f(d.idoso) + f(d.queda) + f(d.dequil) + f(d.vertigem) + f(d.auxilio) + f(d.medequil) + f(d.obst)
                  + f(d.diabetes) + f(d.dvis) + f(d.dneu) + f(d.multimed) + f(d.hiosc))
    d["alto"] = (d.score >= 4).astype(float)
    d["risco_g"] = pd.cut(d.score, [-1, 1, 3, 99], labels=["Baixa", "Moderada", "Alta"])
    for k in BINLBL:
        d["b_" + k] = d[k].map({1: "Sim", 0: "Não"})
    return d


df_all = carregar()

# métricas: chave -> (rótulo, tipo, grupo)  tipo: p=proporção, m=média, med=mediana
M = {}
for k in ["queda", "lesao", "recorrente", "dequil", "vertigem", "auxilio", "medequil", "obst", "orient"]:
    M[k] = (BINLBL[k] + " (%)", "p", "Quedas e equilíbrio")
M["alto"] = ("Vulnerabilidade alta (%)", "p", "Quedas e equilíbrio")
M["score"] = ("Índice de vulnerabilidade (0–12)", "m", "Quedas e equilíbrio")
M["nquedas"] = ("Nº médio de quedas", "m", "Quedas e equilíbrio")
for k in ["dsis", "hipert", "diabetes", "osteoart", "cardio", "respir", "depans", "dneu", "avc", "dvis",
          "catarata", "glaucoma", "refrativo", "malaria"]:
    M[k] = (BINLBL[k] + " (%)", "p", "Saúde")
M["pahas"] = ("PA aferida ≥140/90 (%)", "p", "Saúde")
M["multimed"] = ("Usa ≥2 medicamentos (%)", "p", "Saúde")
M["ncom"] = ("Nº médio de condições crônicas", "m", "Saúde")
M["idoso"] = ("Idosas ≥60 (%)", "p", "Perfil")
M["idade"] = ("Idade média", "m", "Perfil")
M["renda"] = ("Renda média (SM)", "m", "Perfil")
M["r1"] = ("Renda < 1 SM (%)", "p", "Perfil")
M["baixaesc"] = ("Até fund. incompleto (%)", "p", "Perfil")
M["imc"] = ("IMC médio", "m", "Perfil")
M["obes"] = ("Obesidade IMC≥30 (%)", "p", "Perfil")
M["cint88"] = ("Cintura >88 cm (%)", "p", "Perfil")
M["pant31"] = ("Panturrilha <31 cm (%)", "p", "Perfil")
M["sonoc"] = ("Sono < 6 h (%)", "p", "Perfil")
for k in ["ativ", "alcool", "fumo", "agrotox", "quimicos"]:
    M[k] = (BINLBL[k] + " (%)", "p", "Hábitos e exposições")
for k, l in [("area_oa", "Área elipse OA"), ("area_of", "Área elipse OF"), ("dt_oa", "Desvio total OA"),
             ("dt_of", "Desvio total OF"), ("romberg", "Quociente de Romberg")]:
    M[k] = (l + " (mediana)", "med", "Estabilometria")
M["hiosc"] = ("Oscilação OA no quartil superior (%)", "p", "Estabilometria")

DIMS = {
    "macro_lbl": "Macrorregião", "reg": "Região de saúde", "mun": "Município", "faixa": "Faixa etária",
    "cor": "Cor/raça", "civil": "Estado civil", "esc_g": "Escolaridade", "renda_g": "Renda familiar",
    "imc_g": "IMC", "cint_g": "Cintura", "pant_g": "Panturrilha", "pa_g": "Pressão aferida", "sono_g": "Sono",
    "nmed_g": "Nº de medicamentos", "ncom_g": "Nº de condições crônicas", "risco_g": "Vulnerabilidade a quedas",
    "oscil_g": "Oscilação OA",
}
DIMS.update({"b_" + k: v + " (Sim/Não)" for k, v in BINLBL.items()})


def agg(s: pd.Series, k: str):
    t = M[k][1]
    s = s.dropna()
    if s.empty:
        return np.nan
    return s.median() if t == "med" else s.mean()


def fmt(k, v):
    if v is None or pd.isna(v):
        return "–"
    t = M[k][1]
    if t == "p":
        return f"{v*100:.1f}%".replace(".", ",")
    if k in ("area_oa", "area_of"):
        return f"{v:.4f}".replace(".", ",")
    if k in ("dt_oa", "dt_of"):
        return f"{v:.0f}"
    return f"{v:.2f}".replace(".", ",")


def tabela(d, by, ks):
    g = d.groupby(by, observed=True)
    out = pd.DataFrame({"n": g.size()})
    for k in ks:
        out[k] = g[k].median() if M[k][1] == "med" else g[k].mean()
    return out


def escala(k):
    return 100 if M[k][1] == "p" else 1


def sel_metrica(label, default="queda", key=None):
    ks = list(M)
    return st.selectbox(label, ks, index=ks.index(default), format_func=lambda k: f"{M[k][2]} · {M[k][0]}", key=key)


def rp(d, exp, out="queda"):
    x = d[[exp, out]].dropna() if isinstance(exp, str) else None
    a = ((x[exp] == 1) & (x[out] == 1)).sum(); n1 = (x[exp] == 1).sum()
    c = ((x[exp] == 0) & (x[out] == 1)).sum(); n0 = (x[exp] == 0).sum()
    if n1 < 1 or n0 < 1 or a < 1 or c < 1:
        return None
    p1, p0 = a / n1, c / n0
    r = p1 / p0
    se = np.sqrt((1 - p1) / a + (1 - p0) / c)
    return dict(a=a, n1=n1, c=c, n0=n0, p1=p1, p0=p0, rp=r, lo=r * np.exp(-1.96 * se), hi=r * np.exp(1.96 * se))


def forest(rows, xlabel):
    rows = sorted(rows, key=lambda r: r["v"])
    cor = ["#c0352b" if r["lo"] > 1 else "#1f7a3a" if r["hi"] < 1 else "#8a9693" for r in rows]
    fig = go.Figure(go.Scatter(
        x=[r["v"] for r in rows], y=[r["nome"] for r in rows], mode="markers",
        marker=dict(color=cor, size=9, symbol="square"),
        error_x=dict(type="data", symmetric=False, array=[r["hi"] - r["v"] for r in rows],
                     arrayminus=[r["v"] - r["lo"] for r in rows], color="#8a9693", thickness=1.5),
        hovertext=[r.get("hover", "") for r in rows], hoverinfo="text"))
    fig.add_vline(x=1, line_dash="dash", line_color="#7a8583")
    fig.update_layout(height=max(320, 24 * len(rows) + 80), xaxis_type="log", xaxis_title=xlabel,
                      margin=dict(l=10, r=10, t=10, b=40), showlegend=False)
    return fig


# ---------------------------------------------------------------- filtros
st.sidebar.title("Filtros")
f_macro = st.sidebar.multiselect("Macrorregião", [f"Macro {m}" for m in MACROS], placeholder="Todos")
muns_disp = sorted(df_all[df_all.macro_lbl.isin(f_macro)].mun.unique() if f_macro else df_all.mun.unique())
f_mun = st.sidebar.multiselect("Município", muns_disp, placeholder="Todos")
f_faixa = st.sidebar.multiselect("Faixa etária", list(df_all.faixa.cat.categories), placeholder="Todos")
f_cor = st.sidebar.multiselect("Cor/raça", sorted(df_all.cor.dropna().unique()), placeholder="Todos")
f_esc = st.sidebar.multiselect("Escolaridade", list(df_all.esc_g.cat.categories), placeholder="Todos")
f_renda = st.sidebar.multiselect("Renda familiar", list(df_all.renda_g.cat.categories), placeholder="Todos")
cond_opts = [k for k in BINLBL if k not in ("orient", "ativ", "suplem", "med")]
f_cond = st.sidebar.selectbox("Só quem tem", [""] + cond_opts, format_func=lambda k: "Todas" if not k else BINLBL[k])


def filtrar(d, ignorar=()):
    if f_macro and "macro" not in ignorar: d = d[d.macro_lbl.isin(f_macro)]
    if f_mun and "mun" not in ignorar: d = d[d.mun.isin(f_mun)]
    if f_faixa and "faixa" not in ignorar: d = d[d.faixa.isin(f_faixa)]
    if f_cor: d = d[d.cor.isin(f_cor)]
    if f_esc: d = d[d.esc_g.isin(f_esc)]
    if f_renda: d = d[d.renda_g.isin(f_renda)]
    if f_cond: d = d[d[f_cond] == 1]
    return d


df = filtrar(df_all)
st.sidebar.metric("Participantes no recorte", f"{len(df):,}".replace(",", "."))
st.sidebar.caption(f"de {len(df_all):,} no total".replace(",", "."))

# ---------------------------------------------------------------- cabeçalho
st.title("Observatório de Quedas · Pará")
st.caption("Projeto “De mãos dadas para não deixar cair” · LEMOH/UFPA · FAPESPA 105/2024 · "
           f"{df_all.mun.nunique()} municípios · {len(df_all)} participantes")
if df.empty:
    st.warning("Nenhuma participante no recorte. Ajuste os filtros na barra lateral.")
    st.stop()

abas = st.tabs(["Panorama", "Municípios", "Ciclo de vida", "Fatores de risco", "Equilíbrio postural",
                "Cruzamentos", "Método"])

# ================================================================ PANORAMA
with abas[0]:
    kp = [("queda", "Caíram nos últimos 12 meses"), ("lesao", "A queda causou lesão"),
          ("recorrente", "Quedas recorrentes (≥2)"), ("dequil", "Dificuldade de equilíbrio"),
          ("vertigem", "Vertigem recente"), ("obst", "Obstáculos de risco em casa"),
          ("alto", "Vulnerabilidade alta"), ("orient", "Já recebeu orientação")]
    cols = st.columns(4)
    for i, (k, l) in enumerate(kp):
        v, vt = df[k].mean(), df_all[k].mean()
        delta = None if len(df) == len(df_all) else f"{(v - vt)*100:+.1f} p.p. vs total".replace(".", ",")
        cols[i % 4].metric(l, fmt(k, v), delta, delta_color="inverse" if k != "orient" else "normal")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Mapa do risco")
        mk = sel_metrica("Indicador do mapa", key="mapa")
        t = tabela(df, "mun", [mk]).reset_index()
        t["lat"] = t.mun.map(lambda m: COORD[m][0]); t["lon"] = t.mun.map(lambda m: COORD[m][1])
        t["valor"] = t[mk] * escala(mk)
        t["macro"] = t.mun.map(df_all.drop_duplicates("mun").set_index("mun").macro_lbl)
        fig = px.scatter_geo(t, lat="lat", lon="lon", size="n", color="valor", hover_name="mun",
                             hover_data={"macro": True, "n": True, "valor": ":.2f", "lat": False, "lon": False},
                             text="mun", color_continuous_scale=HEAT, size_max=28, labels={"valor": M[mk][0]})
        fig.update_traces(textposition="top center", textfont_size=10, marker_line_color="#4c5856")
        fig.update_geos(fitbounds="locations", showcountries=True, showsubunits=True, subunitcolor="#bbb",
                        showrivers=True, rivercolor="#9ec5f4", showland=True, landcolor="#f6f7f5", resolution=50)
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Principais achados")
        st.caption("Amostra completa, sem filtros.")
        a = df_all
        mq = a.groupby("mun").queda.mean().sort_values(ascending=False)
        fq = a.groupby("faixa", observed=True).queda.mean()
        dia, obs = rp(a, "diabetes"), rp(a, "obst")
        orf = a[a.queda == 1].orient.mean()
        st.markdown(f"""
- **Território:** maiores taxas de queda em {', '.join(f'{m} ({v*100:.0f}%)' for m, v in mq.head(3).items())}; menores em {', '.join(f'{m} ({v*100:.0f}%)' for m, v in mq.tail(2).items())}.
- **Idade:** a queda vai de {fq.iloc[0]*100:.0f}% (18–29) a {fq.iloc[-1]*100:.0f}% (70+).
- **Diabetes:** quedas {dia['rp']:.1f}× mais frequentes ({dia['p1']*100:.0f}% vs {dia['p0']*100:.0f}%); efeito mantido no modelo ajustado.
- **Casa:** {a.obst.mean()*100:.0f}% relatam obstáculos; quedas {obs['rp']:.1f}× mais frequentes. É o fator modificável mais forte.
- **Cuidado:** só {orf*100:.0f}% das mulheres que caíram receberam orientação profissional sobre quedas.
- **Macrorregião:** a diferença de queda entre macros não se sustenta após ajuste por idade e clínica (aba Fatores de risco).
""")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Indicadores de queda por macrorregião")
        ks = ["queda", "lesao", "dequil", "vertigem", "obst", "alto"]
        t = tabela(df, "macro_lbl", ks)[ks].mul(100).reset_index().melt("macro_lbl", var_name="ind", value_name="pct")
        t["ind"] = t.ind.map(lambda k: M[k][0].replace(" (%)", ""))
        fig = px.bar(t, x="ind", y="pct", color="macro_lbl", barmode="group", color_discrete_map=MACRO_COR,
                     labels={"pct": "%", "ind": "", "macro_lbl": ""})
        fig.update_layout(height=380, margin=dict(t=10, b=10), legend_orientation="h")
        st.plotly_chart(fig, width="stretch")
    with c4:
        st.subheader("Lacuna de cuidado")
        st.caption("Entre as que caíram: receberam orientação profissional sobre quedas?")
        t = df[df.queda == 1].groupby("macro_lbl").orient.agg(["mean", "size"]).reset_index()
        t["Receberam"] = t["mean"] * 100; t["Não receberam"] = 100 - t.Receberam
        fig = px.bar(t, y="macro_lbl", x=["Receberam", "Não receberam"], orientation="h",
                     color_discrete_sequence=["#0d6b6e", "#c2410c"], labels={"value": "%", "macro_lbl": "", "variable": ""},
                     hover_data={"size": True})
        fig.update_layout(height=380, margin=dict(t=10, b=10), legend_orientation="h", xaxis_range=[0, 100])
        st.plotly_chart(fig, width="stretch")

# ================================================================ MUNICÍPIOS
with abas[1]:
    st.subheader("Ranking de municípios")
    mk = sel_metrica("Indicador", key="rank")
    t = tabela(df, "mun", [mk]).reset_index().dropna()
    t["valor"] = t[mk] * escala(mk)
    t["macro"] = t.mun.map(df_all.drop_duplicates("mun").set_index("mun").macro_lbl)
    t = t.sort_values("valor", ascending=False)
    fig = px.bar(t, x="mun", y="valor", color="macro", color_discrete_map=MACRO_COR, hover_data=["n"],
                 labels={"valor": M[mk][0], "mun": "", "macro": ""})
    media = agg(df[mk], mk) * escala(mk)
    fig.add_hline(y=media, line_dash="dash", line_color="#4c5856", annotation_text=f"Média {fmt(mk, agg(df[mk], mk))}")
    fig.update_layout(height=440, margin=dict(t=10), xaxis_categoryorder="total descending")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Painel comparativo")
    st.caption("Índice de prioridade = média dos escores-z de queda, dificuldade de equilíbrio, vulnerabilidade alta "
               "e falta de orientação (maior = mais prioritário).")
    grp = st.radio("Agrupar por", ["mun", "reg", "macro_lbl"], horizontal=True,
                   format_func={"mun": "Município", "reg": "Região de saúde", "macro_lbl": "Macrorregião"}.get)
    ks = ["idade", "idoso", "renda", "queda", "lesao", "recorrente", "dequil", "vertigem", "obst", "orient",
          "diabetes", "hipert", "dvis", "multimed", "pant31", "alto", "area_oa", "romberg"]
    t = tabela(df, grp, ks)
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)
    t["prioridade"] = (z(t.queda) + z(t.dequil) + z(t.alto) - z(t.orient)) / 4
    t = t.sort_values("prioridade", ascending=False)
    show = t.copy()
    for k in ks:
        if M[k][1] == "p":
            show[k] = show[k] * 100
    show = show.rename(columns={k: M[k][0] for k in ks})
    pcols = [M[k][0] for k in ks if M[k][1] == "p" and k != "orient"] + ["prioridade"]
    st.dataframe(show.style.format(precision=1).format({M["area_oa"][0]: "{:.4f}", M["romberg"][0]: "{:.2f}",
                                                         "prioridade": "{:.2f}"})
                 .background_gradient(cmap="Oranges", subset=pcols),
                 width="stretch", height=620)

# ================================================================ CICLO DE VIDA
with abas[2]:
    dv = filtrar(df_all, ignorar=("faixa",))
    st.subheader("O risco ao longo da vida")
    modo = st.radio("Modo", ["Vários indicadores", "Um indicador × macrorregião"], horizontal=True)
    if modo == "Vários indicadores":
        opts = ["queda", "lesao", "recorrente", "dequil", "vertigem", "obst", "orient", "alto", "hiosc", "dsis",
                "hipert", "diabetes", "dvis", "multimed", "pant31", "ativ"]
        sel = st.multiselect("Indicadores", opts, default=["queda", "lesao", "dequil", "vertigem", "alto"],
                             format_func=lambda k: M[k][0])
        if sel:
            t = tabela(dv, "faixa", sel)[sel].mul(100).reset_index().melt("faixa", var_name="ind", value_name="pct")
            t["ind"] = t.ind.map(lambda k: M[k][0].replace(" (%)", ""))
            fig = px.line(t, x="faixa", y="pct", color="ind", markers=True, labels={"pct": "%", "faixa": "", "ind": ""})
            fig.update_layout(height=440, legend_orientation="h")
            st.plotly_chart(fig, width="stretch")
    else:
        mk = sel_metrica("Indicador", key="ciclo")
        t = dv.groupby(["macro_lbl", "faixa"], observed=True)[mk].agg(
            ["median" if M[mk][1] == "med" else "mean", "size"]).reset_index()
        t.columns = ["macro", "faixa", "v", "n"]
        t = t[t.n >= 5]; t["v"] = t.v * escala(mk)
        fig = px.line(t, x="faixa", y="v", color="macro", markers=True, color_discrete_map=MACRO_COR,
                      hover_data=["n"], labels={"v": M[mk][0], "faixa": "", "macro": ""})
        fig.update_layout(height=440, legend_orientation="h")
        st.plotly_chart(fig, width="stretch")
        st.caption("Pontos com menos de 5 participantes são omitidos.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Oscilação postural por idade")
        t = dv.groupby("faixa", observed=True)[["area_oa", "area_of"]].median().reset_index()
        t = t.melt("faixa", var_name="cond", value_name="area")
        t["cond"] = t.cond.map({"area_oa": "Olhos abertos", "area_of": "Olhos fechados"})
        fig = px.bar(t, x="faixa", y="area", color="cond", barmode="group",
                     color_discrete_sequence=["#2a78d6", "#eb6834"],
                     labels={"area": "Área elipse 95% (mediana, u.a.)", "faixa": "", "cond": ""})
        fig.update_layout(height=360, legend_orientation="h")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Comorbidades por idade")
        ks = ["hipert", "diabetes", "dvis", "osteoart"]
        t = dv.groupby("faixa", observed=True)[ks].mean().mul(100).reset_index().melt("faixa", var_name="c", value_name="pct")
        t["c"] = t.c.map(BINLBL)
        fig = px.line(t, x="faixa", y="pct", color="c", markers=True, labels={"pct": "%", "faixa": "", "c": ""})
        fig.update_layout(height=360, legend_orientation="h")
        st.plotly_chart(fig, width="stretch")

    ks = ["queda", "lesao", "recorrente", "dequil", "vertigem", "obst", "orient", "alto", "hipert", "diabetes",
          "dvis", "multimed", "pant31", "ativ", "area_oa", "area_of", "romberg"]
    t = tabela(dv, "faixa", ks)
    for k in ks:
        if M[k][1] == "p":
            t[k] = t[k] * 100
    st.dataframe(t.rename(columns={k: M[k][0] for k in ks}).style.format(precision=1)
                 .format({M["area_oa"][0]: "{:.4f}", M["area_of"][0]: "{:.4f}", M["romberg"][0]: "{:.2f}"}),
                 width="stretch")

# ================================================================ FATORES
EXPOS = {"idoso": "Idade ≥60", "r1": "Renda < 1 SM", "baixaesc": "Até fund. incompleto", "obes": "Obesidade (IMC≥30)",
         "cint88": "Cintura >88 cm", "pant31": "Panturrilha <31 cm", "pahas": "PA aferida ≥140/90",
         "sonoc": "Sono < 6 h", "multimed": "≥2 medicamentos", "hiosc": "Oscilação OA alta (Q4)"}
EXPOS.update({k: BINLBL[k] for k in ["hipert", "diabetes", "osteoart", "cardio", "respir", "depans", "dneu", "avc",
                                     "dvis", "catarata", "glaucoma", "refrativo", "cores", "fibro", "tireoide",
                                     "malaria", "dequil", "vertigem", "medequil", "auxilio", "obst", "alcool", "fumo",
                                     "agrotox", "quimicos", "ativ", "orient"]})


@st.cache_data
def modelos():
    d = df_all.copy()
    d["idade10"] = d.idade / 10
    for m in ["II", "III", "IV"]:
        d["macro_" + m] = (d.macro == m).astype(int)
    d["log_area_oa"] = np.log(d.area_oa)
    lab = {"idade10": "Idade (+10 anos)", "macro_II": "Macrorregião II (vs I)", "macro_III": "Macrorregião III (vs I)",
           "macro_IV": "Macrorregião IV (vs I)", "esc": "Escolaridade (+1 nível)", "renda": "Renda (+1 SM)",
           "obes": "Obesidade (IMC ≥30)", "pant31": "Panturrilha <31 cm", "dvis": "Doença visual",
           "hipert": "Hipertensão", "diabetes": "Diabetes", "osteoart": "Doença osteoarticular",
           "depans": "Depressão/ansiedade", "dneu": "Doença neurológica", "multimed": "Uso de ≥2 medicamentos",
           "obst": "Obstáculos em casa", "ativ": "Atividade física", "alcool": "Consumo de álcool",
           "vertigem": "Vertigem recente", "dequil": "Dificuldade de equilíbrio",
           "log_area_oa": "Oscilação OA (log área elipse)"}
    base = ["idade10", "macro_II", "macro_III", "macro_IV", "esc", "renda", "obes", "pant31", "dvis", "hipert",
            "diabetes", "osteoart", "depans", "dneu", "multimed", "obst", "ativ", "alcool"]
    sets = {"A · Clínico": base, "B · + sintomas": base + ["vertigem", "dequil"],
            "C · + estabilometria": base + ["vertigem", "dequil", "log_area_oa"]}
    res = {}
    for nome, X in sets.items():
        dd = d[["queda"] + X].dropna()
        m = sm.Logit(dd.queda, sm.add_constant(dd[X])).fit(disp=0)
        ci = m.conf_int()
        res[nome] = dict(n=len(dd), ev=int(dd.queda.sum()), r2=m.prsquared,
                         rows=[dict(nome=lab[v], v=np.exp(m.params[v]), lo=np.exp(ci.loc[v, 0]),
                                    hi=np.exp(ci.loc[v, 1]), p=m.pvalues[v]) for v in X])
    ols = {}
    Xo = ["idade10", "macro_II", "macro_III", "macro_IV", "esc", "renda", "obes", "hipert", "diabetes", "dvis",
          "queda", "dequil"]
    lab["queda"] = "Queda nos últimos 12 meses"
    for y in ["area_oa", "area_of"]:
        dd = d[[y] + Xo].dropna()
        m = sm.OLS(np.log(dd[y]), sm.add_constant(dd[Xo])).fit()
        ols[y] = pd.DataFrame({"Fator": [lab[v] for v in Xo], "coef": m.params[Xo].values, "p": m.pvalues[Xo].values})
        ols[y + "_meta"] = (len(dd), m.rsquared)
    return res, ols


with abas[3]:
    c1, c2 = st.columns([7, 5])
    with c1:
        st.subheader("O que acompanha o desfecho")
        st.caption("Razão de prevalência (RP) bruta com IC 95%, recalculada com os filtros. Vermelho = aumenta; "
                   "verde = reduz; cinza = sem significância.")
        outc = {"queda": "Queda (12 meses)", "lesao": "Queda com lesão", "recorrente": "Quedas recorrentes",
                "dequil": "Dificuldade de equilíbrio", "hiosc": "Oscilação OA alta (Q4)"}
        ok = st.selectbox("Desfecho", list(outc), format_func=outc.get)
        rows = []
        for k, nome in EXPOS.items():
            if k == ok:
                continue
            r = rp(df, k, ok)
            if r and r["a"] >= 5:
                rows.append(dict(nome=nome, v=r["rp"], lo=r["lo"], hi=r["hi"],
                                 hover=f"{nome}<br>{r['p1']*100:.1f}% com ({r['a']}/{r['n1']}) vs "
                                       f"{r['p0']*100:.1f}% sem ({r['c']}/{r['n0']})<br>RP {r['rp']:.2f} "
                                       f"(IC {r['lo']:.2f}–{r['hi']:.2f})"))
        if rows:
            st.plotly_chart(forest(rows, "Razão de prevalência (escala log)"), width="stretch")
    with c2:
        st.subheader("Efeito independente sobre quedas")
        st.caption("Regressão logística multivariada na amostra completa. OR = razão de chances ajustada.")
        res, ols = modelos()
        mn = st.radio("Modelo", list(res), index=1, horizontal=True)
        m = res[mn]
        for r in m["rows"]:
            r["hover"] = f"{r['nome']}<br>OR {r['v']:.2f} (IC {r['lo']:.2f}–{r['hi']:.2f})<br>p = {r['p']:.3f}"
        st.plotly_chart(forest(m["rows"], "OR ajustada (escala log)"), width="stretch")
        st.caption(f"{m['n']} participantes com dados completos, {m['ev']} com queda. Pseudo-R² {m['r2']*100:.1f}%.")

    st.subheader("Carga de comorbidades")
    ks = ["hipert", "diabetes", "dvis", "refrativo", "catarata", "glaucoma", "osteoart", "respir", "depans", "dneu",
          "cardio", "tireoide", "fibro", "avc", "malaria"]
    lin = []
    for k in ks:
        r = rp(df, k)
        if r and r["n1"] >= 5:
            lin += [(BINLBL[k], "Prevalência da condição", r["n1"] / (r["n1"] + r["n0"]) * 100),
                    (BINLBL[k], "Queda entre quem TEM", r["p1"] * 100),
                    (BINLBL[k], "Queda entre quem NÃO tem", r["p0"] * 100)]
    t = pd.DataFrame(lin, columns=["cond", "serie", "pct"])
    ordem = t[t.serie == "Prevalência da condição"].sort_values("pct", ascending=False).cond.tolist()
    fig = px.bar(t, x="cond", y="pct", color="serie", barmode="group",
                 color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a"], category_orders={"cond": ordem},
                 labels={"pct": "%", "cond": "", "serie": ""})
    fig.update_layout(height=440, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

# ================================================================ EQUILÍBRIO
with abas[4]:
    st.subheader("Estabilometria por grupo")
    st.caption("Medianas. Valores maiores = mais oscilação = pior controle postural.")
    c1, c2 = st.columns(2)
    dim = c1.selectbox("Comparar por", list(DIMS), format_func=DIMS.get, key="eqdim")
    var = c2.selectbox("Medida", ["area", "dt", "rmsap", "rmsml"], format_func={
        "area": "Área da elipse 95%", "dt": "Desvio total", "rmsap": "RMS antero-posterior",
        "rmsml": "RMS médio-lateral"}.get)
    t = df.groupby(dim, observed=True)[[var + "_oa", var + "_of"]].median().reset_index()
    n = df.groupby(dim, observed=True)[var + "_oa"].count().rename("n").reset_index()
    t = t.merge(n, on=dim).melt([dim, "n"], var_name="cond", value_name="v")
    t["cond"] = t.cond.map(lambda c: "Olhos abertos" if c.endswith("_oa") else "Olhos fechados")
    fig = px.bar(t, x=dim, y="v", color="cond", barmode="group", hover_data=["n"],
                 color_discrete_sequence=["#2a78d6", "#eb6834"], labels={"v": "Mediana", dim: "", "cond": ""})
    fig.update_layout(height=420, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Quociente de Romberg")
        st.caption("Área OF ÷ área OA (mediana). Abaixo de 1: oscilação já alta com olhos abertos.")
        t = df.groupby(dim, observed=True).romberg.median().reset_index()
        t["cor"] = np.where(t.romberg < 1, "< 1", "≥ 1")
        fig = px.bar(t, x=dim, y="romberg", color="cor", color_discrete_map={"< 1": "#c2410c", "≥ 1": "#0d6b6e"},
                     labels={dim: "", "romberg": "Romberg", "cor": ""})
        fig.add_hline(y=1, line_dash="dash")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Idade × oscilação (olhos abertos)")
        t = df.dropna(subset=["idade", "area_oa", "queda"]).copy()
        t["Queda"] = t.queda.map({1: "Com queda", 0: "Sem queda"})
        fig = px.scatter(t, x="idade", y="area_oa", color="Queda", log_y=True, opacity=.65,
                         color_discrete_map={"Sem queda": "#2a78d6", "Com queda": "#eb6834"},
                         hover_data=["mun"], labels={"idade": "Idade (anos)", "area_oa": "Área elipse OA (log)"})
        fig.update_layout(height=380, legend_orientation="h")
        st.plotly_chart(fig, width="stretch")

# ================================================================ CRUZAMENTOS
with abas[5]:
    st.subheader("Tabela de cruzamentos")
    st.caption("Combine quaisquer duas dimensões e um indicador. Células com menos de 10 participantes ficam em branco.")
    c1, c2, c3 = st.columns(3)
    lk = list(DIMS)
    rdim = c1.selectbox("Linhas", lk, index=lk.index("faixa"), format_func=DIMS.get)
    cdim = c2.selectbox("Colunas", ["(nenhuma)"] + lk, index=1 + lk.index("macro_lbl"),
                        format_func=lambda k: DIMS.get(k, k))
    mk = c3.selectbox("Indicador", list(M), format_func=lambda k: f"{M[k][2]} · {M[k][0]}", key="pv")
    fn = "median" if M[mk][1] == "med" else "mean"
    if cdim == "(nenhuma)":
        pv = df.groupby(rdim, observed=True)[mk].agg([fn, "count"])
        val = pv[fn].where(pv["count"] >= 10).to_frame(M[mk][0]) * escala(mk)
        cnt = pv[["count"]]
    else:
        val = df.pivot_table(index=rdim, columns=cdim, values=mk, aggfunc=fn, observed=True)
        cnt = df.pivot_table(index=rdim, columns=cdim, values=mk, aggfunc="count", observed=True)
        val = val.where(cnt >= 10) * escala(mk)
        val["Total"] = df.groupby(rdim, observed=True)[mk].agg(fn) * escala(mk)
    prec = 4 if mk in ("area_oa", "area_of") else 1
    st.dataframe(val.style.format(precision=prec, na_rep="–").background_gradient(cmap="Oranges", axis=None),
                 width="stretch")
    with st.expander("Ver número de participantes por célula"):
        st.dataframe(cnt, width="stretch")
    plot = val.drop(columns="Total", errors="ignore").reset_index().melt(rdim, var_name="col", value_name="v")
    fig = px.bar(plot, x=rdim, y="v", color="col" if cdim != "(nenhuma)" else None, barmode="group",
                 color_discrete_map=MACRO_COR if cdim == "macro_lbl" else None,
                 labels={"v": M[mk][0], rdim: "", "col": DIMS.get(cdim, "")})
    fig.update_layout(height=440, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    st.download_button("Baixar tabela (CSV)", val.to_csv().encode("utf-8-sig"), "cruzamento.csv", "text/csv")

# ================================================================ MÉTODO
with abas[6]:
    st.subheader("Como os dados foram tratados")
    mac = df_all.drop_duplicates("mun").groupby("macro_lbl").apply(
        lambda g: f"**{g.name}** ({', '.join(sorted(g.reg.unique()))}): {', '.join(sorted(g.mun))}")
    st.markdown(f"""
**Fonte:** {len(df_all)} formulários de {df_all.mun.nunique()} municípios. Amplia o relatório técnico parcial de agosto/2026
(11 municípios, 626 participantes). Nomes, telefones e endereços foram removidos.

**Macrorregiões de Saúde** (SESPA/CIB-PA, 13 regiões em 4 macros):
""" + "\n".join(f"- {x}" for x in mac) + """

**Variáveis derivadas**
- Renda em salários mínimos (R$ 1.518). Bolsa Família isolado = 0,4 SM; “menos de 1 SM” = 0,6.
- Comorbidades específicas por palavras-chave no campo “qual” e nos nomes de medicamentos (ex.: losartana → hipertensão).
- Pressão aferida: elevada ≥130/80; hipertensiva ≥140/90. Cintura >88 cm = obesidade abdominal; panturrilha <31 cm = baixa massa muscular.
- **Índice de vulnerabilidade a quedas** (exploratório, não validado): soma de 12 marcadores (idade ≥60, queda prévia,
  dificuldade de equilíbrio, vertigem, auxílio para caminhar, medicação que altera equilíbrio, obstáculos em casa, diabetes,
  doença visual, doença neurológica, ≥2 medicamentos, oscilação OA no quartil superior). Baixa 0–1, moderada 2–3, alta ≥4.

**Estatística**
- Estabilometria resumida por mediana (distribuições com cauda longa).
- Razão de prevalência com IC 95% pelo método do log; fatores com menos de 5 casos expostos são omitidos.
- Regressão logística (queda) e linear (log da área da elipse OA/OF) com statsmodels, casos completos.
""")
    res, ols = modelos()
    st.markdown("**Regressão linear da oscilação (log da área da elipse)**")
    t = ols["area_oa"].merge(ols["area_of"], on="Fator", suffixes=(" OA", " OF"))
    st.dataframe(t.style.format({"coef OA": "{:+.3f}", "coef OF": "{:+.3f}", "p OA": "{:.3f}", "p OF": "{:.3f}"}),
                 width="stretch", hide_index=True)
    (na, ra), (nf, rf) = ols["area_oa_meta"], ols["area_of_meta"]
    st.caption(f"OA: n={na}, R²={ra*100:.1f}%. OF: n={nf}, R²={rf*100:.1f}%. Coeficiente +0,10 ≈ 10% mais área.")
    st.download_button("Baixar base desidentificada (CSV)",
                       pd.read_csv("dados_quedas.csv").to_csv(index=False).encode("utf-8-sig"),
                       "dados_quedas.csv", "text/csv")
