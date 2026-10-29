"""
Observatório de Quedas · Pará
Projeto "De mãos dadas para não deixar cair" — LEMOH/UFPA · FAPESPA 105/2024

Painel interativo (Streamlit) sobre risco de quedas em mulheres paraenses:
território, ciclo de vida, comorbidades e equilíbrio postural (estabilometria).
Dados lidos da planilha única (versão anonimizada) por preparar_dados.py.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="Observatório de Quedas · Pará", page_icon="📊", layout="wide")
pd.set_option("styler.format.precision", 1)
pd.set_option("styler.format.decimal", ",")
pd.set_option("styler.format.thousands", ".")

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
    from preparar_dados import preparar
    d = preparar()
    # colunas do celular (versões antigas do CSV não as têm): cria vazias para o app não quebrar
    for b in ["RMS AP", "RMS ML", "Desvio total", "Área elipse", "Freq. mediana AP", "Freq. mediana ML",
              "Energia baixa AP", "Energia média AP", "Energia alta AP", "Energia baixa ML", "Energia média ML",
              "Energia alta ML"]:
        for c in ["OA", "OF"]:
            if f"s_{b} ({c})" not in d.columns:
                d[f"s_{b} ({c})"] = np.nan
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
    # Índice de Risco de Quedas (IRQ-Pará), pontos derivados da regressão logística
    d["irq"] = (np.where(d.idade >= 60, 3, np.where(d.idade >= 40, 2, 0)) + 2 * f(d.diabetes) + 2 * f(d.dequil)
                + 2 * f(d.obst) + f(d.vertigem) + 3 * f(d.auxilio) + 2 * f(d.cores))
    d["irq_g"] = pd.cut(d.irq, [-1, 2, 4, 6, 99], labels=["Baixo", "Moderado", "Alto", "Muito alto"])
    d["irq_alto"] = d.irq_g.isin(["Alto", "Muito alto"]).astype(float)
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
M["irq"] = ("IRQ-Pará médio (0–15)", "m", "Quedas e equilíbrio")
M["irq_alto"] = ("IRQ-Pará alto ou muito alto (%)", "p", "Quedas e equilíbrio")
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
    "oscil_g": "Oscilação OA", "irq_g": "IRQ-Pará (classe de risco)",
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



# ---------------------------------------------------------------- mapa
import json as _json

CLS_COR = {"hi": "#d64541", "mid": "#f2b33d", "lo": "#3a9d63", "off": "#e4e8e6"}
BONS = {"orient", "ativ", "renda"}
# deslocamento do rótulo (pixels) para evitar sobreposição: (ax, ay)
OFF_MAIN = {"Oriximiná": (0, -30), "Óbidos": (60, -20), "Juruti": (-70, 30), "Santarém": (40, -35),
            "Itaituba": (-40, 30), "Altamira": (-30, 70), "Vitória do Xingu": (70, -45), "Oeiras do Pará": (40, -30),
            "Tucuruí": (-60, -20), "Itupiranga": (-90, -25), "Nova Ipixuna": (90, -25), "Marabá": (80, 20),
            "Parauapebas": (-60, 35)}
OFF_INSET = {"Belém": (-60, -40), "Ananindeua": (-40, 45), "Marituba": (50, 55), "Castanhal": (0, -45),
             "Capanema": (-40, -55), "Tracuateua": (-30, -80), "Bragança": (40, -60), "Augusto Corrêa": (90, -35),
             "Cachoeira do Piriá": (90, 10)}


@st.cache_data
def carregar_geo():
    return _json.load(open("pa_municipios.geojson", encoding="utf-8"))


GEO = carregar_geo()


@st.cache_data
def carregar_validacao():
    return _json.load(open("sensor_validacao.json", encoding="utf-8"))


def auc_mw(x, y):
    """AUC pela estatística U de Mann-Whitney (probabilidade de um caso ter valor maior que um não caso)."""
    from scipy import stats
    a, b = x[y == 1], x[y == 0]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b)
    return u / (len(a) * len(b)), p





def _xy(feat):
    xs, ys = [], []
    for poly in feat["geometry"]["coordinates"]:
        ring = poly[0]
        xs += [p[0] for p in ring] + [None]
        ys += [p[1] for p in ring] + [None]
    return xs, ys


def mapa_fig(fc, vals, mk, altura, offsets, destaque=False, sem_rotulo=()):
    fig = go.Figure()
    xs, ys = [], []
    for f in fc["features"]:
        if f["properties"]["mun"] not in vals:
            x, y = _xy(f); xs += x; ys += y
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=CLS_COR["off"], mode="lines",
                             line=dict(color="white", width=0.8), hoverinfo="skip", showlegend=False))
    for f in fc["features"]:
        m = f["properties"]["mun"]
        if m not in vals:
            continue
        v = vals[m]
        x, y = _xy(f)
        info = f"<b>{m}</b><br>{M[mk][0]}: {fmt(mk, v[mk])}<br>n = {v['n']}"
        fig.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor=CLS_COR[v["cls"]], mode="lines",
                                 line=dict(color="white", width=1), hoveron="fills", text=info, hoverinfo="text",
                                 showlegend=False))
        cx, cy = f["properties"]["c"]
        if m in sem_rotulo:
            continue
        ax, ay = offsets.get(m, (0, -30))
        fig.add_annotation(x=cx, y=cy, text=f"<b>{m}</b>  {fmt(mk, v[mk])}", showarrow=True, ax=ax, ay=ay,
                           arrowhead=0, arrowwidth=1.3, arrowcolor="#14201f", bgcolor="rgba(255,255,255,.94)",
                           bordercolor=CLS_COR[v["cls"]], borderwidth=2.5, borderpad=4, font=dict(size=13, color="#14201f"))
    if destaque:
        fig.add_shape(type="rect", x0=-49.05, x1=-45.8, y0=-2.3, y1=-0.5, line=dict(color="#4c5856", dash="dash", width=1.5))
        fig.add_annotation(x=-47.4, y=-0.35, text="área ampliada abaixo", showarrow=False, font=dict(size=11, color="#4c5856"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=altura, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", dragmode="pan")
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

abas = st.tabs(["Panorama", "Índice de risco", "Municípios", "Ciclo de vida", "Fatores de risco", "Perfil de risco",
                "Equilíbrio postural", "Sensor do celular", "Cruzamentos", "Método"])

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

    st.subheader("Mapa do risco no Pará")
    st.caption("Cada município pesquisado é pintado conforme o indicador: vermelho = situação pior, amarelo = "
               "intermediária, verde = melhor, comparando os municípios entre si (terços). Cinza: sem coleta.")
    mk = sel_metrica("Indicador do mapa", key="mapa")
    dmap = filtrar(df_all, ignorar=("mun",))
    t = tabela(dmap, "mun", [mk]).dropna()
    bom = mk in BONS
    q1, q2 = t[mk].quantile([1 / 3, 2 / 3]).values
    def classe(v):
        i = 0 if v < q1 else 1 if v < q2 else 2
        return (["hi", "mid", "lo"] if bom else ["lo", "mid", "hi"])[i]
    t["cls"] = t[mk].map(classe)
    vals = t.to_dict("index")
    st.plotly_chart(mapa_fig(GEO["main"], vals, mk, 860, OFF_MAIN, destaque=True, sem_rotulo=set(OFF_INSET)), width="stretch",
                    config={"displayModeBar": False})
    nomes = {"hi": "Pior terço", "mid": "Intermediário", "lo": "Melhor terço"}
    faixas = {"hi": (q2, t[mk].max()), "mid": (q1, q2), "lo": (t[mk].min(), q1)} if not bom else \
             {"lo": (q2, t[mk].max()), "mid": (q1, q2), "hi": (t[mk].min(), q1)}
    st.markdown("&nbsp;&nbsp;".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px">'
        f'<span style="width:16px;height:16px;border-radius:4px;background:{CLS_COR[c]}"></span>'
        f'{nomes[c]}: {fmt(mk, faixas[c][0])} a {fmt(mk, faixas[c][1])}</span>' for c in ["hi", "mid", "lo"])
        + f'<span style="display:inline-flex;align-items:center;gap:6px"><span style="width:16px;height:16px;'
          f'border-radius:4px;background:{CLS_COR["off"]};border:1px solid #ccc"></span>Sem coleta</span>'
          f'<br><small>Média dos municípios do recorte: <b>{fmt(mk, agg(dmap[mk], mk))}</b></small>',
        unsafe_allow_html=True)
    st.markdown("**Região Metropolitana e Nordeste paraense (ampliado)**")
    st.plotly_chart(mapa_fig(GEO["inset"], vals, mk, 460, OFF_INSET), width="stretch",
                    config={"displayModeBar": False})

    st.markdown("**Ranking dos municípios**")
    r = t.sort_values(mk, ascending=bom).reset_index()
    r["Posição"] = range(1, len(r) + 1)
    r["Situação"] = r.cls.map({"hi": "🔴 Pior terço", "mid": "🟡 Intermediário", "lo": "🟢 Melhor terço"})
    r["valor"] = r[mk] * escala(mk)
    st.dataframe(r[["Posição", "mun", "Situação", "valor", "n"]].rename(columns={"mun": "Município", "valor": M[mk][0]}),
                 hide_index=True, width="stretch", height=min(36 * len(r) + 40, 520),
                 column_config={M[mk][0]: st.column_config.ProgressColumn(
                     M[mk][0], format="%.1f" if M[mk][1] != "med" else "%.4f",
                     min_value=0, max_value=float(r["valor"].max()))})

    st.subheader("Principais achados")
    st.caption("Amostra completa, sem filtros.")
    a = df_all
    mq = a.groupby("mun").queda.mean().sort_values(ascending=False)
    fq = a.groupby("faixa", observed=True).queda.mean()
    dia, obs = rp(a, "diabetes"), rp(a, "obst")
    orf = a[a.queda == 1].orient.mean()
    f1, f2 = st.columns(2)
    f1.markdown(f"""
- **Território:** maiores taxas de queda em {', '.join(f'{m} ({v*100:.0f}%)' for m, v in mq.head(3).items())}; menores em {', '.join(f'{m} ({v*100:.0f}%)' for m, v in mq.tail(2).items())}.
- **Idade:** a queda vai de {fq.iloc[0]*100:.0f}% (18–29) a {fq.iloc[-1]*100:.0f}% (70+).
- **Diabetes:** quedas {dia['rp']:.1f}× mais frequentes ({dia['p1']*100:.0f}% vs {dia['p0']*100:.0f}%); efeito mantido no modelo ajustado.
""")
    f2.markdown(f"""
- **Casa:** {a.obst.mean()*100:.0f}% relatam obstáculos; quedas {obs['rp']:.1f}× mais frequentes. É o fator modificável mais forte.
- **Cuidado:** só {orf*100:.0f}% das mulheres que caíram receberam orientação profissional sobre quedas.
- **Macrorregião:** a diferença de queda entre macros não se sustenta após ajuste por idade e clínica (aba Fatores de risco).
- **IRQ-Pará:** o índice de 8 perguntas classifica o risco com AUC ≈ 0,70; no risco muito alto, {a[a.irq_g=="Muito alto"].queda.mean()*100:.0f}% caíram, contra {a[a.irq_g=="Baixo"].queda.mean()*100:.0f}% no baixo (aba Índice de risco).
- **Celular:** mede de fato o controle postural (separa bem idade e diabetes), mas sozinho prevê pouco a queda; tende a funcionar melhor em quem já tem fragilidade clínica (aba Sensor do celular).
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

# ================================================================ ÍNDICE DE RISCO (IRQ-Pará)
IRQ_PTS = [("Idade 40–59 anos", 2), ("Idade ≥ 60 anos", 3), ("Diabetes", 2), ("Dificuldade de equilíbrio ou coordenação", 2),
           ("Obstáculos com risco de queda em casa", 2), ("Vertigem recente", 1), ("Usa auxílio para caminhar", 3),
           ("Dificuldade em discriminar cores", 2)]
IRQ_COR = {"Baixo": "#3a9d63", "Moderado": "#f2b33d", "Alto": "#e8743b", "Muito alto": "#d64541"}
IRQ_ORD = ["Baixo", "Moderado", "Alto", "Muito alto"]

with abas[1]:
    st.subheader("IRQ-Pará: Índice de Risco de Quedas")
    st.caption("Escore de pontos construído com os dados do projeto: cada item do questionário recebe pontos proporcionais "
               "ao seu peso na regressão logística (quanto aumenta a chance de queda, ajustado pelos demais). Soma de 0 a 15. "
               "Não exige equipamento: pode ser aplicado por agente comunitário de saúde em 2 minutos.")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Como pontuar**")
        st.dataframe(pd.DataFrame(IRQ_PTS, columns=["Item", "Pontos"]), hide_index=True, width="stretch")
        st.markdown("**Classes**")
        st.markdown("🟢 **Baixo** 0–2 · 🟡 **Moderado** 3–4 · 🟠 **Alto** 5–6 · 🔴 **Muito alto** ≥ 7")
    with c2:
        st.markdown("**Calculadora**")
        k1, k2 = st.columns(2)
        idade_c = k1.number_input("Idade (anos)", 18, 110, 55, key="irq_idade")
        itens = {}
        for i, (nome, p) in enumerate(IRQ_PTS[2:]):
            itens[nome] = (k1 if i % 2 else k2).checkbox(f"{nome} (+{p})", key=f"irq_{i}")
        pts = (3 if idade_c >= 60 else 2 if idade_c >= 40 else 0) + sum(p for (n, p) in IRQ_PTS[2:] if itens[n])
        cls = "Baixo" if pts <= 2 else "Moderado" if pts <= 4 else "Alto" if pts <= 6 else "Muito alto"
        ref = df_all[df_all.irq_g == cls]
        st.markdown(
            f'<div style="padding:14px 18px;border-radius:10px;border:2px solid {IRQ_COR[cls]};margin-top:6px">'
            f'<div style="font-size:2.2rem;font-weight:700;line-height:1">{pts} pontos · '
            f'<span style="color:{IRQ_COR[cls]}">{cls}</span></div>'
            f'<div style="margin-top:6px">No projeto, mulheres nesta classe: <b>{ref.queda.mean()*100:.0f}%</b> caíram nos '
            f'últimos 12 meses, <b>{ref.lesao.mean()*100:.0f}%</b> com lesão (n = {len(ref)}).</div></div>',
            unsafe_allow_html=True)
        recs = {"Baixo": "Orientação geral sobre prevenção; reavaliar em 1 ano.",
                "Moderado": "Orientação sobre quedas e revisão do ambiente doméstico; reavaliar em 6 meses.",
                "Alto": "Avaliação de equilíbrio (celular/teste clínico), revisão de medicamentos e controle de doenças; adaptação da casa.",
                "Muito alto": "Prioridade: encaminhar para fisioterapia/avaliação multiprofissional, adaptação da casa e acompanhamento próximo."}
        st.caption("Sugestão de conduta: " + recs[cls])

    st.markdown("#### Desempenho do índice")
    g = df.groupby("irq_g", observed=True).agg(n=("queda", "size"), queda=("queda", "mean"), lesao=("lesao", "mean"),
                                               recorrente=("recorrente", "mean"), orient=("orient", "mean")).reindex(IRQ_ORD)
    c1, c2 = st.columns([3, 2])
    with c1:
        t = g.reset_index().melt(["irq_g", "n"], ["queda", "lesao", "recorrente"], var_name="ind", value_name="pct")
        t["pct"] *= 100
        t["ind"] = t.ind.map({"queda": "Queda (12 meses)", "lesao": "Queda com lesão", "recorrente": "Quedas recorrentes"})
        fig = px.bar(t, x="irq_g", y="pct", color="ind", barmode="group", hover_data=["n"],
                     text=t.pct.round(0).astype("Int64").astype(str) + "%", category_orders={"irq_g": IRQ_ORD},
                     color_discrete_sequence=["#d64541", "#f2b33d", "#7a3b8f"],
                     labels={"irq_g": "Classe do IRQ-Pará", "pct": "% das mulheres na classe", "ind": ""})
        fig.update_layout(height=400, legend_orientation="h", legend_y=-0.2)
        st.plotly_chart(fig, width="stretch")
    with c2:
        v = carregar_validacao()["irq"]
        x = df[["irq", "queda"]].dropna()
        a_app = auc_mw(x.irq.values, x.queda.values)[0] if x.queda.nunique() == 2 else np.nan
        st.metric("AUC no recorte atual", f"{a_app:.2f}".replace(".", ","))
        st.metric("AUC com validação cruzada (amostra completa)", f"{v['cv_queda'][0]:.2f}".replace(".", ","),
                  f"IC {v['cv_queda'][1]:.2f}–{v['cv_queda'][2]:.2f}".replace(".", ","), delta_color="off")
        st.caption(f"Para quedas recorrentes: AUC {v['cv_recorrente'][0]:.2f}; com lesão: {v['cv_lesao'][0]:.2f}. "
                   f"Acrescentar o celular (área OA ≥ {v['thr_area']:.3f}) leva a {v['cv_queda_cel'][0]:.2f}: ganho desprezível. "
                   "Para comparação: só a idade dá 0,63 e o celular sozinho 0,60.")

    st.markdown("**Escolha do ponto de corte**")
    x = df[["irq", "queda"]].dropna()
    linhas = []
    for c in range(2, 9):
        pos = x.irq >= c
        tp, fn = (pos & (x.queda == 1)).sum(), (~pos & (x.queda == 1)).sum()
        tn, fp = (~pos & (x.queda == 0)).sum(), (pos & (x.queda == 0)).sum()
        linhas.append({"Corte (IRQ ≥)": c, "Mulheres sinalizadas (%)": pos.mean() * 100,
                       "Sensibilidade (%)": tp / max(tp + fn, 1) * 100, "Especificidade (%)": tn / max(tn + fp, 1) * 100,
                       "Quedas entre sinalizadas (%)": tp / max(tp + fp, 1) * 100})
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.0f") for k in
                                ["Mulheres sinalizadas (%)", "Sensibilidade (%)", "Especificidade (%)", "Quedas entre sinalizadas (%)"]})
    st.caption("Corte ≥ 3 serve para triagem ampla (encontra ~8 em cada 10 que caíram); corte ≥ 5 equilibra melhor; "
               "corte ≥ 7 seleciona poucas mulheres com risco muito alto (mais da metade caiu).")

    st.markdown("#### Onde estão as mulheres de risco alto")
    t = df.groupby("mun").irq_alto.agg(["mean", "size"]).reset_index().sort_values("mean", ascending=False)
    t["pct"] = t["mean"] * 100
    t["macro"] = t.mun.map(df_all.drop_duplicates("mun").set_index("mun").macro_lbl)
    fig = px.bar(t, x="mun", y="pct", color="macro", color_discrete_map=MACRO_COR, hover_data={"size": True},
                 labels={"pct": "% com IRQ alto ou muito alto", "mun": "", "macro": ""})
    fig.update_layout(height=400, xaxis_categoryorder="total descending")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Orientação profissional sobre quedas já recebida: {g.orient.iloc[0]*100:.0f}% no risco baixo e "
               f"{g.orient.iloc[-1]*100:.0f}% no muito alto: a orientação não está chegando a quem mais precisa.")
    st.warning("**Limitações:** o índice foi construído e testado nos mesmos dados (a validação cruzada reduz, mas não "
               "elimina, o otimismo) e o estudo é transversal: a queda foi perguntada sobre o ano anterior. Para uso "
               "clínico, o IRQ-Pará precisa ser validado prospectivamente (quedas nos 12 meses seguintes) e em outras amostras.")

# ================================================================ MUNICÍPIOS
with abas[2]:
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
    fmt_map = {c: "{:.1f}" for c in show.columns if c != "n"}
    fmt_map.update({M["area_oa"][0]: "{:.4f}", M["romberg"][0]: "{:.2f}", "prioridade": "{:.2f}", "n": "{:.0f}"})
    st.dataframe(show.style.format(fmt_map, na_rep="–")
                 .background_gradient(cmap="Oranges", subset=pcols),
                 width="stretch", height=620)


    st.subheader("O que os municípios com mais quedas têm em comum")
    st.caption("Correlação entre municípios (ecológica): cada ponto é um município. Mostra características do território, "
               "não de cada mulher.")
    caracts = {"idade": "Idade média", "idoso": "% com 60+ anos", "renda": "Renda média (SM)", "esc": "Escolaridade média",
               "ncom": "Nº médio de doenças crônicas", "diabetes": "% com diabetes", "dequil": "% com dificuldade de equilíbrio",
               "vertigem": "% com vertigem", "obst": "% com obstáculos em casa", "orient": "% que recebeu orientação",
               "multimed": "% com ≥2 medicamentos", "area_oa": "Oscilação OA (mediana)", "irq": "IRQ-Pará médio"}
    agg_m = df.groupby("mun").agg(queda=("queda", "mean"), n=("queda", "size"),
                                  **{k: (k, "median" if k == "area_oa" else "mean") for k in caracts})
    agg_m = agg_m[agg_m.n >= 10]
    if len(agg_m) >= 6:
        from scipy import stats as _st
        cor = []
        for k, l in caracts.items():
            x = agg_m[[k, "queda"]].dropna()
            r, p = _st.spearmanr(x[k], x.queda)
            cor.append(dict(c=l, k=k, rho=r, p=p))
        C = pd.DataFrame(cor).sort_values("rho")
        C["sig"] = np.where(C.p < 0.05, "p < 0,05", "não significativo")
        c1, c2 = st.columns(2)
        fig = px.bar(C, x="rho", y="c", color="sig", orientation="h", hover_data={"p": ":.3f"},
                     color_discrete_map={"p < 0,05": "#0d6b6e", "não significativo": "#b8c2bf"},
                     labels={"rho": "Correlação com % de quedas (Spearman ρ)", "c": "", "sig": ""})
        fig.add_vline(x=0, line_color="#7a8583")
        fig.update_layout(height=460, legend_orientation="h", legend_y=1.08, xaxis_range=[-1, 1])
        c1.plotly_chart(fig, width="stretch")
        kx = c2.selectbox("Ver em detalhe", list(caracts), index=list(caracts).index("idade"), format_func=caracts.get,
                          key="eco_x")
        t = agg_m.reset_index()
        t["macro"] = t.mun.map(df_all.drop_duplicates("mun").set_index("mun").macro_lbl)
        t["queda_pct"] = t.queda * 100
        fig = px.scatter(t, x=kx, y="queda_pct", text="mun", color="macro", color_discrete_map=MACRO_COR, size="n",
                         size_max=16, labels={kx: caracts[kx], "queda_pct": "% de quedas", "macro": ""})
        fig.update_traces(textposition="top center", textfont_size=10)
        z = np.polyfit(t[kx].astype(float), t.queda_pct, 1)
        xs = np.linspace(t[kx].min(), t[kx].max(), 20)
        fig.add_scatter(x=xs, y=np.polyval(z, xs), mode="lines", line=dict(dash="dash", color="#4c5856"),
                        showlegend=False, hoverinfo="skip")
        fig.update_layout(height=430, legend_orientation="h")
        c2.plotly_chart(fig, width="stretch")
        st.caption("Na amostra completa: municípios com população mais velha, mais doenças crônicas, mais dificuldade de "
                   "equilíbrio, menor renda e menor escolaridade têm mais quedas. Com só 22 municípios, correlações "
                   "abaixo de ~0,42 não atingem significância.")

# ================================================================ CICLO DE VIDA
with abas[3]:
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
    st.dataframe(t.rename(columns={k: M[k][0] for k in ks}).style.format({M["area_oa"][0]: "{:.4f}", M["area_of"][0]: "{:.4f}", M["romberg"][0]: "{:.2f}"}),
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


with abas[4]:
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


    st.subheader("Gradientes: o risco sobe degrau a degrau")
    c1, c2 = st.columns(2)
    t = df.assign(ncomg=df.ncom.clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})).groupby("ncomg").queda.agg(["mean", "size"]).reset_index()
    fig = px.bar(t, x="ncomg", y=t["mean"] * 100, text=(t["mean"] * 100).round(0).astype(int).astype(str) + "%",
                 hover_data={"size": True}, color_discrete_sequence=["#d64541"],
                 labels={"ncomg": "Nº de doenças crônicas", "y": "% que caiu"})
    fig.update_xaxes(type="category"); fig.update_layout(height=340, yaxis_title="% que caiu")
    c1.plotly_chart(fig, width="stretch")
    t = df.groupby("faixa", observed=True).queda.agg(["mean", "size"]).reset_index()
    fig = px.bar(t, x="faixa", y=t["mean"] * 100, text=(t["mean"] * 100).round(0).astype(int).astype(str) + "%",
                 hover_data={"size": True}, color_discrete_sequence=["#d64541"], labels={"faixa": "Faixa etária"})
    fig.update_layout(height=340, yaxis_title="% que caiu")
    c2.plotly_chart(fig, width="stretch")

    st.subheader("Correlações individuais")
    st.caption("Spearman entre características de cada participante. |ρ| < 0,1 desprezível; 0,1–0,3 fraca; 0,3–0,5 moderada. "
               "Em desfechos com muitas causas, como a queda, correlações fracas são esperadas.")
    cvars = {"idade": "Idade", "renda": "Renda", "esc": "Escolaridade", "imc": "IMC", "cintura": "Cintura",
             "panturrilha": "Panturrilha", "pas": "PA sistólica", "sono": "Horas de sono", "nmed": "Nº medicamentos",
             "ncom": "Nº doenças", "irq": "IRQ-Pará", "queda": "Queda", "nquedas": "Nº quedas", "dequil": "Dif. equilíbrio",
             "vertigem": "Vertigem", "area_oa": "Área OA", "area_of": "Área OF", "romberg": "Romberg"}
    Cm = df[list(cvars)].corr("spearman").rename(index=cvars, columns=cvars)
    fig = px.imshow(Cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-0.6, zmax=0.6, aspect="auto",
                    labels={"color": "ρ"})
    fig.update_layout(height=640)
    st.plotly_chart(fig, width="stretch")

# ================================================================ PERFIL DE RISCO
MARC = {"idoso": "Idade ≥ 60", "diabetes": "Diabetes", "dequil": "Dificuldade de equilíbrio", "obst": "Obstáculos em casa",
        "com2": "≥ 2 doenças crônicas", "multimed": "≥ 2 medicamentos"}

with abas[5]:
    st.subheader("Perfil de risco: quanto mais fatores, mais quedas")
    st.caption("Seis marcadores simples, todos obtidos no questionário: idade ≥ 60, diabetes, dificuldade de equilíbrio "
               "relatada, obstáculos em casa, duas ou mais doenças crônicas e uso de dois ou mais medicamentos. "
               "São os fatores que se mantiveram associados à queda no modelo ajustado ou que mostraram gradiente claro.")
    dp = df.copy()
    dp["com2"] = (dp.ncom >= 2).astype(float)
    dp["nf"] = sum(dp[k].fillna(0) for k in MARC)
    dp["nf_g"] = dp.nf.clip(upper=4).map({0: "0", 1: "1", 2: "2", 3: "3", 4: "4 ou mais"})
    ordem_nf = ["0", "1", "2", "3", "4 ou mais"]
    g = dp.groupby("nf_g").agg(queda=("queda", "mean"), lesao=("lesao", "mean"), recorrente=("recorrente", "mean"),
                               n=("queda", "size")).reindex(ordem_nf).reset_index()
    c1, c2 = st.columns([3, 2])
    with c1:
        t = g.melt(["nf_g", "n"], ["queda", "lesao", "recorrente"], var_name="ind", value_name="pct")
        t["pct"] *= 100
        t["ind"] = t.ind.map({"queda": "Queda (12 meses)", "lesao": "Queda com lesão", "recorrente": "Quedas recorrentes"})
        fig = px.bar(t, x="nf_g", y="pct", color="ind", barmode="group", hover_data=["n"],
                     text=t.pct.round(0).astype("Int64").astype(str) + "%",
                     color_discrete_sequence=["#d64541", "#f2b33d", "#7a3b8f"],
                     category_orders={"nf_g": ordem_nf},
                     labels={"nf_g": "Número de marcadores de risco", "pct": "%", "ind": ""})
        fig.update_xaxes(type="category")
        fig.update_layout(height=420, legend_orientation="h", legend_y=-0.2)
        st.plotly_chart(fig, width="stretch")
    with c2:
        base = g.queda.iloc[0]
        st.markdown("**Como ler**")
        st.markdown("\n".join(
            f"- **{r.nf_g} marcador{'es' if r.nf_g != '1' else ''}:** {r.queda*100:.0f}% caíram"
            + (f" ({r.queda/base:.1f}× o grupo sem marcadores)" if i and base > 0 else "") + f" · n = {r.n}"
            for i, r in enumerate(g.itertuples()) if r.n > 0))
        st.caption("Na amostra completa, a queda vai de cerca de 13% (nenhum marcador) a 55% (4 ou mais): "
                   "o gradiente mais forte de todo o estudo.")

    st.markdown("#### Quem são as mulheres em cada nível de risco")
    t = dp.groupby("nf_g").agg(idade=("idade", "mean"), renda=("renda", "median"), esc=("baixaesc", "mean"),
                               orient=("orient", "mean"), alto=("alto", "mean"), area=("area_oa", "median"),
                               n=("queda", "size")).reindex(ordem_nf)
    t["esc"] *= 100; t["orient"] *= 100
    st.dataframe(t.drop(columns="alto").rename(columns={"idade": "Idade média", "renda": "Renda mediana (SM)",
                 "esc": "Até fund. incompleto (%)", "orient": "Recebeu orientação (%)", "area": "Área elipse OA (mediana)",
                 "n": "n"}).rename_axis("Marcadores de risco").style.format({"Idade média": "{:.0f}", "Renda mediana (SM)": "{:.1f}",
                 "Até fund. incompleto (%)": "{:.0f}", "Recebeu orientação (%)": "{:.0f}",
                 "Área elipse OA (mediana)": "{:.4f}"}), width="stretch")
    st.caption("A orientação profissional sobre quedas não acompanha o risco: quem acumula mais fatores não "
               "recebe proporcionalmente mais orientação.")

    st.markdown("#### Combinação idade × carga de doenças")
    dp["ncom_g"] = dp.ncom.clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})
    dp["faixa3"] = pd.cut(dp.idade, [0, 39.99, 59.99, 200], labels=["< 40", "40–59", "≥ 60"])
    pv = dp.pivot_table(index="faixa3", columns="ncom_g", values="queda", aggfunc="mean", observed=True) * 100
    cn = dp.pivot_table(index="faixa3", columns="ncom_g", values="queda", aggfunc="count", observed=True)
    fig = px.imshow(pv.where(cn >= 10), text_auto=".0f", color_continuous_scale=[[0, "#3a9d63"], [.5, "#f2b33d"], [1, "#d64541"]],
                    labels={"x": "Nº de doenças crônicas", "y": "Idade", "color": "% queda"}, aspect="auto")
    fig.update_xaxes(type="category"); fig.update_yaxes(type="category")
    fig.update_layout(height=320)
    st.plotly_chart(fig, width="stretch")
    st.caption("Células com menos de 10 participantes ficam em branco.")

    st.markdown("#### O celular funciona melhor no perfil de risco?")
    st.caption("Para cada subgrupo: AUC da área da elipse de olhos abertos para separar quem caiu (0,5 = acaso) e "
               "quanto a chance de queda sobe a cada aumento de 1 desvio-padrão na oscilação (OR).")
    import statsmodels.formula.api as smf
    dp["z_area"] = np.log(dp.area_oa)
    dp["z_area"] = (dp.z_area - dp.z_area.mean()) / dp.z_area.std()
    subs = [("Todas", dp.index == dp.index), ("Sem dificuldade de equilíbrio", dp.dequil == 0),
            ("Com dificuldade de equilíbrio", dp.dequil == 1), ("Sem diabetes", dp.diabetes == 0),
            ("Com diabetes", dp.diabetes == 1), ("0–1 doença crônica", dp.ncom <= 1), ("≥ 2 doenças crônicas", dp.ncom >= 2),
            ("Sem vertigem", dp.vertigem == 0), ("Com vertigem", dp.vertigem == 1), ("< 60 anos", dp.idade < 60),
            ("≥ 60 anos", dp.idade >= 60), ("0–1 marcador de risco", dp.nf <= 1), ("≥ 2 marcadores de risco", dp.nf >= 2)]
    lin = []
    for nome, mask in subs:
        x = dp.loc[mask, ["z_area", "queda"]].dropna()
        if x.queda.sum() < 10 or (x.queda == 0).sum() < 10:
            continue
        a, p = auc_mw(x.z_area.values, x.queda.values)
        try:
            m = smf.logit("queda ~ z_area", data=x).fit(disp=0)
            o, lo, hi = np.exp(m.params.z_area), *np.exp(m.conf_int().loc["z_area"])
        except Exception:
            o = lo = hi = np.nan
        lin.append(dict(Subgrupo=nome, n=len(x), Quedas=int(x.queda.sum()), AUC=a, OR=o, lo=lo, hi=hi))
    T = pd.DataFrame(lin)
    if not T.empty:
        T["grupo"] = np.where(T.Subgrupo.str.startswith(("Com ", "≥")), "Perfil de risco",
                              np.where(T.Subgrupo == "Todas", "Todas", "Sem o fator"))
        fig = px.scatter(T, x="AUC", y="Subgrupo", color="grupo", size="n", size_max=18,
                         color_discrete_map={"Perfil de risco": "#d64541", "Sem o fator": "#3a9d63", "Todas": "#4c5856"},
                         hover_data={"n": True, "Quedas": True, "OR": ":.2f"}, labels={"grupo": ""})
        fig.add_vline(x=0.5, line_dash="dash", line_color="#7a8583", annotation_text="acaso")
        fig.update_layout(height=460, xaxis_range=[0.4, 0.75], legend_orientation="h",
                          yaxis_categoryorder="array", yaxis_categoryarray=T.Subgrupo.tolist()[::-1])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(T.drop(columns="grupo").rename(columns={"lo": "OR IC inf.", "hi": "OR IC sup."}), hide_index=True,
                     width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f")
                                                     for c in ["AUC", "OR", "OR IC inf.", "OR IC sup."]})
    st.info("**Leitura:** nos grupos com fragilidade clínica (dificuldade de equilíbrio, diabetes, várias doenças, "
            "vertigem) a oscilação medida pelo celular separa melhor quem caiu (AUC ≈ 0,61–0,65 contra ≈ 0,55 nos "
            "demais) e pesa mais na chance de queda. A idade sozinha não muda o desempenho. A diferença entre grupos "
            "ainda não é estatisticamente confirmada (interação p ≈ 0,2–0,3), por isso é uma tendência a confirmar "
            "com mais mulheres frágeis e acompanhamento de quedas futuras.")

    st.markdown("#### Recomendação: triagem em duas etapas")
    st.markdown("""
1. **Questionário rápido para todas** (os 6 marcadores acima). Com 2 ou mais marcadores, a mulher entra no grupo de atenção.
2. **Exame com o celular no grupo de atenção**, onde a oscilação medida discrimina melhor, para graduar a prioridade.
3. **Ação imediata em quem tem ≥ 3 marcadores**: orientação sobre quedas, revisão de medicamentos, controle do diabetes e adaptação da casa (o único fator forte totalmente modificável).
""")

# ================================================================ EQUILÍBRIO
with abas[6]:
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

# ================================================================ SENSOR DO CELULAR
SENS_BASE = ["RMS AP", "RMS ML", "Desvio total", "Área elipse", "Freq. mediana AP", "Freq. mediana ML",
             "Energia baixa AP", "Energia média AP", "Energia alta AP", "Energia baixa ML", "Energia média ML",
             "Energia alta ML"]
SENS_OUT = {"queda": "Queda (12 meses)", "recorrente": "Quedas recorrentes (≥2)", "lesao": "Queda com lesão",
            "dequil": "Dificuldade de equilíbrio (relato)", "idoso": "Idade ≥ 60", "diabetes": "Diabetes",
            "dvis": "Doença visual"}



with abas[7]:
    st.subheader("O celular ajuda a identificar quem está em risco?")
    st.caption("O smartphone (aplicativo Momentum Sensors, na lombar, 60 s de olhos abertos e 60 s de olhos fechados) "
               "gera 12 medidas por condição: amplitude (RMS, desvio total, área da elipse), frequência mediana e "
               "energia em três faixas (0–0,5 Hz, 0,5–2 Hz, 2–8 Hz), nos eixos antero-posterior (AP) e médio-lateral (ML).")

    if df_all["s_RMS AP (OA)"].notna().sum() == 0:
        st.error("A planilha do repositório não traz as medidas do celular. "
                 "Substitua-a pela versão mais recente para ver as análises desta aba.")
    val = carregar_validacao()
    st.markdown("#### 1. Capacidade de distinguir, testada com validação cruzada")
    st.caption("AUC = probabilidade de o modelo dar escore maior a quem teve o desfecho do que a quem não teve. "
               "0,5 = sorteio; 0,6 = fraco; 0,7 = razoável; 0,8 = bom. Validação cruzada 5×10 (amostra completa).")
    linhas = []
    for y, lab in [("queda", "Queda"), ("recorrente", "Quedas recorrentes"), ("lesao", "Queda com lesão"),
                   ("idoso", "Idade ≥ 60"), ("diabetes", "Diabetes")]:
        for nome, v in val["cv"][y].items():
            linhas.append(dict(desfecho=lab, modelo=nome, auc=v[0], lo=v[1], hi=v[2]))
    t = pd.DataFrame(linhas)
    cores_mod = {"Só idade": "#9aa5a2", "Questionário (sem sintomas)": "#86b6ef", "Questionário completo": "#2a78d6",
                 "Só celular": "#eb6834", "Só celular (random forest)": "#f2a07a", "Questionário + celular": "#0d6b6e",
                 "Idade + celular": "#0d6b6e"}
    fig = px.bar(t, x="auc", y="desfecho", color="modelo", barmode="group", orientation="h",
                 error_x=t.hi - t.auc, error_x_minus=t.auc - t.lo, color_discrete_map=cores_mod,
                 labels={"auc": "AUC (validação cruzada)", "desfecho": "", "modelo": ""})
    fig.add_vline(x=0.5, line_dash="dash", line_color="#7a8583", annotation_text="acaso")
    fig.add_vline(x=0.7, line_dash="dot", line_color="#3a9d63", annotation_text="razoável")
    fig.update_layout(height=520, xaxis_range=[0.4, 0.85], legend_orientation="h", legend_y=-0.15)
    st.plotly_chart(fig, width="stretch")
    q = val["cv"]["queda"]
    st.info(
        f"**Leitura:** para quedas, o celular sozinho acerta pouco acima do acaso (AUC {q['Só celular'][0]:.2f}); "
        f"o questionário vai melhor ({q['Questionário completo'][0]:.2f}) e somar o celular ao questionário "
        f"não melhora ({q['Questionário + celular'][0]:.2f}). Onde o sinal do celular é mais forte é no "
        f"envelhecimento (AUC {val['cv']['idoso']['Só celular'][0]:.2f} para separar quem tem 60+), ou seja, "
        "ele mede de fato uma perda de controle postural, mas essa perda, num único exame de 60 s, ainda não "
        "identifica sozinha a pessoa que vai cair.")

    st.markdown("#### 2. Qual medida do celular separa melhor? (recalculado com os filtros)")
    c1, c2 = st.columns([1, 2])
    yk = c1.selectbox("Desfecho", list(SENS_OUT), format_func=SENS_OUT.get, key="s_out")
    dS = df.copy()
    dS["idoso"] = (dS.idade >= 60).astype(float).where(dS.idade.notna())
    lin = []
    for b in SENS_BASE:
        for cond in ["OA", "OF"]:
            col = f"s_{b} ({cond})"
            x = dS[[col, yk]].dropna()
            if x[yk].nunique() < 2:
                continue
            a, p = auc_mw(x[col].values, x[yk].values)
            lin.append(dict(medida=b, cond="Olhos abertos" if cond == "OA" else "Olhos fechados", auc=a, p=p,
                            n=len(x), casos=int(x[yk].sum()),
                            med_com=x.loc[x[yk] == 1, col].median(), med_sem=x.loc[x[yk] == 0, col].median()))
    A = pd.DataFrame(lin)
    if A.empty:
        st.warning("Poucos casos no recorte para calcular.")
    else:
        A["força"] = pd.cut((A.auc - 0.5).abs(), [-1, 0.03, 0.07, 0.12, 1],
                            labels=["Nenhuma", "Muito fraca", "Fraca", "Moderada"])
        A["sig"] = np.where(A.p < 0.05, "p < 0,05", "n.s.")
        fig = px.bar(A.sort_values("auc"), x="auc", y="medida", color="cond", barmode="group", orientation="h",
                     color_discrete_sequence=["#2a78d6", "#eb6834"], hover_data=["p", "n", "casos"],
                     labels={"auc": "AUC (>0,5 = valor maior em quem tem o desfecho)", "medida": "", "cond": ""})
        fig.add_vline(x=0.5, line_dash="dash", line_color="#7a8583")
        fig.update_layout(height=560, xaxis_range=[0.3, 0.8], legend_orientation="h")
        c2.plotly_chart(fig, width="stretch")
        top = A.assign(d=(A.auc - .5).abs()).sort_values("d", ascending=False).head(6)
        c1.markdown("**Melhores medidas**")
        c1.dataframe(top[["medida", "cond", "auc", "força", "sig"]].rename(columns={"cond": "condição", "auc": "AUC"}),
                     hide_index=True, width="stretch", column_config={"AUC": st.column_config.NumberColumn(format="%.2f")})
        c1.caption("As medidas do eixo **antero-posterior** (frente-trás) e de **baixa/média frequência** "
                   "costumam ser as mais informativas.")

    st.markdown("#### 3. Existe um ponto de corte? Quedas por quintil da medida")
    c1, c2 = st.columns([1, 2])
    mb = c1.selectbox("Medida", SENS_BASE, index=SENS_BASE.index("Área elipse"), key="s_q")
    cnd = c1.radio("Condição", ["OA", "OF"], horizontal=True, key="s_qc",
                   format_func={"OA": "Olhos abertos", "OF": "Olhos fechados"}.get)
    yq = c1.selectbox("Desfecho", list(SENS_OUT)[:4], format_func=SENS_OUT.get, key="s_qy")
    x = df[[f"s_{mb} ({cnd})", yq, "idade"]].dropna()
    if len(x) >= 50:
        x["quintil"] = pd.qcut(x.iloc[:, 0], 5, labels=["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (maior)"])
        g = x.groupby("quintil", observed=True).agg(pct=(yq, "mean"), n=(yq, "size"), idade=("idade", "mean")).reset_index()
        g["pct"] *= 100
        fig = px.bar(g, x="quintil", y="pct", text=g.pct.round(1).astype(str) + "%", hover_data=["n", "idade"],
                     labels={"pct": f"{SENS_OUT[yq]} (%)", "quintil": f"Quintil de {mb} ({cnd})"},
                     color_discrete_sequence=["#0d6b6e"])
        fig.update_layout(height=360)
        c2.plotly_chart(fig, width="stretch")
        c1.caption("Na amostra completa, a área da elipse de olhos abertos mostra um **degrau**: ~16–22% de quedas "
                   "nos três quintis inferiores e ~27–32% nos dois superiores. Acima do percentil 60 a "
                   "sensibilidade é ~51% e a especificidade ~63%, útil para triagem de grupo, insuficiente para "
                   "diagnóstico individual.")

    st.markdown("#### 4. O que interfere no sinal do celular")
    fat = {"idade": "Idade", "imc": "IMC", "cintura": "Cintura", "panturrilha": "Panturrilha", "renda": "Renda",
           "esc": "Escolaridade", "ncom": "Nº doenças crônicas", "nmed": "Nº medicamentos", "nquedas": "Nº de quedas"}
    mat = []
    for b in SENS_BASE:
        for cnd in ["OA", "OF"]:
            col = f"s_{b} ({cnd})"
            mat.append([df[[col, f]].dropna().corr("spearman").iloc[0, 1] for f in fat])
    H = pd.DataFrame(mat, index=[f"{b} ({c})" for b in SENS_BASE for c in ["OA", "OF"]], columns=list(fat.values()))
    fig = px.imshow(H, color_continuous_scale="RdBu_r", zmin=-0.4, zmax=0.4, text_auto=".2f", aspect="auto",
                    labels={"color": "Spearman ρ"})
    fig.update_layout(height=720)
    st.plotly_chart(fig, width="stretch")
    st.caption("Destaques na amostra completa: a **energia de alta frequência AP cai com o IMC e a cintura** "
               "(ρ ≈ −0,33): o tecido da região lombar amortece o sinal do aparelho, então o corpo da participante "
               "interfere na medida (o mesmo vale para jerk e entropia amostral). A **energia de média frequência AP "
               "sobe com a idade** (ρ ≈ +0,23) e é a melhor marcadora de envelhecimento do equilíbrio.")

    st.markdown("#### 5. Efeito do local de coleta")
    t = df[["mun", "s_Área elipse (OA)"]].dropna()
    ordem = t.groupby("mun")["s_Área elipse (OA)"].median().sort_values().index.tolist()
    fig = px.box(t, x="mun", y="s_Área elipse (OA)", log_y=True, category_orders={"mun": ordem}, points=False,
                 color_discrete_sequence=["#0d6b6e"], labels={"mun": "", "s_Área elipse (OA)": "Área elipse OA (log)"})
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("O município explica ~8% da variação da área da elipse (Kruskal-Wallis p < 0,001). Ajustando idade e "
               "IMC, Santarém mede ~1,8× e Itupiranga ~1,6× a área de Belém. Em Santarém os valores ainda vêm da "
               "planilha original (protocolo de 30 s, convertidos por calibração), então a diferença deve ser lida "
               "com cautela; nos demais, pode refletir piso, ruído, aparelho ou diferenças reais.")

    st.markdown("#### 6. As 24 medidas repetem a mesma informação")
    st.caption("Quem balança muito tem, ao mesmo tempo, RMS alto, desvio total alto, área grande e energia de baixa "
               "frequência alta. Correlações acima de 0,85 significam que, sabendo uma, praticamente se sabe a outra.")
    c1, c2 = st.columns([3, 2])
    cols_oa = [f"s_{b} (OA)" for b in SENS_BASE]
    R = df[cols_oa].corr("spearman")
    R.index = R.columns = SENS_BASE
    fig = px.imshow(R, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    labels={"color": "ρ"})
    fig.update_layout(height=520, title="Correlação entre as medidas (olhos abertos)")
    c1.plotly_chart(fig, width="stretch")
    pv = carregar_validacao()["pca"][:8]
    t = pd.DataFrame({"Componente": [f"C{i+1}" for i in range(len(pv))], "Informação (%)": np.array(pv) * 100})
    t["Acumulado (%)"] = t["Informação (%)"].cumsum()
    fig = px.bar(t, x="Componente", y="Informação (%)", text=t["Acumulado (%)"].round(0).astype(int).astype(str) + "% acum.",
                 color_discrete_sequence=["#0d6b6e"])
    fig.update_layout(height=360, title="Quanto cada 'super-medida' resume (24 medidas)")
    c2.plotly_chart(fig, width="stretch")
    c2.markdown("""
**Na prática, o celular mede 4 aspectos:**
1. **Quanto o corpo balança**: área, RMS, desvio total, energia baixa (o principal)
2. **Quão rápido oscila**: frequência mediana
3. **Tremores rápidos**: energia alta (2–8 Hz), sensível ao IMC
4. **Quanto piora ao fechar os olhos**: Romberg (OF ÷ OA)

Para o artigo, use um representante de cada aspecto em vez das 24 medidas.
""")

    # ---------------------------------------------------------------- sinais brutos
    st.markdown("---")
    st.markdown("#### 7. Reanálise dos sinais brutos do acelerômetro")
    try:
        SB = _json.load(open("sinais_brutos.json", encoding="utf-8"))
    except Exception:
        SB = None
    if SB is None or "r_rms_ap_oa" not in df_all.columns:
        st.info("Envie o arquivo **sinais_brutos.json** e a planilha anonimizada atualizada para ver esta seção.")
    else:
        q = SB["quality"]
        st.caption("Os arquivos originais do aplicativo (aceleração a ~200 Hz, 60 s por condição) foram reprocessados do zero: "
                   "reamostragem a 100 Hz, correção da inclinação e da orientação do celular pela gravidade, filtro 0,1–10 Hz e "
                   "cálculo de 26 medidas por condição, incluindo medidas que não existiam na planilha (jerk, velocidade média, "
                   "entropia amostral, frequência de 95% e entropia espectral).")
        k = st.columns(4)
        k[0].metric("Arquivos lidos", f"{q['n_files_total']:,}".replace(",", "."))
        k[1].metric("Gravações únicas", f"{q['n_unique']:,}".replace(",", "."), "o restante eram cópias .csv/.txt", delta_color="off")
        k[2].metric("Participantes ligadas à planilha", q["n_people"])
        k[3].metric("Com OA e OF válidos", q["people_with_good_both"])

        st.markdown("**Controle de qualidade: o que foi encontrado**")
        st.markdown(f"""
- **Celular deitado de lado:** 297 gravações, quase todas de Itaituba, foram feitas com o aparelho na horizontal. Na tabulação original
  isso não foi corrigido; na reanálise os eixos foram recalculados pela gravidade e essas gravações passaram a ser aproveitadas.
- **Protocolo diferente em Santarém:** as gravações de Santarém têm **30 s** (e não 60 s), formato de arquivo e versão do aplicativo
  diferentes (2026). Isso ajuda a explicar por que Santarém aparecia com a maior oscilação da amostra: parte da diferença é do
  método de coleta. Essas gravações ficaram fora da comparação principal.
- **Movimento durante a coleta:** só {q['bad_spk']} gravações tiveram picos de movimento; {q['bad_dur']} foram curtas demais.
- **Efeito do município caiu para {SB['eta']['sinais']*100:.0f}%** da variação da área de oscilação depois da reanálise
  (antes, {SB['eta']['tabulada']*100:.0f}%): parte da "diferença entre territórios" era diferença de medição.
""")
        st.markdown("**A medida é repetível?**")
        rel = SB["reliability"]
        nomes_rel = {"rms_ap": "RMS antero-posterior", "rms_ml": "RMS médio-lateral", "area95": "Área da elipse 95%",
                     "range_ap": "Amplitude AP", "jerk_ap": "Jerk AP", "path_vel": "Velocidade média",
                     "f50_ap": "Frequência mediana AP", "f95_ap": "Frequência 95% AP", "sampen_ap": "Entropia amostral AP",
                     "sampen_ml": "Entropia amostral ML", "spent_ap": "Entropia espectral AP"}
        t = pd.DataFrame([{"Medida": v, "Teste-reteste OA (ICC)": rel["OA"]["retest"].get(k_),
                           "Teste-reteste OF (ICC)": rel["OF"]["retest"].get(k_)} for k_, v in nomes_rel.items()])
        c1, c2 = st.columns([3, 2])
        fig = px.bar(t.melt("Medida", var_name="cond", value_name="ICC"), x="ICC", y="Medida", color="cond", barmode="group",
                     orientation="h", color_discrete_sequence=["#2a78d6", "#eb6834"], labels={"cond": ""})
        fig.add_vline(x=0.75, line_dash="dash", line_color="#3a9d63", annotation_text="boa")
        fig.add_vline(x=0.5, line_dash="dot", line_color="#f2b33d", annotation_text="moderada")
        fig.update_layout(height=420, xaxis_range=[0, 1], legend_orientation="h")
        c1.plotly_chart(fig, width="stretch")
        c2.markdown(f"""
- **{rel['OA']['n_retest']} mulheres** tiveram duas gravações válidas na mesma condição: dá para medir a repetibilidade de verdade.
- As medidas de **amplitude** (RMS, área, amplitude AP) e a **entropia amostral** têm ICC ≈ 0,8: **boa repetibilidade**.
- As de **frequência** são moderadas (0,55–0,75).
- Dentro da mesma gravação, os três terços de 20 s concordam com ICC ≈ {rel['OA']['thirds_area']:.2f}.

""")

        st.markdown("**Os sinais brutos discriminam melhor?** (validação cruzada, mesmas participantes)")
        lin = []
        for y, lab in [("queda", "Queda"), ("recorrente", "Quedas recorrentes"), ("lesao", "Queda com lesão"),
                       ("idoso", "Idade ≥ 60"), ("diabetes", "Diabetes")]:
            for nome, v in SB["cv"][y].items():
                lin.append(dict(desfecho=lab, modelo=nome, auc=v[0], lo=v[1], hi=v[2]))
        t = pd.DataFrame(lin)
        cores_m = {"IRQ": "#7a3b8f", "Celular sinais brutos": "#eb6834",
                   "Celular sinais brutos (random forest)": "#f2a07a", "IRQ + sinais brutos": "#0d6b6e"}
        fig = px.bar(t, x="auc", y="desfecho", color="modelo", barmode="group", orientation="h",
                     error_x=t.hi - t.auc, error_x_minus=t.auc - t.lo, color_discrete_map=cores_m,
                     labels={"auc": "AUC (validação cruzada)", "desfecho": "", "modelo": ""})
        fig.add_vline(x=0.5, line_dash="dash", line_color="#7a8583")
        fig.update_layout(height=520, xaxis_range=[0.4, 0.9], legend_orientation="h", legend_y=-0.15)
        st.plotly_chart(fig, width="stretch")
        cvq = SB["cv"]
        st.info(f"**Leitura:** com as 26 medidas do sinal bruto, o celular separa queda com AUC {cvq['queda']['Celular sinais brutos'][0]:.2f}, "
                f"quedas recorrentes com {cvq['recorrente']['Celular sinais brutos'][0]:.2f} e, sobretudo, **envelhecimento com "
                f"{cvq['idoso']['Celular sinais brutos'][0]:.2f}**. O IRQ-Pará ({cvq['queda']['IRQ'][0]:.2f}) continua melhor para "
                "identificar quem cai, e somar o celular a ele não aumenta o acerto.")

        st.markdown("**Medidas novas: qual separa melhor? (recalculado com os filtros)**")
        nomes_f = {"rms_ap": "RMS AP", "rms_ml": "RMS ML", "rms_r": "RMS resultante", "area95": "Área elipse 95%",
                   "range_ap": "Amplitude AP", "jerk_ap": "Jerk AP", "jerk_ml": "Jerk ML", "path_vel": "Velocidade média",
                   "f50_ap": "Freq. mediana AP", "f95_ap": "Freq. 95% AP", "fcent_ap": "Freq. centroide AP",
                   "sampen_ap": "Entropia amostral AP", "sampen_ml": "Entropia amostral ML", "spent_ap": "Entropia espectral AP",
                   "pw_b1_ap": "Potência 0,1–0,5 Hz AP", "pw_b3_ap": "Potência 2–10 Hz AP"}
        c1, c2 = st.columns([1, 2])
        yr = c1.selectbox("Desfecho", list(SENS_OUT), format_func=SENS_OUT.get, key="r_out")
        cnd = c1.radio("Condição", ["oa", "of", "rom"], horizontal=True, key="r_cond",
                       format_func={"oa": "Olhos abertos", "of": "Olhos fechados", "rom": "Razão OF/OA"}.get)
        dR = df.copy(); dR["idoso"] = (dR.idade >= 60).astype(float).where(dR.idade.notna())
        lin = []
        for k_, nm in nomes_f.items():
            col = f"r_{k_}_{cnd}"
            if col not in dR.columns:
                continue
            x = dR[[col, yr]].dropna()
            if x[yr].nunique() < 2 or x[yr].sum() < 5:
                continue
            a, p = auc_mw(x[col].values, x[yr].values)
            lin.append(dict(Medida=nm, AUC=a, p=p, n=len(x), sentido="maior em quem tem" if a >= 0.5 else "menor em quem tem",
                            forca=abs(a - 0.5)))
        Tm = pd.DataFrame(lin)
        if not Tm.empty:
            Tm = Tm.sort_values("forca")
            fig = px.bar(Tm, x="AUC", y="Medida", color="sentido", orientation="h", hover_data={"p": ":.3f", "n": True},
                         color_discrete_map={"maior em quem tem": "#d64541", "menor em quem tem": "#2a78d6"},
                         labels={"sentido": ""})
            fig.add_vline(x=0.5, line_dash="dash", line_color="#7a8583")
            fig.update_layout(height=520, xaxis_range=[0.3, 0.75], legend_orientation="h")
            c2.plotly_chart(fig, width="stretch")
        c1.markdown("""
**Destaque: entropia amostral.** Mede o quanto o balanço é irregular e adaptável. Valores **menores** (balanço mais
rígido e repetitivo) aparecem em quem caiu, nas diabéticas e nas mais velhas, o padrão de "perda de complexidade" descrito
na literatura de envelhecimento. É uma das medidas mais repetíveis (ICC ≈ 0,8) e não existia na planilha.

**Jerk** mede a brusquidão das correções do corpo; **velocidade média**, quanto o corpo se desloca por segundo.
""")

        st.markdown("**E no perfil de risco?**")
        sc = SB["sub_cv"]
        t = pd.DataFrame([{"Subgrupo": k_, "AUC (sinais brutos, validação cruzada)": v} for k_, v in sc.items()])
        t["grupo"] = np.where(t.Subgrupo.str.startswith(("Com ", "≥", "IRQ ≥")), "Perfil de risco",
                              np.where(t.Subgrupo == "Todas", "Todas", "Sem o fator"))
        fig = px.bar(t, x="AUC (sinais brutos, validação cruzada)", y="Subgrupo", color="grupo", orientation="h",
                     color_discrete_map={"Perfil de risco": "#d64541", "Sem o fator": "#3a9d63", "Todas": "#4c5856"},
                     labels={"grupo": ""})
        fig.add_vline(x=0.5, line_dash="dash", line_color="#7a8583")
        fig.update_layout(height=420, xaxis_range=[0.4, 0.75], legend_orientation="h")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"O sinal funciona claramente melhor nas mulheres que **relatam dificuldade de equilíbrio** "
                   f"(AUC {sc.get('Com dif. equilíbrio', float('nan')):.2f} contra {sc.get('Sem dif. equilíbrio', float('nan')):.2f}), "
                   "confirmando com os sinais brutos a tendência vista antes. Para diabetes e multimorbidade a vantagem não se confirmou.")

    st.markdown("#### 8. Conclusões sobre o uso do celular")
    st.markdown(f"""
- **O sinal é real:** as medidas do celular variam com idade, diabetes (sobretudo de olhos fechados), renda e histórico de quedas, na direção esperada.
- **Mas, sozinho, discrimina pouco no nível individual:** AUC ≈ 0,58–0,60 para queda; as perguntas simples do questionário vão melhor e o celular não acrescenta ao questionário.
- **Melhores candidatas:** amplitude e energia de baixa/média frequência no eixo antero-posterior, de olhos abertos.
- **As 24 medidas são muito redundantes:** o 1º componente principal resume {val['pca'][0]*100:.0f}% da informação e 4 componentes, {sum(val['pca'][:4])*100:.0f}%.
- **Repetibilidade boa:** com os sinais brutos, duas gravações da mesma mulher concordam com ICC ≈ 0,8 nas medidas de amplitude (seção 7). A baixa correlação entre olhos abertos e fechados reflete condições diferentes, não ruído.
- **Para ganhar poder individual:** padronizar a duração (60 s) e a orientação do aparelho em todos os municípios; tarefas mais desafiadoras (pés juntos, tandem, apoio unipodal, dupla tarefa); padronizar piso, posição e aparelho; registrar IMC para corrigir o amortecimento; e acompanhar as participantes prospectivamente (quedas nos 12 meses **seguintes**), que é o teste definitivo de predição.
""")

# ================================================================ CRUZAMENTOS
with abas[8]:
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
    st.dataframe(val.style.format(precision=prec, na_rep="–", decimal=",").background_gradient(cmap="Oranges", axis=None),
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
with abas[9]:
    st.subheader("Como os dados foram tratados")
    mac = df_all.drop_duplicates("mun").groupby("macro_lbl").apply(
        lambda g: f"**{g.name}** ({', '.join(sorted(g.reg.unique()))}): {', '.join(sorted(g.mun))}")
    st.markdown(f"""
**Fonte:** {len(df_all)} formulários de {df_all.mun.nunique()} municípios. Amplia o relatório técnico parcial de agosto/2026
(11 municípios, 626 participantes). Nomes, telefones e endereços foram removidos.

**Estabilometria:** os valores vêm do **sinal bruto do celular reprocessado** (reamostragem a 100 Hz, correção da orientação
pela gravidade, filtro 0,1–10 Hz) para {(df_all.fonte_oa=="Sinal bruto reprocessado").sum()} participantes. Para as
{(df_all.fonte_oa=="Planilha original").sum()} sem arquivo bruto válido (principalmente Santarém, gravada com protocolo de 30 s),
usa-se o valor original da planilha convertido para a mesma escala por calibração log-linear.

**Macrorregiões de Saúde** (SESPA/CIB-PA, 13 regiões em 4 macros):
""" + "\n".join(f"- {x}" for x in mac) + """

**Variáveis derivadas**
- Renda em salários mínimos (R$ 1.518). Bolsa Família isolado = 0,4 SM; “menos de 1 SM” = 0,6.
- Comorbidades específicas por palavras-chave no campo “qual” e nos nomes de medicamentos (ex.: losartana → hipertensão).
- Pressão aferida: elevada ≥130/80; hipertensiva ≥140/90. Cintura >88 cm = obesidade abdominal; panturrilha <31 cm = baixa massa muscular.
- **Índice de vulnerabilidade a quedas** (exploratório, não validado): soma de 12 marcadores (idade ≥60, queda prévia,
  dificuldade de equilíbrio, vertigem, auxílio para caminhar, medicação que altera equilíbrio, obstáculos em casa, diabetes,
  doença visual, doença neurológica, ≥2 medicamentos, oscilação OA no quartil superior). Baixa 0–1, moderada 2–3, alta ≥4.

**IRQ-Pará (Índice de Risco de Quedas):** regressão logística com os itens do questionário; os coeficientes foram
divididos pelo menor efeito relevante e arredondados para pontos inteiros (idade 40–59 = 2, ≥60 = 3, diabetes = 2,
dificuldade de equilíbrio = 2, obstáculos em casa = 2, vertigem = 1, auxílio para caminhar = 3, dificuldade com cores = 2).
Desempenho por validação cruzada 5×20 (pontos recalculados em cada repetição).

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
                       df_all.to_csv(index=False).encode("utf-8-sig"),
                       "dados_quedas.csv", "text/csv")
