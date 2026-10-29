"""Lê a planilha única dos 22 municípios e prepara as variáveis do Observatório de Quedas.

Estabilometria: para quem tem o sinal bruto do celular reprocessado, a planilha já traz os valores novos;
para quem só tem o valor original, ele é convertido para a escala do sinal bruto por calibração log-linear
(calibracao_estabilometria.json), deixando todas as participantes comparáveis. A coluna
"Fonte da estabilometria OA/OF" informa a origem de cada valor.
"""
import json
import os
import warnings
import re
import unicodedata

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

PLANILHAS = ["Planilha_Unica_22Municipios_anonimizada.xlsx", "Planilha_Unica_22Municipios_sinais_brutos.xlsx",
             "Planilha_Unica_22Municipios.xlsx"]

MUN = {'AUGUSTO CORREA': ('Augusto Corrêa', 'II', 'Rio Caetés', -1.0219, -46.6347),
       'CAPANEMA': ('Capanema', 'II', 'Rio Caetés', -1.1969, -47.1806),
       'BRAGANCA': ('Bragança', 'II', 'Rio Caetés', -1.0536, -46.7656),
       'TRACUATEUA': ('Tracuateua', 'II', 'Rio Caetés', -1.0767, -46.9031),
       'CACHOEIRA DO PIRIA': ('Cachoeira do Piriá', 'II', 'Rio Caetés', -1.7597, -46.5458),
       'CASTANHAL': ('Castanhal', 'II', 'Metropolitana III', -1.2939, -47.9264),
       'BELEM': ('Belém', 'I', 'Metropolitana I', -1.4558, -48.4902),
       'ANANINDEUA': ('Ananindeua', 'I', 'Metropolitana I', -1.3656, -48.3722),
       'MARITUBA': ('Marituba', 'I', 'Metropolitana I', -1.3550, -48.3419),
       'OEIRAS DO PARA': ('Oeiras do Pará', 'I', 'Tocantins', -2.0033, -49.8547),
       'ALTAMIRA': ('Altamira', 'III', 'Xingu', -3.2033, -52.2064),
       'VITORIA DO XINGU': ('Vitória do Xingu', 'III', 'Xingu', -2.8792, -52.0089),
       'JURUTI': ('Juruti', 'III', 'Baixo Amazonas', -2.1522, -56.0922),
       'OBIDOS': ('Óbidos', 'III', 'Baixo Amazonas', -1.9178, -55.5181),
       'ORIXIMINA': ('Oriximiná', 'III', 'Baixo Amazonas', -1.7656, -55.8661),
       'SANTAREM': ('Santarém', 'III', 'Baixo Amazonas', -2.4431, -54.7083),
       'ITAITUBA': ('Itaituba', 'III', 'Tapajós', -4.2761, -55.9836),
       'ITUPIRANGA': ('Itupiranga', 'IV', 'Lago de Tucuruí', -5.1347, -49.3269),
       'TUCURUI': ('Tucuruí', 'IV', 'Lago de Tucuruí', -3.7661, -49.6725),
       'NOVA IPIXUNA': ('Nova Ipixuna', 'IV', 'Lago de Tucuruí', -4.9172, -49.0822),
       'MARABA': ('Marabá', 'IV', 'Carajás', -5.3686, -49.1178),
       'PARAUAPEBAS': ('Parauapebas', 'IV', 'Carajás', -6.0675, -49.9022)}

# 12 medidas da planilha -> nome curto usado no app
EST = {'RMS AP': 'RMS AP', 'RMS ML': 'RMS ML', 'Desvio total': 'Desvio total', 'Area elipse 95%': 'Área elipse',
       'Frequencia mediana AP (Hz)': 'Freq. mediana AP', 'Frequencia mediana ML (Hz)': 'Freq. mediana ML',
       'Energia baixa AP (0-0.5 Hz)': 'Energia baixa AP', 'Energia media AP (0.5-2 Hz)': 'Energia média AP',
       'Energia alta AP (2-8 Hz)': 'Energia alta AP', 'Energia baixa ML (0-0.5 Hz)': 'Energia baixa ML',
       'Energia media ML (0.5-2 Hz)': 'Energia média ML', 'Energia alta ML (2-8 Hz)': 'Energia alta ML'}
# medidas novas do sinal bruto (colunas acrescentadas na planilha) -> chave usada no app
NOVAS = {'RMS resultante': 'rms_r', 'RMS vertical': 'rms_vt', 'Amplitude AP': 'range_ap', 'Amplitude ML': 'range_ml',
         'Jerk AP': 'jerk_ap', 'Jerk ML': 'jerk_ml', 'Velocidade média': 'path_vel', 'Freq. 95% AP (Hz)': 'f95_ap',
         'Freq. 95% ML (Hz)': 'f95_ml', 'Freq. centroide AP (Hz)': 'fcent_ap', 'Freq. centroide ML (Hz)': 'fcent_ml',
         'Entropia espectral AP': 'spent_ap', 'Entropia espectral ML': 'spent_ml', 'Entropia amostral AP': 'sampen_ap',
         'Entropia amostral ML': 'sampen_ml'}


def norm(s):
    if pd.isna(s):
        return ''
    s = str(s).strip().lower()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def yn(v):
    s = norm(v)
    if s.startswith('sim') or s.startswith('as vezes') or s == 'amitil':
        return 1
    if s.startswith('nao') or s == 'nn':
        return 0
    return None


def num(v):
    s = norm(v).replace(',', '.')
    m = re.search(r'\d+(\.\d+)?', s)
    return float(m.group()) if m else None


def esc(v):
    s = norm(v)
    if not s or 'sem registro' in s:
        return None
    if 'pos' in s:
        return 7
    if 'nao alfab' in s:
        return 0
    if s.startswith('sup') or 'superior' in s:
        return 5 if 'incomp' in s else 6
    if s.startswith('med') or 'medio' in s:
        return 3 if 'incomp' in s else 4
    if s.startswith('fund'):
        if 'med' in s:
            return 4
        return 1 if 'incomp' in s else 2
    return None


SM = 1518
WORDS = {'um': 1, 'uma': 1, 'dois': 2, 'duas': 2, 'tres': 3, 'quatro': 4, 'cinco': 5, 'seis': 6, 'meio': 0.5}


def renda(v):
    s = norm(v)
    if not s or 'sem registro' in s or s in ('nao', 'nao sabe', 'nao informou'):
        return None
    if 'menos de 1' in s or 'menos de um' in s:
        return 0.6
    m = re.search(r'mais de (\d+)', s)
    if m:
        return float(m.group(1)) + 0.5
    if 'salario' in s or re.fullmatch(r'\d+(\.\d)?\s*sm', s):
        m = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:a\s*(\d+))?\s*(?:salario|sm)', s)
        if m:
            a = float(m.group(1).replace(',', '.'))
            if m.group(2):
                a = (a + float(m.group(2))) / 2
            if a < 20:
                return a
        for w, n in WORDS.items():
            if re.search(r'\b' + w + r'\b', s):
                return float(n)
    m = re.search(r'(\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?', s)
    if m:
        val = float(m.group(1).replace('.', ''))
        if val >= 100:
            return round(val / SM, 2)
        if val < 20 and 'bolsa' not in s:
            return val
    if 'bolsa' in s or 'beneficio' in s or 'bpc' in s or 'aposent' in s or 'pensao' in s:
        if 'aposent' in s or 'bpc' in s or 'pensao' in s:
            return 1.0
        return 0.4
    return None


def alt(v):
    x = num(v)
    if x is None:
        return None
    if x > 3:
        x = x / 100
    return x if 1.2 < x < 2.0 else None


def pa(v):
    s = norm(v)
    m = re.search(r'(\d{2,3})\s*[x/]\s*(\d{1,3})', s)
    if not m:
        return (None, None)
    a, b = int(m.group(1)), int(m.group(2))
    if a < 30:
        a *= 10
    if b < 20:
        b *= 10
    if not (70 <= a <= 260 and 40 <= b <= 150):
        return (None, None)
    return (a, b)


def sono(v):
    s = norm(v)
    if not s or 'sem' in s:
        return None
    nums = [float(x) for x in re.findall(r'\d+', s)]
    if not nums:
        return None
    if 'min' in s and len(nums) >= 2:
        x = nums[0] + nums[1] / 60
    elif re.search(r'\d+\s*(a|-|ou)\s*\d+', s) and len(nums) >= 2:
        x = (nums[0] + nums[1]) / 2
    else:
        x = nums[0]
    return x if 2 <= x <= 14 else None


def kw(series, pats):
    return series.str.contains('|'.join(pats), regex=True).astype(int)


def ler_planilha(caminho):
    x = pd.ExcelFile(caminho)
    frames = []
    for s in x.sheet_names:
        d = x.parse(s, header=None)
        cols = [re.sub(r'(\d)Hz\)', r'\1 Hz)', str(c)).replace('Área', 'Area').replace('Frequência', 'Frequencia')
                .replace('Energia média', 'Energia media') if isinstance(c, str) else c for c in d.iloc[1]]
        d = d.iloc[2:].copy()
        d.columns = cols
        d = d[d['ID do formulário'].notna()]
        d = d.loc[:, ~pd.Index(d.columns).duplicated()]
        d['Municipio_sheet'] = s
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def encontrar_planilha(pasta="."):
    for nome in PLANILHAS:
        p = os.path.join(pasta, nome)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Nenhuma planilha encontrada. Coloque 'Planilha_Unica_22Municipios_anonimizada.xlsx' na pasta do app.")


def preparar(caminho=None, calibracao="calibracao_estabilometria.json"):
    caminho = caminho or encontrar_planilha()
    df = ler_planilha(caminho)
    o = pd.DataFrame()
    o['id'] = df['ID do formulário'].astype(str).values
    o['mun'] = df.Municipio_sheet.map(lambda k: MUN[k][0]).values
    o['macro'] = df.Municipio_sheet.map(lambda k: MUN[k][1]).values
    o['reg'] = df.Municipio_sheet.map(lambda k: MUN[k][2]).values
    age = df['Idade'].map(num)
    o['idade'] = age.where((age >= 15) & (age <= 105)).values
    c = df['Cor'].map(norm)
    o['cor'] = c.map({'parda': 'Parda', 'branca': 'Branca', 'preta': 'Preta', 'indigena': 'Indígena', 'amarela': 'Amarela'}).values
    ec = df['Estado civil'].map(norm)
    o['civil'] = np.select([ec.str.startswith('casad') | ec.str.contains('uniao'), ec.str.startswith('solteir'),
                            ec.str.startswith('viuv'), ec.str.startswith('divorc'), ec.str.startswith('outro')],
                           ['Casada/união', 'Solteira', 'Viúva', 'Divorciada', 'Outro'], None)
    o['esc'] = df['Escolaridade'].map(esc).values
    o['renda'] = df['Renda familiar (média de salários)'].map(renda).values
    h = df['Altura (m)'].map(alt)
    w = df['Peso (kg)'].map(num)
    w = w.where((w > 30) & (w < 200))
    bmi = w / (h ** 2)
    o['imc'] = bmi.where((bmi >= 10) & (bmi <= 70)).round(1).values
    cc = df['Circunferência da cintura (cm)'].map(num)
    o['cintura'] = cc.where((cc > 45) & (cc < 180)).values
    cp = df['Circunferência da panturrilha (cm)'].map(num)
    o['panturrilha'] = cp.where((cp > 15) & (cp < 70)).values
    p = df['Pressão arterial'].map(pa)
    o['pas'] = p.map(lambda t: t[0]).values
    o['pad'] = p.map(lambda t: t[1]).values
    o['sono'] = df['Horas de sono'].map(sono).values
    binmap = {'cores': 'Dificuldade em discriminar cores', 'colirio': 'Faz uso de colírio com frequência',
              'famcego': 'Familiar cego ou com grave perda visual', 'dvis': 'Doenças visuais confirmadas',
              'dsis': 'Doenças sistêmicas crônicas', 'dinf': 'Doenças infecciosas', 'dneu': 'Doenças neurológicas',
              'cirurgia': 'Cirurgia realizada', 'med': 'Uso de medicação', 'fumo': 'Consumo de cigarro',
              'alcool': 'Consumo de álcool', 'drogas': 'Consumo de drogas de abuso', 'agrotox': 'Exposição a agrotóxico',
              'quimicos': 'Exposição a outros agentes químicos', 'fresco': 'Alimento fresco', 'enlatado': 'Alimento enlatado',
              'frutas': 'Frutas e verduras', 'ativ': 'Realiza atividade física', 'alergia_al': 'Possui alergia alimentar',
              'suplem': 'Toma suplementação', 'queda': 'Teve queda nos últimos 12 meses', 'lesao': 'Queda resultou em lesão',
              'dequil': 'Dificuldade de equilíbrio ou coordenação', 'auxilio': 'Usa auxílio para caminhar',
              'vertigem': 'Tem sentido vertigem ultimamente', 'medequil': 'Usa medicação que altera equilíbrio',
              'obst': 'Existem obstáculos com risco de queda em casa',
              'orient': 'Já recebeu orientação profissional sobre risco de queda'}
    for k, cn in binmap.items():
        o[k] = df[cn].map(yn).values
    o.loc[o.queda == 0, 'lesao'] = 0
    qv = df['Queda - quantas vezes']
    nq = qv.map(lambda v: None if not re.search(r'\d', norm(v)) else num(v))
    nq = nq.where(nq <= 30)
    nq[qv.map(norm).isin(['uma', 'uma vez'])] = 1
    nq = nq.values
    o['nquedas'] = np.where(o.queda == 0, 0, np.where(o.queda == 1, nq, np.nan))
    o.loc[(o.queda == 1) & (o.nquedas.isna() | (o.nquedas == 0)), 'nquedas'] = np.nan
    o['recorrente'] = np.where(o.queda == 0, 0, np.where(o.nquedas >= 2, 1, np.where(o.nquedas == 1, 0, np.nan)))
    sq = df['Doenças sistêmicas - qual'].map(norm)
    sis = sq + ' | ' + df['Medicação - listar todos'].map(norm)
    o['hipert'] = kw(sis, [r'hipert', r'\bhas\b', r'pressao alta', r'losart', r'captopril', r'enalapril', r'anlodip',
                           r'hidroclorot', r'atenolol', r'valsart', r'ramipril', r'olmesart', r'propranol']).values
    o['diabetes'] = kw(sis, [r'diabet', r'\bdm\b', r'metformin', r'glibencl', r'glimepir', r'insulin', r'glicazid', r'gliclazid']).values
    o['osteoart'] = kw(sq, [r'artros', r'artrit', r'reumat', r'osteopor', r'osteopen', r'hernia de disco', r'bico de papagaio',
                            r'coluna', r'gota', r'lombalg']).values
    o['fibro'] = kw(sq, [r'fibromial']).values
    o['cardio'] = kw(sq, [r'cardi', r'coracao', r'arritm', r'insuficiencia card', r'sopro', r'infarto']).values
    o['respir'] = kw(sq, [r'asma', r'bronqu', r'respirat', r'dpoc', r'rinite', r'sinusite', r'pulm']).values
    o['tireoide'] = kw(sis, [r'tireo', r'tiroid', r'levotirox', r'puran', r'hipotireo', r'hipertireo']).values
    o['depans'] = kw(sis + ' | ' + df['Doenças neurológicas - qual'].map(norm),
                     [r'depress', r'ansiedad', r'fluoxet', r'sertral', r'escitalo', r'amitrip', r'clonazep', r'rivotril',
                      r'bipolar', r'paroxet', r'venlafax', r'diazep', r'alprazol']).values
    o['colest'] = kw(sis, [r'colesterol', r'dislipid', r'sinvast', r'rosuvast', r'atorvast', r'triglic']).values
    vq = df['Doenças visuais - qual'].map(norm)
    o['catarata'] = kw(vq, ['catarat']).values
    o['glaucoma'] = kw(vq, ['glaucom', 'pressao ocular', 'pressao alta ocular']).values
    o['refrativo'] = kw(vq, ['miop', 'astigm', 'hipermet', 'presbi', 'ceratoc']).values
    o['avc'] = kw(df['Doenças neurológicas - qual'].map(norm), [r'\bave\b', r'\bavc\b', r'\bait\b']).values
    o.loc[o.dsis == 0, ['osteoart', 'fibro', 'cardio', 'respir']] = 0
    iq = df['Doenças infecciosas - qual'].map(norm)
    o['malaria'] = kw(iq, ['malari']).values
    o['dengue'] = kw(iq, ['dengue', 'chikung', 'zika']).values
    ml = df['Medicação - listar todos'].values

    def nmed(med, lista):
        if med == 0:
            return 0
        s = norm(lista)
        if not s or s in ('sem registro', 'nao'):
            return None
        s = re.sub(r'\(.*?\)', '', s)
        parts = [p for p in re.split(r',|;|\+|\be\b|/|\n', s) if len(p.strip()) > 2]
        return len(parts) if parts else 1
    o['nmed'] = [nmed(m, l) for m, l in zip(o.med.values, ml)]
    o['nmed'] = pd.to_numeric(o['nmed'], errors='coerce')
    o['polifarm'] = np.where(o.nmed.isna(), np.nan, (o.nmed >= 5).astype(float))

    # ---------- estabilometria (sinal bruto, com calibração dos valores originais restantes)
    cal = {}
    if calibracao and os.path.exists(calibracao):
        cal = json.load(open(calibracao, encoding='utf-8'))
    for cnd in ['OA', 'OF']:
        fonte_col = f'Fonte da estabilometria {cnd}'
        fonte = df[fonte_col] if fonte_col in df.columns else pd.Series('Planilha original', index=df.index)
        o[f'fonte_{cnd.lower()}'] = fonte.values
        orig = (fonte == 'Planilha original').values
        for col, lab in EST.items():
            v = pd.to_numeric(df.get(f'{col} ({cnd})'), errors='coerce').values.astype(float)
            c = cal.get(f'{col} ({cnd})')
            if c and orig.any():
                with np.errstate(divide='ignore', invalid='ignore'):
                    conv = np.exp(c['a'] + c['b'] * np.log(v))
                v = np.where(orig & (v > 0), conv, v)
            o[f's_{lab} ({cnd})'] = v
        # medidas novas (só existem para quem tem sinal bruto)
        for col, key in NOVAS.items():
            o[f'r_{key}_{cnd.lower()}'] = pd.to_numeric(df.get(f'{col} ({cnd})'), errors='coerce').values
        bruto = (fonte == 'Sinal bruto reprocessado').values
        o[f'r_rms_ap_{cnd.lower()}'] = np.where(bruto, o[f's_RMS AP ({cnd})'], np.nan)
        o[f'r_rms_ml_{cnd.lower()}'] = np.where(bruto, o[f's_RMS ML ({cnd})'], np.nan)
        o[f'r_area95_{cnd.lower()}'] = np.where(bruto, o[f's_Área elipse ({cnd})'], np.nan)
        o[f'r_f50_ap_{cnd.lower()}'] = np.where(bruto, o[f's_Freq. mediana AP ({cnd})'], np.nan)
        o[f'r_f50_ml_{cnd.lower()}'] = np.where(bruto, o[f's_Freq. mediana ML ({cnd})'], np.nan)
        tot = o[f's_Energia baixa AP ({cnd})'] + o[f's_Energia média AP ({cnd})'] + o[f's_Energia alta AP ({cnd})']
        o[f'r_pw_b1_ap_{cnd.lower()}'] = np.where(bruto, o[f's_Energia baixa AP ({cnd})'] / tot, np.nan)
        o[f'r_pw_b3_ap_{cnd.lower()}'] = np.where(bruto, o[f's_Energia alta AP ({cnd})'] / tot, np.nan)
    for key in ['rms_ap', 'rms_ml', 'rms_r', 'area95', 'jerk_ap', 'jerk_ml', 'path_vel', 'sampen_ap', 'sampen_ml',
                'f50_ap', 'f50_ml', 'range_ap', 'f95_ap', 'fcent_ap', 'spent_ap', 'pw_b1_ap', 'pw_b3_ap']:
        with np.errstate(divide='ignore', invalid='ignore'):
            o[f'r_{key}_rom'] = o[f'r_{key}_of'] / o[f'r_{key}_oa']
    if 'Nº de gravações do celular' in df.columns:
        o['r_n_rec'] = pd.to_numeric(df['Nº de gravações do celular'], errors='coerce').values
        o['r_tilt'] = pd.to_numeric(df['Inclinação do celular (graus)'], errors='coerce').values
    # aliases usados nas análises gerais
    for k, lab in [('rmsap', 'RMS AP'), ('rmsml', 'RMS ML'), ('dt', 'Desvio total'), ('area', 'Área elipse'),
                   ('fap', 'Freq. mediana AP'), ('fml', 'Freq. mediana ML')]:
        for cnd in ['OA', 'OF']:
            o[f'{k}_{cnd.lower()}'] = o[f's_{lab} ({cnd})']
    o['romberg'] = (o.area_of / o.area_oa).where((o.area_oa > 0) & (o.area_of > 0))
    return o
