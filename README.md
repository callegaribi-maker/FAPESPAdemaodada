# Observatório de Quedas · Pará

Painel interativo (Streamlit) do projeto **"De mãos dadas para não deixar cair: o perfil do equilíbrio estático e o risco de quedas das mulheres paraenses"** — LEMOH/UFPA · FAPESPA, Termo de Outorga 105/2024.

- 1.093 participantes · 22 municípios · 4 Macrorregiões de Saúde do Pará
- Abas: Panorama (mapa), Municípios, Ciclo de vida, Fatores de risco (RP bruta e regressão logística), Equilíbrio postural (estabilometria OA/OF), Cruzamentos livres e Método
- Filtros por macrorregião, município, faixa etária, cor/raça, escolaridade, renda e condição de saúde
- Dados lidos de `Planilha_Unica_22Municipios_anonimizada.xlsx` (mesma estrutura da planilha do projeto, com nome trocado pelo ID e telefone/endereço apagados), processados por `preparar_dados.py`
- Estabilometria: valores do sinal bruto do celular reprocessado; quem não tem arquivo bruto válido mantém o valor original, convertido por `calibracao_estabilometria.json`

## Arquivos do repositório
`app.py`, `preparar_dados.py`, `Planilha_Unica_22Municipios_anonimizada.xlsx`, `calibracao_estabilometria.json`, `sensor_validacao.json`, `sinais_brutos.json`, `pa_municipios.geojson`, `requirements.txt`, `.streamlit/config.toml`

## Rodar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar e gerar o link (Streamlit Community Cloud, gratuito)
1. Envie esta pasta para um repositório no GitHub (pode ser privado).
2. Acesse https://share.streamlit.io, entre com a conta do GitHub e clique em **Create app**.
3. Escolha o repositório, branch `main` e arquivo `app.py`. Clique em **Deploy**.
4. O link fica no formato `https://<nome-escolhido>.streamlit.app`.

Para restringir o acesso, em **Settings → Sharing** do app deixe-o privado e convide os e-mails autorizados.

Coordenação: Profa. Bianca Callegari — Laboratório de Estudos da Motricidade Humana (LEMOH/UFPA).
