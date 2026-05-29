"""
app_dashboard.py — Dashboard Jurimétrico ANLA
Etapa 4B del prototipo de tesis.
"""
import os
import sys
import json
import glob
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------------------------
# Configuración (RUTAS CORREGIDAS)
# -------------------------------------------------------------------------
BASE_DIR        = "/content/drive/MyDrive/4_Semestre_4/PLN/Trabajo final"
SEGMENT_DIR     = os.path.join(BASE_DIR, "Segmentación")
RETRIEVAL_DIR   = os.path.join(BASE_DIR, "Retrieval")
CHROMA_DIR      = os.path.join(RETRIEVAL_DIR, "chroma_db")
GENERACION_DIR  = os.path.join(BASE_DIR, "Generacion")

sys.path.append(RETRIEVAL_DIR)

st.set_page_config(
    page_title="Dashboard Jurimétrico ANLA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Carga de datos (cacheada)
# -------------------------------------------------------------------------
@st.cache_data
def cargar_corpus_chunks():
    ruta = os.path.join(SEGMENT_DIR, "corpus_chunks_consolidado.json")
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def cargar_respuestas():
    last = os.path.join(GENERACION_DIR, "last_respuestas.json")
    if os.path.exists(last):
        with open(last, "r", encoding="utf-8") as f:
            ruta_actual = json.load(f)["ruta_actual"]
        ruta_full = os.path.join(GENERACION_DIR, ruta_actual)
    else:
        archivos = sorted(glob.glob(os.path.join(GENERACION_DIR, "respuestas_*.json")))
        if not archivos:
            return None
        ruta_full = archivos[-1]
    with open(ruta_full, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_archivo"] = os.path.basename(ruta_full)
    return data

@st.cache_data
def cargar_estadisticas_json():
    datos = []
    rutas = glob.glob(os.path.join(SEGMENT_DIR, "*.json"))
    
    for ruta in rutas:
        if "corpus_chunks_consolidado" in ruta:
            continue
        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                
                # Búsqueda robusta: busca en la raíz o dentro de "metadata"
                empresa = doc.get('empresa') or doc.get('metadata', {}).get('empresa')
                fecha = doc.get('fecha_documento') or doc.get('metadata', {}).get('fecha_documento')
                materias = doc.get('materias') or doc.get('metadata', {}).get('materias', [])
                
                if empresa:
                    datos.append({
                        'Empresa': empresa,
                        'Fecha': fecha,
                        'Materias': materias
                    })
        except Exception:
            continue
            
    df = pd.DataFrame(datos)
    if not df.empty:
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df['Año'] = df['Fecha'].dt.year
    return df

@st.cache_resource
def cargar_retriever():
    from retriever_utils import Retriever
    return Retriever(chroma_dir=CHROMA_DIR)

# -------------------------------------------------------------------------
# Sidebar — navegación
# -------------------------------------------------------------------------
st.sidebar.title("⚖️ Dashboard Jurimétrico")
st.sidebar.markdown("**Autos de Apertura ANLA**")
st.sidebar.markdown("_Sector hidrocarburos_")
st.sidebar.markdown("---")

vista = st.sidebar.radio(
    "Selecciona una vista",
    [
        "📊 Vista General",
        "🤖 Comparación de Modelos",
        "🔍 Consulta Interactiva",
        "📈 Análisis Jurimétrico",
        "ℹ️ Acerca del Prototipo",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>⚠ <b>Advertencia:</b> las respuestas son generadas por IA. No constituyen asesoría jurídica.</small>",
    unsafe_allow_html=True,
)

corpus = cargar_corpus_chunks()
respuestas_data = cargar_respuestas()
df_chunks = pd.DataFrame(corpus["chunks"])

# =========================================================================
# VISTA 1: VISTA GENERAL
# =========================================================================
if vista == "📊 Vista General":
    st.title("📊 Vista General del Corpus")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Documentos", corpus["total_documentos"])
    c2.metric("🧩 Chunks totales", corpus["total_chunks"])
    c3.metric("📝 Chunks de texto", corpus["chunks_texto"])
    c4.metric("📊 Chunks de tabla", corpus["chunks_tabla"])
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Chunks por documento")
        df_por_doc = df_chunks.groupby(["filename", "tipo_chunk"]).size().reset_index(name="chunks")
        fig = px.bar(df_por_doc, x="filename", y="chunks", color="tipo_chunk", barmode="stack", color_discrete_map={"texto": "#2E74B5", "tabla": "#E67E22"})
        fig.update_layout(xaxis_tickangle=-30, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Distribución de tamaño de chunks (tokens)")
        fig = px.histogram(df_chunks, x="n_tokens", color="tipo_chunk", nbins=30, color_discrete_map={"texto": "#2E74B5", "tabla": "#E67E22"})
        fig.update_layout(height=400, barmode="overlay")
        fig.update_traces(opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# VISTA 2: COMPARACIÓN DE MODELOS
# =========================================================================
elif vista == "🤖 Comparación de Modelos":
    st.title("🤖 Comparación Llama 3 vs Mistral")
    if respuestas_data is None:
        st.warning("No se encontraron respuestas.")
        st.stop()

    respuestas = respuestas_data["respuestas"]
    df_resp = pd.DataFrame([{ "consulta_id": r["consulta_id"], "modelo": r["modelo"], "tiempo_generacion_s": r["tiempo_generacion_s"], "longitud_chars": len(r["respuesta"]) } for r in respuestas])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df_resp.groupby("modelo")["tiempo_generacion_s"].mean().reset_index(), x="modelo", y="tiempo_generacion_s", color="modelo", text_auto=".2f", title="⏱ Tiempo de generación promedio")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(df_resp.groupby("modelo")["longitud_chars"].mean().reset_index(), x="modelo", y="longitud_chars", color="modelo", text_auto=".0f", title="📝 Longitud de respuesta promedio (chars)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    consulta_sel = st.selectbox("Selecciona una consulta", sorted({r["consulta_id"] for r in respuestas}))
    respuestas_consulta = [r for r in respuestas if r["consulta_id"] == consulta_sel]
    if respuestas_consulta:
        st.info(f"**Pregunta:** {respuestas_consulta[0]['query']}")
        cols = st.columns(len(respuestas_consulta))
        for i, r in enumerate(respuestas_consulta):
            with cols[i]:
                st.markdown(f"### {r['modelo']}")
                st.write(r["respuesta"])

# =========================================================================
# VISTA 3: CONSULTA INTERACTIVA
# =========================================================================
elif vista == "🔍 Consulta Interactiva":
    st.title("🔍 Consulta Interactiva")
    query = st.text_input("Escribe tu consulta jurídica:")

    if query:
        with st.spinner("Cargando motor de retrieval..."):
            retriever = cargar_retriever()
        with st.spinner("Recuperando fragmentos..."):
            resultados = retriever.buscar(query, top_k=5)
            
        if resultados:
            st.success(f"✓ {len(resultados)} fragmentos recuperados")
            for i, r in enumerate(resultados, 1):
                md = r["metadata"]
                with st.expander(f"**[{i}]** sim={r['similitud']:.3f} | {md['filename']} | numeral {md['numeral']}"):
                    st.write(r["texto"])

# =========================================================================
# VISTA 4: ANÁLISIS JURIMÉTRICO
# =========================================================================
elif vista == "📈 Análisis Jurimétrico":
    st.title("📈 Análisis Jurimétrico")
    
    tab_procesos, tab_tecnico = st.tabs(["🏛️ Expedientes (Metadatos)", "⚙️ Corpus (Chunks y Tablas)"])

    with tab_procesos:
        df_stats = cargar_estadisticas_json()
        if not df_stats.empty:
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Expedientes", len(df_stats))
            kpi2.metric("Empresas Únicas", df_stats['Empresa'].nunique())
            rango_anios = f"{int(df_stats['Año'].min())} - {int(df_stats['Año'].max())}" if pd.notna(df_stats['Año'].min()) else "N/A"
            kpi3.metric("Rango de Años", rango_anios)
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🏢 Empresas con más Procesos")
                df_empresas = df_stats['Empresa'].value_counts().reset_index().head(10)
                df_empresas.columns = ['Empresa', 'Cantidad']
                
                df_empresas['Empresa_Corta'] = df_empresas['Empresa'].apply(lambda x: x[:35] + '...' if len(x) > 35 else x)
                
                fig_emp = px.bar(
                    df_empresas, 
                    x='Cantidad', 
                    y='Empresa_Corta', 
                    orientation='h', 
                    color='Cantidad', 
                    color_continuous_scale='Blues',
                    text='Cantidad',
                    hover_data={'Empresa': True, 'Empresa_Corta': False}
                )
                
                fig_emp.update_layout(
                    yaxis={'categoryorder':'total ascending', 'title': None},
                    xaxis={'title': 'Cantidad de Procesos'},
                    coloraxis_showscale=False,
                    height=400,
                    margin=dict(l=0, r=20, t=20, b=0)
                )
                fig_emp.update_traces(textposition='outside')
                st.plotly_chart(fig_emp, use_container_width=True)

            with col2:
                st.subheader("📅 Aperturas por Año")
                df_anio = df_stats['Año'].value_counts().reset_index().sort_values('Año')
                df_anio.columns = ['Año', 'Cantidad']
                fig_anio = px.area(df_anio, x='Año', y='Cantidad', markers=True)
                fig_anio.update_xaxes(dtick=1)
                st.plotly_chart(fig_anio, use_container_width=True)

            st.markdown("---")
            st.subheader("📑 Temáticas Frecuentes")
            df_materias = df_stats.explode('Materias')
            df_materias['Materias'] = df_materias['Materias'].astype(str).str.replace('_', ' ').str.title()
            df_top_materias = df_materias['Materias'].value_counts().reset_index().head(10)
            df_top_materias.columns = ['Temática', 'Frecuencia']
            fig_mat = px.pie(df_top_materias, names='Temática', values='Frecuencia', hole=0.4)
            st.plotly_chart(fig_mat, use_container_width=True)
        else:
            st.info("No se encontraron metadatos para graficar.")

    with tab_tecnico:
        df_tablas = df_chunks[df_chunks["tipo_chunk"] == "tabla"].copy()
        df_textos = df_chunks[df_chunks["tipo_chunk"] == "texto"].copy()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Tablas indexadas", len(df_tablas))
        c2.metric("📝 Bloques de texto", len(df_textos))
        if len(df_tablas) > 0:
            c3.metric("📏 Tokens promedio (tablas)", f"{df_tablas['n_tokens'].mean():.0f}")

        if len(df_tablas) > 0:
            st.subheader("Tablas por documento")
            tablas_por_doc = df_tablas.groupby("filename").size().reset_index(name="tablas")
            fig = px.bar(tablas_por_doc, x="filename", y="tablas", color="tablas", color_continuous_scale="Oranges", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# VISTA 5: ACERCA DEL PROTOTIPO (IDÉNTICO A LA IMAGEN)
# =========================================================================
else:
    st.title("ℹ️ Acerca del Prototipo")
    
    st.subheader("Arquitectura técnica")
    st.markdown("""
    | Etapa | Componente | Tecnología |
    |---|---|---|
    | 1 | Captura y preparación | Selenium + parser PDF→JSON |
    | 2 | Segmentación + embeddings | Longformer (4096 tokens), chunks 2048 + overlap 20% |
    | 3 | Retrieval | ChromaDB (coseno) + LangChain + parent-document |
    | 4A | Generación | Llama 3 8B + Mistral 7B (4-bit) |
    | 4B | Dashboard | Streamlit + Plotly |
    """)
    
    st.subheader("Consideraciones éticas")
    st.markdown("""
    * **Fuentes públicas:** los Autos provienen de la Gaceta Ambiental de la ANLA, repositorio de acceso público.
    * **Presunción de inocencia:** un Auto de Apertura no constituye decisión sancionatoria definitiva.
    * **No automatización de decisión jurídica:** este sistema es una herramienta de apoyo, no sustituye el criterio profesional del abogado.
    * **Trazabilidad:** todas las respuestas incluyen citas al documento, numeral y página de origen.
    * **Mitigación de alucinaciones:** el prompt jurídico obliga al LLM a responder solo con base en el contexto recuperado.
    """)
    
    st.subheader("Limitaciones reconocidas")
    st.markdown("""
    * Prueba de concepto sobre 5 documentos del sector hidrocarburos
    * Longformer no está especializado en lenguaje jurídico colombiano
    * Evaluación realizada por el autor (single-annotator), sin gold-standard amplio
    """)
    
    st.subheader("Autor")
    st.markdown("**Oswaldo Salgado Gómez** — Maestría en Analítica de Datos, Universidad Central (2026) Director: Profesor Miguel Ángel Rippe Espinosa")
