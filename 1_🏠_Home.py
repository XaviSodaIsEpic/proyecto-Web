import streamlit as st

st.set_page_config(
    page_title="Proyecto",
    page_icon="📊"
)

colT1,colT2 = st.columns([1,3])
colT2.title("PÁGINA PRINCIPAL")

st.write('El proyecto realizado consiste en el análisis de la influencia de las redes sociales en la salud mental de las personas y realizar una comparación para detectar las diferencias entre los usuarios de las redes sociales más populares.')

st.write('Esta preocupación nace del escaso conocimiento que tenemos sobre el uso de estas herramientas, se han popularizado tanto y a un ritmo tan acelerado que por ejemplo es difícil pensar que una persona no posea un teléfono inteligente en el que tenga instalado como mínimo redes sociales que se pueden considerar más esenciales en Europa como WhatsApp; sin embargo las consecuencias reales tanto en nuestra salud física mental todavía se están estudiando.')
st.write('Gracias al avance de la tecnología y sobre todo la relacionada con las ciencias de la información, se están realizando estudios sanitarios y sociales que pretenden ser capaces de detectar patrones de escritura que relacionen los pensamientos y comentarios en texto de los usuarios con el posible diagnóstico de una enfermedad mental tal como la depresión, anorexia, etc.')

st.write('En este contexto, el estudio se ha centrado en analizar el estado de salud de usuarios que utilizan diferentes tipos de redes sociales, las comparaciones entre ellos y sobre todo con personas que directamente no usan redes o lo hacen de forma responsable.')

st.divider()
colT1,colT2 = st.columns([1,3])
colT2.page_link("pages/2_📊_Proyecto.py", label="INICIAR TEST", icon="➡")