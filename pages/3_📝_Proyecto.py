import streamlit as st

with st.sidebar:
    st.image("logo.png")
    

colT1,colT2 = st.columns([1,3])
colT2.title("PROYECTO")

st.write('El proyecto realizado consiste en el análisis de la influencia de las redes sociales en la salud mental de las personas y realizar una comparación para detectar las diferencias entre los usuarios de las redes sociales más populares.')
