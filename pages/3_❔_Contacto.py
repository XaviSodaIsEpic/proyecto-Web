import streamlit as st

colT1,colT2 = st.columns([1,3])
colT2.title("CONTACTO 📫")

contact_form = """
<form action="https://formsubmit.co/xaviven05@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder = "Nombre" required>
     <input type="email" name="email" placeholder="Email" required>
     <textarea name="message" placeholder="Escribe tu mensaje"></textarea>
     <button type="submit">Enviar</button>
</form>
"""

st.markdown(contact_form, unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style/style.css")