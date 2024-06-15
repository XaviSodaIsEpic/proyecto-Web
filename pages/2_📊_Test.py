import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

colT1,colT2 = st.columns([1,3])
colT2.title("TEST SALUD MENTAL")

with st.sidebar:
    st.image("logo.png")
    
respuestas = {}

tipo_pregunta = {
    "pregunta_1": ['ADHD'],
    "pregunta_2": ['ADHD'],
    "pregunta_3": ['Ansiedad'],
    "pregunta_4": ['ADHD'],
    "pregunta_5": ['Ansiedad'],
    "pregunta_6": ['ADHD'],
    "pregunta_7": ['Autoestima'],
    "pregunta_8": ['Autoestima'],
    "pregunta_9": ['Autoestima'],
    "pregunta_10": ['Depresion'],
    "pregunta_11": ['Depresion'],
    "pregunta_12": ['Depresion'],
}



respuestasPorEnfermadad = {}

    

with st.form("Cuestionario"): 
    st.subheader('Pregunta 1')
    pregunta_1 = st.slider("¿Con qué frecuencia utilizas las redes sociales sin un propósito específico?",1,5)
    
    st.subheader('Pregunta 2')
    pregunta_2 = st.slider("¿Con qué frecuencia te distraes con las redes sociales cuando estás ocupado haciendo algo?",1,5)
    
    st.subheader('Pregunta 3')
    pregunta_3 = st.slider("¿Te sientes inquieto si hace tiempo que no utilizas las redes sociales?",1,5)
    
    st.subheader('Pregunta 4')
    pregunta_4 = st.slider("En una escala del 1 al 5, ¿con qué facilidad te distraes?",1,5)
    
    st.subheader('Pregunta 5')
    pregunta_5 = st.slider("En una escala del 1 al 5, ¿cuánto le molestan las preocupaciones?",1,5)
    
    st.subheader('Pregunta 6')
    pregunta_6 = st.slider("¿Le resulta difícil concentrarse en las cosas?",1,5)
    
    st.subheader('Pregunta 7')
    pregunta_7 = st.slider("En una escala del 1 al 5, ¿con qué frecuencia te comparas con otras personas exitosas mediante el uso de las redes sociales?'",1,5)
    
    st.subheader('Pregunta 8')
    pregunta_8 = st.slider("Siguiendo la pregunta anterior, ¿cómo te sientes acerca de estas comparaciones, en términos generales?",1,5)
    
    st.subheader('Pregunta 9')
    pregunta_9 = st.slider("¿Con qué frecuencia buscas validación en las funciones de las redes sociales?",1,5)
    
    st.subheader('Pregunta 10')
    pregunta_10 = st.slider("¿Con qué frecuencia te sientes deprimido o decaído?",1,5)
    
    st.subheader('Pregunta 11')
    pregunta_11 = st.slider("En una escala del 1 al 5, ¿con qué frecuencia fluctúa su interés en las actividades diarias?",1,5)
    
    st.subheader('Pregunta 12')
    pregunta_12 = st.slider("En una escala del 1 al 5, ¿con qué frecuencia tiene problemas relacionados con el sueño?",1,5)
    
    
    for pregunta, tipo_lista in tipo_pregunta.items():
            for enfermedad in tipo_lista:
                if enfermedad not in respuestasPorEnfermadad:
                    respuestasPorEnfermadad[enfermedad] = 0
   
    submitted = st.form_submit_button("Enviar")
    if submitted:
        with st.spinner('Recogiendo datos...'):
            time.sleep(3)

        respuestas['pregunta_1']=pregunta_1
        respuestas['pregunta_2']=pregunta_2
        respuestas['pregunta_3']=pregunta_3
        respuestas['pregunta_4']=pregunta_4
        respuestas['pregunta_5']=pregunta_5
        respuestas['pregunta_6']=pregunta_6
        respuestas['pregunta_7']=pregunta_7
        respuestas['pregunta_8']=pregunta_8
        respuestas['pregunta_9']=pregunta_9
        respuestas['pregunta_10']=pregunta_10
        respuestas['pregunta_11']=pregunta_11
        respuestas['pregunta_12']=pregunta_12
        
        for pregunta, tipo_lista in tipo_pregunta.items():
            for enfermedad in tipo_lista:
                if pregunta in respuestas:
                    respuestasPorEnfermadad[enfermedad] += respuestas[pregunta]
            


        total = sum(respuestas.values())
        
        porcetanje = total/(5*12)*100
        if total > 40:
            st.write(f'#### Tienes alta probabilidad de contraer una enfermedad mental ({porcetanje}%)')
        else:
            st.write(f'#### Tienes baja probabilidad de contraer una enfermedad mental ({porcetanje}%)')
        
        tab1, tab2 = st.tabs(["📈 Gráficas", "🗃 Data"])

        
        chart_data = pd.DataFrame(data=[respuestasPorEnfermadad])
        tab2.dataframe(chart_data)



        #st.bar_chart(chart_data)

        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].pie(list(respuestasPorEnfermadad.values()),labels=list(respuestasPorEnfermadad.keys()),autopct='%1.1f%%',
            shadow=True, startangle=90)
        ax[1].bar(range(len(respuestasPorEnfermadad)),list(respuestasPorEnfermadad.values()),tick_label=list(respuestasPorEnfermadad.keys()))

        fig.suptitle('RESULTADOS')
        
        col1, col2 = st.columns(2)

        tab1.pyplot(fig)
        
        
