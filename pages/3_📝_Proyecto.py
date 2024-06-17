import streamlit as st

with st.sidebar:
    st.image("logo.png")
    st.image("UPV.png")
    

colT1,colT2 = st.columns([1,3])
colT2.title("PROYECTO")

st.write("En este página de nuestra página web se explicará y profundizaremos en más aspectos a la hora del estudio de este.")


st.header("Estudio y gráficas")


with st.echo():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # loading the dataset used in for the project

    data=pd.read_csv('smmh.csv')
    # Setting to display all columns
    pd.set_option("display.max_columns", None)
    # Printing first 5 entries in the data set
    data.head()
    data.rename(columns = {'1. What is your age?':'Age','2. Gender':'Sex','3. Relationship Status':'Relationship Status',
                        '4. Occupation Status':'Occupation',
                        '5. What type of organizations are you affiliated with?':'Affiliations',
                        '6. Do you use social media?':'Social Media User?',
                        '7. What social media platforms do you commonly use?':'Platforms Used',
                        '8. What is the average time you spend on social media every day?':'Time Spent',
                        '9. How often do you find yourself using Social media without a specific purpose?':'ADHD Q1',
                        '10. How often do you get distracted by Social media when you are busy doing something?':'ADHD Q2',
                        "11. Do you feel restless if you haven't used Social media in a while?":'Anxiety Q1',
                        '12. On a scale of 1 to 5, how easily distracted are you?':'ADHD Q3',
                        '13. On a scale of 1 to 5, how much are you bothered by worries?':'Anxiety Q2',
                        '14. Do you find it difficult to concentrate on things?':'ADHD Q4',
                        '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?':'Self Esteem Q1',
                        '16. Following the previous question, how do you feel about these comparisons, generally speaking?':'Self Esteem Q2',
                        '17. How often do you look to seek validation from features of social media?':'Self Esteem Q3',
                        '18. How often do you feel depressed or down?':'Depression Q1',
                        '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?':'Depression Q2',
                        '20. On a scale of 1 to 5, how often do you face issues regarding sleep?':'Depression Q3' },inplace=True)
    titles = list(data.columns)
    titles[11], titles[12] = titles[12], titles[11]
    titles[12], titles[14] = titles[14], titles[12]
    titles[13], titles[14] = titles[14], titles[13]
    data = data[titles]
    #List all the unique Gender/Sex entries.
    Genders = set(data['Sex'])
    data.drop(data.loc[data['Sex'] =='There are others???'].index, inplace=True)
    Genders = set(data['Sex'])
    #Combining the unique entries that all fall under the "Others" category
    data.replace('Non-binary','Others', inplace=True)
    data.replace('Nonbinary ','Others', inplace=True)
    data.replace('NB','Others', inplace=True)
    data.replace('unsure ','Others', inplace=True)
    data.replace('Non binary ','Others', inplace=True)
    data.replace('Trans','Others', inplace=True)

    #Converting Age from float64 to int64 and displaying record # 382
    data['Age'] = data['Age'].astype('int64')
    #setting scores of 3,4 and 5 to 0.
    data.loc[data['Self Esteem Q2'] == 3, 'Self Esteem Q2'] = 0
    data.loc[data['Self Esteem Q2'] == 4, 'Self Esteem Q2'] = 0
    data.loc[data['Self Esteem Q2'] == 5, 'Self Esteem Q2'] = 0
    #Setting scores of '1' to '4' and '2' to '2'.
    data.loc[data['Self Esteem Q2'] == 1, 'Self Esteem Q2'] = 4
    data.loc[data['Self Esteem Q2'] == 2, 'Self Esteem Q2'] = 2

    #Summing scores from ADHD, Anxiety, Self Esteem and Depression individually and creating a new column

    ADHD = ['ADHD Q1', 'ADHD Q2', 'ADHD Q3', 'ADHD Q4']
    data['ADHD Score'] = data[ADHD].sum(axis=1)

    Anxiety = ['Anxiety Q1', 'Anxiety Q2']
    data['Anxiety Score'] = data[Anxiety].sum(axis=1)

    SelfEsteem = ['Self Esteem Q1', 'Self Esteem Q2','Self Esteem Q3']
    data['Self Esteem Score'] = data[SelfEsteem].sum(axis=1)

    Depression = ['Depression Q1', 'Depression Q2','Depression Q3']
    data['Depression Score'] = data[Depression].sum(axis=1)

    Total = ['ADHD Score', 'Anxiety Score','Self Esteem Score','Depression Score']
    data['Total Score'] = data[Total].sum(axis=1)

    #Deleting question columns and timestamp columns as they are no longer used
    adicionales = data.iloc[:, 9:21] #para el final
    data.drop(data.iloc[:, 9:21], inplace = True, axis = 1)
    data.drop(['Timestamp'], inplace = True, axis = 1)

    def map_score(score):
        if score < 40:
            return "0"
        elif score >= 40:
            return "1"

    data['Outcome']= data['Total Score'].apply(lambda score: map_score(score))
    data['Outcome'] = data['Outcome'].astype('int64')


    #Drop Total score column and display correlation plot
    data.drop(['Total Score'], inplace = True, axis = 1)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

with st.echo():
    for index, row in data.iterrows():
        data.loc[index, 'Discord'] = 'Discord' in row['Platforms Used']
        data.loc[index, 'Reddit'] = 'Reddit' in row['Platforms Used']
        data.loc[index, 'Facebook'] = 'Facebook' in row['Platforms Used']
        data.loc[index, 'Twitter'] = 'Twitter' in row['Platforms Used']
        data.loc[index, 'Pinterest'] = 'Pinterest' in row['Platforms Used']
        data.loc[index, 'Instagram'] = 'Instagram' in row['Platforms Used']
        data.loc[index, 'Youtube'] = 'Youtube' in row['Platforms Used']
        data.loc[index, 'Snapchat'] = 'Snapchat' in row['Platforms Used']
        data.loc[index, 'TikTok'] = 'TikTok' in row['Platforms Used']
    Twitter = data.loc[data['Twitter'] == True]
    TTwitter = Twitter.loc[:, ['Anxiety Score', 'ADHD Score', 'Self Esteem Score', 'Depression Score']].mean()
    Ninguna = data.loc[(data['Social Media User?'] == 'No') | (data['Time Spent'] == 'Less than an Hour')]
    TNinguna = Ninguna.loc[:, ['Anxiety Score', 'ADHD Score', 'Self Esteem Score', 'Depression Score']].mean()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(TNinguna.index, TNinguna.values, color = (0,0,0.8))
    ax.bar(TTwitter.index, TTwitter.values, bottom = TNinguna.values, color = (0.1, 0.63, 0.95))
    st.write(fig)

with st.echo():
    bar_colors = [(0.11, 0.63, 0.95), (0,0,0.8, 0.9)]
    Total = TTwitter.copy()
    Total['Anxiety Score NRRSS'] = TNinguna['Anxiety Score']
    Total['ADHD NRRSS'] = TNinguna['ADHD Score']
    Total['Self Esteem NRRSS'] = TNinguna['Self Esteem Score']
    Total['Depression Score NRRSS'] = TNinguna['Depression Score']
    Total = Total.reindex(['Anxiety Score', 'Anxiety Score NRRSS', 'ADHD Score', 'ADHD NRRSS', 'Self Esteem Score', 'Self Esteem NRRSS', 'Depression Score', 'Depression Score NRRSS'])

    fig, ax = plt.subplots(figsize=(17,6))
    ax.bar(Total.index, Total.values, color = bar_colors)
    st.write(fig)

with st.echo():
    TikTok = data.loc[data['TikTok'] == True]
    TTikTok = TikTok.loc[:, ['Anxiety Score', 'ADHD Score', 'Self Esteem Score', 'Depression Score']].mean()
    fig, ax = plt.subplots(figsize=(17,6))
    ax.bar(TNinguna.index, TNinguna.values, color = (0,0,0.8, 0.9))
    ax.bar(TTikTok.index, TTikTok.values, bottom = TNinguna.values, color = (0,0,0, 0.9))
    st.write(fig)

with st.echo():
    bar_colors = [(0,0,0, 0.9), (0,0,0.8, 0.9)]
    Total = TTikTok.copy()
    Total['Anxiety Score NRRSS'] = TNinguna['Anxiety Score']
    Total['ADHD NRRSS'] = TNinguna['ADHD Score']
    Total['Self Esteem NRRSS'] = TNinguna['Self Esteem Score']
    Total['Depression Score NRRSS'] = TNinguna['Depression Score']
    Total = Total.reindex(['Anxiety Score', 'Anxiety Score NRRSS', 'ADHD Score', 'ADHD NRRSS', 'Self Esteem Score', 'Self Esteem NRRSS', 'Depression Score', 'Depression Score NRRSS'])

    fig, ax = plt.subplots(figsize=(17,6))
    ax.bar(Total.index, Total.values, color = bar_colors)
    st.write(fig)

with st.echo():
    Insta = data.loc[data['Instagram'] == True]
    TInstagram = Insta.loc[:, ['Anxiety Score', 'ADHD Score', 'Self Esteem Score', 'Depression Score']].mean()
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(TNinguna.index, TNinguna.values, color = (0,0,0.8, 0.9))
    ax.bar(TInstagram.index, TInstagram.values, bottom = TNinguna.values, color=(1, 0, 0.6, 0.8))
    st.write(fig)

with st.echo():
    bar_colors = [(1, 0, 0.6, 0.8), (0,0,0.8, 0.9)]
    Total = TInstagram.copy()
    Total['Anxiety Score NRRSS'] = TNinguna['Anxiety Score']
    Total['ADHD NRRSS'] = TNinguna['ADHD Score']
    Total['Self Esteem NRRSS'] = TNinguna['Self Esteem Score']
    Total['Depression Score NRRSS'] = TNinguna['Depression Score']
    Total = Total.reindex(['Anxiety Score', 'Anxiety Score NRRSS', 'ADHD Score', 'ADHD NRRSS', 'Self Esteem Score', 'Self Esteem NRRSS', 'Depression Score', 'Depression Score NRRSS'])

    fig, ax = plt.subplots(figsize=(17,6))
    ax.bar(Total.index, Total.values, color = bar_colors)
    st.write(fig)

with st.echo():
    Todo = TTwitter.copy()
    Todo = Todo.drop(labels = ['Anxiety Score', 'ADHD Score', 'Self Esteem Score', 'Depression Score'])

    Todo['ADHD Twitter'] = TTwitter['ADHD Score']
    Todo['ADHD Instagram'] = TInstagram['ADHD Score']
    Todo['ADHD TikTok'] = TTikTok['ADHD Score']
    Todo['ADHD NRSS'] = TNinguna['ADHD Score']

    Todo['Anxiety Twitter'] = TTwitter['Anxiety Score']
    Todo['Anxiety Instagram'] = TInstagram['Anxiety Score']
    Todo['Anxiety TikTok'] = TTikTok['Anxiety Score']
    Todo['Anxiety NRSS'] = TNinguna['Anxiety Score']

    Todo['Self esteem Twitter'] = TTwitter['Self Esteem Score']
    Todo['Self esteem Instagram'] = TInstagram['Self Esteem Score']
    Todo['Self esteem TikTok'] = TTikTok['Self Esteem Score']
    Todo['Self esteem NRSS'] = TNinguna['Self Esteem Score']

    Todo['Depression Twitter'] = TTwitter['Depression Score']
    Todo['Depression Instagram'] = TInstagram['Depression Score']
    Todo['Depression TikTok'] = TTikTok['Depression Score']
    Todo['Depression NRSS'] = TNinguna['Depression Score']

    enfermedades = ("TDAH", "Anxiety", "Self Esteem", "Depression")
    valoresRRSS = {
        'Twitter': (Todo['ADHD Twitter'],Todo['Anxiety Twitter'],Todo['Self esteem Twitter'],Todo['Depression Twitter']),
        'Instagram': (Todo['ADHD Instagram'], Todo['Anxiety Instagram'], Todo['Self esteem Instagram'], Todo['Depression Instagram']),
        'TikTok': (Todo['ADHD TikTok'], Todo['Anxiety TikTok'], Todo['Self esteem TikTok'], Todo['Depression TikTok']),
        'NRSS': (Todo['ADHD NRSS'], Todo['Anxiety NRSS'], Todo['Self esteem NRSS'], Todo['Depression NRSS'])
    }

    x = np.arange(len(enfermedades))
    width = 0.2  #ponemos 1/5
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    colores = [(0.1, 0.63, 0.95), (1, 0, 0.6, 0.8), (0, 0, 0, 0.9), (0, 0, 0.8, 0.9)]
    indice = 0

    for attribute, measurement in valoresRRSS.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color = colores[indice])
        #ax.bar_label(rects, padding=3) pone números encima de cada columna
        multiplier += 1
        indice += 1

    ax.set_title('Media de valores en redes sociales por enfermedad mental')
    ax.set_xticks(x + width+0.1, enfermedades) #0.1 para ponerlos en medio
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 20)

    st.write(fig)

st.header("Modelo predictivo")


with st.echo():
    #Codificando la database para el árbol de decisión:

    #Male = 0, Female = 1
    data['Sex'] = data['Sex'].map({'Male': 0, 'Female': 1, 'Others': 2})

    #Single = 0, In a relationship = 1, Married = 3, Divorced = 4
    data['Relationship Status'] = data['Relationship Status'].map({'Single': 0, 'In a relationship': 1, 'Married': 2, 'Divorced': 3})

    #School student = 0, University Student = 1, Salaried Worker = 2, Retired = 3
    data['Occupation'] = data['Occupation'].map({'School Student': 0, 'University Student': 1, 'Salaried Worker': 2, 'Retired': 3})

    #Affiliations
    for index, row in data.iterrows():
        if type(row['Affiliations']) != float:
            data.loc[index, 'Affiliation: School'] = 'School' in row['Affiliations']
            data.loc[index, 'Affiliation: University'] = 'University' in row['Affiliations']
            data.loc[index, 'Affiliation: Company'] = 'Company' in row['Affiliations']
            data.loc[index, 'Affiliation: Private'] = 'Private' in row['Affiliations']
            data.loc[index, 'Affiliation: Goverment'] = 'Goverment' in row['Affiliations']
        else:
            data.loc[index, 'Affiliation: School'] = 0
            data.loc[index, 'Affiliation: University'] = 0
            data.loc[index, 'Affiliation: Company'] = 0
            data.loc[index, 'Affiliation: Private'] = 0
            data.loc[index, 'Affiliation: Goverment'] = 0
    affiliations = ['Affiliation: School', 'Affiliation: University', 'Affiliation: Company', 'Affiliation: Private', 'Affiliation: Goverment']
    for affiliation in affiliations:
        data[affiliation] = data[affiliation].map({False: 0, True: 1})
    del(data['Affiliations'])
    #No = 0, Yes = 1
    data['Social Media User?'] = data['Social Media User?'].map({'No': 0, 'Yes': 1})

    #Quitamos Platforms used porque ya las tenemos una por una
    del(data['Platforms Used'])

    #Less than an Hour = 0, Between 1 and 2 hours = 1, Between 2 and 3 hours = 2,  Between 3 and 4 hours = 3, Between 4 and 5 hours = 4, More than 5 hours = 5
    data['Time Spent'] = data['Time Spent'].map({'Less than an Hour': 0, 'Between 1 and 2 hours': 1, 'Between 2 and 3 hours': 2, 'Between 3 and 4 hours': 3, 'Between 4 and 5 hours': 4, 'More than 5 hours': 5})

    #Las diferentes redes sociales
    redes_sociales = ['Discord', 'Reddit', 'Facebook', 'Twitter', 'Pinterest', 'Instagram', 'Youtube', 'Snapchat', 'TikTok']
    for red_social in redes_sociales:
        data[red_social] = data[red_social].map({False: 0, True: 1})
    for column in adicionales.columns:
        data[column] = adicionales[column]
    data.drop(data.iloc[:, 6:10], inplace = True, axis = 1)
    st.dataframe(data)

with st.echo():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    X = data.drop(columns='Outcome')
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test.shape
    dt = DecisionTreeClassifier(random_state=12, max_depth = 4)
    dt.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score
    y_predicted = dt.predict(X_test)
    ac = accuracy_score(dt.predict(X_train), y_train)
    st.metric(label="Accuracy Score", value=f'{round(ac*100,4)}%')
