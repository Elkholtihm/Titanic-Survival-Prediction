import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer 
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures, LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff


# configuration de la page -------------------
st.set_page_config(
    page_title='Titanic Survival prediction',
    page_icon='🚢',
    layout='centered',
    initial_sidebar_state='auto',
    menu_items={
        'Get help' : 'https://github.com/Elkholtihm'
    }
)


# writing a header-----------------------------
st.header('This is an App for predicting suvivors and a dashboard of titanic dataset', 
          help='visit this website https://github.com/Elkholtihm', divider='violet')

# display video------------------
st.image(r"images/titanic_ship.jpeg")

#define a function to process data as the data the model trained on --------------------------------
def processor(train_data, prediction = False):
    #ticket

    #adding columns for duplicated ticket's value
    dup = train_data.groupby('Ticket').size()
    train_data['duplication'] = train_data['Ticket'].map(dup)

    #
    train_data.loc[train_data['Ticket'] == 'LINE', 'Ticket'] = 'LINE 0' 

    #
    def prefix(ticket):
        prefix = ticket.split(' ')[0][0]
        if prefix.isalpha():
            return ticket.split(' ')[0]
        else:
            return 'digits'
    
    #
    train_data['Ticket_prefix'] = train_data['Ticket'].apply(prefix)
    train_data['number_digits'] = train_data['Ticket'].apply(lambda x: len(str(x.split(' ')[-1])))

    #
    prefix_counts = train_data['Ticket_prefix'].value_counts()
    prefix_groups = {
    'Additional': prefix_counts[prefix_counts <= 12].index.tolist()}

    prefix_to_group = {prefix : group for group, prefixes in prefix_groups.items() for prefix in prefixes}
    train_data['Ticket_prefix'] = train_data['Ticket_prefix'].replace(prefix_to_group)

    #
    train_data['first_digit'] = train_data['Ticket'].apply(lambda x: int(x.split(' ')[-1][0]))

    #Title
    train_data['Title'] = train_data['Name'].str.split('[,\.]', expand = True)[1].str.strip()

    #
    prefix_groups = {
    'Additional': ['Dr','Dona', 'Rev', 'Mlle', 'Major', 'Col', 'Countess', 'Capt', 'Ms', 'Sir', 'Lady', 'Mme', 'Don', 'Jonkheer', 'the Countess']}
    prefix_to_group = {prefix : group for group, prefixes in prefix_groups.items() for prefix in prefixes}
    train_data['Title'] = train_data['Title'].replace(prefix_to_group)

    #Cabin
    #
    train_data['Cabin'] = train_data['Cabin'].str[0]

    #
    train_data['Cabin'] = train_data['Cabin'].replace({'T' : 'A'})

    #
    train_data['Cabin'] = train_data['Cabin'].fillna('Not_provided')

    #
    # The absence of cabin information is significant because it correlates with the majority of passengers who did not survive
    # thats why i kept the Cabin column
    print(train_data[train_data['Cabin'].isnull() & train_data['Survived'] == 0].shape[0], 
          train_data[train_data['Cabin'].isnull() & train_data['Survived'] == 1].shape[0])
    
    #
    train_data['SibSp'] = train_data['SibSp'].replace({
        i:j for i, j in zip(range(9), [f'SibSp{i}' for i in range(3)]+['SibSp_more_than3']*6)
        })
    
    #
    train_data['Parch'] = train_data['Parch'].replace({
    i:j for i, j in zip(range(7), [f'Parch{i}' for i in range(3)]+['Parch_more_than3']*4)
    })

    # delete useless columns
    train_data = train_data.drop(['Name', 'PassengerId', 'Ticket'], axis = 1)

    #
    encoder = LabelEncoder()
    encoders = {}
    for col in ['Sex', 'Cabin', 'Embarked', 'Title', 'SibSp', 'Parch', 'Ticket_prefix']:
        train_data[col] = encoder.fit_transform(train_data[col])

    #
    transformer = RobustScaler()
    for col in ['Fare', 'Age']:
        train_data[col] = transformer.fit_transform(train_data[[col]])

    #
    imputer1 = KNNImputer(n_neighbors = 3)
    train_data = pd.DataFrame(imputer1.fit_transform(train_data), columns=train_data.columns)

    #
    train_data['is_alone'] = train_data.apply(lambda x: 'alone' if x['SibSp']==0 and x['Parch']==0 else 'not_alone', axis=1)

    #
    train_data['is_alone'] = encoder.fit_transform(train_data['is_alone'])

    #
    final_data = train_data.copy()
    if prediction:
        cols = ['Pclass', 'Sex', 'Cabin', 'Title', 'SibSp', 'Parch', 'is_alone', 'first_digit', 'duplication', 'Ticket_prefix', 'Embarked']
        final_data[cols] = final_data[cols].astype(str)

        #
        dummies = pd.get_dummies(final_data[cols])
        col_to_keep = ['Survived', 'Age', 'Fare', 'number_digits']
        final_data = pd.concat([dummies, final_data[col_to_keep]], axis = 1)

        #
        final_data = final_data.astype(int)
        final_data.columns = ['Pclass_1.0', 'Pclass_2.0', 'Pclass_3.0', 'Sex_0.0', 'Sex_1.0',
        'Cabin_0.0', 'Cabin_1.0', 'Cabin_2.0', 'Cabin_3.0', 'Cabin_4.0',
        'Cabin_5.0', 'Cabin_6.0', 'Cabin_7.0', 'Title_0.0', 'Title_1.0',
        'Title_2.0', 'Title_3.0', 'Title_4.0', 'SibSp_0.0', 'SibSp_1.0',
        'SibSp_2.0', 'SibSp_3.0', 'Parch_0.0', 'Parch_1.0', 'Parch_2.0',
        'Parch_3.0', 'is_alone_0', 'is_alone_1', 'first_digit_0.0',
        'first_digit_1.0', 'first_digit_2.0', 'first_digit_3.0',
        'first_digit_4.0', 'first_digit_5.0', 'first_digit_6.0',
        'first_digit_7.0', 'first_digit_8.0', 'first_digit_9.0',
        'duplication_1.0', 'duplication_2.0', 'duplication_3.0',
        'duplication_4.0', 'duplication_5.0', 'duplication_6.0',
        'duplication_7.0', 'Ticket_prefix_0.0', 'Ticket_prefix_1.0',
        'Ticket_prefix_2.0', 'Ticket_prefix_3.0', 'Embarked_0.0',
        'Embarked_1.0', 'Embarked_2.0', 'Embarked_3.0', 'Age', 'Fare',
        'number_digits', 'Survived']
    final_data = final_data.drop(['Survived'], axis=1)
    return final_data


# diplay data---------------------------------------------------------------
st.subheader('DataFrame used for Training')
st.page_link('https://www.kaggle.com/competitions/titanic', label='Dataset Link')
df = pd.read_csv(r"C:\Users\user\Desktop\ci1\ProjetRealisee\titanic\data\train.csv")
st.dataframe(df.head())

# predict the survivors of draged data-------------------------------
st.subheader('Prediction')
data = st.file_uploader("drag you're data to make predictions on it : ", type=['xls', 'csv', 'xlsx'])

# convert the data to dataframe, process and predict the data-----------------------
if data is not None:
    if data.name.endswith('.csv'):
        data = pd.read_csv(data)
    else:
        data = pd.read_excel(data, engine='xlsxwriter')

    with st.spinner('please wait...'):
        with open('xgb.pkl', 'rb') as file:
            model = pickle.load(file)
            processed = processor(data, prediction = True)
            prediction = model.predict(processed)
            output_data = pd.concat([data.reset_index(drop = True), pd.DataFrame(prediction, columns = ['Prediction'])], axis=1)

    # download data with predictions
    st.download_button(label = "Download you're data with prediction column", 
                       data = output_data.to_csv(index=False), file_name = 'output_data.csv')

# Manage page---------------------------------------------------------------------
col1,_ = st.columns([0.7, 0.3])
st.divider()
met1, met2, met3, met4=st.columns(4)
st.divider()
col4,_ , col5=st.columns([0.3,0.3, 0.3])
st.divider()
col6, col7=st.columns([0.5, 0.3])
st.divider()
col8, col9=st.columns([0.5, 0.3])
st.divider()
col10, col11=st.columns([0.5, 0.3])

# Filters
with col1:
    selected_data = st.selectbox('select data to analyse', ("Titanic_data", "you're data"))

with col4:
    SelectedCategoricalCol = st.selectbox('Survived VS ', ('Pclass', 'Sex', 'Parch', 'SibSp', 'Embarked'))



# display charts based on selected column (Survived - selected_column)------------------------------------------------------
# get data based on selection
if selected_data == "Titanic_data":
    # procces data and add prediction column
    with open('xgb.pkl', 'rb') as file:
        model = pickle.load(file)
        col = df['Survived']
        df = processor(df)
        df = pd.concat([df.reset_index(drop = True), col], axis=1)
        dataToUse = df
        print(dataToUse.head())

elif selected_data == "you're data" and data is None:
    dataToUse = df

elif selected_data == "you're data" and data is not None:
    dataToUse = data

numberMales = dataToUse[dataToUse['Sex'] == 'male'].shape[0]
numberFemales = dataToUse[dataToUse['Sex'] == 'female'].shape[0]
numberSurvivors = dataToUse[dataToUse['Survived'] == 1].shape[0]
numberNSurvivors = dataToUse[dataToUse['Survived'] == 0].shape[0]


# number of males and females
with met1:
    st.markdown('<span style="color:blue;">Male</span>', unsafe_allow_html=True)
    st.metric(label="",value = f'{numberMales}')

with met2:
    st.markdown('<span style="color:pink;">Female</span>', unsafe_allow_html=True)
    st.metric(label="",value = f'{numberFemales}')

with met3:
    st.markdown('<span style="color:green;">Survived</span>', unsafe_allow_html=True)
    st.metric(label="",value = f'{numberSurvivors}')

with met4:
    st.markdown('<span style="color:red;">Not Survived</span>', unsafe_allow_html=True)
    st.metric(label="",value = f'{numberNSurvivors}')

with col5:
    SelectedNumericalCol = st.selectbox('Survived VS ', ('Age', 'Fare'))

# dispaly charts based on selected data
with col6:
    fig = px.histogram(dataToUse, x='Survived', color=SelectedCategoricalCol)
    fig.update_layout(bargap=0.05) 
    st.plotly_chart(fig)

with col7:
    if SelectedNumericalCol == 'Fare':
        fig = px.strip(dataToUse, x='Survived', y=SelectedNumericalCol, color = 'Survived')
        st.plotly_chart(fig)
    
    if SelectedNumericalCol == 'Age':
        fig = px.box(dataToUse, x='Survived', y=SelectedNumericalCol, color = 'Survived')
        st.plotly_chart(fig)

with col8:
    fig = px.violin(dataToUse, y='Age', x='Pclass', box=True, color_discrete_sequence=['green'])
    st.plotly_chart(fig)

with col9:
    fig = px.box(dataToUse, x='Pclass', y='Fare')
    st.plotly_chart(fig)

# draw a heatmap
with st.container():
    corr_matrix = dataToUse.corr()              
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='blues',  
        annotation_text=corr_matrix.values.round(2),  
        showscale=True  
    )

    fig.update_layout(
        title='Correlation Heatmap', 
        xaxis=dict(title='Features'), 
        yaxis=dict(title='Features')  
    )
    st.plotly_chart(fig, use_container_width=True)

