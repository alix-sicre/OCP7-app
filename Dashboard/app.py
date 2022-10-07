import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import plotly.express as px
import requests

#streamlit run Stream_app.py


st.set_page_config(page_title='Project 7 Dashboard"', page_icon=':smiley', 
                   layout="wide", initial_sidebar_state='expanded')

sns.set_theme(style="white")

def count_plot(df):
    fig, ax = plt.subplots(figsize=(5,5))
    sns.countplot(data=df, x='TARGET')
    return fig

def distribution(df, column_name, client_value=0, hue=None):
    fig, ax = plt.subplots(figsize=(5,5))
    sns.kdeplot(data=df, x=column_name,  fill=True, hue=hue)
    plt.axvline(client_value, color='red')
    return fig

def correlation_matrix(df):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(5,5))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return fig



def get_model_prediction(idx):
    r = requests.get(url='https://ocp7-app.herokuapp.com//predict?status='+str(int(idx)))
    acc = [r.json()['Accept']]
    ref = [r.json()['Refuse']]
    dic = {"Accept":acc,"Refuse":ref}
    df_predictions = pd.DataFrame.from_dict(dic, orient='index', columns = ['values']).reset_index()
    fig, ax = plt.subplots(figsize=(10,10))
    sns.barplot(x='index', y='values', data=df_predictions)
    return fig
 
def shap_plot(idx):
    r = requests.get(url='https://ocp7-app.herokuapp.com//explain?status='+str(int(idx)))
    dic = r.json()['0']
    print(dic)
    shap_df = pd.DataFrame.from_dict(dic, orient='index', columns = ['values']).reset_index()
    shap_df.info()
    fig, ax = plt.subplots(figsize=(10,10))
    sns.barplot(data=shap_df, x=shap_df["values"], y='index')
    return fig

def main():


    st.title("Project 7 Dashboard")

    st.header("Cleaned Data:")
    X = pd.read_csv("X.csv")
    X = X.drop(columns = ['Unnamed: 0'])

    X_w_ID = pd.read_csv("X_w_ID.csv")
    X_w_ID = X_w_ID.drop(columns = ['Unnamed: 0'])

    Y = pd.read_csv("Y.csv")
    Y = Y.drop(columns = ['Unnamed: 0'])

    Train_df = X_w_ID.join(Y)
    Train_df = Train_df.set_index('SK_ID_CURR')
    

    df = pd.read_csv("Test_df_1.csv")
    df = df.drop(columns = ['Unnamed: 0'])
    df = df.set_index('SK_ID_CURR')


    #idx = st.number_input('Client idx', value=100001)

    
    st.dataframe(Train_df)

    st.header("Target repartition:")
    fig0 = count_plot(Y)
    st.plotly_chart(fig0,  use_container_width=True)

    if st.checkbox('Show correlation matrix'):
        fig5 = correlation_matrix(df)
        st.pyplot(fig5) #,  use_container_width=True

    idx = st.selectbox('Wich ID ?',df.index)

    st.header("Prediction :")
    fig1 = get_model_prediction(idx)
    st.plotly_chart(fig1,  use_container_width=True)

    st.header("Explanation :")
    fig2 = shap_plot(idx)
    st.pyplot(fig2)
    #st.plotly_chart(fig2, use_container_width=True)



if __name__ == '__main__':
    main()