# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request
import json
import requests
import pandas as pd 
import joblib
import streamlit as st
import pandas as pd
import random
import pickle
import shap



app = Flask(__name__)


data = pd.read_csv("Test_df_1.csv")
data = data.drop(columns = ['Unnamed: 0'])
data.drop(columns = ['TARGET'], inplace = True)
ls_SK_ID = data['SK_ID_CURR'].to_list()



clf = joblib.load('Model.sav')



# http://127.0.0.1:5000/predict?status=1


with open( 'shap.pickle', 'rb') as handle:
    shap_pickle = pickle.load(handle)


@app.route("/")
def hello():
    return "Welcome on my webpage"

@app.route('/predict', methods = ['GET'])

def Predict_proba():
    args = request.args
    i = args.get('status',default = 0, type = int)

    Test_dfi = data.loc[data.SK_ID_CURR == i]
    Test_dfi = Test_dfi.drop(columns = ['SK_ID_CURR'])
    d = {}
    d['Accept'] = clf.predict_proba(Test_dfi)[0][0]
    d['Refuse'] = clf.predict_proba(Test_dfi)[0][1]
    return d


@app.route('/explain', methods = ['GET'])

def explain_shap():
    args = request.args
    i = args.get('status',default = 0, type = int)
    Test_dfi = data.loc[data.SK_ID_CURR == i]
    Df_Link_ID_Index = data['SK_ID_CURR']
    Df_Link_ID_Index = Df_Link_ID_Index.to_frame()
    i = Df_Link_ID_Index.loc[Df_Link_ID_Index.SK_ID_CURR == int(i)]
    Test_dfi = Test_dfi.drop(columns = ['SK_ID_CURR'])
    #shap_df = pd.DataFrame(shap_pickle.values[i][:,0], index = Test_dfi.columns)
    shap_df = pd.DataFrame(shap_pickle.values[i.index[0]][:,0], index = Test_dfi.columns)

    
    return shap_df.to_dict() 


if __name__ == "__main__":
    app.run(debug=True)