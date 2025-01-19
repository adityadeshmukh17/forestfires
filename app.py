import pickle
from flask import Flask,render_template,jsonify,request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

ridgemodel=pickle.load(open('C:/Users/Aditya Deshmukh/OneDrive/Desktop/data science/linear reg/models/ridge.pkl','rb'))
scalermodel=pickle.load(open('C:/Users/Aditya Deshmukh/OneDrive/Desktop/data science/linear reg/models/scaler1.pkl','rb'))

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('templates/index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_data():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=scalermodel.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridgemodel.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
