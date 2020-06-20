from flask import Flask,request,render_template,url_for
import requests
import numpy as np
import pandas as pd 

# Sklearn libraries
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,LabelEncoder
# import statsmodels.api as sm
from sklearn.metrics import r2_score,mean_squared_error

# Serializing
import pickle

# Reference dictionaries
fuel = pickle.load(open('fuel2.pkl','rb'))
fl = list(fuel.keys())
company = pickle.load(open('company2.pkl','rb'))
comp = list(company.keys())
comp.remove('well')
comp_new = comp
year = pickle.load(open('year.pkl','rb'))
yr = list(set(year))
# Regressor
reg = pickle.load(open('mlp3.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',comp=comp_new,yr=yr)

@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form.get('item')
    yop = int(request.form.get('inputYear'))
    dist = int(request.form.get('distance'))
    ful = request.form.get('fuel')
    # Encoding brand and fuel
    brand_enc = company[brand]
    fuel_enc = fuel[ful]
    inp = [fuel_enc,brand_enc,yop,dist]
    # Prediction results
    price = (int(reg.predict(np.array(inp).reshape(1,4))[0]))*10
    # Label

    return render_template('predict.html',price=price)

if __name__=="__main__":
    app.run(debug=True)
