from flask import Flask, render_template, request, redirect

import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
        id = request.form['id']
        sts_cnt = request.form['sts_cnt']
        flrs_cnt = request.form['flrs_cnt']
        frds_cnt = request.form['frds_cnt']
        fvts_cnt = request.form['fvts_cnt']
        lstd_cnt = request.form['lstd_cnt']
        lng_cde = request.form['lng_cde']

        arr = np.array([[id,sts_cnt,flrs_cnt,frds_cnt,fvts_cnt,lstd_cnt,lng_cde]])
        preds = model.predict(arr)

        return render_template('after.html', data=preds)

if __name__ == "__main__":
    app.run(debug=True)