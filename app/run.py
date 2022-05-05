import sys
import os
from pathlib import Path
from joblib import load
import pandas as pd

from flask import Flask, request, render_template

# Add util directory to path
curr_dir = sys.path[0]
parent_dir = Path(curr_dir).parents[0]
dir = os.path.join(parent_dir, 'util')
sys.path.append(dir)

from custom_transformer import NumericalFeatures, CategoricalFeatures

app = Flask(__name__)
model = load('../model/model.pkl')


@app.route('/')
@app.route('/index.html')
def display():
    return render_template("index.html", title="Heart Disease Predictor")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    form_data = request.form.to_dict()
    for key, value in form_data.items():
        form_data[key] = convert_to_float(value)

    test_df = pd.DataFrame()
    for key, value in form_data.items():
        test_df[key] = [value]

    test_df = test_df[['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking'\
                      ,'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma'\
                      ,'KidneyDisease', 'SkinCancer']]

    result = model.predict(test_df)

    if result[0] == 0:
        result_str = "Good News, You are not expected to have Heart Disease"
    else:
        result_str = "Please make an appt. with your doctor. This model predicts that you have a heart disease"

    return render_template("index.html", title="Prediction", result=result_str)


def convert_to_float(s):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return s


def main():
    app.run(port=3201, debug=True)


if __name__ == '__main__':
    main()
