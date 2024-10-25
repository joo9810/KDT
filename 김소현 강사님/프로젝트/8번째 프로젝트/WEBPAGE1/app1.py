import pandas as pd
from flask import Flask, request, render_template
from KDTModule import *
import joblib

app = Flask(__name__)


# index.html을 렌더링하는 경로
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    
    if request.form['hair_length'] == 'Yes':
        long_hair = 1
    else:
        long_hair = 0

    forehead_width_cm = float(request.form['forehead_width'])
    forehead_height_cm = float(request.form['forehead_height'])

    if request.form['nose_wide'] == 'Yes':
        nose_wide = 1
    else:
        nose_wide = 0

    if request.form['nose_long'] == 'Yes':
        nose_long = 1
    else:
        nose_long = 0

    if request.form['lips_thin'] == 'Yes':
        lips_thin = 1
    else:
        lips_thin = 0

    if request.form['distance_nose_to_lip_long'] == 'Yes':
        distance_nose_to_lip_long = 1
    else:
        distance_nose_to_lip_long = 0

    # predicted_value = [long_hair, forehead_width_cm, forehead_height_cm, nose_wide,
    #                    nose_long, lips_thin, distance_nose_to_lip_long]

    input_data = pd.DataFrame({
        'long_hair' : [long_hair],
        'forehead_width_cm' : [forehead_width_cm],
        'forehead_height_cm' : [forehead_height_cm],
        'nose_wide' : [nose_wide],
        'nose_long' : [nose_long],
        'lips_thin' : [lips_thin],
        'distance_nose_to_lip_long' : [distance_nose_to_lip_long]
    })

    mm_scaler = joblib.load('/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/8번째 프로젝트/WEBPAGE1/MinMaxScaler.joblib')

    input_data_scaled = mm_scaler.transform(input_data[['forehead_width_cm', 'forehead_height_cm']])
    input_data[['forehead_width_cm', 'forehead_height_cm']] = input_data_scaled

    # 모델 불러오기
    dt = joblib.load('/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/8번째 프로젝트/WEBPAGE1/DecisionTree.joblib')
    predicted_value = dt.predict(input_data)

    if predicted_value == 0:
        predicted_value = '남자'
    else:
        predicted_value = '여자'

    return render_template('result.html', predicted_value=predicted_value)

if __name__ == '__main__':
    app.run(debug=True, port=5001)