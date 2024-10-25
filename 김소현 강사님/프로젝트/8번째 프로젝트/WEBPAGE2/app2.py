from flask import Flask, request, render_template
import torch
import torch.nn as nn 
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from KDTModule import *
import pickle
from konlpy.tag import Okt
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)


# index.html을 렌더링하는 경로
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    sentence = request.form['tweet'] # index.html에서 전송한 텍스트

    sentenceDF = pd.DataFrame([sentence], columns=['clean_text'])
    loaded_vectorizer = joblib.load('/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/8번째 프로젝트/WEBPAGE2/tfid_vectorizer.pkl')
    input_vector = loaded_vectorizer.transform(sentenceDF['clean_text']).toarray()
    input_vectorDF = pd.DataFrame(input_vector)
    best_model = LSTMModel(input_size = 8000, output_size = 3, hidden_list = [100, 80, 60, 40, 20],
                    act_func=F.relu, model_type='multiclass', num_layers=1)
    best_model.load_state_dict(torch.load('/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/8번째 프로젝트/WEBPAGE2/Best_LSTM_Model.pth', weights_only=True))
    predicted_value = predict_value(input_vectorDF, best_model, dim=3)

    if predicted_value == 1:
        predicted_value = '긍정'
    else:
        predicted_value = '부정'

    return render_template('result.html', predicted_value=predicted_value)

if __name__ == '__main__':
    app.run(debug=True, port=5005)