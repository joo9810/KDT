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


sentence = request.form['tweet'] # index.html에서 전송한 텍스트

class TextModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_list, act_func, model, num_layers=1):
        super().__init__()
        # 입력층 (LSTM)
        if model == 'lstm':
            self.lstm_layer = nn.LSTM(input_size, hidden_list[0], num_layers, batch_first=True)
        elif model == 'rnn':
            self.rnn_layer = nn.RNN(input_size, hidden_list[0], num_layers, batch_first=True)
        elif model == 'gru':
            self.gru_layer = nn.GRU(input_size, hidden_list[0], num_layers, batch_first=True)
        # 은닉층
        self.hidden_layer_list = nn.ModuleList()
        for i in range(len(hidden_list)-1):
            self.hidden_layer_list.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
        # 출력층
        self.output_layer = nn.Linear(hidden_list[-1], output_size)

        self.act_func = act_func
        self.dropout = nn.Dropout(0.5)
        self.model = model
        
    def forward(self, x):
        # 입력층
        if self.model == 'lstm':
            lstm_out, (hn, cn) = self.lstm_layer(x) # lstm_out : 모든 타입스텝 출력
            x = lstm_out[:, -1, :] # 마지막 타입스텝 출력
        elif self.model == 'rnn':
            rnn_out, hn = self.rnn_layer(x) # rnn_out : 모든 타입스텝 출력
            x = rnn_out[:, -1, :] # 마지막 타입스텝 출력
        elif self.model == 'gru':
            gru_out, hn = self.gru_layer(x) # gru_out : 모든 타입스텝 출력
            x = gru_out[:, -1, :] # 마지막 타입스텝 출력
        # 은닉층
        for layer in self.hidden_layer_list:
            x = layer(x)
            x = self.act_func(x)
            x = self.dropout(x)
        # 출력층
        return self.output_layer(x) # 로짓값

with open('stopword.txt', 'r', encoding='utf-8') as f:
    stopwords = [i.strip() for i in f.readlines()]

token = Okt().morphs(sentence)
clean_token = remove_punctuation(token)
clean_token = remove_stopwords(clean_token, stopwords)
clean_text = ' '.join(clean_token)
textDF = pd.DataFrame({'clean_text' : [clean_text]})

loaded_vectorizer = joblib.load('/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/8번째 프로젝트/WEBPAGE5/tfid_vectorizer.pkl')
text_vector = loaded_vectorizer.transform(textDF['clean_text']).toarray()
text_vectorDF = pd.DataFrame(text_vector)

best_model = TextModel(input_size = 5000, output_size = 4, hidden_list = [100, 50],
                    act_func=F.relu, model='lstm', num_layers=2)

checkpoint = torch.load('/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/8번째 프로젝트/WEBPAGE5/Best_LSTM_Model.pth')
best_model.load_state_dict(checkpoint['model_state_dict'])

predicted_value = predict_value(text_vectorDF, best_model, dim=3)

if predicted_value == 0:
    predicted_value = '전체 관람가'
elif predicted_value == 1:
    predicted_value = '12세이상관람가'
elif predicted_value == 2:
    predicted_value = '15세이상관람가'
elif predicted_value == 3:
    predicted_value = '청소년 관람불가'
