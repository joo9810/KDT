#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cgi
import cgitb
import sys
import codecs
import torch
from konlpy.tag import Okt
import joblib
import torch
import torch.nn as nn

class SentenceClassifier(nn.Module):
    def __init__(self, n_vocab, hidden_dim, embedding_dim, n_layers, n_classes, dropout=0.5, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.lstm(embeddings)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)
        return logits

# 숫자와 카테고리 매핑 딕셔너리
category_map = {
    0: '육아,결혼',
    1: '반려동물',
    2: '패션',
    3: '인테리어',
    4: '요리',
    5: '상품리뷰',
    6: '원예'
}


# 디버깅을 위한 CGI 트레이스백 활성화
cgitb.enable()

# 표준 출력의 인코딩을 UTF-8로 설정
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# 모델 및 필요한 데이터 파일의 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'bestmodel.pth')
token_to_id_path = os.path.join(current_dir, 'token_to_id.joblib')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
try:
    model = torch.load(model_path, map_location=device)
    model.eval()  # 평가 모드로 설정
    print("Model loaded successfully")
except Exception as e:
    print("Content-Type: text/html\n")
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

# 토크나이저 및 필요한 데이터 로드
tokenizer = Okt()
token_to_id = joblib.load(token_to_id_path)
le = joblib.load(label_encoder_path)
max_length = 32  # 또는 저장된 값에서 로드

def predict_category(model, text, tokenizer, token_to_id, max_length, device):
    tokens = tokenizer.morphs(text)
    ids = [token_to_id.get(token, token_to_id['<unk>']) for token in tokens]
    ids = ids[:max_length] + [token_to_id['<pad>']] * max(0, max_length - len(ids))
    input_tensor = torch.tensor([ids]).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    _, predicted = torch.max(output, 1)
    numeric_result = predicted.item()
    return numeric_result, category_map.get(numeric_result, "알 수 없는 카테고리")

def showHTML(text, msg):
    print("Content-Type: text/html; charset=utf-8")
    print()
    print(f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>블로그 제목 분류</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 600px; margin: auto; }}
            textarea {{ width: 100%; height: 100px; margin-bottom: 10px; }}
            input[type="submit"] {{ background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }}
            .result {{ margin-top: 20px; padding: 10px; background-color: #f0f0f0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>블로그 제목 분류</h1>
            <form method="post">
                <textarea name="text" placeholder="텍스트를 입력하세요...">{text}</textarea>
                <input type="submit" value="분류하기">
            </form>
            <div class="result">
                <h2>분류 결과</h2>
                <p>{msg}</p>
            </div>
        </div>
    </body>
    </html>
    """)

# 메인 로직
form = cgi.FieldStorage()
text = form.getvalue("text", "")

if text:
    try:
        numeric_result, category = predict_category(model, text, tokenizer, token_to_id, max_length, device)
        msg = f"입력 텍스트의 예측 카테고리 번호: {numeric_result}<br>카테고리: {category}"
    except Exception as e:
        msg = f"예측 중 오류 발생: {str(e)}"
else:
    msg = "텍스트를 입력해주세요."

showHTML(text, msg)