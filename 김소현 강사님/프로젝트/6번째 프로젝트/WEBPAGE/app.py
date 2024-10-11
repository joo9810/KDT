from flask import Flask, request, render_template, redirect
import os
from PIL import Image
import torch
import torch.nn as nn 
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from KDTModule import *
import pickle
import sys
import codecs
from konlpy.tag import Okt
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)


# index.html을 렌더링하는 경로
@app.route('/')
def index():
    return render_template('index.html')


# 각 범주에 맞는 HTML 페이지로 이동
@app.route('/entertainment', methods=['GET'])
def entertainment():
    return render_template('entertainment.html')

@app.route('/lifestyle', methods=['GET'])
def lifestyle():
    return render_template('lifestyle.html')

@app.route('/hobby', methods=['GET'])
def hobby():
    return render_template('hobby.html')

@app.route('/knowledge', methods=['GET'])
def knowledge():
    return render_template('knowledge.html')


# 입력된 데이터 처리하고 result.html로 넘기는 경로
# 취미/여가/여행 (안효준)
@app.route('/result_hobby', methods=['POST'])
def result_hobby():
    input_text = request.form['input_text'] # index.html에서 전송한 텍스트

    token = Okt().morphs(input_text)
    clean_text = remove_stopwords(token, 'stopword.txt')
    clean_text = remove_punctuation(clean_text)
    join_clean_text = ' '.join(clean_text)

    inputDF = pd.DataFrame([join_clean_text], columns=['clean_text'])

    best_model = LSTMModel(input_size = 5000, output_size = 8, hidden_list = [100, 30],
                        act_func=F.relu, model_type='multiclass', num_layers=1)
    # 본인 pth 경로로 바꾸기
    pth_PATH = '/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/best_model_tf-idf_0.8203.pth'
    best_model.load_state_dict(torch.load(pth_PATH, weights_only=True))

    best_model.eval()
    # 본인 pkl 경로로 바꾸기
    pkl_PATH = '/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/WEBPAGE/tfid_vectorizer_5000.pkl'
    loaded_vectorizer = joblib.load(pkl_PATH)
    input_vector = loaded_vectorizer.transform(inputDF['clean_text'].values)
    input_tensor_vector = torch.FloatTensor(input_vector.toarray())
    with torch.no_grad(): # 불필요한 그레디언트 계산 끄기 (메모리 사용량 줄이기) *켜도 eval 결과에 영향은 없음*
        input_logit = best_model(input_tensor_vector.unsqueeze(1))
        pred = torch.argmax(input_logit, dim=1).item()

    section_dict = {0 : '국내 여행', 1 : '게임', 2 : '취미', 3 : '해외 여행',
                    4 : '사진', 5 : '맛집', 6 : '스포츠', 7 : '자동차'}

    pred_section = section_dict[pred]

    return render_template('result_hobby.html', pred_section=pred_section)


# 엔터테이먼트/예술 (김민석)
def remove_special_characters(text):
    text = re.sub(r'[^가-힣A-Za-z\s]', '', text)
    return text.strip()

def remove_stopwords2(titles):
    nltk.download('stopwords')
    english_stopwords = set(stopwords.words('english'))
    korean_stopwords = [
        '이', '가', '은', '는', '을', '를', '에', '의', '와', '과',
        '하다', '하', '있다', '없다', '도', '한', '로', '에서', '께',
        '부터', '까지', '안', '그', '저', '이것', '그것', '저것', 
        '우리', '너', '당신', '이런', '저런', '그러나', '그리고', 
        '그러므로', '왜냐하면', '하지만', '또한', '즉', '그렇지만',
        '따라서', '때문에', '같은', '듯한', '처럼', '보다', '이렇게',
        '저렇게', '할', '할지', '같은', '그런', '어떤', '모든'
    ]
    all_stopwords = english_stopwords.union(korean_stopwords)

    cleaned_titles = []
    for title in titles:
        words = title.split()
        filtered_words = [word for word in words if word not in all_stopwords]
        cleaned_titles.append(' '.join(filtered_words))

    return cleaned_titles

class TextClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
def predict_label(new_sentence):
    model = TextClassificationModel(input_size=joblib.load('/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/WEBPAGE/vectorizerkimms.pkl').vocabulary_.__len__(), num_classes=9)
    model.load_state_dict(torch.load('/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/best_modelkimms.pth'))
    model.eval()

    vectorizer = joblib.load('/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/WEBPAGE/vectorizerkimms.pkl')
    label_encoder = joblib.load('/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/label_encoderkimms.pkl')

    cleaned_sentence = remove_special_characters(new_sentence)
    sentence_vector = vectorizer.transform([cleaned_sentence]).toarray()

    with torch.no_grad():
        sentence_tensor = torch.tensor(sentence_vector, dtype=torch.float32)
        output = model(sentence_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # 예측된 라벨을 원래의 라벨로 디코딩
    predicted_label = label_encoder.inverse_transform([prediction])
    if predicted_label[0]==5:
        return f'문학, 책'
    if predicted_label[0]==6:
        return f'영화'
    if predicted_label[0]==8:
        return f'미술, 디자인'
    if predicted_label[0]==7:
        return f'공연, 전시'
    if predicted_label[0]==11:
        return f'음악'
    if predicted_label[0]==9:
        return f'드라마'
    if predicted_label[0]==12:
        return f'스타, 연예인'
    if predicted_label[0]==13:
        return f'만화, 애니'
    if predicted_label[0]==10:
        return f'방송'

@app.route('/result_entertainment', methods=['POST'])
def result_entertainment():
    input_text = request.form['input_text'] # index.html에서 전송한 텍스트

    pred_section = predict_label(input_text)

    return render_template('result_entertainment.html', pred_section=pred_section)


# 생활/노하우/쇼핑 (한세진)
tokenizer1 = Okt()

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
category_map1 = {
    0: '육아,결혼',
    1: '반려동물',
    2: '패션',
    3: '인테리어',
    4: '요리',
    5: '상품리뷰',
    6: '원예'
}


model_path1 = os.path.join(os.path.dirname(__file__),'/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/bestmodel.pth')
token_to_id_path1 = os.path.join(os.path.dirname(__file__),'/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/token_to_id.joblib')
label_encoder_path1 = os.path.join(os.path.dirname(__file__),'/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/label_encoder.joblib')

# 디바이스 설정
device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
try:
    model1 = torch.load(model_path1, map_location=device1)
    model1.eval()  # 평가 모드로 설정
    print("Model loaded successfully")
except Exception as e:
    print("Content-Type: text/html\n")
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

# 토크나이저 및 필요한 데이터 로드
token_to_id1 = joblib.load(token_to_id_path1)
le1 = joblib.load(label_encoder_path1)
max_length1 = 32  # 또는 저장된 값에서 로드

def predict_category(model, text, tokenizer, token_to_id, max_length, device):
    tokens = tokenizer.morphs(text)
    ids = [token_to_id.get(token, token_to_id['<unk>']) for token in tokens]
    ids = ids[:max_length] + [token_to_id['<pad>']] * max(0, max_length - len(ids))
    input_tensor = torch.tensor([ids]).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    
    _, predicted = torch.max(output, 1)
    numeric_result = predicted.item()
    return numeric_result, category_map1.get(numeric_result, "알 수 없는 카테고리")


@app.route('/result_lifestyle', methods=['POST'])
def result_lifestyle():
    text1 = request.form['input_text'] # index.html에서 전송한 텍스트

    if text1:
        try:
            numeric_result, category = predict_category(model1, text1, tokenizer1, token_to_id1, max_length1, device1)
            msg = f"입력 텍스트의 예측 카테고리: {category}"
        except Exception as e:
            msg = f"예측 중 오류 발생: {str(e)}"
    else:
        msg = "텍스트를 입력해주세요."

    pred_section = msg

    return render_template('result_lifestyle.html', pred_section=pred_section)


# 지식/동향 (황은혁)
# 형태소
okt = Okt()

# 모델 클래스
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        output = self.fc(pooled)
        return output

# 텍스트를 시퀀스로 변환
def text_to_sequence(text, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in text.split()]

MODEL_PATH = os.path.join(os.path.dirname(__file__), '/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/WEBPAGE/text_classification_model.pth')
VOCAB_PATH = os.path.join(os.path.dirname(__file__), '/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/WEBPAGE/vocab.pkl')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), '/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/6번째 프로젝트/WEBPAGE/label_encoder.pkl')

with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

device = torch.device('cpu')

# 모델 
model = TextClassifier(len(vocab), 100, len(le.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 텍스트 처리 
def classify_text(text):
    morphemes = ' '.join(okt.morphs(text))  # 형태소 분석
    sequence = text_to_sequence(morphemes, vocab)
    sequence = [min(t, len(vocab) - 1) for t in sequence]  # 인덱스 범위 확인
    sequence = torch.tensor([sequence]).to(device)

    with torch.no_grad():
        output = model(sequence)
        _, predicted = torch.max(output.data, 1)  # 가장 높은 확률의 클래스를 예측
    
    return predicted.item()

@app.route('/result_knowledge', methods=['POST'])
def result_knowledge():
    input_text = request.form['input_text'] # index.html에서 전송한 텍스트

    if input_text:
        # 입력된 텍스트 분류
        predicted_label = classify_text(input_text)
        
        # 예측된 결과를 해석
        if predicted_label == 0:
            pred_section = "건강, 의학 블로그입니다."
        elif predicted_label == 1:
            pred_section = "교육, 학문 블로그입니다."
        elif predicted_label == 2:
            pred_section = "IT, 컴퓨터 블로그입니다."
        else:
            pred_section = "알 수 없는 카테고리입니다."

    return render_template('result_knowledge.html', pred_section=pred_section, predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True, port=5001)