from flask import Flask, request, render_template, redirect
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

import sys
sys.path.append('/Users/anhyojun/WorkSpace/KDT/MyModule')
from KDTModule import *


app = Flask(__name__)

# 업로드할 이미지 저장 경로
UPLOAD_FOLDER = '/Users/anhyojun/WorkSpace/KDT/김소현 강사님/프로젝트/5번째 프로젝트/WEBPAGE/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 업로드 폴더가 없으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')  # 'file' 키의 파일 가져오기

    # 파일이 존재하면 저장하고 텐서로 변환
    if file and file.filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # 파일 저장

        name_dict = {0 : '맹구', 1 : '유리', 2: '짱구' , 3 : '철수', 4 : '훈이'}

        best_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) # resnet50 모델 불러오기
        best_model.fc = nn.Linear(best_model.fc.in_features, 5) # 전결합층 입력 출력 변경
        pth_PATH = '/Users/anhyojun/WorkSpace/KDT/김소현 강사님/프로젝트/5번째 프로젝트/WEBPAGE/best_model_epoch_41.pth'
        best_model.load_state_dict(torch.load(pth_PATH, weights_only=True)) # 모델에 가중치 설정

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 정규화 값
        ])

        image = Image.open(file_path).convert('RGB')
        tensor_image = transform(image)
        input_tensor = tensor_image.unsqueeze(0)
        best_model.eval()
        pred = torch.argmax(best_model(input_tensor), dim=1).item()
        pred_answer = name_dict[pred]      

        background_image = f'../static/{pred_answer}.jpg'

        return render_template('result.html', pred_answer=pred_answer, background_image=background_image)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
