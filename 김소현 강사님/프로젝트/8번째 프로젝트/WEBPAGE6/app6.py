from flask import Flask, request, render_template, redirect
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, DenseNet169_Weights
from KDTModule import *

app = Flask(__name__)

# 업로드할 이미지 저장 경로
UPLOAD_FOLDER = '/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/8번째 프로젝트/WEBPAGE6/uploads'
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

        best_model = models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        best_model.classifier = nn.Sequential(
                                nn.Linear(best_model.classifier.in_features, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 128),
                                nn.ReLU(),
                                nn.Linear(128, 7)
                            ) # 전결합층 입력 출력 변경
        pth_PATH = '/Users/anhyojun/WorkSpace/KDT2/김소현 강사님/프로젝트/개인 프로젝트/DenseNet169.pth'
        checkpoint = torch.load(pth_PATH)
        best_model.load_state_dict(checkpoint['model_state_dict']) # 모델에 가중치 설정

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

        name_dict = {0 : '애니메이션', 1 : '액션,어드벤처,판타지,SF', 2 : '공포,미스터리', 3 : '다큐멘터리',
                     4 : '드라마,멜로·로맨스', 5 : '범죄,스릴러,서스펜스', 6 : '코미디'}

        pred_answer = name_dict[pred]


        return render_template('result.html', pred_answer=pred_answer)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True, port=5007)
