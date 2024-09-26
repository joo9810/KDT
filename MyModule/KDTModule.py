import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torchmetrics.regression import *
from torchmetrics.classification import *
from torchmetrics.functional.regression import r2_score
from torchmetrics.functional.classification import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ----------------------------------------------------
# 클래스 목적 : 학습용 데이터셋 텐서화 및 전처리
# 클래스 이름 : CustomDataSet
# 부모 클래스 : torch.utils.data.Dataset
# 매개   변수 : featureDF, targetDF
# ----------------------------------------------------

class CustomDataset(Dataset):
    # 데이터 로딩 및 전처리 진행과 인스턴스 생성 메서드
    def __init__(self, feature, target, feature_dim=1):
        super().__init__()
        self.feature = feature
        self.target = target
        self.n_rows = feature.shape[0]
        self.feature_dim = feature_dim
        if isinstance(feature, pd.DataFrame): # 입력이 데이터 프레임이라면
            self.n_features = feature.shape[1]

    # 데이터의 개수 반환 메서드
    def __len__(self):
        return self.n_rows

    # 특정 index의 데이터와 타겟 반환 메서드 => Tensor 반환!!!
    def __getitem__(self, idx): # 클래스 인스턴스 생성하면 자동으로 호출되는 콜백 메서드
        if isinstance(self.feature, pd.DataFrame):
            if self.feature_dim == 1:
                featureTS = torch.FloatTensor(self.feature.iloc[idx].values)
            elif self.feature_dim == 2:
                featureTS = torch.FloatTensor(self.feature.iloc[idx].values).unsqueeze(0)
            targetTS = torch.FloatTensor(self.target.iloc[idx].values)
            return featureTS, targetTS
        elif isinstance(self.feature, np.ndarray):
            if self.feature_dim == 1:
                featureTS = torch.FloatTensor(self.feature[idx])
            elif self.feature_dim == 2:
                featureTS = torch.FloatTensor(self.feature[idx]).unsqueeze(0)
            targetTS = torch.FloatTensor(self.target)[[idx]]
            return featureTS, targetTS

# -----------------------------------------------------------------
# 사용자 정의 모델 클래스
# -----------------------------------------------------------------
# 부모 클래스 : nn.Module
# 필수 오버라이딩 : 
#     => __init__()  : 모델 층 구성 (설계)
#     => forward()   : 순방향 학습 진행 코드 구현
# -----------------------------------------------------------------

# cpu로 할지 gpu로 할지
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 입력 피쳐 수, 은닉층 퍼셉트론 수, 은닉층 개수가 모두 동적인 모델
class LinearModel(nn.Module):
    def __init__(self, input_in, output_out, hidden_list,
                 act_func, model_type):
        super().__init__() # 부모 클래스 생성

        # 입력층
        self.input_layer = nn.Linear(input_in, hidden_list[0])
        # 은닉층
        self.hidden_layer_list = nn.ModuleList()
        for i in range(len(hidden_list)-1):
            self.hidden_layer_list.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
        # 출력층
        self.output_layer = nn.Linear(hidden_list[-1], output_out)

        self.act_func = act_func
        self.model_type = model_type

    # 순방향/전방향 학습 진행 시 자동 호출되는 메서드 (콜백 함수: 시스템에서 호출되는 함수)
    def forward(self, x):
        # 입력층 학습
        x = self.input_layer(x)
        x = self.act_func(x)

        # 은닉층 학습
        for i in self.hidden_layer_list:
            x = i(x)
            x = self.act_func(x)

        if self.model_type == 'regression': # 회귀
            return self.output_layer(x) # 활성화 함수 안거치고 출력
        elif self.model_type == 'binary': # 이진 분류
            return torch.sigmoid(self.output_layer(x)) # 시그모이드
        elif self.model_type == 'multiclass': # 다중 분류
            return self.output_layer(x) # 소프트맥스
            # 다중 분류의 경우 CrossEntropyLoss에서 내부적으로 log-softmax를 처리하기 때문에
            # 모델의 마지막 출력에서 소프트맥스를 적용하지 않고 바로 전달하면 됨


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_list, act_func, model_type, num_layers=1):
        super().__init__()
        
        # LSTM 레이어 (입력 크기, 은닉 크기, 레이어 수, batch_first를 True로 설정하여 배치가 첫 번째 차원이 되게 함)
        self.lstm = nn.LSTM(input_size, hidden_list[0], num_layers=num_layers, batch_first=True)
        
        # 은닉층
        self.hidden_layer_list = nn.ModuleList()
        for i in range(len(hidden_list)-1):
            self.hidden_layer_list.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
        
        # 출력층
        self.output_layer = nn.Linear(hidden_list[-1], output_size)

        self.act_func = act_func
        self.model_type = model_type

    def forward(self, x):
        # LSTM 레이어를 통과 (x: 배치 크기, 시퀀스 길이, 입력 크기)
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out은 모든 타임스텝의 출력을 포함, hn은 마지막 타임스텝의 출력
        
        # LSTM의 마지막 타임스텝 출력만 사용
        x = lstm_out[:, -1, :]  # 마지막 타임스텝의 출력을 사용
        
        # 은닉층
        for layer in self.hidden_layer_list:
            x = layer(x)
            x = self.act_func(x)

        # 출력층
        if self.model_type == 'regression':  # 회귀
            return self.output_layer(x)
        elif self.model_type == 'binary':  # 이진 분류
            return torch.sigmoid(self.output_layer(x))
        elif self.model_type == 'multiclass':  # 다중 분류
            return self.output_layer(x)  # CrossEntropyLoss에서 log-softmax 처리


class CNNModel(nn.Module):
    def __init__(self, input_cnn1, output_cnn1, output_cnn2, hidden_list,
                 output_classes ,kernel_size, padding1, padding2, dropout_prob,
                 image_height_size, image_width_size):
        super().__init__()

        # 첫 번째 합성곱 계층 정의 (동적으로 입력 채널과 은닉 레이어 적용)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_cnn1, out_channels=output_cnn1,
                      kernel_size=kernel_size, padding=padding1),
            nn.BatchNorm2d(output_cnn1), # 배치 정규화
            nn.ReLU(), # 활성화 함수
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 출력크기 = (입력크기 - 커널크기 + 2*패딩크기)//스트라이드크기 + 1
        conv1_output_height = (image_height_size - kernel_size + 2 * padding1) // 2 + 1
        conv1_output_width = (image_width_size - kernel_size + 2 * padding1) // 2 + 1

        # 두 번째 합성곱 계층 정의
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=output_cnn1, out_channels=output_cnn2,
                      kernel_size=kernel_size, padding=padding2),
            nn.BatchNorm2d(output_cnn2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        conv2_output_height = (conv1_output_height - kernel_size + 2*padding2) // 2 + 1
        conv2_output_width = (conv1_output_width - kernel_size + 2*padding2) // 2 + 1

        # 평탄화 후 전결합층의 입력 크기 계산
        self.fc_input_size = output_cnn2 * conv2_output_height * conv2_output_width

        # 전결합층 정의
        self.drop = nn.Dropout(dropout_prob) # 드롭아웃 설정
        self.fc1 = nn.Linear(in_features=self.fc_input_size, out_features=hidden_list[0])
        self.fc2_list = nn.ModuleList()
        for i in range(len(hidden_list)-1):
            self.fc2_list.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
        self.fc3 = nn.Linear(in_features=hidden_list[-1], out_features=output_classes)

    def forward(self, x):
        # 합성곱 계층 통과
        x = self.layer1(x)
        x = self.layer2(x)

        # 1D로 평탄화 (배치 크기, 채널 수, 높이, 너비) => (배치 크기, -1)
        x = x.view(x.shape[0], -1)

        # 드롭아웃 및 전결합층 통과
        x = self.drop(x)

        x = self.fc1(x)
        x = F.relu(x)

        for i in self.fc2_list:
            x = i(x)
            x = F.relu(x)

        x = self.fc3(x)
        return x
        

# -----------------------------------------------------------------
## 테스트/검증 함수 
# ==> 가중치, 절편 업데이트 X, 최적화 미진행
# ==> 현재 가중치와 절편값으로 테스트 진행
# -----------------------------------------------------------------
def testing(test_DataLoader, model, model_type, num_classes=None):
    model.eval() # 검증 모드임을 명시적으로 선언 (검증용 통계치 사용 및 드롭아웃 비활성화)
    total_loss_test = 0
    total_score_test = 0

    with torch.no_grad(): # 가중치 업데이트 없이 테스트 진행
        for X_batch, y_batch in test_DataLoader:
            # (1) 순전파 (평가)
            pred_test_y = model(X_batch)
            # (2) 손실 함수 계산

            if model_type == 'regression': # 회귀일 때
                loss_test = F.mse_loss(pred_test_y, y_batch)
                score_test = r2_score(pred_test_y, y_batch)
            elif model_type == 'binary': # 이진 분류일 때
                loss_test = F.binary_cross_entropy(pred_test_y, y_batch)
                score_test = f1_score(pred_test_y, y_batch, task='binary')
            elif model_type == 'multiclass': # 다중 분류일 때
                y_batch1D = y_batch.reshape(-1) # 다중 분류는 y가 반드시 1차원이어야 함.. (너무 불친절)
                loss_test = F.cross_entropy(pred_test_y, y_batch1D.long())
                pred_test_labels = torch.argmax(pred_test_y, dim=1)
                # dim=1을 해야 한 행 내에서 가장 큰 원소의 인덱스를 가져옴
                score_test = f1_score(pred_test_labels, y_batch1D,
                                      task='multiclass', num_classes=num_classes)
                # 다중 분류는 long타입으로 전달해야 하는듯

            total_loss_test += loss_test.item()
            total_score_test += score_test.item()

    loss_test_avg = total_loss_test / len(test_DataLoader)
    score_test_avg = total_score_test / len(test_DataLoader)
    return loss_test_avg, score_test_avg

# -----------------------------------------------------------------
# 모델 학습 함수
# -----------------------------------------------------------------
def training(train_DataLoader, test_DataLoader, model, model_type, optimizer,
             epoch = 1000, endurance_cnt = 10, view_epoch=1, num_classes=None, SAVE_PATH=None):
    model.train() # 학습 모드임을 명시적으로 선언 (학습용 통계치 사용)
    loss_train_history = []
    loss_test_history = []
    score_train_history = []
    score_test_history = []

    EARLY_STOP_LOSS_CNT = 0

    for i in range(1, epoch+1): # 에포크 횟수만큼 반복
        total_loss_train = 0 # 한 에포크당 합산할 손실값 (나중에 평균 계산)
        total_score_train = 0 # 한 에포크당 합산할 손실값 (나중에 평균 계산)

        for X_batch, y_batch in train_DataLoader:
            # (1) 순전파 (학습)
            pred_train_y = model(X_batch)
            # (2) 손실 함수 계산
            if model_type == 'regression': # 회귀일 때
                loss_train = F.mse_loss(pred_train_y, y_batch)
                score_train = r2_score(pred_train_y, y_batch)
            elif model_type == 'binary': # 이진 분류일 때
                loss_train = F.binary_cross_entropy(pred_train_y, y_batch)
                score_train = f1_score(pred_train_y, y_batch, task='binary')
            elif model_type == 'multiclass': # 다중 분류일 때
                y_batch1D = y_batch.reshape(-1) # 다중 분류는 y가 반드시 1차원이어야 함.. (너무 불친절)
                loss_train = F.cross_entropy(pred_train_y, y_batch1D.long())
                pred_train_labels = torch.argmax(pred_train_y, dim=1)
                # dim=1을 해야 한 행 내에서 가장 큰 원소의 인덱스를 가져옴
                score_train = f1_score(pred_train_labels, y_batch1D,
                                       task='multiclass', num_classes=num_classes)
                # 다중 분류는 long타입으로 전달해야 하는듯

            # (3) 최적화
            optimizer.zero_grad() # 그레디언트 초기화
            loss_train.backward() # 역전파 하면서 그레디언트 계산
            optimizer.step() # 가중치, 절편 업데이트
            # (4) 손실값 합산
            total_loss_train += loss_train.item()
            total_score_train += score_train.item()

        loss_train_avg = total_loss_train / len(train_DataLoader)
        score_train_avg = total_score_train / len(train_DataLoader)
        
        # 한 에포크마다 테스트 실행
        if model_type == 'regression':
            loss_test_avg, score_test_avg = testing(test_DataLoader, model, model_type='regression')
        elif model_type == 'binary':
            loss_test_avg, score_test_avg = testing(test_DataLoader, model, model_type='binary')
        elif model_type == 'multiclass':
            loss_test_avg, score_test_avg = testing(test_DataLoader, model,
                                    model_type='multiclass', num_classes=num_classes)

        loss_train_history.append(loss_train_avg)
        loss_test_history.append(loss_test_avg)
        score_train_history.append(score_train_avg)
        score_test_history.append(score_test_avg)

        if len(loss_test_history) == 1: # 첫 에포크일때
            best_loss = loss_test_avg
            if SAVE_PATH != None:
                torch.save(model.state_dict(), f'{SAVE_PATH}/best_model_epoch_{i}.pth')
            print(f"[EPOCH] : {i}에서 모델 저장 완료.")

        else:
            if best_loss > loss_test_avg: # 손실값이 개선 됐다면
                best_loss = loss_test_avg
                EARLY_STOP_LOSS_CNT = 0 # 개선되면 카운트 초기화
                if SAVE_PATH != None:
                    torch.save(model.state_dict(), f'{SAVE_PATH}/best_model_epoch_{i}.pth')
                print(f"[EPOCH] : {i}에서 모델 저장 완료.")
            else:                         # 손실값이 개선되지 않았다면
                EARLY_STOP_LOSS_CNT += 1

        if EARLY_STOP_LOSS_CNT == endurance_cnt:
            print(f'[Loss]값의 개선이 이루어지지 않아 [{i}] EPOCH에서 학습을 종료합니다.')
            break

        # (4) 학습 결과 출력
        if i % view_epoch == 0:
            print(f"[Loss : {i}/{epoch}] Train : {loss_train_avg:.4f}, Test : {loss_test_avg:.4f}")
            print(f"[Score  : {i}/{epoch}] Train : {score_train_avg:.4f}, Test : {score_test_avg:.4f}")
    
    return loss_train_history, loss_test_history, score_train_history, score_test_history

# -----------------------------------------------------------------
# 손실값과 스코어 그려주는 함수
# -----------------------------------------------------------------

def DrawPlot(result):
    fig, axs = plt.subplots(1, 2, figsize = (14, 5))

    label_list = ['Loss', 'Score']

    LENGTH = len(result[0])

    for i in range(2):
        axs[i].plot(range(1, LENGTH+1), result[2*i], label = f'Train {label_list[i]}')
        axs[i].plot(range(1, LENGTH+1), result[2*i+1], label = f'Valid {label_list[i]}')
        axs[i].set_title(label_list[i])
        axs[i].set_xlabel('EPOCH')
        axs[i].set_ylabel('Loss')
        axs[i].legend()
    return fig, axs

# -----------------------------------------------------------------
# 예측값 출력하는 함수
# -----------------------------------------------------------------

def predict_value(test_inputDF, model, dim):
    if dim == 2:
        test_inputTS = torch.FloatTensor(test_inputDF.values)
        return torch.argmax(model(test_inputTS), dim=1)
    elif dim == 3:
        test_inputTS = torch.FloatTensor(test_inputDF.values).reshape(1,1,-1)
        return torch.argmax(model(test_inputTS), dim=1)