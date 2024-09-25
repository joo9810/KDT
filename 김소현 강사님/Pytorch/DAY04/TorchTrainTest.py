import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from torchmetrics.regression import *
from torchmetrics.classification import *
from torchmetrics.functional.regression import r2_score
from torchmetrics.functional.classification import f1_score

# -----------------------------------------------------------------
## 테스트/검증 함수 
# ==> 가중치, 절편 업데이트 X, 최적화 미진행
# ==> 현재 가중치와 절편값으로 테스트 진행
# -----------------------------------------------------------------
def testing(test_DataLoader, model, is_reg=True, is_bin=None, num_classes=None):
    model.eval() # 검증 모드임을 명시적으로 선언 (검증용 통계치 사용 및 드롭아웃 비활성화)
    total_loss_test = 0
    total_score_test = 0

    with torch.no_grad(): # 가중치 업데이트 없이 테스트 진행
        for X_batch, y_batch in test_DataLoader:
            # (1) 순전파 (평가)
            pred_test_y = model(X_batch)
            # (2) 손실 함수 계산

            if is_reg == True: # 회귀일 때
                loss_test = F.mse_loss(pred_test_y, y_batch)
                score_test = r2_score(pred_test_y, y_batch)
            elif is_reg == False and is_bin == True: # 이진 분류일 때
                loss_test = F.binary_cross_entropy(pred_test_y, y_batch)
                score_test = f1_score(pred_test_y, y_batch)
            elif is_reg == False and is_bin == False: # 다중 분류일 때
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
def training(train_DataLoader, test_DataLoader, model, optimizer,epoch = 1000,
             view_epoch=1, is_reg=True, is_bin=None, num_classes=None):
    model.train() # 학습 모드임을 명시적으로 선언 (학습용 통계치 사용)
    loss_train_history = []
    loss_test_history = []
    score_train_history = []
    score_test_history = []

    for i in range(1, epoch+1): # 에포크 횟수만큼 반복
        total_loss_train = 0 # 한 에포크당 합산할 손실값 (나중에 평균 계산)
        total_score_train = 0 # 한 에포크당 합산할 손실값 (나중에 평균 계산)

        for X_batch, y_batch in train_DataLoader:
            # (1) 순전파 (학습)
            pred_train_y = model(X_batch)
            # (2) 손실 함수 계산
            if is_reg == True: # 회귀일 때
                loss_train = F.mse_loss(pred_train_y, y_batch)
                score_train = r2_score(pred_train_y, y_batch)
            elif is_reg == False and is_bin == True: # 이진 분류일 때
                loss_train = F.binary_cross_entropy(pred_train_y, y_batch)
                score_train = f1_score(pred_train_y, y_batch)
            elif is_reg == False and is_bin == False: # 다중 분류일 때
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
        if is_reg == True:
            loss_test_avg = testing(test_DataLoader, model, is_reg=True)[0]
            score_test_avg = testing(test_DataLoader, model, is_reg=True)[1]
        elif is_reg == False and is_bin == True:
            loss_test_avg = testing(test_DataLoader, model, is_reg=False, is_bin=True)[0]
            score_test_avg = testing(test_DataLoader, model, is_reg=False, is_bin=True)[1]
        elif is_reg == False and is_bin == False:
            loss_test_avg = testing(test_DataLoader, model,
                                    is_reg=False, is_bin=False, num_classes=num_classes)[0]
            score_test_avg = testing(test_DataLoader, model,
                                     is_reg=False, is_bin=False, num_classes=num_classes)[1]

        loss_train_history.append(loss_train_avg)
        loss_test_history.append(loss_test_avg)
        score_train_history.append(score_train_avg)
        score_test_history.append(score_test_avg)

        # (4) 학습 결과 출력
        if i % view_epoch == 0:
            print(f"[Loss : {i}/{epoch}] Train : {loss_train_avg:.4f}, Test : {loss_test_avg:.4f}")
            print(f"[Score  : {i}/{epoch}] Train : {score_train_avg:.4f}, Test : {score_test_avg:.4f}")
    
    return loss_train_history, loss_test_history, score_train_history, score_test_history