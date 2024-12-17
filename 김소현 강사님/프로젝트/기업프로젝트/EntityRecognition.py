from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from torch.optim import AdamW

class NERTrainer:
    def __init__(self, model_name='klue/bert-base', num_labels=3, label_names=['그 외', '매장명', '음식명']):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.label_names = label_names

    def make_new_dataset(self, file_path, max_length=128):
        dataset = load_dataset('json', data_files=file_path)

        tokens_list = []
        ner_tags_list = []
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        labels_list = []

        # file_name = file_path.split('.')[0] # 'train' 글자 추출 용도
        file_name = 'train'

        for i in range(len(dataset[file_name])):
            tokens = dataset[file_name][i]['tokens'] # dataset['train'][0]['tokens'] 형식
            tokens_list.append(tokens)

            ner_tags = dataset[file_name][i]['ner_tags']
            ner_tags_list.append(ner_tags)

            tokenized_inputs = self.tokenizer(tokens,
                                              truncation=True,
                                              padding='max_length',
                                              max_length=max_length,
                                              is_split_into_words=True)
            # truncation=True : 시퀀스가 모델의 최대 길이를 초과할 경우, 초과 부분을 잘라냄
            input_ids = tokenized_inputs['input_ids']
            input_ids_list.append(input_ids)

            token_type_ids = tokenized_inputs['token_type_ids']
            token_type_ids_list.append(token_type_ids)

            attention_mask = tokenized_inputs['attention_mask']
            attention_mask_list.append(attention_mask)

            word_ids = tokenized_inputs.word_ids(batch_index=0)
            aligned_labels = [ner_tags[word_id] if word_id is not None else -100 for word_id in word_ids]
            labels_list.append(aligned_labels)

        data_dict = {
            'tokens' : tokens_list,
            'ner_tags' : ner_tags_list,
            'input_ids' : input_ids_list,
            'token_type_ids' : token_type_ids_list,
            'attention_mask' : attention_mask_list,
            'labels' : labels_list
        }

        new_dataset = Dataset.from_dict(data_dict)

        return new_dataset
    
    def compute_metrics(pred):
        predictions = pred.predictions
        label_ids = pred.label_ids

        # 로짓에서 가장 높은 점수의 클래스를 선택
        preds = np.argmax(predictions, axis=2)

        pred_list, target_list = [], []
        for pred, label in zip(preds, label_ids):
            pred_list.extend(pred)
            target_list.extend([0 if i==-100 else i for i in label])

        f1 = f1_score(target_list, pred_list, average='weighted')
        accuracy = accuracy_score(target_list, pred_list)

        return {'f1' : f1, 'accuracy' : accuracy}

    def split_training(self, new_dataset, optimizer, test_size=0.2, batch_size=1,
                       epoch=3, weight_decay=0.01, compute_metrics=compute_metrics):
        split_dataset = new_dataset.train_test_split(test_size=test_size)

        training_args = TrainingArguments(
            output_dir="./results",                 # 모델 학습 결과를 저장할 경로
            # learning_rate=lr,                       # 학습률 (Trainer에 optimizer를 명시적으로 준 경우에는 불필요)
            per_device_train_batch_size=batch_size,          # 학습 시 배치 크기
            per_device_eval_batch_size=batch_size,           # 평가 시 배치 크기
            num_train_epochs=epoch,                     # 총 학습 epoch 수
            weight_decay=weight_decay,                      # 가중치 감소율 (과적합 방지)
            eval_strategy="epoch",                  # 매 epoch 마다 평가 진행
            save_strategy="epoch",                   # 매 epoch 마다 checkpoint에 모델 저장
            save_total_limit=1,                      # 모델을 1개만 저장 (최고 성능 모델)
            load_best_model_at_end=True,             # 학습 종료 후 가장 좋은 모델 불러오기
            metric_for_best_model="f1",              # 기준 메트릭
            greater_is_better=True,                 # 기준 메트릭이 높을수록 좋은 경우
            seed=2024
        )

        self.trainer = Trainer(
            model=self.model,                       # 훈련할 모델
            args=training_args,                     # 학습 파라미터
            train_dataset=split_dataset["train"],   # 훈련 데이터셋
            eval_dataset=split_dataset["test"],     # 평가 데이터셋
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # 3번 연속 개선이 없으면 종료
            optimizers=(optimizer, None)
        )

        self.trainer.train()

        best_model_path = training_args.output_dir
        self.model.save_pretrained(best_model_path + "/model")
        self.tokenizer.save_pretrained(best_model_path + "/tokenizer")

        return split_dataset

    def evaluate(self, dataset):
        predictions, label_ids, metrics = self.trainer.predict(dataset)
        preds = np.argmax(predictions, axis=2)

        pred_list, target_list = [], []
        for pred, label in zip(preds, label_ids):
            pred_list.extend(pred)
            target_list.extend([0 if i==-100 else i for i in label])

        report = classification_report(target_list, pred_list, target_names=self.label_names)
        return report