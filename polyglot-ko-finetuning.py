from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import time
import numpy as np
import random
import os

# 환경 변수 설정 (Tokenizer 병렬 처리 비활성화)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 랜덤 시드 설정
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# 토크나이저 및 모델 불러오기
model_name = "EleutherAI/polyglot-ko-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)

# 데이터셋 불러오기 및 전처리
def preprocess_function(examples):
    inputs = [f"{original} -> {reference} {tokenizer.eos_token}" for original, reference in zip(examples['original'], examples['reference'])]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    # Add labels for language modeling
    model_inputs['labels'] = model_inputs['input_ids'].copy()
    return model_inputs

dataset = load_dataset("csv", data_files="ParaKQC_v1_pair.csv")['train']
dataset = dataset.train_test_split(test_size=0.2, seed=random_seed)
train_dataset = dataset['train']
test_dataset = dataset['test']

train_valid_split = train_dataset.train_test_split(test_size=0.1, seed=random_seed)
train_dataset = train_valid_split['train']
valid_dataset = train_valid_split['test']

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# 데이터셋 확인
for i in range(5):
    print(dataset['train'][i])

# 에포크 시간 콜백 정의
class EpochTimerCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {state.epoch} took {epoch_time:.2f} seconds")

# 데이터 콜레이터 정의
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # Polyglot-Ko 1.3B 모델은 큰 메모리를 요구하므로 batch size를 줄입니다.
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=2,
    fp16=True  # Mixed precision training을 사용하여 메모리 절약
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EpochTimerCallback()]
)

# # 모델 학습
trainer.train()

# # 학습된 모델 저장
model.save_pretrained("fine-tuned-try01")
tokenizer.save_pretrained("fine-tuned-try01")

# 테스트용 문장
input_text = "메일을 다 비울까 아니면 안읽은 것만 지울까?"

# 토큰화 및 모델 추론
input_ids = tokenizer.encode(input_text + " ->", return_tensors="pt", truncation=True, padding="max_length", max_length=128)  # input_ids를 GPU로 이동
outputs = model.generate(
    input_ids, 
    max_new_tokens=50,  # max_new_tokens로 생성 길이 설정
    pad_token_id=tokenizer.pad_token_id, 
    eos_token_id=tokenizer.eos_token_id, 
    do_sample=True,  # sampling을 사용할 경우
    top_k=50,  # top-k sampling 설정
    top_p=0.95  # top-p (nucleus) sampling 설정
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

import evaluate
# 평가 라이브러리 불러오기
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# 테스트 데이터셋에 대한 예측 및 점수 계산
test_predictions = []
test_references = []

for example in test_dataset:
    input_text = example['original'] + " ->"
    reference_text = example['reference']

    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)  # input_ids를 GPU로 이동
    outputs = model.generate(
        input_ids, 
        max_new_tokens=50,  # max_new_tokens로 생성 길이 설정
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id=tokenizer.eos_token_id, 
        do_sample=True,  # sampling을 사용할 경우
        top_k=50,  # top-k sampling 설정
        top_p=0.95  # top-p (nucleus) sampling 설정
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    test_predictions.append(generated_text)
    test_references.append(reference_text)

# BLEU Score 계산
bleu_result = bleu.compute(predictions=test_predictions, references=[[ref] for ref in test_references])
print(f"BLEU score: {bleu_result['bleu']}")

# BERTScore 계산
bertscore_result = bertscore.compute(predictions=test_predictions, references=test_references, lang="ko")
print(f"BERTScore - Precision: {np.mean(bertscore_result['precision'])}, Recall: {np.mean(bertscore_result['recall'])}, F1: {np.mean(bertscore_result['f1'])}")