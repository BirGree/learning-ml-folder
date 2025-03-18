import os
import torch
#from datasets import load_dataset
from transformers import BertTokenizer
#创建自定义数据集
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

'''尝试在线下载HUGGING face的数据集不成功，改用本地读取的方式
dataset = load_dataset("imdb")'''
def load_local_imdb_data(direc):
    texts= []
    labels= []
    #本地读取正面和负面影评并标签
    positive_dir= os.path.join(direc, 'pos')
    for filename in os.listdir(positive_dir):
        with open(os.path.join(positive_dir, filename), "r", encoding="utf-8") as file:
            texts.append(file.read().strip()) 
            labels.append(1) 
    
    negative_dir= os.path.join(direc, 'neg')
    for filename in os.listdir(negative_dir):
        with open(os.path.join(negative_dir, filename), "r", encoding="utf-8") as file:
            texts.append(file.read().strip()) 
            labels.append(0) 
    
    return texts, labels

dataset_path = "aclImdb"  #数据集的路径

#加载本地数据集
train_texts, train_labels = load_local_imdb_data(os.path.join(dataset_path, "train"))
test_texts, test_labels = load_local_imdb_data(os.path.join(dataset_path, "test"))

#print(train_texts[1])
#这是一个预训练模型，分词器将文本转化为向量
MODEL_NAME = "bertmodel"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

#继承了Dataset这个类
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

#创建Dataset对象，把数据集变为pytorch能读取的形式
train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer)

#加载预训练的BERT，添加了一个二分类的分类层
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

#使用dataloader加载数据，分批次批量加载
#批次的大小
BATCH_SIZE = 8
#训练集和测试集，shuffle用于自动打乱数据
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#用于管理训练的超参数，如batch size和epochs，控制训练的过程
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=1,  # 训练轮数
    weight_decay=0.01,
    logging_dir="./logs",
)
#自动训练
trainer = Trainer(
    model=model,
    #训练参数
    args=training_args,
    #训练数据
    train_dataset=train_dataset,
    #测试数据
    eval_dataset=test_dataset,
)


trainer.train()

def predict_sentiment(texts, model, tokenizer):
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return ["Positive" if p == 1 else "Negative" for p in predictions]

sample_texts = ["This movie was absolutely fantastic!", "The film was boring and too long."]
predictions = predict_sentiment(sample_texts, model, tokenizer)

for text, sentiment in zip(sample_texts, predictions):
    print(f"Review: {text} | Sentiment: {sentiment}")