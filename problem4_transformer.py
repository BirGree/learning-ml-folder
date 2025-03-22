#第四题：使用transformer模型完成情感分析任务
import os
import torch
#from datasets import load_dataset
#这个是用于加载预训练模型的，但是因为在线数据集无法在云主机中访问，没用上
from transformers import BertTokenizer
#BertTokenizer是分词器，用于将文本转化为向量
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification
#AutoModelForSequenceClassification是一个类，用于加载预训练模型
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
#Trainer是一个类，用于训练模型
#TrainingArguments是一个类，用于管理训练的超参数

'''尝试在线下载HUGGING face的数据集不成功，改用本地读取的方式
dataset = load_dataset("imdb")'''
def load_local_imdb_data(direc):
    #读取本地数据集的函数
    texts= []
    labels= []
    #影评文本和标签，分别存储在texts和labels这两个列表中

    '''以下几行，本地读取正面和负面影评并标签'''
    positive_dir= os.path.join(direc, 'pos')
    #积极影评的目录

    for filename in os.listdir(positive_dir):
        #os.listdir()函数返回目录下的所有文件和文件夹，是一个列表，列表的项是文件名
        with open(os.path.join(positive_dir, filename), "r", encoding="utf-8") as file:
            #打开文件，r是读取模式，encoding是编码格式设为utf-8
            texts.append(file.read().strip())
            #texts是之前定义的列表，append()函数是在列表末尾添加新的对象
            #file.read()用于读取文件内容，strip()用于去掉字符串首尾的空格和换行符
            labels.append(1)
            #积极影评标签设为1
    
    negative_dir= os.path.join(direc, 'neg')
    for filename in os.listdir(negative_dir):
        with open(os.path.join(negative_dir, filename), "r", encoding="utf-8") as file:
            texts.append(file.read().strip()) 
            labels.append(0) 
    #消极的同理
    return texts, labels

dataset_path = "aclImdb/aclImdb"
#数据集的路径，train和test文件夹的上一层，是相对路径

train_texts, train_labels = load_local_imdb_data(os.path.join(dataset_path, "train"))
test_texts, test_labels = load_local_imdb_data(os.path.join(dataset_path, "test"))
#调用读取本地数据集的函数，分别读取训练集和测试集

#print(train_texts[1])
MODEL_NAME = "Untitled_Folder"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
#这是一个预训练模型，分词器将文本转化为向量
#BertTokenizer是一个类，from_pretrained()函数是加载预训练模型

class IMDBDataset(Dataset):
    #继承了Dataset这个类
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    #初始化函数，传入参数是文本，标签，分词器和最大长度

    def __len__(self):
        return len(self.texts)
    #文本的长度

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

train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer)
#创建Dataset对象，把数据集变为pytorch能读取的形式

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
#加载预训练的BERT，添加了一个二分类的分类层

#使用dataloader加载数据，分批次批量加载
BATCH_SIZE = 16
#批次的大小

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#训练集和测试集，shuffle用于自动打乱数据

#管理训练的超参数，如batch size批次大小和epochs轮数，控制训练的过程
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=1,  #训练轮数
    weight_decay=0.01,  #权重衰减，是一种正则化方法，正则化是为了防止过拟合
    logging_dir="./logs", #日志文件的路径
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
    #评估指标
)


trainer.train()
#训练模型，会有一些输出，包括loss值，训练时间等，让我们知道训练的进度
#训练完成后，会在output_dir指定的路径下生成一个checkpoint文件，这个文件就是训练好的模型