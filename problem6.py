import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

num_workers = 0
batch_size = 16
validation_ratio = 0.2

#对数据集的标准化，转化为张量
#参数 (0.5, 0.5, 0.5) 表示每个通道的均值，(0.5, 0.5, 0.5) 是每个通道的标准差
#这样做的目的是将图像像素值从 [0, 1] 映射到 [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#获取训练和测试集
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

#训练集的数量
num_train = len(train_data)
#获取全部索引并且打乱
indices = list(range(num_train))
np.random.shuffle(indices)
#按照一个比例把数据集划分为训练和测试集
split = int(np.floor(validation_ratio * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
#这些sampler都使用划分后的索引，保证批次数据随机选取
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#数据加载器，把数据集加载成了批次，线程数可以自己设置
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

#图像显示的函数，把标准化的图像回到[0,1]范围
#转化为matplotlib需要的(H,W,C)格式
def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

#使用for循环迭代数据加载器
for images, labels in train_loader:
    #确保图片在CPU上并转换为NumPy数组
    images = images.cpu().numpy()
    #只获取一个batch进行显示
    break

#绘制图像
fig = plt.figure(figsize=(25, 4))
#创建2x8图形，显示18张图，每个图像显示类标签
for idx in np.arange(16):
    ax = fig.add_subplot(2, 8, idx + 1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

#plt.show()
#这里只把第四张图像的每个通道的图像取出，并显示为灰度图像
rgb_img = np.squeeze(images[3])
channels = ['red channel', 'green channel', 'blue channel']

fig = plt.figure(figsize = (36, 36)) 
for idx in np.arange(rgb_img.shape[0]):
    ax = fig.add_subplot(1, 3, idx + 1)
    img = rgb_img[idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(channels[idx])
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center', size=8,
                    color='white' if img[x][y]<thresh else 'black')
plt.savefig ('D:/Documents/directory/problem6/one_channel.png',bbox_inches='tight')


#定义CNN类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层(32x32x3的图像)，
        # 参数为输入图像的通道数，输出的通道数，卷积核大小，对图像进行0填充
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 卷积层(16x16x16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 卷积层(8x8x32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # 最大池化层，大小2x2，步幅2，将图像大小减半了
        self.pool = nn.MaxPool2d(2, 2)
        # 线性层即为全连接层(64 * 4 * 4 -> 500)
        # 将卷积层提取的特征映射到类别输出
        # 第一个为特征图展平
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # 线性层(500 -> 10)输出10个类别
        self.fc2 = nn.Linear(500, 10)
        # dropout层 (p=0.3) 随机丢弃神经元，是百分比，防止过拟合
        self.dropout = nn.Dropout(0.4)
    # 这个方法用于前向传播
    def forward(self, x):
        # 输入的x通过卷积层和relu激活函数
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # 图像输出展平，便于输入到全连接层
        x = x.view(-1, 64 * 4 * 4)
        # 全连接层前添加dropout层
        x = self.dropout(x)
        # 全连接层
        x = F.relu(self.fc1(x))
        # dropout层
        x = self.dropout(x)
        # 全连接层
        x = self.fc2(x)
        return x

#CNN模型被定义
model = Net()
#print(model)

# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 使用随机梯度下降优化器(SGD)，学习率lr=0.01
#optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)

#训练模型
n_epochs = 16
train_losses = []
valid_losses = []

valid_loss_min = np.inf #跟踪损失函数的变化

for epoch in range(1, n_epochs+1):

    #训练和验证的损失函数
    train_loss = 0.0
    valid_loss = 0.0
    
    # 训练集的模型，将模式设为训练
    model.train()
    for data, target in train_loader:
        
        # 将之前的梯度清除
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 计算批次损失
        loss = criterion(output, target)
        # 反向传播，计算损失的梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 计算累计的损失
        train_loss += loss.item()*data.size(0)
         
    # 验证集的模型，将模式设为评估
    model.eval()
    for data, target in valid_loader:

        # 前向传播
        output = model(data)
        # 计算批次损失
        loss = criterion(output, target)
        # 计算累计的损失
        valid_loss += loss.item()*data.size(0)
    
    # 计算平均损失
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # 显示训练集与验证集的损失函数 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # 打印损失函数的下降过程
    # 如果验证集损失函数减少，就保存模型。
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # 设置模型保存的目录和保存的名字
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

#绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, n_epochs + 1), valid_losses, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig ('D:/Documents/directory/problem6/loss_graph.png',bbox_inches='tight')

# 加载保存好的模型
model.load_state_dict(torch.load('model_cifar.pt'))
# 初始化测试值的变量
test_loss = 0.0 #测试样本的损失
class_correct = list(0. for i in range(10)) #正确的预测数
class_total = list(0. for i in range(10)) #全部的样本数

# 模型设置为评估模式，禁用dropout和批归一化等操作
model.eval()
# 计算测试集的损失和准确性
for data, target in test_loader:

    # 前向传播，获取输出
    output = model(data)
    # 计算批次损失
    loss = criterion(output, target)
    # 计算累计损失
    test_loss += loss.item()*data.size(0)
    # 获取预测的类别
    _, pred = torch.max(output, 1)    
    # 比较预测标签和真正的标签，转换为numpy数组
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    
    # 按类别统计正确数和总数
    for i in range(batch_size):
        # 获取标签
        label = target.data[i]
        #预测正确就计数加一，总数也加
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# 计算平均测试损失
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

#每个类别的准确性（百分比）
for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
#整体准确性（百分比）
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

#显示一批测试图像
for images, labels in test_loader:
    #确保图片在CPU上并转换为NumPy数组
    images = images.cpu().numpy()
    break  #只获取一个batch进行显示

# 获取输出样本
images_tensor = torch.tensor(images).float()
# 用模型预测那些样本
output = model(images_tensor)
# 获取预测类别，最大概率值对应的索引，并转为numpy数组
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.cpu().numpy())

# 含标签，显示图像并保存
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(16):
    ax = fig.add_subplot(2, 8, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))

plt.savefig ('D:/Documents/directory/problem6/test_accuracy.png',bbox_inches='tight')
