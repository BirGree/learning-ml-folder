#第五题：生成对抗网络（GAN）
#第一次编辑，使用的损失函数是二分类交叉熵损失函数
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import random
os.makedirs('output_images/fake_images', exist_ok=True)
file_path = '/root/autodl-tmp/output_images'

seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

dataroot = '/root/autodl-tmp/mnist_jpg' #数据集的路径
workers = 2 #加载数据的线程数
batch_size = 128 #每个批次的大小
image_size = 28 #图像的大小
nc = 1 #图像的通道数
nz = 100 #噪声的大小
ngf = 28 #生成器的特征图大小
ndf = 28 #判别器的特征图大小
num_epochs = 51 #训练的轮数
lr = 0.0002 #学习率
beta1 = 0.5 #Adam优化器的参数
ngpu = 1 #GPU的数量

class MNISTdataset(Dataset):
    def __init__(self, image_folder, transform=None):
        """
        Args:
            image_folder (str): 存储图片的文件夹路径。
            transform (callable, optional): 可选的转换函数。
        """
        self.image_folder = image_folder
        self.transform = transform
        
        # 获取文件夹中所有的图片文件名
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        
        # 提取文件名中的标签（最后一个'_'后的数字）
        self.labels = [int(f.split('_')[-1].split('.')[0]) for f in self.image_files]
    
    def __len__(self):
        # 数据集的大小
        return len(self.image_files)

    def __getitem__(self, idx):
        # 根据索引加载图像
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name).convert('L')  #灰度图像
        
        # 应用预处理（如果有）
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = self.labels[idx]
        
        return image, label


#以下是对数据集的预处理
transform = transforms.Compose([
    transforms.CenterCrop(image_size), #中心裁剪，确保是正方形
    transforms.ToTensor(), #将图片转换为张量
    transforms.Normalize(mean=[0.5], std=[0.5]) #标准化像素值，使网络更稳定的训练
])

dataset = MNISTdataset(dataroot, transform=transform)

dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=workers)
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

# 获取一个批次的训练图像
real_batch = next(iter(dataloader))
# 创建图形窗口，显示前64张训练集里的图像，并保存为png图像。
# 因为远程主机我没找到如何show，选择了保存。
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

outputimage = os.path.join(file_path, 'input_image.png')
plt.savefig(outputimage)

#初始化权重的函数
def weights_init(m): 
    classname = m.__class__.__name__ #获取类名
    if classname.find('Conv') != -1: #如果是卷积层
        nn.init.normal_(m.weight.data, 0.0, 0.02) #初始化权重
    elif classname.find('BatchNorm') != -1: #如果是BatchNorm层
        nn.init.normal_(m.weight.data, 1.0, 0.02) #初始化权重
        nn.init.constant_(m.bias.data, 0) #初始化偏置

#生成器
class Generator(nn.Module):
    def __init__(self,ngpu): 
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            # 第一层：从潜在向量（100）生成 256 个通道的特征图
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),  # 输出大小: [256, 4, 4]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 第二层：将特征图的大小放大
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 输出大小: [128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 第三层：继续放大
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 输出大小: [64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 第四层：放大到 28x28，适配 MNIST 图像
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),  # 输出大小: [1, 28, 28]
            nn.Tanh()  # 使用 Tanh 将输出的像素值限制在 [-1, 1] 范围内
        )

    def forward(self, x):
        return self.model(x)
    

#判别器
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入尺寸为 (nc) x 28 x 28，nc = 1，因为 MNIST 是灰度图像
            nn.Conv2d(1, 64, 3, 2, 1, bias=False),  # 输出尺寸 (64) x 14 x 14
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出尺寸 (64) x 14 x 14
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # 输出尺寸 (128) x 7 x 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出尺寸 (128) x 7 x 7
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),  # 输出尺寸 (256) x 4 x 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出尺寸 (256) x 4 x 4
            nn.Conv2d(256, 1, 3, 1, 0, bias=False),  # 输出尺寸 (1) x 1 x 1
            nn.Sigmoid()  # 输出一个概率，判断图像是真实的还是生成的
        )

    def forward(self, input):
        return self.main(input)
    

netG = Generator(ngpu).to(device) #将生成器加载到设备上
netG.apply(weights_init) #初始化权重
netD = Discriminator(ngpu).to(device) #将判别器加载到设备上
netD.apply(weights_init) #初始化权重

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999)) #生成器的优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) #判别器的优化器

real_label = 1 #真实图像的标签
fake_label = 0 #生成图像的标签

criterion = nn.BCELoss() #二分类交叉熵损失函数
fixed_noise = torch.randn(64, nz, 1, 1, device=device) #固定的噪声，用于生成图像

G_losses = [] #保存生成器的损失
D_losses = [] #保存判别器的损失

#训练网络
#第一层循环为轮数
for epoch in range(num_epochs):
    #第二层为当前批次的索引
    for i, data in enumerate(dataloader, 0):
        #训练判别器
        #先清空判别器的梯度
        netD.zero_grad()
        
        # 真实图像
        real_images, _ = data
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        output = netD(real_images)
        # 创建标签为真实的目标值
        label = torch.full((batch_size,), real_label, device=device)
        label = label.to(torch.float)
        # 通过 unsqueeze 扩展 label 的维度，确保与模型输出形状一致
        label = label.view(batch_size, 1, 1, 1).expand_as(output)  
        # 扩展为 [batch_size, 1, 2, 2]
        
        # 计算判别器对真实图像的损失
        errD_real = F.binary_cross_entropy(output, label)
        # 反向传播计算梯度
        errD_real.backward()
        
        
        #生成假的图像
        #创建随机噪声
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        #生成器生成假图像
        fake_images = netG(noise)
        
        #标签为假的目标值
        label.fill_(fake_label)
        label = label.to(torch.float)
        #计算判别器对生成图像的损失
        output = netD(fake_images.detach())
        errD_fake = F.binary_cross_entropy(output, label)
        #反向传播计算梯度
        errD_fake.backward()

        #计算总损失
        errD = errD_real + errD_fake
        #更新判别器参数
        optimizerD.step()

        #训练生成器
        #清空生成器的梯度
        netG.zero_grad()
        #伪造标签为1，希望判别器误判
        label.fill_(real_label)
        label = label.to(torch.float)
        #假图像传入判别器，得到预测值
        output = netD(fake_images)
        
        # 计算生成器的损失
        errG = F.binary_cross_entropy(output, label)
        #还是反向传播计算梯度
        errG.backward()

        # 更新生成器的权重
        optimizerG.step()

        # 记录损失，生成器和判别器
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 每训练一定步数，打印并保存生成的图像
        if i % 100 == 0:
            print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item()}, Loss_G: {errG.item()}")

    # 每个epoch后保存生成的图像
    if epoch % 10 == 0 and epoch != 0:  # 每10个epoch保存一次生成图像
        with torch.no_grad():
            fake_images = netG(fixed_noise).detach().cpu()
        save_image(fake_images, f"output_images/fake_images/fake_images_epoch_{epoch}.png", normalize=True)
        # 可视化训练过程中的损失曲线
        plt.figure(figsize=(10, 5))
        plt.title(f"Training Losses at Epoch {epoch}")
        plt.plot(G_losses, label="Generator")
        plt.plot(D_losses, label="Discriminator")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"output_images/loss_epoch_{epoch}.png")
        plt.close()
    
    # 可选：你也可以使用 Matplotlib 来显示部分生成的图像
    if epoch % 50 == 0:  # 每50个epoch显示一次生成图像
        with torch.no_grad():
            fake_images = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(vutils.make_grid(fake_images, padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.title(f"Generated Images at Epoch {epoch}")
        plt.axis('off')
        plt.savefig(f"output_images/fake_epoch_{epoch}.png")