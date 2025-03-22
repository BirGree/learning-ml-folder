#第一题：文件分类，熟悉文件操作，文件夹操作，时间操作
import os
import shutil
import glob
from datetime import datetime
firstenv = os.getcwd()
os.chdir(os.getcwd() + "\\incoming_data")
#print(os.getcwd())
#os.getcwd()函数返回当前的工作目录
env = os.getcwd()
for file in glob.glob(os.path.join(env,'*.*')):
    #os.path.join()函数连接目录和文件名，glob.glob()函数返回所有匹配的文件路径列表，*.*表示所有文件
    #遍历文件夹中的全部文件，不遍历子文件夹里的文件
    if os.path.basename(file)[0]=='.':
        continue
        #basename()函数返回文件名，是字符串，如果文件名第一个字符是.，则表示是隐藏的文件
        #忽略掉隐藏文件
    ext=os.path.splitext(file)
    #splitext()函数分离文件名和扩展名，返回一个元组，如('C:/Users/xxx/xxx/xxx','txt')
    #获取拓展名
    #print(ext[-1])
    if ext[-1] == ".quantum":
        #判断文件类型
        if not os.path.exists('quantum_core/SECTOR-7G'):
            #os.path.exists()函数判断文件夹是否存在
            os.makedirs('quantum_core/SECTOR-7G')
        #目录不存在将被创建
        #os,makedirs()可以创建多级目录，os.mkdir()只能创建一级目录
        dest = env+"\\quantum_core/SECTOR-7G"
        shutil.move(file, dest)
        #shutil.move()函数移动文件，第一个参数是文件名，第二个参数是目标文件夹
        #移动文件
    elif ext[-1] == ".holo":
        #其余类型同理
        if not os.path.exists('hologram_vault/CHAMBER-12F'):
            os.makedirs('hologram_vault/CHAMBER-12F')
        dest = env+"\\hologram_vault/CHAMBER-12F"
        shutil.move(file, dest)
    elif ext[-1] == ".exo":
        if not os.path.exists('exobiology_lab/POD-09X'):
            os.makedirs('exobiology_lab/POD-09X')
        dest = env+"\\exobiology_lab/POD-09X"
        shutil.move(file, dest)
    elif ext[-1] == ".chrono":
        if not os.path.exists('temporal_archive/VAULT-00T'):
            os.makedirs('temporal_archive/VAULT-00T')
        dest = env+"\\temporal_archive/VAULT-00T"
        shutil.move(file, dest)
    else:
        #剩下的为未知类型
        if not os.path.exists('quantum_quarantine'):
            os.mkdir('quantum_quarantine')
        dest = env+"\\quantum_quarantine"
        new = 'ENCRYPTED_'+os.path.basename(file)
        #os.path.basename()函数返回文件名，是字符串，加上前缀ENCRYPTED_
        #加上标签即重命名
        os.renames(file,new)
        #os.renames()函数重命名文件，第一个参数是旧文件名，第二个参数是新文件名
        shutil.move(new, dest)

os.chdir(firstenv)
#返回原来的工作目录
with open('hologram_log.txt','w',encoding='utf-8') as log:
    log.write("┌──────────────────────────────┐\n│ 🛸 Xia-III 空间站数据分布全息图 │\n└──────────────────────────────┘\n\n├─🚀 incoming_data\n")
#切换目录并创建日志文件，切换编码格式为utf-8，否则无法写入火箭符号

def gen_tree(direc , prefix = '│ ' ,log=None):
    #遍历文件夹的函数，direc是文件夹路径，prefix是前缀，log是日志文件
    dir_all = sorted(os.listdir(direc))
    #os.listdir()函数返回目录下的所有文件和文件夹，返回一个列表
    #sorted()函数返回一个排序后的列表
    #dir_all是一个包含了文件夹下的所有文件和文件夹的列表
    dir_all = [f for f in dir_all if not f.startswith(".")]
    #f for f in dir_all if not f.startswith(".")是一个列表推导式，过滤掉这个列表里的隐藏文件
    #列表推导式是一种快速生成列表的方法，写法是[表达式 for 变量 in 列表 if 条件]
    for index, entry in enumerate(dir_all):
        #enumerate()函数返回一个枚举对象，包含了索引和值
        #index是索引，entry是值
        path = os.path.join(direc, entry)
        #遍历的目录（文件或文件夹）
        if os.path.isdir(path):
            #文件夹标注成舱室
            starting = '├─🚀 '
        elif os.path.splitext(path)[-1] in ['.holo', '.chrono', '.quantum', '.exo']:
            starting = '├─🔮️ '+datetime.now().strftime('%Y%m%d%H%M%S')+'_'
            #已知的文件类型,然后加上时间戳
        else:
            starting = '├─⚠️ '
            #未知的文件
        #这个if针对不同的文件类型，加上不同的标签
        log.write(prefix+starting+entry+'\n')
        #log.write()函数写入日志文件
        #prefix是前缀，starting是标签，entry是文件名，即返回枚举对象的值，\n是换行符
        if os.path.isdir(path):
            #如果遍历到文件夹，重复调用这个函数
            #这种方法叫做递归
            new_prefix = prefix+"│ "
            #新的前缀，多一条竖线的文件夹，表示文件夹里的文件
            gen_tree(path , new_prefix , log)

def write_to_log(direc,tolog):
    #定义一个将日志写进txt文件中的函数
    with open(tolog, 'a', encoding='utf-8') as log:
        #with open()函数打开文件，a模式是追加模式，encoding是编码格式
        #a模式是在文件末尾追加内容，如果文件不存在，创建一个新文件
        gen_tree(direc,log=log)
        #gen_tree()函数是刚刚定义的遍历文件夹函数，log=log是将日志文件传入函数
        #调用遍历

write_to_log(env,'hologram_log.txt')
#调用写入的函数，完成日志的创建，env是文件夹路径，'hologram_log.txt'是日志文件名
with open('hologram_log.txt','a',encoding='utf-8') as log:
    log.write("\n🤖 SuperNova · 地球标准时 "+datetime.now().strftime('%Y-%m-%dT%H:%M:%S')+"\n⚠️ 警告：请勿直视量子文件核心\n")
#最后，在文件中写入时间戳和警告信息

