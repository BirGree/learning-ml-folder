import os
import shutil
import glob
from datetime import datetime
firstenv = os.getcwd()
os.chdir(os.getcwd() + "\\incoming_data")
#print(os.getcwd())
#当前的工作目录
env = os.getcwd()
for file in glob.glob(os.path.join(env,'*.*')):
    #遍历文件夹中的全部文件，不遍历子文件夹里的文件
    if os.path.basename(file)[0]=='.':
        continue
        #忽略掉隐藏文件
    ext=os.path.splitext(file)
    #获取拓展名
    #print(ext[-1])
    if ext[-1] == ".quantum":
        #判断文件类型
        if not os.path.exists('quantum_core/SECTOR-7G'):
            os.makedirs('quantum_core/SECTOR-7G')
        #目录不存在将被创建
        dest = env+"\\quantum_core/SECTOR-7G"
        shutil.move(file, dest)
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
        #加上标签即重命名
        os.renames(file,new)
        shutil.move(new, dest)

os.chdir(firstenv)
with open('hologram_log.txt','w',encoding='utf-8') as log:
    log.write("┌──────────────────────────────┐\n│ 🛸 Xia-III 空间站数据分布全息图 │\n└──────────────────────────────┘\n\n├─🚀 incoming_data\n")
#切换目录并创建日志文件，切换编码格式为utf-8，否则无法写入火箭符号

def gen_tree(direc , prefix = '│ ' ,log=None):
    dir_all = sorted(os.listdir(direc))
    #all directories
    dir_all = [f for f in dir_all if not f.startswith(".")]
    #ignore hidden files
    for index, entry in enumerate(dir_all):
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
        log.write(prefix+starting+entry+'\n')
        if os.path.isdir(path):
            #如果遍历到文件夹，重复调用这个函数
            new_prefix = prefix+"│ "
            gen_tree(path , new_prefix , log)

def write_to_log(direc,tolog):
    #将日志写进txt文件中的函数
    with open(tolog, 'a', encoding='utf-8') as log:
        #此处写入内容时不覆盖，使用open函数的a模式
        gen_tree(direc,log=log)
        #调用遍历

write_to_log(env,'hologram_log.txt')
#调用写入的函数，完成日志的创建
with open('hologram_log.txt','a',encoding='utf-8') as log:
    log.write("\n🤖 SuperNova · 地球标准时 "+datetime.now().strftime('%Y-%m-%dT%H:%M:%S')+"\n⚠️ 警告：请勿直视量子文件核心\n")


