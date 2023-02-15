# coding=utf-8
import os
 
 
def renaming(file):
    """修改后缀"""
    ext = os.path.splitext(file)    # 将文件名路径与后缀名分开
 
    if ext[1] == '.JPG':                    # 文件名：ext[0]
        new_name = ext[0] + '.jpg'         # 文件后缀：ext[1]
        os.rename(file, new_name)           # tree()已切换工作地址，直接替换后缀
    elif ext[1] == '.html':
        new_name = ext[0] + '.txt'
        os.rename(file, new_name)
 
 
def tree(path):
    """递归函数"""
    files = os.listdir(path)    # 获取当前目录的所有文件及文件夹
    for file in files:
        file_path = os.path.join(path, file)  # 获取该文件的绝对路径
        if os.path.isdir(file_path):    # 判断是否为文件夹
            tree(file_path)     # 开始递归
        else:
            os.chdir(path)      # 修改工作地址（相当于文件指针到指定文件目录地址）
            renaming(file)      # 修改后缀
 
 
#this_path = os.getcwd()    # 获取当前工作文件的绝对路径（文件夹)
# from pdb import set_trace
# set_trace()

all = ['导地线','附属设施','杆塔','基础','绝缘子','通道环境','小金具','大金具', '接地装置']
paths = ["/data/wulianjun/datasets/SG5/{}".format(part) for part in all]
[tree(path) for path in paths]

