#coding=utf-8
from optparse import Values
import os
import os.path as osp
from symbol import import_as_name
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
uesd_val_label = dict()
train_labels = dict()
val_labels = dict()
all_val_labels = dict()
non_valid_xml = []

def read_xml(xml_path,dataset_class,relabel=False,save_path=None,index=0,val=False,ref_object=None):
    DomTree = ET.parse(xml_path)
    file_name = osp.basename(xml_path)
    root = DomTree.getroot()
    size = root.find('size')
    if size is not None:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    if not relabel:
        for objects in root.findall('object'):
            label = objects.find('name').text
            
            if label in dataset_class:

                if label not in train_labels:
                    train_labels[label] = 1
                else:
                    train_labels[label] += 1


                # bnd_box = objects.find('bndbox')
                # bbox = [
                #             int(float(bnd_box.find('xmin').text)),
                #             int(float(bnd_box.find('ymin').text)),
                #             int(float(bnd_box.find('xmax').text)),
                #             int(float(bnd_box.find('ymax').text))
                #         ]
                # # from pdb import set_trace
                # # set_trace()
                # if bbox[0]>=bbox[2] or bbox[1]>=bbox[3] or bbox[0]<0 or bbox[1]<0 or bbox[2]>width or bbox[3]>height:
                #     non_valid_xml.append(xml_path)

                return xml_path
    else:
        return_xml=False
        for objects in root.findall('object'):
            label = objects.find('name').text
            label_ = label

            #for tongdao

            if label == '吊车' or label == '060800011 通道环境 线下施工外破隐患 吊车':

                label_ = '塔吊'
            #for ganta

            if label == '角杆塔鸟巢' or label == '角杆塔异物' or label == '砼杆鸟巢'  or label == '砼杆异物' or label == '钢管塔鸟巢' or label == '钢管塔异物' or \
                label == '010000021 角杆塔 塔身 异物' or label == '\ufeff010000021 角杆塔 塔身 异物' or \
                label=='010100061\t杆塔\t钢管塔\t异物' or label =='010200111\t杆塔\t钢管杆\t异物' or label=='010100061 杆塔 钢管塔 异物'\
                or label =='010200111 杆塔 钢管杆 异物' or label =='010300091 砼杆 塔身 异物' or label == '010300091 砼杆\t塔身 异物' \
                    or label == '010300091 砼杆 塔身 异物' or label=='010300091 砼杆\t塔身 异物':
                label_ = '塔身异物'

            if label == '角杆塔缺螺栓' or label == '砼杆缺螺栓':
                label_ = '杆塔缺螺栓'

            if label == '钢管塔塔材变形' or label == '角杆塔塔材变形' or label == '010002031 角杆塔 塔材 变形':
                label_ = '塔材变形'

            if label == '砼杆锈蚀' or label == '角杆塔塔材锈蚀' or label == '砼杆叉梁抱箍锈蚀' or \
                label=='角杆塔脚钉锈蚀' or label =='010003021 角杆塔 脚钉 锈蚀' or label=='010002051 角杆塔 塔材 锈蚀' or label =='010302021 砼杆 叉梁 抱箍锈蚀':
                label_ = '塔材锈蚀'

            if  label=='010000031 角杆塔 塔身 锈蚀' or label=='钢管塔锈蚀' or label=='角杆塔塔身锈蚀':
                label_ = '塔身锈蚀'

            #for daodixian
            if label == '普通地线异物' or label == '引流线异物' or label == '020100051 普通地线 异物' \
                or label == '020001061 导线 引流线 异物' or label == '导线本体异物' or label =='020000111 导线 本体 异物':
                label_ = '线异物'

            if  label == '020000021 导线 本体 损伤' or label == '020001021 导线 引流线 损伤' \
                or label=='引流线损伤' or label =='普通地线损伤' or label == '导线本体损伤':
                label_ = '线损伤'

            if  label == '导线本体松股' or label == '020000031 导线 本体 松股'   or label=='引流线松股' :
                label_ = '线松股'
            
            if label == '导线本体断股'  or label == '引流线断股'  \
                  or label =='普通地线断股' :
                label_ = '线断股'

            # for fushusheshi
            # ['防鸟设施松动',
            # '防鸟设施损坏',
            # '杆号牌图文不清',
            # '杆号牌破损',
            # '警告牌图文不清',
            # '色标牌褪色',
            # '色标牌破损',]

            if label == '杆号牌图文不清' or label =='070000011 标志牌 杆号牌（含相序） 图文不清':
                label_ = '杆号牌图文不清'
            
            if label == '杆号牌破损' or label == '070000021 标志牌 杆号牌（含相序） 破损':
                label_ = '杆号牌破损'

            if label == '警告牌破损'  \
                or label == '070002011 标志牌 警告牌\t图文不清'  or label=='070002021 标志牌 警告牌\t破损' or label=='070002021 标志牌 警告牌 破损':
                label_ = '警告牌图文不清'

            if label == '070001011 标志牌 色标牌\t破损':
                label_ = '色标牌破损'

            if label == '色标牌退色'  or \
                label =='070001041 标志牌 色标牌 退色' or label =='070001041 标志牌 色标牌\t退色':
                label_ = '色标牌褪色'


            if label =='防鸟刺未打开' or label =='070400021 驱鸟刺 未打开' or label =="070400021 防鸟刺 未打开":
                label_ = '防鸟设施松动'

            if label =='驱鸟器损坏' or label =='070400021 防鸟刺 损坏' or label =='070400021 驱鸟器 损坏':
                label_ = '防鸟设施损坏'
            ## for xiaojinju


            if label == '缺螺母' or label ==  '040501021缺螺母' or label == '040501021 缺螺母':
                label_ = '螺栓缺螺母'

            if label == '缺销' or label ==  '040500011 缺销钉' or label == '040500011\t金具\t小金具\t销钉\t缺销 ':
                label_ = '销钉缺销'

        
            if label == '玻璃自爆':
                label_ = '玻璃绝缘子自爆'
            
            # if label == '030200021 绝缘子复合串倾斜':
            #     label_ = '复合绝缘子串倾斜'
            


            if label == '玻璃污秽' or label =='030100071 绝缘子玻璃锈蚀':
                label_ = '玻璃绝缘子污秽'

            if label == '复合伞裙破损' or label ==  '复合伞裙变形' or label ==  '030200061 绝缘子复合伞裙变形' or label == '030200061 绝缘子复合伞裙破损':
                label_ = '复合绝缘子伞裙破损'
            
            if label == '瓷质污秽' or label ==  '030000011 绝缘子瓷质污秽' or label == '030000011 绝缘子\t瓷质污秽':
                label_ = '瓷质绝缘子污秽'

            if label == '瓷质锈蚀' or label ==  '030000071 绝缘子瓷质锈蚀':
                label_ = '瓷质绝缘子锈蚀'

            if label == '瓷质破损' or label ==  '030000081 绝缘子瓷质破损':
                label_ = '瓷质绝缘子破损'
            
            if label == '瓷质釉表面灼伤' or label ==  '030000041 绝缘子瓷质釉表面灼伤':
                label_ = '瓷质绝缘子釉表面灼伤'
            
            if label == '复合均压环移位' or label ==  '030300091 均压环移位' or label=='030200131 绝缘子复合均压环移位' or label=='030300091均压环移位':
                label_ = '均压环移位'

            if label == '复合均压环脱落' or label ==  '030200172 绝缘子复合均压环脱落':
                label_ = '均压环脱落'

            if label == '复合均压环灼伤' or label ==  '030200111 绝缘子复合均压环灼伤' or label ==  '030000101 绝缘子瓷质均压环灼伤':
                label_ = '均压环灼伤'

            if label == '复合均压环反装' or label ==  '030200162 绝缘子复合均压环反装':
                label_ = '均压环反装'

            # for jichu
            #     label_ = '接地螺栓缺失'

            if label == '复合均压环反装' or label ==  '030200162 绝缘子复合均压环反装':
                label_ = '均压环反装'

            if label == '复合均压环反装' or label ==  '030200162 绝缘子复合均压环反装':
                label_ = '均压环反装'                                

            if index == 0:
                if label not in all_val_labels:
                    all_val_labels[label] = 1
                else:
                    all_val_labels[label] += 1
            label_ = label_.replace('\t','')
            label_ = label_.replace(' ','')
            new_dataset_class = dataset_class.copy()

            flag = np.array([c in label_ for c in new_dataset_class])
            if flag.sum()>1:
                from pdb import set_trace
                set_trace()
            if flag.any():
                if not val:
                    if label_ not in train_labels:
                        train_labels[label_] = 1
                    else:
                        train_labels[label_] += 1

                new_label_name = np.array(new_dataset_class)[flag].item()


                objects.find('name').text = new_label_name
                return_xml = True

                if label not in uesd_val_label:
                    uesd_val_label[label] = [new_label_name,1]
                else:
                    # if new_label_name not in uesd_val_label[label_]:
                    uesd_val_label[label][-1] += 1

                if new_label_name not in val_labels:
                    val_labels[new_label_name] = [[label],[1]]
                else:
                    # if new_label_name not in uesd_val_label[label_]:
                    if label in val_labels[new_label_name][0]:
                        val_labels[new_label_name][-1][val_labels[new_label_name][0].index(label)] += 1
                    else:
                        val_labels[new_label_name][0].append(label)
                        val_labels[new_label_name][-1].append(1)

                
        if os.path.exists(osp.join(save_path.replace('Annotations','JPEGImages'),file_name.replace('.xml','.jpg'))):
            os.makedirs(save_path,exist_ok=True)
            if not os.path.exists(osp.join(save_path,file_name)):
                os.mknod(osp.join(save_path,file_name))

        if return_xml:
            assert save_path is not None
            new_tree = ET.ElementTree(root)
            new_tree.write(osp.join(save_path,file_name),encoding='utf-8')
            return xml_path
        else:
            width = root.find('size').find('width').text 
            height = root.find('size').find('height').text
            if ref_object is None:
                ref_object = root.findall('object')[0]
            
            ref_object.find('name').text = 'others'
            bnd_box = ref_object.find('bndbox')
            bnd_box.find('xmin').text = str(1)
            bnd_box.find('ymin').text = str(1)
            bnd_box.find('xmax').text = str(int(width)-1)
            bnd_box.find('ymax').text = str(int(height)-1)
            root.append(ref_object)
            new_tree = ET.ElementTree(root)
            new_tree.write(osp.join(save_path,file_name),encoding='utf-8')
    return False

# dataset_train_classes=[['推土机','塔吊','吊车','挖掘机'],
#     ['缺销','缺垫片','螺母安装不规范','销钉安装不到位','螺母锈蚀',
#  '垫片锈蚀','螺帽锈蚀','缺螺母','螺丝安装不规范','缺螺栓','螺栓锈蚀','销钉锈蚀'],
#  ['复合均压环反装',
# '玻璃污秽',
# '玻璃自爆',
# '复合均压环损坏',
# '复合伞裙变形',
# '瓷质釉表面灼伤',
# '瓷质锈蚀',
# '复合均压环移位',
# '复合伞裙破损',
# '复合均压环脱落',
# '复合均压环灼伤',
# '瓷质破损',
# '防污罩脱落',
# '防污罩移位',
# '防污罩破损',
# '瓷质自爆',
# '瓷长棒破损',
# '复合均压环锈蚀',
# '瓷质污秽',
# '防污罩锈蚀',
# '玻璃锈蚀',
# '玻璃釉表面灼伤',
# '瓷质防污闪涂料失效'],
# ['砼杆锈蚀'
# ,'角杆塔鸟巢'
# ,'钢管塔锈蚀'
# ,'角杆塔塔材锈蚀'
#  ,'角杆塔异物'
# ,'角杆塔塔身锈蚀'
# ,'角杆塔缺螺栓'
# ,'角杆塔脚钉锈蚀'
# ,'钢管塔塔材变形'
#  ,'砼杆叉梁抱箍锈蚀'
#  ,'砼杆鸟巢'
# ,'角杆塔塔材变形'
# ,'砼杆异物'
#  ,'钢管塔异物'
# ,'钢管塔鸟巢'
# ,'砼杆塔身裂纹'
# ,'砼杆缺螺栓'],
# ['杂物堆积','立柱淹没','沉降','破损'],
# ['警告牌图文不清'
# ,'警告牌破损'
# ,'色标牌退色'
# ,'驱鸟器损坏'
# ,'杆号牌图文不清'
# ,'防鸟刺未打开'
# ,'杆号牌破损'
# ,'色标牌破损'],
# ['导线本体异物'
# ,'导线本体断股'
# ,'引流线松股'
# ,'导线本体松股'
# ,'普通地线异物'
# ,'引流线断股'
# ,'普通地线锈蚀'
# ,'引流线异物'
# ,'导线本体损伤'
# ,'普通地线断股'
# ,'引流线损伤'
# ,'普通地线损伤']
# ]
# dataset_train_classes = [
# #     ['缺销','缺垫片','螺母安装不规范','销钉安装不到位','螺母锈蚀',
# #  '垫片锈蚀','螺帽锈蚀','缺螺母','螺丝安装不规范','缺螺栓','螺栓锈蚀','销钉锈蚀'],
# ['螺栓缺螺母',
# '螺母安装不规范',
# '螺母锈蚀',
# '销钉锈蚀',
# '销钉缺销',]

#  ]
# dataset_train_classes = [
# [
# '玻璃绝缘子自爆',
# '玻璃绝缘子污秽',
# '复合绝缘子串倾斜',
# '复合绝缘子伞裙破损',
# '瓷质绝缘子污秽',
# '瓷质绝缘子锈蚀',
# '瓷质绝缘子破损',
# '瓷质绝缘子釉表面灼伤',
# '均压环移位',
# '均压环脱落',
# '均压环灼伤',
# '均压环反装'
# ]]


# dataset_train_classes=[['复合均压环反装',
# '玻璃污秽',
# '玻璃自爆',
# '复合均压环损坏',
# '复合伞裙变形',
# '瓷质釉表面灼伤',
# '瓷质锈蚀',
# '复合均压环移位',
# '复合伞裙破损',
# '复合均压环脱落',
# '复合均压环灼伤',
# '瓷质破损',
# '防污罩脱落',
# '防污罩移位',
# '防污罩破损',
# '瓷质自爆',
# '瓷长棒破损',
# '复合均压环锈蚀',
# '瓷质污秽',
# '防污罩锈蚀',
# '玻璃锈蚀',
# '玻璃釉表面灼伤',
# '瓷质防污闪涂料失效']]

# dataset_train_classes=[[
    # '砼杆锈蚀'
# ,'角杆塔鸟巢'
# ,'钢管塔锈蚀'
# ,'角杆塔塔材锈蚀'
#  ,'角杆塔异物'
# ,'角杆塔塔身锈蚀'
# ,'角杆塔缺螺栓'
# ,'角杆塔脚钉锈蚀'
# ,'钢管塔塔材变形'
#  ,'砼杆叉梁抱箍锈蚀'
#  ,'砼杆鸟巢'
# ,'角杆塔塔材变形'
# ,'砼杆异物'
#  ,'钢管塔异物'
# ,'钢管塔鸟巢'
# ,'砼杆塔身裂纹'
# ,'砼杆缺螺栓',
# '杆塔缺螺栓',
# '塔材变形',
# '塔身异物',
# '塔材锈蚀',
# '塔身锈蚀']]
# dataset_train_classes = [['杂物堆积','立柱淹没','沉降','破损']]

# dataset_train_classes =[
# # #     ['警告牌图文不清'
# # # ,'警告牌破损'
# # # ,'色标牌退色'
# # # ,'驱鸟器损坏'
# # # ,'杆号牌图文不清'
# # # ,'防鸟刺未打开'
# # # ,'杆号牌破损'
# # # ,'色标牌破损']

# ['防鸟设施松动',
# '防鸟设施损坏',
# '杆号牌图文不清',
# '杆号牌破损',
# '警告牌图文不清',
# '色标牌褪色',
# '色标牌破损',]

# ]


# dataset_train_classes =[['导线本体异物'
# ,'导线本体断股'
# ,'引流线松股'
# ,'导线本体松股'
# ,'普通地线异物'
# ,'引流线断股'
# ,'普通地线锈蚀'
# ,'引流线异物'
# ,'导线本体损伤'
# ,'普通地线断股'
# ,'引流线损伤'
# ,'普通地线损伤']]

dataset_train_classes =[['异物'
,'断股'
,'松股'
,'损伤']]


# dataset_train_classes = [['接地螺栓缺失'
# '引下线断开'
# '接地体外露']]

# dataset_train_classes=[['推土机','塔吊','挖掘机']]
# sub_data_name = ['通道环境','小金具','绝缘子','杆塔','基础','附属设施','导地线']
sub_data_name = ['导地线']
data_roots = ['/data/wulianjun/datasets/SG5/{}/Annotations'.format(x) for x in sub_data_name]
save_data_roots = ['/data/wulianjun/datasets/SG5/{}/Annotations'.format(x) for x in sub_data_name]
txt_roots = [os.path.dirname(save_data_roots)+'/ImageSets/Main' for save_data_roots in save_data_roots]
for txt_root in txt_roots:
    if not os.path.exists(txt_root):
        os.makedirs(txt_root)
        # if not os.path.exists(os.path.dirname(txt_root)):
        #     os.mkdir(os.path.dirname(txt_root))
        # os.mkdir(txt_root)


for i,(data_root,txt_root,save_data_root) in enumerate(zip(data_roots,txt_roots,save_data_roots)):
    all_files = []
    for root, dirs, files in os.walk(data_root, topdown=False):
        all_files.append(files)
    

# for line in open("/home/lianjun.wu/wulianjun/tongdao_crop2/ImageSets/Main/trainval.txt","r"): 
#     data.append(line)
    
    if not os.path.exists(txt_root+'/train.txt'):
        os.mknod(txt_root+'/train.txt')
    f = open(txt_root+'/train.txt','w')
    for j in tqdm(range(len(all_files[0]))):
        file = all_files[0][j]

        xml_path = os.path.join(data_root,file)
        save_xml = read_xml(xml_path,dataset_train_classes[i],relabel=True,save_path=save_data_root)
        if save_xml:
            f.write(os.path.basename(save_xml).split('.xml')[0]+'\n')
    


    f.close()




# dataset_train_classes=[['推土机','塔吊','吊车','挖掘机']]
# dataset_train_classes = [['杂物堆积','立柱淹没','000000021沉降','000000011破损']]



used_train_label=[]
for key,value in uesd_val_label.items():
    if value[0] not in used_train_label:
        used_train_label.append(value[0])
print(non_valid_xml)



def list_len(List):
    total_len = 0
    for x in List:
        for xx in x:
            if xx >= '\u4e00' and xx <= '\u9fff':
                total_len += 2
            elif xx == '\t':
                total_len += 2
            else:
                total_len += 1
    total_len += (len(List)-1) * 3
    return total_len

#  ln -s /data1/wlj/StateGridv5/验证集/JPEGImages/ /data1/wlj/StateGridv5/验证集/杆塔/JPEGImages
print("*-----train labels----num-----|{}val labels{}num-----*|".format('-'*22,'-'*50))
for label,num in train_labels.items():
    if label not in val_labels:
        print("|    {}{:>{}}    |".format(label,num,20-2*len(label)))
    else:
        print("|    {}{:>{}}    |".format(label,num,20-2*len(label)) + "|    {}{:>{}}    |".format(val_labels[label][0],sum(val_labels[label][1]),100-list_len(val_labels[label][0])))


print("*|-----no uesd val labels{}num-----*|".format('-'*20))
used_val_labels = []
for key,value in val_labels.items():
    used_val_labels.extend(value[0])
for key,num in all_val_labels.items():
    if key not in used_val_labels:
        print("|    {}{:>{}}    |".format(key,num,60-list_len([key])))


uesd_val_label = dict()
val_labels = dict()
all_val_labels = dict()

data_root = '/data/wulianjun/datasets/SG5/新验证集/Annotations'
txt_roots = [os.path.dirname(data_root)+'/{}/ImageSets/Main'.format(x) for x in sub_data_name]
sub_ann_roots = [os.path.dirname(data_root)+'/{}/Annotations'.format(x) for x in sub_data_name]

DomTree = ET.parse('/data/wulianjun/datasets/SG5/附属设施/Annotations/9996.xml')
root = DomTree.getroot()
ref_object = root.findall('object')[0]


for txt_root in txt_roots:
    if not os.path.exists(txt_root):
        os.makedirs(txt_root)
        # if not os.path.exists(os.path.dirname(txt_root)):
        #     os.mkdir(os.path.dirname(txt_root))
        # os.mkdir(txt_root)
for sub_ann_root in sub_ann_roots:
    if not os.path.exists(sub_ann_root):
        os.makedirs(sub_ann_root)

all_files = []
for root, dirs, files in os.walk(data_root, topdown=False):
    all_files.append(files)

for i,(txt_root,sub_ann_root) in enumerate(zip(txt_roots,sub_ann_roots)):

    
    if not os.path.exists(txt_root+'/val.txt'):
        os.mknod(txt_root+'/val.txt')
    f = open(txt_root+'/val.txt','w')

    
    for j in tqdm(range(len(all_files[0]))):
        file = all_files[0][j]
        xml_path = os.path.join(data_root,file)
        save_xml = read_xml(xml_path,dataset_train_classes[i],
            relabel=True,save_path=sub_ann_root,index=i,val=True,ref_object=ref_object)
        if save_xml:
            f.write(os.path.basename(save_xml).split('.xml')[0]+'\n')
    


    f.close()



print("*-----train labels----num-----|{}val labels{}num-----*|".format('-'*22,'-'*50))
for label,num in train_labels.items():
    if label not in val_labels:
        print("|    {}{:>{}}    |".format(label,num,20-2*len(label)))
    else:
        # if list_len(val_labels[label][0])>80:
        #     print("|    {}{:>{}}    |".format(label,num,20-2*len(label)) + "|    {}{:>{}}    |".format(val_labels[label][0],sum(val_labels[label][1]),100-list_len(val_labels[label][0])))

        print("|    {}{:>{}}    |".format(label,num,20-2*len(label)) + "|    {}{:>{}}    |".format(val_labels[label][0],sum(val_labels[label][1]),100-list_len(val_labels[label][0])))


print("*|-----no uesd val labels{}num-----*|".format('-'*20))
used_val_labels = []
for key,value in val_labels.items():
    used_val_labels.extend(value[0])
for key,num in all_val_labels.items():
    if key not in used_val_labels:
        print("|    {}{:>{}}    |".format(key,num,60-list_len([key])))

from pdb import set_trace
set_trace()