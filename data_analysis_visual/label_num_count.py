
import argparse
import os
import xml.etree.ElementTree as ET

def readXml(xmlfile,count):

    DomTree = ET.parse(xmlfile)

    root = DomTree.getroot()
    size = root.find('size')
    if size is not None:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    # objectlist = annotation.getElementsByTagName('object')

    for objects in root.findall('object'):
        difficult = objects.find('difficult').text

        if difficult == '1':
            # if xmlfile not in difficult_obj.keys():
            #     difficult_obj[xmlfile] = 1
            # else:
            #     difficult_obj[xmlfile] += 1
            continue
        else:
            class_label = objects.find('name').text

            # if class_label in classNames:
            if class_label != args.label:
                continue
            else:
                bnd_box = objects.find('bndbox')

                bbox = [
                    int(float(bnd_box.find('xmin').text)),
                    int(float(bnd_box.find('ymin').text)),
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text))
                ]
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or min(bbox)<0:
                    continue
                else:
                    count += 1
    return count

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')

    parser.add_argument(
        '--anno-dir',
        type=str,
        help='path of annotations file')
        
    parser.add_argument(
        '--label',
        type=str,
        help='which label to count')


    args = parser.parse_args()
    return args

args = parse_args()
count = 0
if args.anno_dir.endswith('txt'):
    f = open(args.anno_dir,'r')

    xml_paths = []
    dirname = os.path.dirname(os.path.dirname(os.path.dirname(args.anno_dir)))

    for line in f:
        xml_paths.append(os.path.join(dirname, "Annotations", line.split('\n')[0] + ".xml"))

    f.close()
    for xml_path in xml_paths:
        count = readXml(xml_path,count)
    print(args.label,':',count)
elif os.path.isdir(args.anno_dir):
    all_files = []
    for root, dirs, files in os.walk(args.anno_dir, topdown=False):
        all_files.append(files)
    xml_paths = [os.path.join(args.anno_dir,file) for file in all_files[0]]

    for i,xml_path in enumerate(xml_paths) :
        if i % 1000 ==0:
            print(i) 
        count = readXml(xml_path,count)
    print(args.label,':',count)
