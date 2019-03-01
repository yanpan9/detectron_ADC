#!/usr/bin/env python

# USETC: yanpan
# Email: cnyanpan@gmail.com
# 20190227
##############################################################################

import os
import sys
import shutil

from os import path as osp
from random import sample
from detectron.utils.vocconverter import convertXml2Json

def convert(root_path, train_size=0.8):
    images = os.listdir(osp.join(root_path, "images"))
    xmls = os.listdir(osp.join(root_path, "labels"))
    images = [x[:-4] for x in images if x.endswith(".jpg") or x.endswith(".JPG")]
    xmls = [x[:-4] for x in xmls if x.endswith(".xml")]
    labeled_images = set(images) and set(xmls)
    files = list(labeled_images)
    files.sort()
    
    annotation_path, img_path, val_txt_path, val_anno_path = create_dirs(root_path)
    for name in files:
        file_path = osp.join(root_path, "images", name)
        if osp.exists(file_path+".jpg"):
            shutil.copy(file_path+".jpg", osp.join(img_path, name+".jpg"))
        else:
            shutil.copy(file_path+".JPG", osp.join(img_path, name+".jpg"))
    if len(os.listdir(annotation_path))==0:
        val_files = convert_xml_to_json(root_path, annotation_path, files, train_size)
        with open(osp.join(val_txt_path, "val.txt"), "w") as dst:
            for name in val_files:
                dst.write(name+"\n")
                shutil.copy(osp.join(root_path, "labels", name+".xml"), val_anno_path)
    return annotation_path
    
def create_dirs(root_path):
    dataset_path = osp.join(root_path, "VOC2007")
    if not osp.exists(dataset_path):
        os.makedirs(dataset_path)
    annotation_path = osp.join(dataset_path, "annotations")
    img_path = osp.join(dataset_path, "JPEGImages")
    val_txt_path = osp.join(dataset_path, "VOCdevkit2007", "VOC2007", "ImageSets", "Main")
    val_anno_path = osp.join(dataset_path, "VOCdevkit2007", "VOC2007", "Annotations")
    results_path = osp.join(dataset_path, "VOCdevkit2007", "results", "VOC2007", "Main")
    for path in [annotation_path, img_path, val_txt_path, val_anno_path, results_path]:
        if not osp.exists(path):
            os.makedirs(path)

    return annotation_path, img_path, val_txt_path, val_anno_path

def convert_xml_to_json(root_path, annotation_path, files, train_size):
    train_num = int(len(files)*train_size)
    val_num = len(files)-train_num
    val_files = sample(files, val_num)
    val_files.sort()
    train_files = [name for name in files if name not in val_files]
    converter = convertXml2Json()
    converter.parseXmlFiles(osp.join(root_path, "labels"), train_files)
    converter.writeToJSON(osp.join(annotation_path, "voc_2007_train.json"))
    converter = convertXml2Json()
    converter.parseXmlFiles(osp.join(root_path, "labels"), val_files)
    converter.writeToJSON(osp.join(annotation_path, "voc_2007_val.json"))
    return val_files


if __name__ == "__main__":
    convert("C:\\Users\\yanpan\\Desktop\\Test")