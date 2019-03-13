

import xml.etree.ElementTree as ET
import os
import sys
import json

from os import path as osp


class convertXml2Json:
    def __init__(self):
        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = []

        self.category_set = dict()
        self.image_set = set()

        self.category_item_id = 0
        self.image_id = 20180000000
        self.annotation_id = 0

    def addCatItem(self, name):
        category_item = dict()
        category_item['supercategory'] = 'none'
        self.category_item_id += 1
        category_item['id'] = self.category_item_id
        category_item['name'] = name
        self.coco['categories'].append(category_item)
        self.category_set[name] = self.category_item_id
        return self.category_item_id

    def addImgItem(self, file_name, size):
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        if size['width'] is None:
            raise Exception('Could not find width tag in xml file.')
        if size['height'] is None:
            raise Exception('Could not find height tag in xml file.')
        self.image_id += 1
        image_item = dict()
        image_item['id'] = self.image_id
        image_item['file_name'] = file_name
        image_item['width'] = size['width']
        image_item['height'] = size['height']
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)
        return self.image_id

    def addAnnoItem(self, object_name, category_id, bbox):
        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg = []
        #bbox[] is x,y,w,h
        #left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        #left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        #right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        #right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])

        annotation_item['segmentation'].append(seg)

        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = self.image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        self.annotation_id += 1
        annotation_item['id'] = self.annotation_id
        self.coco['annotations'].append(annotation_item)

    def parseXmlFiles(self, xml_list): 
        for name in xml_list:
            
            bndbox = dict()
            size = dict()
            current_image_id = None
            current_category_id = None
            file_name = None
            size['width'] = None
            size['height'] = None
            size['depth'] = None

            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.tag != 'annotation':
                raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

            #elem is <folder>, <filename>, <size>, <object>
            for elem in root:
                current_parent = elem.tag
                current_sub = None
                object_name = None
                
                if elem.tag == 'folder':
                    continue
                
                if elem.tag == 'filename':
                    file_name = elem.text
                    if file_name in self.category_set:
                        raise Exception('file_name duplicated')
                    
                #add img item only after parse <size> tag
                elif current_image_id is None and file_name is not None and size['width'] is not None:
                    if file_name not in self.image_set:
                        current_image_id = self.addImgItem(file_name, size)
                        print('add image with {} and {}'.format(file_name, size))
                    else:
                        raise Exception('duplicated image: {}'.format(file_name)) 
                #subelem is <width>, <height>, <depth>, <name>, <bndbox>
                for subelem in elem:
                    bndbox ['xmin'] = None
                    bndbox ['xmax'] = None
                    bndbox ['ymin'] = None
                    bndbox ['ymax'] = None
                    
                    current_sub = subelem.tag
                    if current_parent == 'object' and subelem.tag == 'name':
                        object_name = subelem.text
                        if object_name not in self.category_set:
                            current_category_id = self.addCatItem(object_name)
                        else:
                            current_category_id = self.category_set[object_name]

                    elif current_parent == 'size':
                        if size[subelem.tag] is not None:
                            raise Exception('xml structure broken at size tag.')
                        size[subelem.tag] = int(subelem.text)

                    #option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                    for option in subelem:
                        if current_sub == 'bndbox':
                            if bndbox[option.tag] is not None:
                                raise Exception('xml structure corrupted at bndbox tag.')
                            bndbox[option.tag] = int(option.text)

                    #only after parse the <object> tag
                    if bndbox['xmin'] is not None:
                        if object_name is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_image_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_category_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        bbox = []
                        #x
                        bbox.append(bndbox['xmin'])
                        #y
                        bbox.append(bndbox['ymin'])
                        #w
                        bbox.append(bndbox['xmax'] - bndbox['xmin'])
                        #h
                        bbox.append(bndbox['ymax'] - bndbox['ymin'])
                        print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                        self.addAnnoItem(object_name, current_category_id, bbox )
    def writeToJSON(self, json_file):
        for idx,item in enumerate(self.coco["images"]):
            if item["file_name"].endswith("JPG"):
                new_file_name = item["file_name"][:-3]+"jpg"
                self.coco["images"][idx]["file_name"]=new_file_name

        with open(json_file, "w") as dst:
            json.dump(self.coco, dst)

def readCategoryFromJson(filepath):
    with open(filepath, "r") as src:
        json_dict = json.load(src)
        categories = [elem["name"] for elem in json_dict["categories"]]
    return categories

if __name__ == "__main__":
    res = readCategoryFromJson("/home/yanpan/Test/VOC2007/annotations/voc_2007_val.json")
    print(res)
