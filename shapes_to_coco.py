#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np

import sys
sys.path.append('../')
from pycococreatortools import pycococreatortools

ROOT_DIR = '/home/ccchang/disk2_1tb_ssd/robot_dataset/panel_exp_fixed/mycoco/'

###Synthetic images
IMAGE_DIR = os.path.join(ROOT_DIR, "coco_train_200k/")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "coco_label_200k/")

###Real images
# IMAGE_DIR = os.path.join(ROOT_DIR, "coco_real_train/")
# ANNOTATION_DIR = os.path.join(ROOT_DIR, "coco_real_label/")


INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'panel',
        'supercategory': 'shape',
    },
    # {
    #     'id': 2,
    #     'name': 'circle',
    #     'supercategory': 'shape',
    # },
    # {
    #     'id': 3,
    #     'name': 'triangle',
    #     'supercategory': 'shape',
    # },
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg','*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def main_old():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        print(image_files)
        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                print(annotation_files)
                input("stop")

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    # if annotation_filename.find("32_panel_0")==-1:
                    # if annotation_filename.find("32_panel_0")!=-1:
                    #     continue
                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}

                    # print("here0")
                    tmp_img = np.array(Image.open(annotation_filename))
                    tmp_img[tmp_img<255] = 0
                    tmp_img = Image.fromarray(tmp_img)                    

                    # binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)  #Image convert("1") -> grayscale
                    # binary_mask = np.asarray(tmp_img.convert('1')).astype(np.uint8)  
                    binary_mask = np.array(tmp_img).astype(np.uint8)
                    
                    # print("here1")
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    # print("here2")
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/panel_coco.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    all_img_str = []
    all_ann_str = []
    ### img_ann_dict : {'0':['0_panel_0','0_panel_1'], '1':['1_panel_0','1_panel_1']...}
    img_ann_dict = {}

    for img_name in os.listdir(IMAGE_DIR):
        all_img_str.append(img_name)

    for ann_name in os.listdir(ANNOTATION_DIR):
        all_ann_str.append(ann_name)

    for ann_name in all_ann_str:
        pos = ann_name.find('_')
        img_id = ann_name[:pos]

        if img_id in img_ann_dict.keys():
            img_ann_dict[img_id].append(ann_name)
        else:
            img_ann_dict[img_id] = [ann_name]

    # go through each image
    for img_id in img_ann_dict.keys():
        image_filename = IMAGE_DIR + img_id + ".png"
        image = Image.open(image_filename)
        image_info = pycococreatortools.create_image_info(image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)
        ann_files = img_ann_dict[img_id]
        # go through each associated annotation
        for ann_str in ann_files:
            # if annotation_filename.find("32_panel_0")==-1:
            # if annotation_filename.find("32_panel_0")!=-1:
            #     continue
            annotation_filename = ANNOTATION_DIR + ann_str
            # print(annotation_filename)
            class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
            category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}

            tmp_img = np.array(Image.open(annotation_filename))

            tmp_img[tmp_img<255] = 0
            tmp_img = Image.fromarray(tmp_img)                    

            # binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)  #Image convert("1") -> grayscale
            # binary_mask = np.asarray(tmp_img.convert('1')).astype(np.uint8)  
            binary_mask = np.array(tmp_img).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask,
                image.size, tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
            segmentation_id = segmentation_id + 1

        if image_id %1000 == 0:
            print("Finished num:",image_id)
        image_id = image_id + 1

    with open('{}/panel_coco_200k.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

if __name__ == "__main__":
    main()
