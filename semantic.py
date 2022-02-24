from PIL import Image
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import sys
import glob
# from torchvision.io import read_image
from PIL import Image
from pathlib import Path
import os
import requests
import io
matplotlib.use('Agg')



def show_and_save(imgs,img_name,var_name):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False,figsize=(16, 9), dpi=(120))
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            Path('D:/Carla/WindowsNoEditor/Bbox_sem/images/'+ var_name).mkdir(parents=True, exist_ok=True)
            plt.savefig('D:/Carla/WindowsNoEditor/Bbox_sem/images/'+ var_name +'/'+ img_name +'_with_bbox.jpg',bbox_inches='tight')
            # plt.savefig('images/with_bbox/'+ img_name +'_with_bbox.jpg')
            plt.close(fix)

def normalize_image(path): 
    with Image.open(path) as im:
        return im.convert('RGB')
    
def calc_bb(im):
    boxes = []
    boxes_name = []
    observed = {"person","car"}
    obj_list= {"unlabeled": np.array([0, 0, 0]), "building": np.array([70, 70, 70]), "fence": np.array([100, 40, 40]), "other": np.array([55, 90, 80]), "person": np.array([220, 20, 60])
        , "pole": np.array([153, 153, 153]), "roadline": np.array([157, 234, 50]), "road": np.array([128, 64, 128]), "sidewalk": np.array([244, 35, 232]), "vegetation": np.array([107, 142, 35])
        , "car": np.array([0, 0, 142]), "wall": np.array([102, 102, 156]), "signs": np.array([220, 220, 0]), "sky": np.array([70, 130, 180]), "ground": np.array([81, 0, 81])
        , "bridge": np.array([150, 100, 100]), "railtrack": np.array([230, 150, 140]), "guardrail": np.array([180, 165, 180]), "trafficlight": np.array([250, 170, 30])
        , "static": np.array([110, 190, 160]), "dynamic": np.array([170, 120, 50]), "water": np.array([45, 60, 150]), "terrain": np.array([145, 170, 100]) }

    im_ = np.array(im)
    im_ = im_[:,:,:3]

    for key in obj_list:
        if key in observed:
            col_filter = obj_list[key]
            f = np.all(im_ == col_filter, axis=2)
            n, m = f.shape
            top = np.any(f, axis=1).argmax()
            bottom = (n-1) - np.any(f, axis=1)[::-1].argmax()
            left = np.any(f, axis=0).argmax()
            right = (m-1) - np.any(f, axis=0)[::-1].argmax()
            box = [left,top,right,bottom]
            if box != [0,0,m-1,n-1]:
                boxes.append(box)
                boxes_name.append(key)
              
    # r = np.array([220, 20, 60])
    # f = np.all(im_ == r, axis=2)
    # n, m = f.shape
    # top = np.any(f, axis=1).argmax()
    # bottom = (n-1) - np.any(f, axis=1)[::-1].argmax()
    # left = np.any(f, axis=0).argmax()
    # right = (m-1) - np.any(f, axis=0)[::-1].argmax()
    # boxes = [[left,top,right,bottom]]
    print(boxes,boxes_name)
    return boxes, boxes_name

def draw_boxes(image,boxes,label,img_name,var_name):
    img = F.pil_to_tensor(image)
    boxes = torch.Tensor(boxes)
    # label = ['person']
    colors = ['blue'] * len(boxes)
    result = draw_bounding_boxes(img,boxes,label,colors=colors, width=5, font="arial.ttf", font_size=30)
    show_and_save(result,img_name, var_name)

def get_csv(list_images,var_name):
    bbox_df = pd.DataFrame()
    for img in list_images:
#######################################################################    
        # # evaluate specific image 

        # specific_img = os.path.join(images_folder, 'ydist_variation1_00357.jpg')
        # if img == specific_img:
        #     var_name = Path(img).stem[-6]
        #     img_name=Path(img).stem
        #     print(img_name)
        #     # print(calc_bb(img))
        #     img = normalize_image(img)
        #     box,name,score = draw_boxes(img, calc_bb(img), img_name)
        #     df_dict = {img_name: [box,name,score]}
        #     df = pd.DataFrame.from_dict(df_dict, orient='index',
        #                 columns=['BBox', 'Type', 'Score'])
        #     bbox_df = pd.concat([bbox_df,df])
        # else :
        #     continue
#######################################################################
        # evaluate all images
        img_name=Path(img).stem
        print(img_name)
        img = normalize_image(img)
        boxes, name = calc_bb(img)
        draw_boxes(img, boxes, name, img_name, var_name)
        # box,name,score = get_info(calc_bb(img))
        df_dict = {img_name: [boxes,name]}
        df = pd.DataFrame.from_dict(df_dict, orient='index',
                    columns=['BBox','Type'])
        bbox_df = pd.concat([bbox_df,df])
    csv_name = var_name + '.csv'
    csv_path = os.path.join('d:',os.sep,'Carla','WindowsNoEditor','Bbox_sem',csv_name)
    bbox_df.to_csv(csv_path)
    # bbox_df.to_csv('csv_file/testing.csv')    
    print("sucessfuly save"+csv_name)


def main():
    images_variation_folder = os.path.join('d:',os.sep,'Carla','WindowsNoEditor','images_sem')
    images_folder = glob.glob(os.path.join(images_variation_folder,'*'))

    for folder in images_folder:
        list_images=glob.glob(os.path.join(folder, '*.jpg'))
        folder_name = Path(folder).stem
###############################################################################
        # evaluate specific folder
        specific_folder = os.path.join(images_variation_folder, 'xydist_variation2')
        if folder == specific_folder:
            get_csv(list_images,folder_name)
        else:
            continue
###############################################################################
        # # evaluate all folder

        # get_csv(list_images, folder_name)

if __name__ == "__main__":
    main()