import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import pandas as pd
import matplotlib
import sys
import glob
# from torchvision.io import read_image
from PIL import Image
from pathlib import Path
import os
import requests
import io
matplotlib.use('Agg')
observed = {"car", "person", "cat", "truck"}

def draw_boxes(image, boxes, img_name, var_name):
    plt.rcParams["savefig.bbox"] = 'tight'

    def show_and_save(imgs, img_name, var_name):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        Path('D:/Carla/WindowsNoEditor/Bbox_rgb/images/'+ var_name).mkdir(parents=True, exist_ok=True)
        plt.savefig('D:/Carla/WindowsNoEditor/Bbox_rgb/images/'+ var_name +'/'+ img_name +'_with_bbox.jpg')
        # plt.savefig('images/with_bbox/'+ img_name +'_with_bbox.jpg')
        plt.close(fix)

    selected_box = [b for n, b, s in boxes if n in observed]
    box_name = [n for n, b, s in boxes if n in observed]
    box_score = [s for n, b, s in boxes if n in observed]
    label = [f"{n}, {s:.3f}" for n, _, s in boxes if n in observed]

    boxes = torch.tensor(selected_box, dtype=torch.float)
    print(boxes)
    colors = ["blue"] * len(boxes)
    img = F.pil_to_tensor(image)
    result = draw_bounding_boxes(img, boxes,label,colors=colors, width=5, font="arial.ttf", font_size=30)
    show_and_save(result, img_name, var_name)
    
    return selected_box,box_name,box_score


def get_info(boxes):
    selected_box = [b for n, b, s in boxes if n in observed]
    box_name = [n for n, b, s in boxes if n in observed]
    box_score = [s for n, b, s in boxes if n in observed]
    boxes = torch.tensor(selected_box, dtype=torch.float)
    print(boxes)
    return selected_box,box_name,box_score


def calc_bb(image):
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    r = requests.post("http://127.0.0.1:8080/predictions/fastrcnn", data=buf.getvalue())
    
    def unpack(obj):
        score = obj['score']
        [(name, box)] = [(n, b) for n, b in obj.items() if n != "score"]
        return name, box, score

    return [unpack(obj) for obj in r.json()]


# return some image
def normalize_image(path): 
    with Image.open(path) as im:
        return im.convert('RGB')


def get_csv(list_images, var_name):
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
        box,name,score = draw_boxes(img, calc_bb(img), img_name, var_name)
        # box,name,score = get_info(calc_bb(img))
        df_dict = {img_name: [box,name,score]}
        df = pd.DataFrame.from_dict(df_dict, orient='index',
                    columns=['BBox', 'Type', 'Score'])
        bbox_df = pd.concat([bbox_df,df])
    csv_name = var_name + '.csv'
    csv_path = os.path.join('d:',os.sep,'Carla','WindowsNoEditor','Bbox_rgb',csv_name)
    bbox_df.to_csv(csv_path)
    # bbox_df.to_csv('csv_file/testing.csv')    
    print("sucessfuly save"+csv_name)


def main():
    # img = "car.jpg"
    # print(calc_bb(img))
    # img_name=Path(img).stem
    # print(img_name)

    # print(calc_bb(img))
    
    # img = normalize_image(img)
    # draw_boxes(img, calc_bb(img), img_name)

    # images_folder = sys.argv[1]
    # images_folder = os.path.join('d:',os.sep,'PytorchOD','PyServe','images')

    # images_folder = os.path.join('d:',os.sep,'Carla','WindowsNoEditor','images_rgb')
    # list_images=glob.glob(os.path.join(images_folder, '*.jpg'))

    images_variation_folder = os.path.join('d:',os.sep,'Carla','WindowsNoEditor','images_rgb')
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