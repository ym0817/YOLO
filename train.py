# coding=utf-8
import torch
from pygments.styles.dracula import background

from ultralytics import YOLO
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
import sys, os, shutil, cv2
import time, csv
import yaml
import random
import argparse



def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    final_x = 0.0 if x < 0.0 else x
    final_x = 1.0 if final_x > 1.0 else x
    final_y = 0.0 if y < 0.0 else y
    final_y = 1.0 if final_y > 1.0 else y
    final_w = 0.0 if w < 0.0 else w
    final_w = 1.0 if final_w > 1.0 else w
    final_h = 0.0 if h < 0.0 else h
    final_h = 1.0 if final_h > 1.0 else h
    return (final_x, final_y, final_w, final_h)

def convert_annotation(classes, path, savepath):
    filenames = os.listdir(path)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for image_name in filenames:
        # print(image_name)
        in_file = open(os.path.join(path, image_name), 'r', encoding='utf-8')
        xml_text = in_file.read()
        try:
            root = ET.fromstring(xml_text)
        except:
            pass
        in_file.close()
        size = root.find('size')
        w = int(size.find('width').text)
        # print(image_name)
        h = int(size.find('height').text)
        out_file = open(os.path.join(savepath, image_name[:-4] + '.txt'), 'w', encoding='utf-8')
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                print('Not exist in Classes  ' ,image_name, cls)
                continue
            cls_id = classes.index(cls)
            if not obj.find('bndbox'):
                print('Please Check XML Format  ', image_name, cls)
            else:
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))
                bb = convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.close()

def get_backgroundimgs(images_path, xmls_path, background_path):
    if not os.path.exists(background_path):
        os.makedirs(background_path)
    all_xml_names = [xml_file[:-4] for xml_file in os.listdir(xmls_path)]
    all_img_names = [img_file[:-4] for img_file in os.listdir(images_path)]
    background_names = list(set(all_img_names) - set(all_xml_names))
    for img_file in os.listdir(images_path):
        img_path = os.path.join(images_path, img_file)
        if img_file[:-4] in background_names:
            shutil.copy(img_path, os.path.join(background_path, img_file))
    return background_names


def split_data(classes,imgs_path, txts_path, new_data_path, types, ratio):
    random.seed(0)
    data = { }
    for c in classes:
        data[c] = []
    for img_name in os.listdir(imgs_path):
        filename = img_name[:-4]
        img_path = os.path.join(imgs_path, img_name)
        txt_path = os.path.join(txts_path, filename + '.txt')
        if os.path.exists(txt_path):
            image = cv2.imread(img_path)
            in_file = open(txt_path, 'r', encoding='utf-8')
            lines = in_file.readlines()
            in_file.close()
            if image is not None and len(lines) >=1:
                cls_label = lines[0].strip().split()[0]
                cls_name = classes[int(cls_label)]
                data[cls_name].append(img_name)

    # data = {key: value for key, value in data.items() if len(value) >= 110 }
    train_imgs_path = os.path.join(new_data_path, 'images', types[0])
    val_imgs_path = os.path.join(new_data_path, 'images', types[1])
    train_txts_path = os.path.join(new_data_path, 'labels', types[0])
    val_txts_path = os.path.join(new_data_path, 'labels', types[1])
    print("defect-cls-num :  ", len(data))
    total_cnt = 0
    for k, v in data.items():
        total_cnt += len(v)
    print("All useful image-xml-pair count : ", total_cnt)
    for k, v in data.items():
        val_list = random.sample(v, int(ratio[1]*len(v)))
        train_list = [per_v for per_v in v if per_v not in val_list]
        print('cls-name :  ', k, 'train-num :  ', len(train_list), 'val-num :  ', len(val_list))
        for per_v in v:
            txt_name = per_v[:-4] + '.txt'
            if per_v in val_list:
                if not os.path.exists(val_imgs_path):
                    os.makedirs(val_imgs_path)
                if not os.path.exists(val_txts_path):
                    os.makedirs(val_txts_path)
                shutil.copy(os.path.join(imgs_path, per_v), os.path.join(val_imgs_path, per_v))
                shutil.copy(os.path.join(txts_path, txt_name), os.path.join(val_txts_path, txt_name))
            else:
                if not os.path.exists(train_imgs_path):
                    os.makedirs(train_imgs_path)
                if not os.path.exists(train_txts_path):
                    os.makedirs(train_txts_path)
                shutil.copy(os.path.join(imgs_path, per_v), os.path.join(train_imgs_path, per_v))
                shutil.copy(os.path.join(txts_path, txt_name), os.path.join(train_txts_path, txt_name))



def split_data2(classes,imgs_path, txts_path, backgtound_path, new_data_path, types, ratio):
    random.seed(0)
    data = { }
    for c in classes:
        data[c] = []
    data["background"] = []
    for img_name in os.listdir(imgs_path):
        filename = img_name[:-4]
        img_path = os.path.join(imgs_path, img_name)
        txt_path = os.path.join(txts_path, filename + '.txt')
        if os.path.exists(txt_path):
            image = cv2.imread(img_path)
            in_file = open(txt_path, 'r', encoding='utf-8')
            lines = in_file.readlines()
            in_file.close()
            if image is not None and len(lines) >=1:
                cls_label = lines[0].strip().split()[0]
                cls_name = classes[int(cls_label)]
                data[cls_name].append(img_name)
        if img_name in os.listdir(backgtound_path):
            data["background"].append(img_name)

    # data = {key: value for key, value in data.items() if len(value) >= 110 }
    train_imgs_path = os.path.join(new_data_path, 'images', types[0])
    val_imgs_path = os.path.join(new_data_path, 'images', types[1])
    train_txts_path = os.path.join(new_data_path, 'labels', types[0])
    val_txts_path = os.path.join(new_data_path, 'labels', types[1])
    if not os.path.exists(val_imgs_path):
        os.makedirs(val_imgs_path)
    if not os.path.exists(val_txts_path):
        os.makedirs(val_txts_path)
    if not os.path.exists(train_imgs_path):
        os.makedirs(train_imgs_path)
    if not os.path.exists(train_txts_path):
        os.makedirs(train_txts_path)
    print("defect-cls-num :  ", len(data))
    total_cnt = 0
    for k, v in data.items():
        total_cnt += len(v)
    print("All useful image-xml-pair count : ", total_cnt)
    for k, v in data.items():
        if k != "background":
            val_list = random.sample(v, int(ratio[1]*len(v)))
            train_list = [per_v for per_v in v if per_v not in val_list]
            print('cls-name:  ', k, '       train-num:  ', len(train_list), '       val-num:  ', len(val_list))
            for per_v in v:
                txt_name = per_v[:-4] + '.txt'
                if per_v in val_list:
                    shutil.copy(os.path.join(imgs_path, per_v), os.path.join(val_imgs_path, per_v))
                    shutil.copy(os.path.join(txts_path, txt_name), os.path.join(val_txts_path, txt_name))
                else:

                    shutil.copy(os.path.join(imgs_path, per_v), os.path.join(train_imgs_path, per_v))
                    shutil.copy(os.path.join(txts_path, txt_name), os.path.join(train_txts_path, txt_name))
        else:
            for per_v in v:
                shutil.copy(os.path.join(imgs_path, per_v), os.path.join(train_imgs_path, per_v))


def train(model, data_root, batch_size, img_size, max_epoch, projectname, expname, init_lr, final_lr):
    gpu_ids= [i for i in range(torch.cuda.device_count())]
    results = model.train(
        data=data_root,
        epochs=max_epoch,
        imgsz=img_size,
        plots=True,
        batch=batch_size,
        amp=False,
        # project= project_name,
        patience=300,
        # degrees=180,
        # auto_augment=True,
        cache=True,
        hsv_h=0.02,
        translate=0.1,
        flipud=0.5,
        # bgr=0.5,
        close_mosaic=0,
        scale=0.1,
        device='0,1',
        # device=gpu_ids,
        # device='0',
        workers=0,
        # save_dir = save_weight,
        project=projectname,
        name=expname,
        resume=False,
        lr0=init_lr,
        lrf=final_lr

    )

def evaluation(model, data_root, batch_size, img_size):
    model.val(data=data_root,
              # split='test',
              imgsz=img_size,
              batch=batch_size,
              device=0,
              workers=0,
              save=True,
              task="test")

def over_message():
    width, height = 640, 480
    blank_image = np.zeros((height, width, 3), np.uint8)
    text = "over"
    position = (width // 2 - 20, height // 2 + 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    background_color = (0, 0, 0)  # 黑色
    cv2.putText(blank_image, text, position, font, font_scale, font_color, 2)
    cv2.imwrite("Over.jpg", blank_image)

def DisplaySampleInfo(data_path, classname_list):
    types = ['train','val']
    # classname_list = [c+1 for c in range(len(classname_list))]
    train_cnt_list = [0 for i in range(len(classname_list))]
    val_cnt_list = [0 for i in range(len(classname_list))]
    train_sample_cnt, val_sample_cnt = 0, 0
    for type in types:
        labdir_path = os.path.join(data_path, "labels", type)
        for lab in os.listdir(labdir_path):
            lab_path = os.path.join(labdir_path, lab)
            in_file = open(lab_path, 'r', encoding='utf-8')
            lines = in_file.readlines()
            in_file.close()
            if type == 'train':
                train_sample_cnt += 1
            else:
                val_sample_cnt += 1
            for line in lines:
                cls_id = int(line.strip().split()[0])
                if type == 'train':
                    train_cnt_list[cls_id] += 1
                else:
                    val_cnt_list[cls_id] += 1
        # print(classname_list)
        # print(cnt_list)
    samples_dict = {}
    samples_dict['train_sample_count'] = train_sample_cnt
    samples_dict['val_sample_count'] = val_sample_cnt
    # train_list, val_list = {}, {}
    t_list,v_list = [], []
    for k,v in zip(classname_list,train_cnt_list):
        t_w = k + "(" + str(v) + ")"
        t_list.append(t_w)
    for k,v in zip(classname_list,val_cnt_list):
        v_w = k + "(" + str(v) + ")"
        v_list.append(v_w)
    samples_dict['train-category'] = t_list
    samples_dict['val-category'] = v_list
    return samples_dict

def model_decryption(init_files, method, temp_dir):
    encryt_file_list = ["weight.init", "weight1.init"]
    license_list = ["license.txt", "license1.txt"]
    if method < len(encryt_file_list):
        print("select method {}".format(method))
        encryt_file = os.path.join(init_files, encryt_file_list[method])
        license = os.path.join(init_files, license_list[method])
    else:
        print("out of method index, default select method zero")
        encryt_file = os.path.join(init_files, encryt_file_list[0])
        license = os.path.join(init_files, license_list[0])

    out_file = os.path.join(temp_dir, "m.pt")
    with open(license, 'rb') as key:
        key = key.readline()
    with open(encryt_file, 'rb') as fr:
        encrypted_data = fr.read()
    decrypted_data = Fernet(key).decrypt(encrypted_data)
    with open(out_file, 'wb') as ew:
        ew.write(decrypted_data)
    initmodel = YOLO(out_file)
    # os.remove(out_file)
    return initmodel



def main(yaml_file):
    start_time = time.time()
    # file = open(yaml_file, 'r', encoding="utf-8")
    file = open(yaml_file, 'r', encoding="UTF-8")
    file_data = yaml.load(file, Loader = yaml.FullLoader)
    # file_data = yaml.safe_load(file)
    file.close()
    abs_path = file_data["path"]
    samples_path = file_data["samples_path"]
    xlms_path = file_data["xlms_path"]
    # save_modelpath = file_data["save_modelpath"]
    model_backup = file_data["backup"]
    # init_weight_path = file_data["init_weight_path"]
    data_name = file_data["data_name"]
    dataset_types = file_data["dataset_types"]
    dataset_ratio = file_data["dataset_ratio"]
    batch_size = file_data["batch_size"]
    img_size = file_data["img_size"]
    max_epoch = file_data["max_epoch"]
    lr0 = file_data["init_lr"]
    lrf = file_data["final_lr"]
    method_index = file_data["methods"]
    pretrain_weight = file_data["pretrain_weight"]
    is_add_backbound = file_data["addedbackground"]
    names_dict = file_data["names"]
    name_list = []
    for k, v in names_dict.items():
        print(v)
        name_list.append(v)
    imgs_data = os.path.join(abs_path, samples_path)
    xmls_data = os.path.join(abs_path, xlms_path)

    txts_data = os.path.join(os.path.split(xmls_data)[0], "temp_txts")
    background_data = os.path.join(os.path.split(xmls_data)[0], "background_images")
    train_path = os.path.join(os.path.split(xmls_data)[0], data_name)
    if os.path.exists(txts_data):
        shutil.rmtree(txts_data)
    if os.path.exists(background_data):
        shutil.rmtree(background_data)
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    convert_annotation(name_list, xmls_data, txts_data)
    get_backgroundimgs(imgs_data, xmls_data, background_data)
    if not is_add_backbound:
        split_data(name_list, imgs_data, txts_data, train_path, dataset_types, dataset_ratio)
    else:
        split_data2(name_list,imgs_data, txts_data, background_data, train_path, dataset_types, dataset_ratio)
    if os.path.exists(txts_data):
        shutil.rmtree(txts_data)
    if os.path.exists(background_data):
        shutil.rmtree(background_data)

    project_name = "PSL_ADC_V8"
    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = "train_" + current_time
    save_modelpath = os.path.join(abs_path, project_name)
    model = YOLO(pretrain_weight)
    # train(model, yaml_file, batch_size, img_size, max_epoch, save_modelpath, lr0, lrf)
    train(model, yaml_file, batch_size, img_size, max_epoch, project_name, exp_name, lr0, lrf)
    end_time = time.time()
    print(f"训练时长：{(end_time - start_time) / 3600} Hours")
    # re.write('Training-Cost-Time : ' + str((end_time - start_time) / 3600) + ' Hours' + '\n')

if __name__ == '__main__':

    # yaml_file = "config.yaml"
    # yaml_file = "D:\\AlgoData\\Trains\\try\\SN003\\config_1C.yaml"
    yaml_file = "config_1C.yaml"
    # yaml_file = sys.argv[1]
    main(yaml_file)






