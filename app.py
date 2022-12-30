from flask import Flask, render_template, request, send_file, session, Response, send_from_directory, jsonify

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.visualize import image_mask_and_boxes, instances_to_images, create_image_boxed, add_value, \
    get_image_for_blueprint, project_name_correction
from mrcnn.use_tesseract import image_to_string, image_to_txt, image_to_df
# from mrcnn.closes_node import closest_node2

import numpy
import re

from pdf2image import convert_from_bytes, convert_from_path

import skimage.io
import skimage.transform as tr
import skimage.exposure as exposure
from skimage.filters import unsharp_mask

from PIL import Image, ImageEnhance

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import json

import tensorflow as tf
from tensorflow import keras

# tf.compat.v1.disable_eager_execution()
# init = tf.compat.v1.global_variables_initializer()
# graph = tf.compat.v1.get_default_graph()
config = tf.compat.v1.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)
# graph = tf.Graph()
from keras.backend import get_session
from tensorflow.python.keras.backend import set_session

what_cropped = ["Stamp", "Company_name", "Label_title_h", "Label_title_v", "Label_Proj_no_h",
                "Label_Proj_no_v", "Label_dr_no_h", "Label_dr_no_v", "Label_rev_h", "Label_rev_v",
                "Label_scale_h", "Label_scale_v", "Label_date_h", "Label_date_v"]


# gpus = tf.test.gpu_device_name()
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         # for gpu in gpus:
#         #     tf.test.gpu_device_name().set_memory_growth(gpu, True)
#         # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         sess = tf.Session(config=config)
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

# Initialize a shared session for keras / tensorflow operations.
# session = get_session()
# init = tf.global_variables_initializer()
# session.run(init)

def load_the_model(path, config):
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="C:\\Users\\kuanyshov.a\\Anaconda3\\logs")
    model.load_weights(path, by_name=True)
    return model


def crop_instances(image, boxes, class_ids, what_to_crop):
    white_big = skimage.io.imread("white.jpg")
    # print(white_big.shape)
    white_big_rgb = skimage.color.gray2rgb(white_big)
    n_instances = boxes.shape[0]
    for i in range(n_instances):
        if class_ids[i] in what_to_crop:
            y = boxes[i][0]
            x = boxes[i][1]
            height = boxes[i][2]
            width = boxes[i][3]
            # white_small = white_big[0: width - x, 0: height - y]
            image[y:height, x:width] = white_big_rgb[0:height - y, 0: width - x]
    return image, what_to_crop


class InferenceConfigStamp(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "stamp"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 34  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.85

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


modelStamp = load_the_model("C:\\Users\\kuanyshov.a\\Anaconda3\\logs\\stable_versions\\mask_rcnn_stamp4_0010.h5",
                            config=InferenceConfigStamp())
class_names_modelStamp = ["FONT", "Stamp", "Company_name", "Label_title_h", "Label_title_v",
                          "Project_name_h", "Project_name_v", "Label_Proj_no_h",
                          "Label_Proj_no_v", "Proj_no_h", "Proj_no_v", "Label_dr_no_h", "Label_dr_no_v",
                          "Dr_no_h",
                          "Dr_no_v", "Label_rev_h", "Label_rev_v", "REV_h", "REV_v", "Label_scale_h",
                          "Label_scale_v", "Scale_h", "Scale_v", "Label_date_h", "Label_date_v", "Date_h",
                          "Date_v", "Label_by", "Label_chk", "Label_eng",
                          "Label_supv", "Label_oper", "Label_mgr", "Role_input", "Blue_drawing_no"]


# Load the machine learning model and weights only once and use throughout
# application lifetime.
class CustomConfigBlueprint(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "BLUEPRINT2"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95


class InferenceConfigBlueprint(CustomConfigBlueprint):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config2 = InferenceConfigBlueprint()
ROOT_DIR = os.path.abspath("../../")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

model2 = modellib.MaskRCNN(mode="inference", config=config2,
                           model_dir=DEFAULT_LOGS_DIR)
model2.load_weights("C:\\Users\\kuanyshov.a\\Anaconda3\\logs\\stable_versions\\mask_rcnn_blueprint3_0010.h5",
                    by_name=True)
class_names_model2 = ['FONT', 'STAMP_h', 'STAMP_v', 'COMPAS', 'DRAWING NO',
                      'PAPER_SIZE', 'Drawing_no_blue', 'SCALE_BAR']  # Blueprint 2.0 no v_h


# model2.keras_model._make_predict_function()
# graph = tf.get_default_graph()
# set_session(session)


def splash(model, class_names, arr_image):
    if arr_image.shape[-1] == 4:
        arr_image = arr_image[..., :3]
    r = model.detect([arr_image], verbose=1)[0]
    image = image_mask_and_boxes(arr_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    return image


def check_O(text):
    if "0" in text:
        return text.replace("0", "O")
    else:
        return text


def replace_numbers(text):
    """
    Replaces all numbers instead of characters from 'text'.

    :param text: Text string to be filtered
    :return: Resulting number
    """
    list_of_numbers = re.findall(r'[a-zA-Z]+', text)
    result_number = ''.join(list_of_numbers)
    return result_number


def replace_code(text):
    for iter in re.finditer(re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)"), text):
        result_text = ''.join(iter.group())
        return result_text
    return text


def datetime_extract(image):
    patterns = [
        re.compile(r"^[0-9]+[/.\\-][0-9]+[/.\\-][0-9]+$", re.IGNORECASE),
        re.compile(r"^[0-9]+[/.\\-][0-9]+[/.\\-][0-9]+$", re.IGNORECASE)

    ]
    patterns2 = [
        re.compile(r"^[0-9]+[/.\\-][0-9]+$", re.IGNORECASE), re.compile(r"^[0-9]{2}[/.\\-][0-9]+$", re.IGNORECASE)
    ]
    # date_re = re.match([0-9]{2}[-\\. ]{1,2}[0-9]{2}[-\\. ]{1,2}(19|20)[0-9]{2})
    dt = None
    content = image_to_string(image, lang="eng1.9to1.10_3", config=r'--oem 3 --psm 7 -c page_separator=""')
    # print(content)
    for pattern in patterns:
        # print(pattern)
        try:
            dt = re.search(pattern, content).group()
            # print(dt)
            break
        except AttributeError:
            continue
    if dt is None:
        for pattern in patterns2:
            # print(pattern)
            try:
                dt = re.search(pattern, content).group()
                # print(dt)
                if pattern == re.compile(r"^[0-9]{2}[/.\\-][0-9]+$", re.IGNORECASE):
                    dt = dt[0:5] + r"/" + dt[6::]
                    break
                elif pattern == re.compile(r"^[0-9]+[/.\\-][0-9]+$", re.IGNORECASE):
                    dt = dt[0:2] + r"/" + dt[3::]
                    break
            except AttributeError:
                continue
        if dt is None:
            print("No match")
            return content  # Исправлять дт в этом месте два кейса 21117/22 и 21/17121
    return dt


# def instances_to_images(image, boxes, class_ids, names, list_accepted_ids=[1, 2, 8, 10], stamp=False):
#     """Crop the images by their boxes
#     ВОЗВРАЩАЕТ ЛИСТ КАРТИНОК В БОКСАХ
#     """
#     if stamp:
#         indices_role_input = [i for i, x in enumerate(class_ids) if x == 33]
#         role_input_coords = [x for i, x in enumerate(boxes) if i in indices_role_input]  # Все корды инпутов
#     n_instances = boxes.shape[0]
#     dict_images = {}
#     for i in range(n_instances):
#         if list_accepted_ids == [True]:
#             if not np.any(boxes[i]) or class_ids[i] in [1]:
#                 continue
#         else:
#             if not np.any(boxes[i]) or not class_ids[i] in list_accepted_ids:
#                 continue
#         label = names[class_ids[i]]
#         if stamp:
#             if label in ["Label_by", "Label_chk", "Label_eng", "Label_supv", "Label_oper", "Label_mgr"]:
#                 box = boxes[i]  # Корды лэйбла
#                 print(label)
#                 print(type(box))
#                 print(box)
#                 box = closest_node2(box, role_input_coords)[1]
#                 if box is None:
#                     add_value(dict_images, label, {"Coordinates": None})
#                 else:
#                     boxed_image = create_image_boxed(image, box)
#                     dict_roles = {"Coordinates": boxed_image}
#                     add_value(dict_images, label, dict_roles)
#             else:
#                 box = boxes[i]
#                 boxed_image = create_image_boxed(image, box)
#                 key = label
#                 add_value(dict_images, key, boxed_image)
#         else:
#             box = boxes[i]
#             boxed_image = create_image_boxed(image, box)
#             # imshow(boxed_image)
#             # plt.show()
#             # list_images.append(boxed_image)
#             # key = class_ids[i]
#             key = label
#             # dict_images[class_ids[i]] = boxed_image
#             add_value(dict_images, key, boxed_image)
#     return dict_images


def detect1(model, image):
    try:
        with session.as_default():
            with session.graph.as_default():
                r0 = model.detect([image], verbose=1)[0]
                return r0
    except:
        return None
    #             n_instances = r0['rois'].shape[0]
    #             for i in range(n_instances):
    #                 if not np.any(r0['rois'][i]) or not r0['class_ids'][i] in [1, 2]:
    #                     continue
    #                 box = r0['rois'][i]
    #                 boxed_image = create_image_boxed(image, box)
    #             return boxed_image
    # except UnboundLocalError:
    #     return None


def detect2(model, image):
    try:
        with session.as_default():
            with session.graph.as_default():
                r = model.detect([image], verbose=1)[0]
                return r
    except:
        return None


def splash_global(model, class_names, image):
    with session.as_default():
        with session.graph.as_default():
            image0 = splash(model, class_names, image)
            return image0


def array_indexes_of_duplicates_of(arr, what_include=None):
    """
    Возвращает индексы values(повторяющихся) в arr

    Returns
    -------
    list_duplicates : Возвращаем значения id классов которые повторяются e.g. [3, 7, 11] - 3 лейбла повторяются
    """
    if what_include is None:
        what_include = [3, 4, 7, 8, 11, 12, 15, 16, 23, 24]
    arr_copy = [id for id in arr if id in what_include]
    values_arr, counts_arr = np.unique(arr_copy,
                                       return_counts=True)  # values = np.array([ВСЕ УНИКАЛЬНЫЕ ЗНАЧЕНИЯ])       counts_arr = np.arr([кол-во повторений])
    list_duplicates = []
    for i in range(values_arr.shape[0]):
        if counts_arr[i] > 1:
            list_duplicates.append(values_arr[i])
    return list_duplicates


def filter_array_of_array(arr_class_ids, arr_conf, arr_boxes, arr_masks, ids_list):
    list_delete = []
    for id in ids_list:  # Для каждого айди
        indexes_list = np.where(arr_class_ids == id)[0]  # Лист индексов повторяющихся айди
        list_conf = []
        for index in indexes_list:
            list_conf.append(arr_conf[index])  # Грузим лист всех конфинесов повторяющихся айди
        list_conf.remove(max(list_conf))
        if list_conf is not False:
            for conf in list_conf:
                ind_delete = np.where(arr_conf == conf)[0][0]
                list_delete.append(ind_delete)
        else:
            break
    list_delete.sort(reverse=True)
    for i in list_delete:
        arr_conf = np.delete(arr_conf, i)
        arr_class_ids = np.delete(arr_class_ids, i)
        arr_boxes = np.delete(arr_boxes, i, axis=0)
        arr_masks = np.delete(arr_masks, i, axis=2)
    return arr_conf, arr_boxes, arr_class_ids, arr_masks


def predict(pdffile):

    # global graph, session
    # with session.as_default():

    # image_path = "./Static/" + "/images/" + imagefile.filename
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], pdffile.filename)
    pdffile.save(output_image_path)

    image = get_image_for_blueprint(output_image_path, dpi=200)

    r0 = detect1(model2, image)
    if r0 is None:
        return render_template('no_detection.html')
    if 1 in r0['class_ids'] and 2 in r0['class_ids']:  # Оба штампа найдены
        # Выбираем тот в котором больше уверенности
        if r0['scores'][np.where(r0['class_ids'] == 1)[0][0]] > r0['scores'][np.where(r0['class_ids'] == 2)[0][0]]:
            accept = [1]
        else:
            accept = [2]
    else:
        accept = [1, 2]

    what_cropped = ["Stamp", "Company_name", "Label_title_h", "Label_title_v", "Label_Proj_no_h",
                    "Label_Proj_no_v", "Label_dr_no_h", "Label_dr_no_v", "Label_rev_h", "Label_rev_v",
                    "Label_scale_h", "Label_scale_v", "Label_date_h", "Label_date_v"]
    dict_images = instances_to_images(image, r0['rois'], r0['class_ids'], names=class_names_model2,
                                      list_accepted_ids=accept, stamp=False)
    dict_text = {}

    try:
        for id in dict_images:
            r1 = detect2(modelStamp, dict_images[id])
            if r1 is None:
                return render_template('no_detection.html')
            list_dups = array_indexes_of_duplicates_of(r1['class_ids'])
            r1['scores'], r1['rois'], r1['class_ids'], r1['masks'] = filter_array_of_array(r1['class_ids'],
                                                                                           r1['scores'], r1['rois'],
                                                                                           r1['masks'], list_dups)
            stamp_image_copy = dict_images[id].copy()
            image_pure, rfederaciya = crop_instances(stamp_image_copy, r1['rois'], r1['class_ids'],
                                                     [1, 2, 3, 4, 7, 8, 11, 12, 15, 16, 23, 24])
            dict_images_stamp = instances_to_images(image_pure, r1['rois'], r1['class_ids'],
                                                    names=class_names_modelStamp,
                                                    list_accepted_ids=[5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26,
                                                                       27,
                                                                       28,
                                                                       29, 30, 31, 32, 33, 34], stamp=True)
            for id in dict_images_stamp:
                if id in what_cropped:
                    continue
                else:
                    if id == "Date_v" or id == "Date_h":
                        if isinstance(dict_images_stamp[id], list):
                            text = datetime_extract(dict_images_stamp[id][0])
                        else:
                            text = datetime_extract(dict_images_stamp[id])
                        dict_text[id] = text
                    elif id in ["Label_oper", "Label_eng", "Label_mgr", "Label_chk", "Label_by", "Label_supv"]:
                        try:
                            if dict_images_stamp[id]["Coordinates"] is None:
                                dict_text[id] = "Empty"
                            else:
                                text = image_to_string(dict_images_stamp[id]["Coordinates"], lang="eng1.9to1.10_3",
                                                       config=r'--oem 3 --psm 7 -c page_separator=""')
                                print("ОРИГ ЛЭЙБЛА РОЛИ: " + text)
                                text = check_O(text)
                                text = replace_numbers(text)
                                dict_text[id] = text

                        except TypeError:
                            if isinstance(dict_images_stamp[id], list):
                                if dict_images_stamp[id][0]["Coordinates"] is None:
                                    dict_text[id] = "Empty"
                                else:
                                    im = Image.fromarray(dict_images_stamp[id][0]["Coordinates"])
                                    dict_text[id] = replace_numbers(
                                        check_O(image_to_string(im, lang="eng1.9to1.10_3",
                                                                config=r'--oem 3 --psm 7 -c '
                                                                       r'page_separator=""')))
                    elif id == "Role_input":
                        pass
                    elif id in ["REV_h",
                                "REV_v"]:
                        if isinstance(dict_images_stamp[id], list):
                            # text = image_to_string(dict_images_stamp[id][0], lang="eng1.4to_proj_no3",
                            #                        config=r'--oem 3 --psm 8 -c page_separator=""')
                            text = image_to_string(dict_images_stamp[id][0], lang='eng1.9',
                                                   config=r'--oem 3 --psm 8 -c page_separator=""')
                        else:
                            # text = image_to_string(dict_images_stamp[id], lang="eng1.4to_proj_no3",
                            #                        config=r'--oem 3 --psm 8 -c page_separator=""')
                            text = image_to_string(dict_images_stamp[id], lang='eng1.9',
                                                   config=r'--oem 3 --psm 8 -c page_separator=""')
                        text = text.replace("O", "0")
                        dict_text[id] = text
                    elif id in ["Scale_v", "Scale_h"]:
                        if isinstance(dict_images_stamp[id], list):
                            text = image_to_string(dict_images_stamp[id][0], lang="eng1.9+rus1.7",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="eng1.9+rus1.7",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        if text == "":
                            text = "-"
                        elif text is None:
                            text = "-"
                        dict_text[id] = text
                    elif id in ["Project_name_h", "Project_name_v"]:
                        if isinstance(dict_images_stamp[id], list):
                            dict_images_stamp[id] = dict_images_stamp[id][0]
                        # if isinstance(dict_images_stamp[id], list):
                        #     text = image_to_string(dict_images_stamp[id][0], lang="eng1.9to1.10_5+rus1.7to1.9_4",
                        #                            config=r'--oem 3 --psm 6 -c page_separator=""')
                        # else:
                        #     Image.fromarray(dict_images_stamp[id]).show()
                        #     text = image_to_string(dict_images_stamp[id], lang="eng1.9to1.10_5+rus1.7to1.9_4",
                        #                            config=r'--oem 3 --psm 6 -c page_separator=""')
                        df = image_to_df(dict_images_stamp[id], section="text", lang=langeng,
                                         config=r'--oem 3 --psm 6 -c page_separator=""')  # Прокручиваем всё на англ языке
                        a = 0
                        sum_lines = 0
                        line_qt = df.max()['line_num']
                        if df.max()['par_num'] > 1:
                            for i in range(df.max()['par_num']):
                                rslt = df[df['par_num'] == i + 1]
                                try:
                                    sum_lines += rslt.max()['line_num']
                                    # df_orig_par2 = df[df['par_num'] == i]
                                    # df_orig_par2.line_num += df[df['par_num'] == i].max()['line_num']
                                except:
                                    pass
                            line_qt = sum_lines

                        text_eng_name, text_rus_name = project_name_correction(df, line_qt, dict_images_stamp[id],
                                                                               langeng,
                                                                               langrus)

                        # if isinstance(dict_images_stamp[id], list):
                        #     text_name_pr = image_to_string(dict_images_stamp[id][0],
                        #                                    lang="eng1.9to1.10_5+rus1.7to1.9_4",
                        #                                    config=r'--oem 3 --psm 6 -c page_separator=""')
                        # else:
                        #     text_name_pr = image_to_string(dict_images_stamp[id],
                        #                                    lang="eng1.9to1.10_5+rus1.7to1.9_4",
                        #                                    config=r'--oem 3 --psm 6 -c page_separator=""')

                    elif id in ["Proj_no_h", "Proj_no_v"]:
                        if isinstance(dict_images_stamp[id], list):
                            text = image_to_string(dict_images_stamp[id][0], lang="eng1.3to_proj_no3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="eng1.3to_proj_no3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        print("ТЕКСТ ОРИГ ПРОЖ. НОМЕРА:   " + text)
                        patterns0 = [
                            re.compile(r"\b0-|-0-"),  # Паттерн отдельных мнимых 0, в реальности это буква О
                            re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)")
                        ]
                        text = replace_code(text)
                        for match_zeros in re.finditer(patterns0[0], text):
                            text = text[0:match_zeros.start()] + text[
                                                                 match_zeros.start(): match_zeros.end()].replace(
                                "0",
                                "O") + text[
                                       match_zeros.end(): len(
                                           text)]  # На каждый найденный мнимый 0, заменяем его в тексте
                        dict_text['Proj_no_2'] = image_to_string(dict_images_stamp[id], lang="Proj_no_re_1",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        dict_text[id] = text
                    elif id in ["Dr_no_h", "Dr_no_v"]:
                        if isinstance(dict_images_stamp[id], list):
                            text = image_to_string(dict_images_stamp[id][0], lang="Dr_no_3",
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="Dr_no_3",
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
                        patterns0 = [
                            re.compile(r"\b0-|-0-"),  # Паттерн отдельных мнимых 0, в реальности это буква О
                            re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)")
                        ]
                        print("ТЕКСТ ОРИГ ДР НОМЕРА:    os.path.basename(path)" + text)
                        print("НОМЕР ДОКА : " + os.path.basename(output_image_path))
                        text = replace_code(text)
                        for match_zeros in re.finditer(patterns0[0], text):
                            text = text[0:match_zeros.start()] + text[
                                                                 match_zeros.start(): match_zeros.end()].replace(
                                "0",
                                "O") + text[
                                       match_zeros.end(): len(
                                           text)]  # На каждый найденный мнимый 0, заменяем его в тексте
                        dict_text[id] = text
                    else:
                        if isinstance(dict_images_stamp[id], list):
                            list_text = []
                            for i in dict_images_stamp[id]:
                                text = image_to_string(i, lang="eng1.9to1.10_4+rus1.7to1.9_2",
                                                       config=r'--oem 3 --psm 6 -c page_separator=""')
                                list_text.append(text)
                            dict_text[id] = list_text
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="eng1.9to1.10_4+rus1.7to1.9_2",
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
                            dict_text[id] = text

        # dict_text["CHECK_WITH_THIS"] = text_name_pr
        if text_rus_name is not None:
            dict_text["Rus_name"] = text_rus_name
            dict_text["Eng_name"] = text_eng_name
        elif text_rus_name is None and text_eng_name is None:
            dict_text["Rus_name"] = None
            dict_text["eng_name"] = None
        else:
            dict_text["Eng_name"] = text_eng_name
        dict_text["File_name"] = os.path.basename(output_image_path)

        # with open(os.path.join(os.path.dirname(output_image_path), "sample.json"), "w", encoding="utf-8") as outfile:
        #     json.dump(dict_text, outfile, skipkeys=True, indent=4)
        # with open(os.path.join(app.config['UPLOAD_FOLDER'], "sample_no_ascii.json"), "w", encoding="utf-8") as outfile:
        #     json.dump(dict_text, outfile, skipkeys=True, indent=4, ensure_ascii=False)

        # boxed_image = list(dict_images.values())[0]
        # image0 = splash_global(modelStamp, class_names_modelStamp, boxed_image)

        # output_image_path = output_image_path.replace(".pdf", ".jpg")
        #
        # skimage.io.imsave(output_image_path, image0)
        #
        # output_image_path = output_image_path.replace("/Static/", "")
        # print(output_image_path)

        # filename = "/Static/images/" + os.path.basename(image_path)
        #
        # send_file(image_path)

        # , prediction=output_image_path

        # return render_template('indexdownload.html')
        return dict_text
    except:
        # return render_template('no_detection.html')
        return None


UPLOAD_FOLDER = os.path.join('Static', 'images')
app = Flask(__name__,
            static_folder="C:\\Users\\kuanyshov.a\\Documents\\Project\\flask_project\\FirstFlaskWebApp\\Static")
app.config['SECRET_KEY'] = 'tempkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

langeng = 'eng1.9to1.10_5'
langrus = 'rus1.7'


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/detect', methods=['POST', 'GET'])
def send():
    if request.method == 'POST':
        pdffile = request.files['filename']
        dict_text = predict(pdffile)
        if dict_text is None:
            return render_template('no_detection.html')
        else:
            return json.dumps(dict_text, skipkeys=True, indent=4, ensure_ascii=False)


@app.route('/pdf', methods=['POST'])
def reveive():
    return jsonify({"message": "Well received"})


@app.route('/processjson', methods=['POST'])
def process():
    return None


@app.route('/download', methods=['POST', 'GET'])
def download():
    if request.method == 'POST':
        # return send_from_directory(app.config['UPLOAD_FOLDER'], "sample_no_ascii.json")
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], "sample_no_ascii.json"))


@app.route('/get_back', methods=['POST', 'GET'])
def get_back():
    if request.method == 'POST':
        # return send_from_directory(app.config['UPLOAD_FOLDER'], "sample_no_ascii.json")
        return render_template('index.html')


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(port=3000, debug=True)
