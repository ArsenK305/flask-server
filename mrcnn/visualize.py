"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import random
import itertools
import colorsys
import regex as re
import numpy as np
from PIL import ImageEnhance, Image
from pdf2image import convert_from_bytes
from skimage.measure import find_contours
from skimage.io import imshow
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon
import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn.use_tesseract import image_to_df, image_to_string


############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def random_colors2(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def create_image_boxed(image, box):
    y = box[0]
    x = box[1]
    height = box[2]
    width = box[3]
    image = image[y:height, x:width]
    return image


def get_image_for_blueprint(path, dpi):
    """Create image for full blueprint from path
    path: path to pdf file
    """
    images = convert_from_bytes(open(path, 'rb').read(), dpi=dpi)  # list PIL images
    for i in images:
        width, height = i.size
        if width < height:
            continue
        else:
            # print(i.size)  # tuple : (width, height)
            image = i.resize((4096, int(height / width * 4096)))
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2)  # Sharpness
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0)  # black and white
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2)  # Contrast
            image = np.asarray(image)  # array
            image = image.astype(np.uint8)
            return image


def add_value(dict_obj, key, value):
    ''' Adds a key-value pair to the dictionary.
        If the key already exists in the dictionary,
        it will associate multiple values with that
        key instead of overwritting its value'''
    if key not in dict_obj:
        dict_obj[key] = value
    elif isinstance(dict_obj[key], list):
        dict_obj[key].append(value)
    else:
        dict_obj[key] = [dict_obj[key], value]


def closest_node2(node, nodes):
    y0_label = node[0]
    x0_label = node[1]
    nodes = [node for index, node in enumerate(nodes) if nodes[index][0] > y0_label and nodes[index][1] > x0_label]
    if not nodes:
        return None
    else:
        nodes_arr = np.asarray(nodes)
        deltas = nodes_arr - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        index = np.argmin(dist_2)
        return nodes[index]


def sort_lines_and_paragraphs(df):
    if df.max()['par_num'] > 1:
        for unique_par in df['par_num'].unique():
            df_temp = df[df.par_num > unique_par]
            if unique_par == 1:

                df_temp.line_num += (df[df['par_num'] == unique_par]).max()['line_num']
                # df[df.par_num == unique_par] =
                df.update(df_temp)
            else:
                df_temp.line_num += 1
                df.update(df_temp)
    return df


def has_numbers(inputstring):
    return any(char.isdigit() for char in inputstring)


def project_name_correction(df, line_qt, image, langeng, langrus):
    """
    Correct text block step by step
    First 2-3 lines must be on english
    Last 2-4 lines must be rus+eng or simple rus

    Parameters
    ----------
    df : The initial dataframe ocr-ed on english
    line_qt : Total quantity of lines
    image : image of this block of text
    langeng : version of eng language to use
    langrus : version of rus language to use

    Returns
    -------
    text_rus_name : String of a text after correction
    text_eng_name : String of a text after correction

    """
    if isinstance(image, list):
        image = image[0]
    else:
        pass
    df = sort_lines_and_paragraphs(df)
    a = 0
    if line_qt >= 8:
        df = df.drop(df[df.line_num >= 8].index)
        line_qt = df.max()['line_num']
    if line_qt == 6 or line_qt == 7:  # Если 6 или 7 линий всего


        #   ГЕНЕРИРУЕМ БОКС ДЛЯ РУС ТЕКСТА
        for i in list(df.index.values):
            if df['line_num'].iloc[a] == 4 and df['word_num'].iloc[a] == 1:  # Ищем первое слово в 4ой строке
                # print(df['text'].iloc[a - 1])
                y0 = df['top'].iloc[a - 1] + df['height'].iloc[a - 1]  # y0 - координата y под текстом 3 линии
                h0 = df['top'].iloc[a] - y0
                if h0 < 0:
                    y0 = df['top'].iloc[a]
                else:
                    y0 = y0 + h0 / 2  # y0 - координата чуть выше 4ой линии
                # print(y0)
                break
            a += 1
        y1 = df['top'].iloc[-1] + df['height'].iloc[-1] + 10
        heigth = y1 - y0

        box = np.array([int(y0), 0, int(y1), int(image.shape[1])])
        img = create_image_boxed(image, box)

        df_rus = image_to_df(img, section="text_rus", lang=langrus, config=r'--oem 3 --psm 6 -c page_separator=""')
        df_rus = sort_lines_and_paragraphs(df_rus)
        dict_bad_conf_index_to_images = {}

        df_rus['low'] = df_rus['top'] + df_rus['height']
        df_rus['right'] = df_rus['left'] + df_rus['width']

        line1_top_border = df_rus[df_rus['line_num'] == 1].max()[
            'top']  # df вытаскиваем те где лн == 1, выбираем самый большой top
        line1_low_border = df_rus[df_rus['line_num'] == 1].max()[
            'low']  # df вытаскиваем те где лн == 1, выбираем самый большой top + height (low)
        line2_top_border = df_rus[df_rus['line_num'] == 2].min()[
            'top']  # df вытаскиваем те где лн == 2, выбираем самый маленький top
        line2_low_border = df_rus[df_rus['line_num'] == 2].max()[
            'low']  # df вытаскиваем те где лн == 2, выбираем самый большой low
        line3_top_border = df_rus[df_rus['line_num'] == 3].min()[
            'top']  # df вытаскиваем те где лн == 3, выбираем самый маленький top
        line3_low_border = df_rus[df_rus['line_num'] == 3].max()[
            'low']  # df вытаскиваем те где лн == 3, выбираем самый маленький low

        last_line1_num = df_rus[df_rus['line_num'] == 1].max()['word_num']
        last_line2_num = df_rus[df_rus['line_num'] == 2].max()['word_num']
        last_line3_num = df_rus[df_rus['line_num'] == 3].max()['word_num']

        if df_rus.max()['line_num'] == 4:
            line4_top_border = df_rus[df_rus['line_num'] == 4].min()['top']
            line4_low_border = df_rus[df_rus['line_num'] == 4].min()['low']

            last_line4_num = df_rus[df_rus['line_num'] == 4].max()['word_num']

        #   КОРРЕКЦИЯ РУС ТЕКСТА
        for word in df_rus['text_rus']:
            # if df_rus.loc[df_rus.text_rus == word, 'conf'].values[0] < 10:

            if df_rus.loc[df_rus.text_rus == word, 'conf'].values[0] < 93 or has_numbers(word):
                index_row = df_rus.index[df_rus['text_rus'] == word].tolist()[0]
                word_num_inline = df_rus.loc[df_rus.text_rus == word, "word_num"][index_row]
                line_num = df_rus["line_num"][index_row]
                words_in_this_line_qt = len(df_rus[df_rus['line_num'] == line_num])
                if re.search(r"^[ЁёА-я/\\\.-]+$+", df_rus['text_rus'][index_row]) is not None and df_rus['conf'][
                    index_row] > 91:  # Ошибка в самом языке rus(потому что слово состоит только из рус букв)
                    continue
                else:
                    #   ПОЛУЧАЕМ БОКСЫ
                    #   СЛОВА КОТОРЫЕ НАДО ИСПРАВИТЬ В ПЕРВОЙ ЛИНИИ
                    if df_rus['line_num'][index_row] == 1:
                        if line1_top_border - 10 < 0:
                            y0 = 0  # Первая строка
                        else:
                            y0 = line1_top_border - 10
                        if df_rus.loc[df_rus.text_rus == word, "word_num"].iloc[0] == 1:  # Слово первое
                            w0 = (df_rus.loc[index_row, 'right'] + df_rus.loc[
                                df_rus[df_rus['line_num'] == line_num].index[
                                    df_rus[df_rus['line_num'] == line_num]['word_num'] == word_num_inline + 1].tolist()[
                                    0], 'left']) / 2
                            if df_rus.loc[df_rus.text_rus == word, "left"].values[0] < 10:
                                x0 = 0
                            else:
                                x0 = df_rus.loc[df_rus.text_rus == word, "left"].values[0] - 10
                        elif df_rus.loc[df_rus.text_rus == word, "word_num"].iloc[
                            0] == last_line1_num:  # Слово последнее
                            x0 = (df_rus.loc[df_rus[df_rus['line_num'] == line_num].index[
                                                 df_rus[df_rus['line_num'] == line_num][
                                                     'word_num'] == word_num_inline - 1].tolist()[0], 'right'] +
                                  df_rus.loc[index_row, 'left']) / 2
                            if df_rus.loc[df_rus.text_rus == word, "left"].values[0] + \
                                    df_rus.loc[df_rus.text_rus == word, "width"].values[0] + 10 > img.shape[
                                1]:  # Если близко к краю справа
                                w0 = img.shape[1]
                            else:
                                w0 = df_rus.loc[df_rus.text_rus == word, "left"].iloc[0] + \
                                     df_rus.loc[df_rus.text_rus == word, "width"].iloc[0] + 10
                        else:  # Посередине
                            table_with_line = df_rus[df_rus['line_num'] == line_num]
                            index_of_previous = \
                                table_with_line.index[table_with_line['word_num'] == word_num_inline - 1].tolist()[0]
                            index_of_next = \
                                table_with_line.index[table_with_line['word_num'] == word_num_inline + 1].tolist()[0]
                            x0 = (df_rus.loc[index_of_previous, 'right'] + df_rus.loc[index_row, 'left']) / 2
                            # x0 = (df_rus.loc[df_rus[df_rus['line_num'] == line_num].word_num == word_num_inline - 1, "left"] +
                            #       df_rus.loc[df_rus.word_num == word_num_inline - 1, "width"] +
                            #       df_rus['left'][index_row]) / 2
                            w0 = (df_rus.loc[index_row, 'right'] + df_rus.loc[index_of_next, 'left']) / 2
                            # w0 = (df_rus.loc[df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"] + 1, "left"] + df_rus.loc[df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"], "width"] - \
                            #      df_rus.loc[ df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"], "left"]) / 2

                        y1 = (line2_top_border - df_rus.loc[df_rus.text_rus == word, "top"].values[
                            0] -  # между y_line1_low  и y_line2_top
                              df_rus.loc[df_rus.text_rus == word, "height"].values[0]) / 2 + \
                             df_rus.loc[df_rus.text_rus == word, "top"].values[0] + \
                             df_rus.loc[df_rus.text_rus == word, "height"].values[0]

                    # СЛОВА КОТОРЫЕ НУЖНО ИСПРАВИТЬ ВО 2-ой линии
                    elif df_rus['line_num'][index_row] == 2:
                        #   Игрики Y берём у первой и 3ей линии
                        if line1_low_border > line2_top_border:
                            y0 = line2_top_border - 3
                        else:
                            y0 = line1_low_border
                        if line2_low_border > line3_top_border:
                            y1 = line3_top_border
                        else:
                            y1 = line2_low_border + 3

                        # Слово первое
                        if word_num_inline == 1 and words_in_this_line_qt > 1:
                            if df_rus["left"][index_row] < 10:
                                x0 = 0
                            else:
                                x0 = df_rus["left"][index_row] - 10
                            w0 = (df_rus["left"][df_rus[df_rus['line_num'] == 2].index[
                                df_rus[df_rus['line_num'] == 2]['word_num'] == df_rus['word_num'][
                                    index_row + 1]].tolist()[0]] + df_rus['right'][
                                      index_row]) / 2  # Берём левую координату следующего слова


                        # Слово последнее
                        elif word_num_inline == last_line2_num and words_in_this_line_qt > 1:
                            x0 = (df_rus["right"][df_rus[df_rus['line_num'] == 2].index[
                                df_rus[df_rus['line_num'] == 2]['word_num'] == df_rus['word_num'][
                                    index_row - 1]].tolist()[0]] + df_rus['left'][
                                      index_row]) / 2  # Берём правую координату предыдущего
                            if df_rus["right"][index_row] + 10 > img.shape[1]:  # Край справа
                                w0 = img.shape[1]
                            else:
                                w0 = df_rus["right"][index_row] + 10

                        # Слово посередине
                        else:
                            if words_in_this_line_qt == 1:  # Слово единственное (Используем только его коорды)
                                if df_rus['left'][index_row] - 10 > 0:
                                    x0 = df_rus['left'][index_row] - 10
                                else:
                                    x0 = 0
                                if df_rus['right'][index_row] + 10 < img.shape[1]:
                                    w0 = df_rus['right'][index_row] + 10
                                else:
                                    w0 = img.shape[1]
                            else:  # Слово не единственное в строке (Используем соседей)
                                df_cut = df_rus[df_rus['line_num'] == line_num]
                                x0 = df_cut["right"][df_cut.index[
                                    df_cut['word_num'] == df_cut["word_num"][
                                        index_row] - 1].tolist()[
                                    0]] + 1
                                w0 = df_cut["left"][df_cut.index[
                                    df_cut['word_num'] == df_cut["word_num"][
                                        index_row] + 1].tolist()[
                                    0]] - 1

                    #   СЛОВА КОТОРЫЕ НАДО ИСПРАВИТЬ В 3-ей ЛИНИИ
                    elif df_rus['line_num'][index_row] == 3:
                        if line2_low_border > line3_top_border:
                            y0 = line3_top_border - 2
                        else:
                            y0 = line2_low_border
                        if df_rus.max()['line_num'] == 3:  # Считается последней линией
                            if df_rus["low"][index_row] + 10 >= img.shape[0]:
                                y1 = img.shape[0]
                            else:
                                y1 = df_rus["low"][index_row] + 10

                        elif df_rus.max()['line_num'] == 4:  # Есть 4ая линия
                            y1 = line4_top_border - 1

                        if word_num_inline == 1 and words_in_this_line_qt > 1:  # Первое слово в строке (Не единственное!)
                            if df_rus["left"][index_row] < 10:
                                x0 = 0
                            else:
                                x0 = df_rus["left"][index_row] - 10

                            # index_of_previous_word  = df_rus[df_rus['line_num'] == 3].index[df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][index_row + 1]].tolist()[0]
                            w0 = (df_rus["left"][df_rus[df_rus['line_num'] == 3].index[
                                df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][
                                    index_row + 1]].tolist()[0]] + df_rus['right'][
                                      index_row]) / 2  # Берём левую координату следующего слова
                        elif word_num_inline == last_line3_num and words_in_this_line_qt > 1:  # Последнее  (Не единственное)
                            x0 = (df_rus["right"][df_rus[df_rus['line_num'] == 3].index[
                                df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][
                                    index_row - 1]].tolist()[0]] + df_rus['left'][
                                      index_row]) / 2  # Берём правую координату предыдущего
                            if df_rus["right"][index_row] + 10 > img.shape[1]:  # Край справа
                                w0 = img.shape[1]
                            else:
                                w0 = df_rus["right"][index_row] + 10
                        else:  # Посередине
                            if words_in_this_line_qt == 1:  # Слово единственное (Используем только его коорды)
                                if df_rus['left'][index_row] - 10 > 0:
                                    x0 = df_rus['left'][index_row] - 10
                                else:
                                    x0 = 0
                                if df_rus['right'][index_row] + 10 < img.shape[1]:
                                    w0 = df_rus['right'][index_row] + 10
                                else:
                                    w0 = img.shape[1]
                            else:  # Слово не единственное в строке (Используем соседей)
                                df_cut = df_rus[df_rus['line_num'] == line_num]
                                x0 = df_cut["right"][df_cut.index[
                                    df_cut['word_num'] == df_cut["word_num"][index_row] - 1].tolist()[0]] + 1
                                w0 = df_cut["left"][df_cut.index[
                                    df_cut['word_num'] == df_cut["word_num"][index_row] + 1].tolist()[0]] - 1


                    elif df_rus['line_num'][index_row] == 4:  # Слово в 4ой линии
                        y0 = line3_low_border
                        y1 = img.shape[0]
                        if words_in_this_line_qt == 1:  # Единственное в строке(используем только его коорды)
                            if df_rus['left'][index_row] - 10 > 0:
                                x0 = df_rus['left'][index_row] - 10
                            else:
                                x0 = 0
                            if df_rus['right'][index_row] + 10 < img.shape[1]:
                                w0 = df_rus['right'][index_row] + 10
                            else:
                                w0 = img.shape[1]
                        elif word_num_inline == 1 and words_in_this_line_qt > 1:  # не единственное и первое
                            if df_rus["left"][index_row] < 10:
                                x0 = 0
                            else:
                                x0 = df_rus["left"][index_row] - 10

                            # index_of_previous_word  = df_rus[df_rus['line_num'] == 3].index[df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][index_row + 1]].tolist()[0]
                            w0 = (df_rus["left"][df_rus[df_rus['line_num'] == 3].index[
                                df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][
                                    index_row + 1]].tolist()[0]] + df_rus['right'][
                                      index_row]) / 2  # Берём левую координату следующего слова

                        elif word_num_inline == last_line4_num:  # Последнее
                            x0 = (df_rus["right"][df_rus[df_rus['line_num'] == 3].index[
                                df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][
                                    index_row - 1]].tolist()[0]] + df_rus['left'][
                                      index_row]) / 2  # Берём правую координату предыдущего
                            if df_rus["right"][index_row] + 10 > img.shape[1]:  # Край справа
                                w0 = img.shape[1]
                            else:
                                w0 = df_rus["right"][index_row] + 10
                        else:  # Посередине
                            df_cut = df_rus[df_rus['line_num'] == line_num]
                            x0 = df_cut["right"][df_cut.index[
                                df_cut['word_num'] == df_cut["word_num"][index_row] - 1].tolist()[
                                0]] + 1
                            w0 = df_cut["left"][df_cut.index[
                                df_cut['word_num'] == df_cut["word_num"][index_row] + 1].tolist()[
                                0]] - 1

                    try:
                        box = np.array([int(y0), int(x0), int(y1), int(w0)])
                        im1 = create_image_boxed(img, box)
                        text_bad_conf_eng = image_to_df(im1, 'text', lang='eng1.9to1.9_5',
                                                        config=r'--oem 3 --psm 8 -c page_separator=""')
                        text_bad_conf_rus = image_to_df(im1, 'text', lang='rus1.7to1.9_4',
                                                        config=r'--oem 3 --psm 8 -c page_separator=""')
                        if len(text_bad_conf_eng.index) > 1:
                            l1 = [text_bad_conf_eng.conf.iloc[i] for i in range(len(text_bad_conf_eng))]
                            text_bad_conf_eng = pd.DataFrame({'conf': [sum(l1) / len(l1)], 'text': [' '.join([text_bad_conf_eng.text.iloc[i] for i in range(len(text_bad_conf_eng))])]})
                        if len(text_bad_conf_rus.index) > 1:
                            l1 = [text_bad_conf_rus.conf.iloc[i] for i in range(len(text_bad_conf_rus))]
                            text_bad_conf_rus = pd.DataFrame({'conf': [sum(l1) / len(l1)], 'text': [' '.join([text_bad_conf_rus.text.iloc[i] for i in range(len(text_bad_conf_rus))])]})
                    except:
                        continue
                    try:
                        # Сперва определим индекс того же слова в df.
                        # Условия :
                        # 1) line num = 3 + line_num или 2 + line_num для 4/5 строк.
                        # 2) word_num в df == word_num_inline
                        df_cut = df[df['word_num'] == word_num_inline]
                        index_same_word = df_cut.text[df_cut['line_num'] == line_num + 3].index.tolist()[0]
                        if has_numbers(df.text[index_same_word]) and has_numbers(df_rus.text_rus[index_row]):
                            # Если цифры в строке этого слова/слов:
                            # Проверяем каждый случай, с преимуществом англ текста
                            if df.conf[index_same_word] >= df_rus.conf[index_row] - 5:
                                if text_bad_conf_eng.conf.iloc[0] > text_bad_conf_rus.conf.iloc[0]:
                                    if df.conf[index_same_word] > text_bad_conf_eng.conf.iloc[0]:
                                        df_rus.text_rus[index_row] = df.text[index_same_word]
                                    else:
                                        df_rus.text_rus[index_row] = text_bad_conf_eng.text.iloc[0]
                                else:
                                    if df.conf[index_same_word] > text_bad_conf_rus.conf.iloc[0]:
                                        df_rus.text_rus[index_row] = df.text[index_same_word]
                                    else:
                                        df_rus.text_rus[index_row] = text_bad_conf_rus.text.iloc[0]
                            else:
                                if text_bad_conf_eng.conf.iloc[0] > text_bad_conf_rus.conf.iloc[0]:
                                    if text_bad_conf_eng.conf.iloc[0] > df_rus.conf[index_row]:
                                        df_rus.text_rus[index_row] = text_bad_conf_eng.text.iloc[0]
                                    else:
                                        pass
                                else:
                                    if df_rus.conf[index_row] > text_bad_conf_rus.conf.iloc[0]:
                                        pass
                                    else:
                                        df_rus.text_rus[index_row] = text_bad_conf_rus.text.iloc[0]

                                        #   ЕСЛИ df И df_rus оба имеют цифры чекаем у какого из eng,
                            # то есть df или text_bad_conf_eng выше уверенность. Выбираем высшее значение
                            # if df.conf[index_same_word] >= text_bad_conf_eng.conf.iloc[0]:
                            #     df_rus.text_rus[index_row] = df.text[index_same_word]
                            # else:
                            #     df_rus.text_rus[index_row] = text_bad_conf_eng.text.iloc[0]
                        elif text_bad_conf_eng['conf'].iloc[0] > text_bad_conf_rus['conf'].iloc[0]:
                            if df_rus['conf'][index_row] < text_bad_conf_eng['conf'].iloc[0] and not (
                                    text_bad_conf_eng['text'].iloc[0].replace(' ', '').isalpha()):
                                df_rus['text_rus'][index_row] = text_bad_conf_eng['text'].iloc[0]
                        else:
                            if df_rus['conf'][index_row] < text_bad_conf_rus['conf'].iloc[0]:
                                df_rus['text_rus'][index_row] = text_bad_conf_rus['text'].iloc[0]
                    except IndexError:  #Заменить на слово из df(eng)
                        try:
                            df_cut_2nd_part = df.drop(df[df.line_num <= 3].index)   #   Парт рус текста в df
                            index_same_word = df_cut_2nd_part.text[df_cut_2nd_part['line_num'] == line_num + 3].index.tolist()[0]   #   Индекс
                            if df_rus['conf'][index_row] < df_cut_2nd_part['conf'][index_same_word] and \
                                    df_cut_2nd_part.word_num[index_same_word] == df_rus.word_num[index_row]:
                                df_rus['text_rus'][index_row] = df_cut_2nd_part['text'][index_same_word]
                        except:
                            pass


    elif line_qt == 4 or line_qt == 5:  # ЕСЛИ 4 ИЛИ 5 ЛИНИЙ ВСЕГО
        if df.max()['par_num'] == 2:
            df_temp = df[df.par_num == 2]
            df_temp.line_num += 2
            df[df.par_num == 2] = df_temp

        for i in list(df.index.values):
            if df['line_num'].iloc[a] == 3 and df['word_num'].iloc[a] == 1:  # Ищем первое слово в 3ей строке
                # print(df['text'].iloc[a - 1])
                y0 = df['top'].iloc[a - 1] + df['height'].iloc[a - 1]  # y0 - координата y под текстом 2 линии
                h0 = df['top'].iloc[a] - y0
                if h0 < 0:
                    y0 = df['top'].iloc[a]
                else:
                    y0 = y0 + h0 / 2  # y0 - координата чуть выше 3ей линии
                # print(y0)
                break
            a += 1
        y1 = df['top'].iloc[-1] + df['height'].iloc[-1] + 10
        heigth = y1 - y0

        box = np.array([int(y0), 0, int(y1), int(dict_images_stamp[id].shape[1])])
        img = create_image_boxed(dict_images_stamp[id], box)

        df_rus = image_to_df(img, section="text_rus", lang=langrus,
                             config=r'--oem 3 --psm 6 -c page_separator=""')

        dict_bad_conf_index_to_images = {}

        df_rus['low'] = df_rus['top'] + df_rus['height']
        df_rus['right'] = df_rus['left'] + df_rus['width']

        line1_top_border = df_rus[df_rus['line_num'] == 1].max()[
            'top']  # df вытаскиваем те где лн == 1, выбираем самый большой top
        line1_low_border = df_rus[df_rus['line_num'] == 1].max()[
            'low']  # df вытаскиваем те где лн == 1, выбираем самый большой top + height (low)
        line2_top_border = df_rus[df_rus['line_num'] == 2].min()[
            'top']  # df вытаскиваем те где лн == 2, выбираем самый маленький top
        line2_low_border = df_rus[df_rus['line_num'] == 2].max()[
            'low']  # df вытаскиваем те где лн == 2, выбираем самый большой low

        last_line1_num = df_rus[df_rus['line_num'] == 1].max()['word_num']
        last_line2_num = df_rus[df_rus['line_num'] == 2].max()['word_num']

        if df_rus.max()['line_num'] == 4:
            line4_top_border = df_rus[df_rus['line_num'] == 4].min()['top']
            line4_low_border = df_rus[df_rus['line_num'] == 4].min()['low']
            line3_top_border = df_rus[df_rus['line_num'] == 3].min()[
                'top']  # df вытаскиваем те где лн == 3, выбираем самый маленький top
            line3_low_border = df_rus[df_rus['line_num'] == 3].max()[
                'low']  # df вытаскиваем те где лн == 3, выбираем самый маленький low

            last_line3_num = df_rus[df_rus['line_num'] == 3].max()['word_num']
            last_line4_num = df_rus[df_rus['line_num'] == 4].max()['word_num']

        elif df_rus.max()['line_num'] == 3:
            line3_top_border = df_rus[df_rus['line_num'] == 3].min()[
                'top']  # df вытаскиваем те где лн == 3, выбираем самый маленький top
            line3_low_border = df_rus[df_rus['line_num'] == 3].max()[
                'low']  # df вытаскиваем те где лн == 3, выбираем самый маленький low

            last_line3_num = df_rus[df_rus['line_num'] == 3].max()['word_num']

        for word in df_rus['text_rus']:
            if df_rus.loc[df_rus.text_rus == word, 'conf'].values[0] < 91 or has_numbers(word):
                index_row = df_rus.index[df_rus['text_rus'] == word].tolist()[0]
                word_num_inline = df_rus.loc[df_rus.text_rus == word, "word_num"][index_row]
                line_num = df_rus["line_num"][index_row]
                words_in_this_line_qt = len(df_rus[df_rus['line_num'] == line_num])  # Кол-во слов
                if re.search(r"^[ЁёА-я/\\\.-]+$+", df_rus['text_rus'][index_row]) is not None:
                    continue
                else:

                    #   ПОЛУЧАЕМ БОКСЫ
                    #   СЛОВА КОТОРЫЕ НАДО ИСПРАВИТЬ В ПЕРВОЙ ЛИНИИ
                    if df_rus['line_num'][index_row] == 1:
                        if line1_top_border - 10 < 0:
                            y0 = 0  # Первая строка
                        else:
                            y0 = line1_top_border - 10
                        if df_rus.loc[df_rus.text_rus == word, "word_num"].iloc[0] == 1:  # Слово первое
                            w0 = (df_rus.loc[index_row, 'right'] + df_rus.loc[
                                df_rus[df_rus['line_num'] == line_num].index[
                                    df_rus[df_rus['line_num'] == line_num]['word_num'] == word_num_inline + 1].tolist()[
                                    0], 'left']) / 2
                            if df_rus.loc[df_rus.text_rus == word, "left"].values[0] < 10:
                                x0 = 0
                            else:
                                x0 = df_rus.loc[df_rus.text_rus == word, "left"].values[0] - 10
                        elif df_rus.loc[df_rus.text_rus == word, "word_num"].iloc[
                            0] == last_line1_num:  # Слово последнее
                            x0 = (df_rus.loc[df_rus[df_rus['line_num'] == line_num].index[
                                                 df_rus[df_rus['line_num'] == line_num][
                                                     'word_num'] == word_num_inline - 1].tolist()[0], 'right'] +
                                  df_rus.loc[index_row, 'left']) / 2
                            if df_rus.loc[df_rus.text_rus == word, "left"].values[0] + \
                                    df_rus.loc[df_rus.text_rus == word, "width"].values[0] + 10 > img.shape[
                                1]:  # Если близко к краю справа
                                w0 = img.shape[1]
                            else:
                                w0 = df_rus.loc[df_rus.text_rus == word, "left"].iloc[0] + \
                                     df_rus.loc[df_rus.text_rus == word, "width"].iloc[0] + 10
                        else:  # Посередине
                            table_with_line = df_rus[df_rus['line_num'] == line_num]
                            index_of_previous = \
                                table_with_line.index[table_with_line['word_num'] == word_num_inline - 1].tolist()[0]
                            index_of_next = \
                                table_with_line.index[table_with_line['word_num'] == word_num_inline + 1].tolist()[0]
                            x0 = (df_rus.loc[index_of_previous, 'right'] + df_rus.loc[index_row, 'left']) / 2
                            # x0 = (df_rus.loc[df_rus[df_rus['line_num'] == line_num].word_num == word_num_inline - 1, "left"] +
                            #       df_rus.loc[df_rus.word_num == word_num_inline - 1, "width"] +
                            #       df_rus['left'][index_row]) / 2
                            w0 = (df_rus.loc[index_row, 'right'] + df_rus.loc[index_of_next, 'left']) / 2
                            # w0 = (df_rus.loc[df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"] + 1, "left"] + df_rus.loc[df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"], "width"] - \
                            #      df_rus.loc[ df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"], "left"]) / 2

                        y1 = (line2_top_border - df_rus.loc[df_rus.text_rus == word, "top"].values[
                            0] -  # между y_line1_low  и y_line2_top
                              df_rus.loc[df_rus.text_rus == word, "height"].values[0]) / 2 + \
                             df_rus.loc[df_rus.text_rus == word, "top"].values[0] + \
                             df_rus.loc[df_rus.text_rus == word, "height"].values[0]
                    #   СЛОВА КОТОРЫЕ НАДО ИСПРАВИТЬ В 3-ей ЛИНИИ
                    elif df_rus['line_num'][index_row] == 3:
                        y0 = line2_low_border
                        if df_rus.max()['line_num'] == 3:  # Считается последней линией
                            if word_num_inline == 1 and words_in_this_line_qt > 1:  # Первое слово в строке
                                if df_rus["left"][index_row] < 10:
                                    x0 = 0
                                else:
                                    x0 = df_rus["left"][index_row] - 10

                                # index_of_previous_word  = df_rus[df_rus['line_num'] == 3].index[df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][index_row + 1]].tolist()[0]
                                w0 = (df_rus["left"][df_rus[df_rus['line_num'] == 3].index[
                                    df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][
                                        index_row + 1]].tolist()[0]] + df_rus['right'][
                                          index_row]) / 2  # Берём левую координату следующего слова
                            elif word_num_inline == last_line3_num and words_in_this_line_qt > 1:  # Последнее
                                x0 = (df_rus["right"][df_rus[df_rus['line_num'] == 3].index[
                                    df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][
                                        index_row - 1]].tolist()[0]] + df_rus['left'][
                                          index_row]) / 2  # Берём правую координату предыдущего
                                if df_rus["right"][index_row] + 10 > img.shape[1]:  # Край справа
                                    w0 = img.shape[1]
                                else:
                                    w0 = df_rus["right"][index_row] + 10
                            else:  # Посередине
                                if words_in_this_line_qt == 1:  # Слово единственное (Используем только его коорды)
                                    if df_rus['left'][index_row] - 10 > 0:
                                        x0 = df_rus['left'][index_row] - 10
                                    else:
                                        x0 = 0
                                    if df_rus['right'] + 10 < img.shape[1]:
                                        w0 = df_rus['right'] + 10
                                    else:
                                        w0 = img.shape[1]
                                else:  # Слово не единственное в строке (Используем соседей)
                                    df_cut = df_rus[df_rus['line_num'] == line_num]
                                    x0 = df_cut["right"][df_cut.index[
                                        df_cut['word_num'] == df_cut["word_num"][index_row] - 1].tolist()[0]] + 1
                                    w0 = df_cut["left"][df_cut.index[
                                        df_cut['word_num'] == df_cut["word_num"][index_row] + 1].tolist()[0]] - 1

                            if df_rus["low"][index_row] + 10 >= img.shape[0]:
                                y1 = img.shape[0]
                            else:
                                y1 = df_rus["low"][index_row] + 10

                        elif df_rus.max()['line_num'] == 4:  # Есть 4ая линия
                            y1 = line4_top_border - 1
                    elif df_rus['line_num'][index_row] == 4:  # Слово в 4ой линии
                        pass

                    try:
                        box = np.array([int(y0), int(x0), int(y1), int(w0)])
                        im1 = create_image_boxed(img, box)
                        text_bad_conf_eng = image_to_df(im1, 'text', lang=langeng,
                                                        config=r'--oem 3 --psm 8 -c page_separator=""')
                        text_bad_conf_rus = image_to_df(im1, 'text', lang=langrus,
                                                        config=r'--oem 3 --psm 8 -c page_separator=""')
                        dict_bad_conf_index_to_images[index_row] = im1
                    except:
                        continue
                    try:
                        # if has_numbers(str(text_bad_conf_eng['text'].iloc[0])):
                        # if has_numbers(str(df_rus['text_rus'][index_row])):
                        #     df_rus['text_rus'][index_row] = text_bad_conf_eng['text'].iloc[0]
                        # else:
                        if text_bad_conf_eng['conf'].iloc[0] > text_bad_conf_rus['conf'].iloc[0]:
                            if df_rus['conf'][index_row] < text_bad_conf_eng['conf'].iloc[0]:
                                df_rus['text'][index_row] = text_bad_conf_eng['text'].iloc[0]
                        else:
                            if df_rus['conf'][index_row] < text_bad_conf_rus['conf'].iloc[0]:
                                df_rus['text'][index_row] = text_bad_conf_rus['text'].iloc[0]
                    except IndexError:
                        try:
                            df_cut_2nd_part = df.drop(df[df.line_num <= 3].index)
                            index_cut = df_cut_2nd_part.index[df_rus['text_rus'] == word].tolist()[0]
                            if df_rus['conf'][index_row] < df_cut_2nd_part['conf'][index_cut] and \
                                    df_cut_2nd_part.word_num[index_cut] == df_rus.word_num[index_row]:
                                df_rus['text_rus'][index_row] = df['text'][index_row]
                        except:
                            pass
    else:
        if line_qt == 8:
            df = df[df.line_num <= 7]
            return project_name_correction(df, line_qt, image, langeng, langrus)
        else:
            try:
                text_full_name = image_to_string(image, lang="eng1.9to1.10_5+rus1.7to1.9_4",
                                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
            except:
                return None, None


    if line_qt >= 6:
        df = df.drop(df[df.line_num > 3].index)
    else:
        df = df.drop(df[df.line_num > 2].index)
    # df[-len(df_rus):] = df_rus
    text_eng_name = ""
    text_rus_name = ""
    if line_qt == 4 or line_qt == 5:
        pivot = 3
    else:
        pivot = 4
    for word_space_eng in df[df['line_num'] < pivot]['text']:
        if type(word_space_eng) == float:
            word_space_eng = int(word_space_eng)
        text_eng_name = text_eng_name + str(word_space_eng)
        text_eng_name = text_eng_name + " "
    # print(text_eng_name)
    for word_space_rus in df_rus['text_rus']:
        if type(word_space_rus) == float:
            word_space_rus = int(word_space_rus)
        text_rus_name = text_rus_name + str(word_space_rus)
        text_rus_name = text_rus_name + " "
    return text_eng_name, text_rus_name


def instances_to_images(image, boxes, class_ids, names, list_accepted_ids=[1, 2, 8, 10], stamp=False):
    """Crop the images by their boxes
    ВОЗВРАЩАЕТ СЛОВАРЬ КАРТИНОК В БОКСАХ + labels
    """
    if stamp:
        indices_role_input = [i for i, x in enumerate(class_ids) if x == 33]
        role_input_coords = [x for i, x in enumerate(boxes) if i in indices_role_input]  # Все корды инпутов
    n_instances = boxes.shape[0]
    dict_images = {}
    for i in range(n_instances):
        if list_accepted_ids == [True]:
            if not np.any(boxes[i]) or class_ids[i] in [1]:
                continue
        else:
            if not np.any(boxes[i]) or not class_ids[i] in list_accepted_ids:
                continue
        label = names[class_ids[i]]
        if stamp:
            if label in ["Label_by", "Label_chk", "Label_eng", "Label_supv", "Label_oper", "Label_mgr"]:
                box = boxes[i]  # Корды лэйбла
                box = closest_node2(box, role_input_coords)
                if box is None:
                    add_value(dict_images, label, {"Coordinates": None})
                else:
                    boxed_image = create_image_boxed(image, box)
                    dict_roles = {"Coordinates": boxed_image}
                    add_value(dict_images, label, dict_roles)
            elif label in ["Project_name_v", "Project_name_h"]:
                # print("It's HEREEEEEE")
                # for each y from r['rois'] compare to boxes of project_name_v or project_name_h
                box = boxes[i]  # y0,x0,h0,w0
                miny = box[0]  # y0
                minx = box[1]  # x0
                minh = box[2]  # h0
                minw = box[3]  # w0
                list_y_coords = []
                list_h_coords = []
                list_x_coords = []
                list_w_coords = []
                min_x_r_tf = False
                if label == "Project_name_v":
                    for i in range(boxes.shape[0]):
                        if boxes[i][0] > miny and class_ids[i] != 4:
                            list_h_coords.append(boxes[i][0])
                        if boxes[i][3] > minw:
                            list_w_coords.append(boxes[i][3])
                        # if boxes[i][1] < minx:
                        #     list_x_coords.append(boxes[i][1])
                        if class_ids[i] == 4:
                            box[0] = boxes[i][0]
                            box[1] = boxes[i][1]
                        elif 4 not in class_ids:
                            if class_ids[i] == 1:
                                box[0] = boxes[i][2]
                                box[1] = boxes[i][1]
                    if list_h_coords:
                        if miny + (minh - miny) / 3 * 2 >= min(list_h_coords):
                            list_h_coords.remove(min(list_h_coords))
                            minh = min(list_h_coords)
                        else:
                            minh = min(list_h_coords)
                    box[2] = minh
                    if list_w_coords:
                        minw = max(list_w_coords)
                    box[3] = minw
                elif label == "Project_name_h":
                    for i in range(boxes.shape[0]):
                        if boxes[i][2] > minh:
                            list_h_coords.append(boxes[i][2])
                        # if boxes[i][1] > minx and class_ids[i] != 3 and minw * 0.8 < boxes[i][1]:
                        #     list_w_coords.append(boxes[i][1])
                        if 3 not in class_ids:
                            if boxes[i][0] < box[0]:
                                box[0] = boxes[i][0]
                            if boxes[i][3] > minx and boxes[i][1] < minx:
                                min_x_r = boxes[i][3]
                                min_x_r_tf = True
                            elif boxes[i][3] < minx:
                                list_x_coords.append(boxes[i][3])
                        # if boxes[i][1] < minx:
                        #     list_x_coords.append(boxes[i][1])
                        if class_ids[i] == 3:
                            box[0] = boxes[i][0]
                            box[1] = boxes[i][1]
                        if class_ids[i] == 7 or class_ids[i] == 11:
                            list_w_coords.append(boxes[i][1])
                        if 3 not in class_ids and 9 not in class_ids:
                            if boxes[i][0] > miny:
                                list_y_coords.append(boxes[i][0])

                    if min_x_r_tf:
                        minx = min_x_r
                    elif list_x_coords:
                        minx = max(list_x_coords) + 10
                    if list_h_coords:
                        minh = max(list_h_coords)
                    if list_w_coords:
                        minw = sum(list_w_coords) / len(list_w_coords) - 10
                    #     minw = min(list_w_coords) - 10

                    if 3 not in class_ids:
                        box[1] = minx
                        if 9 in class_ids:
                            box[0] = boxes[np.where(class_ids == 9)[0][0]]
                        else:
                            if list_y_coords:
                                box[0] = min(list_y_coords)

                    box[2] = minh
                    if minw > box[1]:
                        box[3] = minw
                boxed_image = create_image_boxed(image, box)
                add_value(dict_images, label, boxed_image)
            elif label in ["Proj_no_v", "Proj_no_h"]:
                try:
                    if dict_images[label] in dict_images.values():
                        continue
                except:
                    box = boxes[i]  # y0,x0,h0,w0
                    miny = box[0]  # y0
                    minx = box[1]  # x0
                    minh = box[2]  # h0
                    minw = box[3]  # w0
                    if label == "Proj_no_v":
                        try:
                            try:
                                box[1] = boxes[np.where(class_ids == 8)[0][0]][1]
                            except:
                                box[1] = boxes[np.where(class_ids == 12)[0][0]][1]
                        except:
                            box[1] = minx
                        # box[0] = boxes[class_ids.index(8)][2]
                        # box[2] = boxes[class_ids.index()][]
                        try:
                            box_rev = boxes[np.where(class_ids == 18)[0][0]]
                            if miny > box_rev[2] or minh < box_rev[0]:  # REV не пересекается
                                box[3] = box_rev[3]
                            elif miny < box_rev[2] and minh > box_rev[0]:  # REV пересекается
                                box[3] = box_rev[1]
                        except:
                            print("No REV found")

                    elif label == "Proj_no_h":
                        try:
                            try:
                                box[1] = boxes[np.where(class_ids == 7)[0][0]][1]
                            except:
                                box[1] = boxes[np.where(class_ids == 11)[0][0]][1]
                        except:
                            pass
                        try:
                            box_rev = boxes[np.where(class_ids == 17)[0][0]]
                            if miny > box_rev[2] or minh < box_rev[0]:  # REV не пересекается
                                box[3] = box_rev[3]
                            elif miny < box_rev[2] and minh > box_rev[0]:  # REV пересекается
                                box[3] = box_rev[1]
                        except:
                            print("No REV found")
                    boxed_image = create_image_boxed(image, box)
                    add_value(dict_images, label, boxed_image)
            elif label in ["Dr_no_v", "Dr_no_h"]:
                try:
                    if dict_images[label] in dict_images.values():
                        continue
                except:
                    box = boxes[i]  # y0,x0,h0,w0
                    miny = box[0]  # y0
                    minx = box[1]  # x0
                    minh = box[2]  # h0
                    minw = box[3]  # w0

                    if label == "Dr_no_v":
                        try:
                            box[1] = boxes[np.where(class_ids == 12)[0][0]][1]
                        except:
                            try:
                                box[1] = boxes[np.where(class_ids == 8)[0][0]][1]
                            except:
                                pass
                        try:
                            box_rev = boxes[np.where(class_ids == 18)[0][0]]
                            if miny > box_rev[2] or minh < box_rev[0]:  # REV не пересекается
                                box[3] = box_rev[3]
                            elif miny < box_rev[2] and minh > box_rev[0]:  # REV пересекается
                                box[3] = box_rev[1]
                        except:
                            print("No REV found")

                    elif label == "Dr_no_h":
                        # Средняя x Label_Proj_no_h и Label_Dr_no_h Сдвинутая на 15 пискселей влево
                        try:
                            try:
                                box[1] = (boxes[np.where(class_ids == 11)[0][0]][1] +
                                          boxes[np.where(class_ids == 7)[0][0]][
                                              1]) / 2 - 15
                            except:
                                try:
                                    box[1] = boxes[np.where(class_ids == 11)[0][0]][1] - 15
                                except:
                                    box[1] = boxes[np.where(class_ids == 7)[0][0]][
                                                 1] - 15
                        except:
                            box[1] = minx
                        try:
                            box_rev = boxes[np.where(class_ids == 17)[0][0]]
                            if miny > box_rev[2] or minh < box_rev[0]:  # REV не пересекается
                                box[3] = box_rev[3]
                            elif miny < box_rev[2] and minh > box_rev[0]:  # REV пересекается
                                box[3] = box_rev[1]
                        except:
                            print("No REV found")

                    boxed_image = create_image_boxed(image, box)
                    add_value(dict_images, label, boxed_image)

            else:
                box = boxes[i]
                boxed_image = create_image_boxed(image, box)
                add_value(dict_images, label, boxed_image)
        else:
            box = boxes[i]
            boxed_image = create_image_boxed(image, box)
            # imshow(boxed_image)
            # plt.show()
            # list_images.append(boxed_image)
            # key = class_ids[i]
            key = label
            # dict_images[class_ids[i]] = boxed_image
            add_value(dict_images, key, boxed_image)
    return dict_images


def show_instances(dict_images):
    import skimage.io
    for id in dict_images:
        if isinstance(dict_images[id], list):
            for i in dict_images[id]:
                imshow(i)
                plt.show()
        else:
            imshow(dict_images[id])
            plt.show()


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    image_copy = np.copy(image)
    for n, c in enumerate(color):
        image_copy[:, :, n] = np.where(mask == 1,
                                       image_copy[:, :, n] *
                                       (1 - alpha) + alpha * c,
                                       image_copy[:, :, n])
    return image_copy


def image_mask_and_boxes(image, boxes, masks, ids, names, scores, crop=False):
    import cv2
    """
        take the image and results and apply the mask, box, and Label in array
    """
    colors = random_colors2(boxes.shape[0])
    # class_dict = {
    #     name:color for name, color in zip(names, colors)
    # }
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    # masked_image = image.astype(np.float32).copy()
    for i in range(n_instances):
        color = colors[i]
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]

        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        # if i == 2:
        #     cv2.imshow('masked image', image)
        #     cv2.waitKey(0)
        mask = masks[:, :, i]
        masked_image = apply_mask(image, mask, color)
        # if i == 2:
        #     cv2.imshow('masked image', masked_image)
        #     cv2.waitKey(0)
        # image = image.transpose((1, 2, 0)).astype(np.uint8).copy()
        # image = apply_mask(image, mask, color)
        image = cv2.rectangle(masked_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
    # cv2.imshow('masked image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.destroyAllWindows()
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title("")

    masked_image = image

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            # print(class_ids)
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='b', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()


def display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.5, score_threshold=0.5):
    """Display ground truth and prediction instances on the same image."""
    # Match predictions to ground truth
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    # Ground truth = green. Predictions = red
    colors = [(0, 1, 0, .8)] * len(gt_match) \
             + [(1, 0, 0, 1)] * len(pred_match)
    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    # Captions per instance show score/IoU
    captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
        pred_score[i],
        (overlaps[i, int(pred_match[i])]
         if pred_match[i] > -1 else overlaps[i].max()))
        for i in range(len(pred_match))]
    # Set title if not provided
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    # Display
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, ax=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)


def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
            [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)
    return masked_image
    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictions and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
        else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with different
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominent each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    display_table(table)
