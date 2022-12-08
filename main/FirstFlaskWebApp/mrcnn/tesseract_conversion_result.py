import IPython
import cv2
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import pytesseract as pt
import pandas
import os, sys
from tabulate import tabulate
import numpy as np
# import PySimpleGUI as sg

pt.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe '
tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
TESSDATA_PREFIX = 'C:\Program Files (x86)\Tesseract-OCR\tessdata'

# DEFAULTS
custom_config = r'--oem 3 --psm 6'
default_element_set = {'NameEng', 'NameRus', 'Code', }
image1 = 'C:\\Users\\trainee2\\Documents\\Project\\PyTests\\test1.1.png'
# image2 = 'C:\\Users\\trainee2\\Documents\\Project\\PyTests\\test1.2.png'
contrast = 2
resize_factor = 2
path = "C:\\Users\\trainee2\\Documents\\Project\\PyTests\\data/"
dirs = os.listdir(path)
lang = 'training'

path = os.getcwd()
path_to_image = path + '\\ImageToOCR.png'
# items = sorted(os.listdir(path))
# valid_name = ["ImageToOCR.png"]
# valid_extension = [".png"]
# for item in os.listdir(path):
#     extension = os.path.splitext(item)[1]
#     if extension.lower() not in valid_extension:
#         continue
#     if item.title().lower() == "imagetoocr.png":
#         image = Image.open(os.path.join(path, item))
#     else:
#         continue
image2 = Image.open(path_to_image)
image2_filepath = path_to_image


# FUNC DEFS
def drawText(text, title):
    background = Image.open('C:\\Users\\trainee2\\Documents\\Project\\PyTests\\textBackground.png')
    font = ImageFont.truetype('C:\\Users\\trainee2\\Documents\\Project\\PyTests\\Roboto-Black.ttf', size=12)
    draw_text = ImageDraw.Draw(background)
    draw_text.text((100, 100),
                   text,
                   font=font,
                   fill='#1C0606')
    background.save(title, quality=100)
    background.show(title=title)


# Image processing(used in convert_blueprint_pic_to_df
def process_image(image, contrast=1):
    # image = Image.open(filepath)
    image = image.resize(
        (874, 614))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    image.show()
    return image

def process_image_cv2():
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread('ImageToOCR2.png')
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1)

    # Dilate to merge into a single contour
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 30))
    dilate = cv2.dilate(thresh, vertical_kernel, iterations=3)

    # Find contours, sort for largest contour and extract ROI
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:-1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 4)
        ROI = original[y:y + h, x:x + w]
        break
    cv2.imwrite('ImageToOCR2.png', ROI)
    image = Image.open(os.getcwd() + '\\ImageToOCR.png')
    image = image.resize(
        (int(image2_w / proportion), int(image2_h / proportion)))
    image.show()
    return image


def process_image_nosize(filepath, contrast=1):
    image = Image.open(filepath)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    # image.show()
    return image


def resize(resize_factor):
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            imResize = im.resize((im.size[0] * resize_factor, im.size[1] * resize_factor), Image.ANTIALIAS)
            enhancer = ImageEnhance.Contrast(imResize)
            imResize = enhancer.enhance(contrast)
            imResize.save(f + ' resized.png', 'PNG', quality=90)


def add_2text_df_to_list(df, index, list_text, column='text'):
    list_text.append(df[column].iloc[index])
    return list_text


def is_family(dataframe, index, column='text'):
    average_height = abs(dataframe['height'].mean())
    average_width = abs(dataframe['width'].mean())
    distv = abs(dataframe['top'].iloc[index + 1] - dataframe.loc[index, 'top'] - dataframe.loc[index, 'height'])
    distg = abs(dataframe['left'].iloc[index + 1] - dataframe['left'].iloc[index] - dataframe['width'].iloc[index])
    block_numi = dataframe['block_num'].iloc[index]
    block_numi1 = dataframe['block_num'].iloc[index + 1]
    heighti = dataframe['height'].iloc[index]
    heighti1 = dataframe['height'].iloc[index + 1]
    widthi = dataframe['width'].iloc[index]
    widthi1 = dataframe['width'].iloc[index + 1]
    word_numi = dataframe['word_num'].iloc[index]
    word_numi1 = dataframe['word_num'].iloc[+ 1]
    print('distv = ' + str(distv) + ' //distg is ' + str(distg) + ' //height is ' + str(heighti) + ' //width is ' + str(
        widthi) + ' //word is ' + dataframe['text'].iloc[index] + ' //avg_h is ' + str(average_height) + ' //avg_w is ' + str(average_width))
    if dataframe['line_num'].iloc[index + 1] == dataframe['line_num'].iloc[
        index]:  # След. слово в той же строке (проверяем гориз расст между словами: True если меньше 14)
        if distg >= widthi:
            return False
        else:
            if block_numi != block_numi1:
                return False
            return True
    else:  # След слово в другой строке (проверяем вертикальное рааст между словами)
        if distv >= heighti:
            return False
        else:
            if block_numi != block_numi1:
                return False
            return True


def classification(df):
    family = True
    list_text = []
    i = 0
    g = 0
    df['class' + str(g)] = 'foo'
    while True:
        print(df.loc[i, 'text'])
        texti = df.loc[i, 'text']
        if i == len(df) - 1:  # слово последнее в файле
            df['class' + str(g + 1)] = 'foo'
            df.loc[i, 'class' + str(g + 1)] = texti  # это слово добавляем в нынешнюю колонну
            break

        if family == False:
            g += 1
            df['class' + str(g)] = 'foo'

        if df['word_num'].iloc[i] == 1:  # это слово первое в строке
            if df['word_num'].iloc[i + 1] == 1:  # это слово первое и последнее в строке
                df.loc[i, 'class' + str(g)] = texti  # это слово добавляем в нынешнюю колонну
                family = is_family(df, i, 'text')
                i += 1
            elif df['word_num'].iloc[i + 1] != 1:  # это слово первое и не единственное в строке
                df.loc[i, 'class' + str(g)] = texti  # это слово добавляем в нынешнюю колонну
                family = is_family(df, i, 'text')
                i += 1
        elif df['word_num'].iloc[i] != 1:  # это слово не первое в строке
            family = is_family(df, i, 'text')
            if df['word_num'].iloc[i + 1] != 1:  # и не последнее
                df.loc[i, 'class' + str(g)] = texti  # это слово добавляем в нынешнюю колонну
                i += 1
            else:  # и последнее
                family = is_family(df, i, 'text')
                df.loc[i, 'class' + str(g)] = texti  # это слово добавляем в нынешнюю колонну
                i += 1


# Converts data from an image to data FRAME...
def convert_blueprint_pic_to_df(filepath, isProcessCv2, contrast=1):
    if not isProcessCv2:
        image = process_image(filepath, contrast)
    else:
        image = process_image_cv2()
    dataframe = pt.image_to_data(image, lang='training', output_type='data.frame')
    dataframe = dataframe[dataframe['conf'] != -1]
    dataframe['text'] = dataframe['text'].apply(lambda x: x.strip())
    dataframe = dataframe[dataframe['text'] != ""]
    dataframe = dataframe[dataframe['text'] != " "]
    result = []
    for number in range(len(dataframe.index)):
        result.append(number)
    dataframe['index'] = result
    dataframe = dataframe.set_index('index')
    # df['text'] = df['text'].apply(lambda x: x.upper())
    # for i in range(len(df) - 1):
    #     dist = df['top'].iloc[i + 1] - df['top'].iloc[i]
    #     if dist < 40 and df['height'].iloc[i + 1] - df['height'].iloc[i] < 10:
    #         if (df['block_num'].iloc[i + 1] == df['block_num'].iloc[i]):
    #             add_2text_df_to_list(df, i, list_text, 'text')
    #             is_merged(df, i, 'text')
    print(tabulate(dataframe, headers='keys', tablefmt='psql'))
    classification(dataframe)
    # df['Name'] = list_text
    # var = df.iloc(i, 11)
    # df.append({'names': var}) #
    # df['names']
    print(type(dataframe))
    print(len(dataframe))

    # IPython.display.display(df)
    return dataframe


def process_blueprint(
        filepath, element_set=None, resize_factor=1):
    if element_set is None:
        element_set = default_element_set
    df = convert_blueprint_pic_to_df(filepath, resize_factor)


# Work with text
# text_orig = pt.image_to_data(im, lang='kaz+rus+eng', config=custom_config, output_type=pt.Output.DATAFRAME)

# ----------------------------->Previous MAIN<-------------------------------
df = text_to_frame = convert_blueprint_pic_to_df(image2_filepath, False, contrast=contrast)
# print(tabulate(df, headers='keys', tablefmt='psql'))

# ----------------------------->Current MAIN<-------------------------------
w, h = image2.size
image2_w = w
image2_h = h
default_text = ''
# for i in range (len(df.columns)):

def createWindow():
    Column1 = [
        [
            sg.Text(text='Please compare and correct(if necessary) with correct variant', key="-IMAGENAME")
        ],
        [
            sg.Image(image2_filepath, background_color='white', key="-IMAGE-")
        ]

    ]

    Column2 = [
        [
            sg.Multiline(default_text=df['text'], key="-OUTPUT-", size=(40, int(image2_h / 10 - image2_h / 20)))
        ],
        [
            sg.RButton(button_text='Correct', key='-VERIF-')
        ]

    ]

    layoutCol = [
        [
            sg.Column(Column1),
            sg.VSeparator(color='red'),
            sg.Column(Column2)
        ]
    ]

    windowCol = sg.Window('Text compare', layoutCol, resizable=True)
    return windowCol

windowCol = createWindow()

while True:
    event, values = windowCol.read()
    windowCol.bring_to_front()
    if event is None:
        break
    if event == '-VERIF-':
        print('cool')
        break
windowCol.close()


wantedSize = 30
proportion = df['height'].mode()[0]/wantedSize
df = text_to_frame = convert_blueprint_pic_to_df(image2_filepath, True, contrast=contrast)
print(pt.image_to_string(process_image_cv2()))

w, h = image2.size
image2_w = w
image2_h = h
default_text = ''

windowColNew = createWindow()

while True:
    event, values = windowColNew.read()
    windowColNew.bring_to_front()
    if event is None:
        break
    if event == '-VERIF-':
        print('cool')
        break
windowColNew.close()

# np.savetxt(r'C:\Users\trainee2\Documents\Project\PyTests\np.txt', df.values, fmt='%s', encoding="utf-8")

# TEXT SHOWN AS IMAGES:
# drawText(text_orig, 'text_orig.png')
