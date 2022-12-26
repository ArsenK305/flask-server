import pytesseract as pt
import pandas as pd
from skimage.io import imread
import os

pt.pytesseract.tesseract_cmd = r'C:\\Users\\kuanyshov.a\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'
tessdata_dir_config = '--tessdata-dir "C:\\Users\\kuanyshov.a\\AppData\\Local\\Tesseract-OCR\\tessdata"'
TESSDATA_PREFIX = 'C:\\Users\\kuanyshov.a\\AppData\\Local\\Tesseract-OCR\\tessdata'
custom_config = r'--oem 3 --psm 13 -c page_separator=""'
default_element_set = {'NameEng', 'NameRus', 'Code', }
text_file_path = "C:\\Users\\kuanyshov.a\\Anaconda3\\envs\\project1\\main\\Text_result\\"

def drawText(text, title):
    with open(title, 'w') as f:
        f.write(text)


def process_image(image):
    return image


def image_to_df(image, section, lang='rus1.8+eng1.9', config=custom_config):
    dataframe = pt.image_to_data(image, lang=lang, output_type='data.frame', config=config)
    dataframe.rename(columns = {'text' : section}, inplace=True)
    # for word in dataframe[section]:
    #     if isinstance(word, float):
    #         dataframe[section]
    j = 0
    for i in dataframe.index:
        print(dataframe.iloc[j][section])
        if pd.isnull(dataframe.iloc[j][section]):
            dataframe.drop(index=i, axis=0, inplace=True)
        else:
            j += 1
    dataframe[section] = pd.to_numeric(dataframe[section], downcast='integer', errors='ignore')
    # for word in dataframe[section]:
    #     if isinstance(word, float):
    #         index = dataframe.index[dataframe[section] == word].tolist()[0]
    #         dataframe[section][index] = int(word)

    # print(dataframe.index)
    return dataframe


def image_to_string(image, lang='eng1.9+rus1.8', config=custom_config):
    text = pt.image_to_string(image, lang=lang, config=config)
    text = text.replace('/f', '')
    # print(text)
    # text_file = open(str(text_file_path) + "sample.txt", "w", encoding="utf-8")
    # text_file.write(text)
    # text_file.close()
    return text


def image_to_txt(image, fp, lang='eng1.3to_proj_no3', config=r'--oem 3 --psm 6 -c page_separator=""'):
    text = pt.image_to_string(image, lang=lang, config=config)
    print(text)
    text = text.replace('/f', '')
    text = text.replace('/n', '')
    text_file = open(fp + ".gt.txt", "w", encoding="utf-8")
    text_file.write(text)
    text_file.close()
    return text

if __name__ == '__main__':
    path = "C:\\Users\\trainee2\\Documents\\Project\\Training_tesseract\\data\\Cropped"

    image_path = "C:\\Users\\trainee2\\Documents\\Project\\Training_tesseract\\data\\Cropped\\test3.1.png"
    image = imread(image_path)
    text = image_to_string(image)
    # df = image_to_df(image, "date")
    # print(df)
    for filename in os.listdir(path):
        if filename.startswith("test"):
            filepath = path + "\\" + filename
            print(image_to_string(filepath))
        else:
            continue

    # text_file = open("sample.txt", "w")
    # n = text_file.write(text)
    # text_file.close()

    # print(dataframe)