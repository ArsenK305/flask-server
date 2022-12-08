import json
with open('C:\\Users\\kuanyshov.a\\Anaconda3\\envs\\project1\\json_data.json', encoding='utf-8') as json_file:
    data = json.load(json_file).decode()
    print(data)