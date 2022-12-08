import numpy as np
from time import time
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    print(dist_2)
    index = np.argmin(dist_2)
    return index, nodes[index]

def closest_node2(node, nodes):
    nodes_arr = np.asarray(nodes)
    deltas = nodes_arr - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    index = np.argmin(dist_2)
    return index, nodes[index]


some_pt = (1, 2)
nodes = [(5, 8), (-5, 3), (1, 1), (43, 15)]

t0 = time()
print(closest_node(some_pt, nodes))
t1 = time() - t0

t00 = time()
print(closest_node2(some_pt, nodes))
t20 = time() - t00
if t20 > t1:
    print("yes")
else:
    print("t20 < t1")

dict_text = {'Scale_v': '1:75\n', 'Project_name_v': 'FIELD OVERPRESSURE MITIGATION\nPIPING GENERAL ARRANGEMENT\nTENGIZ WELLSITE T-110\nПРЕДОСТВРАЩ. ИЗБЫТ. ДАВЛЕНИЯ НА РОМЫСЛЕ\nОБЩЕЕ УСТРОИСТВО ТРУБОПРОВОДОВ\nTEHTM3 CKBAXUHA T-110\n', 'Date_v': '06/10/18', 'Label_mgr': 'Empty', 'Proj_no_v': 'F-005-052-16', 'Label_by': 'KT\n', 'Label_oper': 'Empty', 'Dr_no_v': 'F-2000-L-6765', 'Label_supv': 'BD\n', 'Label_chk': 'YZ\n', 'Label_eng': 'YEI\n', 'REV_v': 'A\n', 'File_name': 'F-2000-L-6765_ReadOnly_27-10-2019_05-42-27.pdf'}
countstr = ""
for id in dict_text:
    if id != "File_name":
        print(dict_text[id])
        countstr += dict_text[id].replace("/n","")
        print(countstr)
print(countstr)
print(len(countstr))

print(type(str))

