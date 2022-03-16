#coding:utf-8


import numpy as np
import os
import glob

path = os.getcwd() #获取当前路径
res_path='result'
res_path = os.path.join(path,res_path)
os.chdir(res_path)
txt_files = glob.glob("*.txt")
txt_files.sort()
print('- - - - - - - - - - - - - - - - -')
print('')
for file in txt_files:
    f = open(res_path +'/'+ str(file), 'r')
    lines = f.readlines()
    res = []
    for line in lines:
        if line == '\n':
            break
        else:
            line = line.strip('\n')
            res.append(float(line))
    start_ind = 0
    end_ind = 10
    while len(res) >=end_ind:
        data_array = np.array(res)[start_ind:end_ind]
        data_mean = np.mean(data_array)
        data_std = np.std(data_array)
        print(file + '\033[1;32m start_ind:%d, data_mean: %f, data_std: %f \033[0m' % (start_ind, data_mean, data_std))
        print(res[start_ind: end_ind])
        print()
        start_ind = end_ind
        end_ind += 10
    if len(res)%10 !=0:
        print(file + '\033[1;32m index [%d:%d], current mean: %f  \033[0m' % (start_ind, len(res), np.mean(res[start_ind: len(res)])))
        print(res[start_ind: len(res)])
        print()


print('testing')

