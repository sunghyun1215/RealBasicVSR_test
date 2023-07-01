import os
import natsort
from os import path


cur_path = os.getcwd()
folder_path = cur_path + '/infrared'

folder_list = os.listdir(folder_path)

folder_list = natsort.natsorted(folder_list)


for num in range(len(folder_list)):
    target_path = folder_path + '/' + folder_list[num]
    file_list = os.listdir(target_path)
    file_list = natsort.natsorted(file_list)

    for num2 in range(len(file_list)):
        # numname = target_path + '/' +  str(num2).rjust(8,'0') + os.path.splitext(file_list[num2])[1]
        numname = target_path + '/' +  str(num2).rjust(8,'0') + '.png'
        file_name = target_path + '/' + file_list[num2]
        os.rename(file_name, numname)

