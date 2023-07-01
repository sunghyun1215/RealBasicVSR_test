import os

current_path = os.getcwd()

print(current_path)


for num in range(201, 300):
    folder_path = current_path + '/' + str(num)
    os.mkdir(folder_path)
