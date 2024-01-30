import cv2
import sys
from yolov7 import runModel
import matplotlib.pyplot as plt

def PrintRowLine():
    print(' ', end='')
    for i in range(0, 5):
        for j in range(0, 8):
            print('-', end='')
        print(' ', end='')
    print()


def PrintColLine():
    for i in range(0, 5):
        print('|', end='')
        for j in range(0, 8):
            print(' ', end='')
    print('|')


file_path = sys.argv[1]
'''
default_box = [[0, 2, 0, 0, 0],
               [2, 0, 0, 0, 4],
               [5, 2, 1, 5, 4],
               [20, 3, 3, 2, 4]
               ]
'''
default_box = [[0, 2, 0, 0, 0],
               [2, 0, 0, 0, 4],
               [4, 2, 1, 6, 2],
               [11, 3, 5, 2, 5]
               ]


if file_path:
    image = cv2.imread(file_path)
    if image is not None:
        '''
        進行圖片矯正的部分
        '''
        box_num, img = runModel(image)#輸入yolov7模型偵測
        tmp = 0
        for i in range(0, 4):#輸出偵測的結果
            PrintRowLine()
            PrintColLine()
            for j in range(0, 5):
                if default_box[i][j] == 0:
                    print("|  None  ", end='')
                else:
                    print("|  {:02d}/{:02d} ".format(box_num[tmp], default_box[i][j]), end='')
                tmp += 1
            print('|')
            PrintColLine()
        PrintRowLine()
        plt.imshow(img)#偵測結果圖片
        plt.pause(0)
            
    else:
        print("Can't read image!")
else:
    print("File Not Exit!")
