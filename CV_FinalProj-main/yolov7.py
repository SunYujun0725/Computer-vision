import cv2
import time
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict,namedtuple

from detection_error import error
cuda = False

w = "model_data/yolov7.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)
boxes = [0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0]

# 儲存偵測出的藥品名稱以及x, y座標
detection_results = []

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def detection(img):

    error_img = img
    boxes = [0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0]
    names = ['med1','med2','med3','med4','med5','med6','med7','med8','med9','med10','med11','med12','med13','med14','med15','med16','med17','med18','med19','med20']
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    outname = [i.name for i in session.get_outputs()]
    outname

    inname = [i.name for i in session.get_inputs()]
    inname

    inp = {inname[0]:im}

    outputs = session.run(outname, inp)[0]
    outputs

    ori_images = [img.copy()]

    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        #print(cls_id)
        score = round(float(score),3)
        boxes[cls_id] = boxes[cls_id] +1
        name = names[cls_id]
        color = colors[name]
        #print(score)
        name += ' '+str(score)
        cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),color,2)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
        '''
        在這個部分會知道標記框的x,y座標，預計在這邊紀錄是否有擺放錯誤
        '''
        #將偵測出來的藥品名稱, x,y座標存到detection_results的list
        detection_results.append((name, box[0], box[1], box[2], box[3]))
        #print(f"{name}: x={box[0]}, y={box[1]}")

    #判斷是否有放錯位置
    error_data = error(detection_results, boxes)
    for entry in error_data:
        name, x0, y0, x1, y1 = entry
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(error_img, (x0, y0), (x1, y1), color, 5)
        cv2.putText(error_img,name,(x0, y0 - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
    cv2.imwrite("./detection_error.jpg", error_img)
        
    img = Image.fromarray(ori_images[0])
    img.save('Result.jpg')
    return boxes, img

            
def runModel(image):
    start = time.time()
    res = detection(image)
    end = time.time()
    #print("cost:\n", end-start)
    return res
