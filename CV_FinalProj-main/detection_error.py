from sklearn.cluster import KMeans
import numpy as np
import math

#ground truth
gt_box = [0, 2, 0, 0, 0, 2, 0, 0, 0, 4,
        4, 2, 1, 6, 2, 11, 3, 5, 2, 5]

# 模型偵測錯藥品名稱
error_data = []

#判斷是否有偵測錯誤/放錯
def error(detection_results, boxes):
    # boxes : model偵測出來的數量
    # 按照藥品名稱排序
    sorted_results = sorted(detection_results)
    '''
    print(boxes)
    for result in sorted_results:
        print(result)
    '''
    # 存放扣掉第一階段偵測錯的數據
    second_results = []
    # 存放扣掉第一階段偵測錯的數量
    second_box = boxes

    # 存放扣掉第二階段偵測錯的數據
    final_results = second_results
    # 存放扣掉第二階段偵測錯的數量
    final_box = second_box


    # 第一階段：依照距離來判斷是否放錯位置或者model偵測錯種類
    for i in range(len(boxes)):
        if (boxes[i] != 0 and boxes[i] != 1):
            med_name = "med" + str(i+1)
            # 先定義正確的位置
            for med in range(len(sorted_results)):
                if med_name in sorted_results[med][0]:
                    correct_x, correct_y = float(sorted_results[med + boxes[i] - 1][1]), float(sorted_results[med + boxes[i] - 1][2])
                    break
            # 再計算相對位置
            for med in range(len(sorted_results)):
                if med_name in sorted_results[med][0]:
                    current_x, current_y = float(sorted_results[med][1]), float(sorted_results[med][2])
                    # Calculate the distance
                    distance = math.sqrt((correct_x - current_x)**2 + (correct_y - current_y)**2)
                    if distance >= 1000 :
                        #print(sorted_results[med])
                        error_data.append(sorted_results[med])
                        second_box[i] -= 1
                    else:
                        # 剩下距離正確的加入第二階段要偵測的list
                        second_results.append(sorted_results[med])
    
    # 第二階段：依照藥品數量來判斷是否偵測錯誤
    for i in range(len(gt_box)):
        #偵測出來的比gt的還多，代表model有偵測錯藥品種類
        if gt_box[i] < second_box[i]:
            #刪除多偵測出來的數量
            error_num = second_box[i] - gt_box[i]
            med_name = "med" + str(i+1)
            for med in range(len(second_results)):
                if med_name in second_results[med][0]:
                    error_data.append(second_results[med])
                    #print(second_results[med])
                    #刪除錯誤的數據(之後要計算擺放位置)
                    del final_results[med]
                    final_box[i] -= 1
                    error_num -= 1
                    if (error_num == 0):
                        break
    

    print("藥品位置放錯或模型偵測種類錯誤的數據:") 
    for result in error_data:
        print(result)
    '''
    print("----------------------")            
    for result in final_results:
        print(result)
    '''

    return error_data



   