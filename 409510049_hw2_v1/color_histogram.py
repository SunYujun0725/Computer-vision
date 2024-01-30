
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import time

def extract_color_histogram(img_path):

    img = cv2.imread(img_path)
    img = cv2.resize(img, (256,256))
    # 將圖像轉換為HSV顏色空間
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 計算顏色直方圖
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [36, 36, 36], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    return hist

def calculate_distance(hist1, hist2, distance_metric='sad'):
    if distance_metric == 'sad':
        # SAD - Sum of Absolute Differences
        distance = np.sum(np.abs(hist1 - hist2))
    elif distance_metric == 'ssd':
        # SSD - Sum of Squared Differences
        distance = np.sum((hist1 - hist2) ** 2)
    else:
        raise ValueError("Invalid distance metric. Use 'sad' or 'ssd'.")
    
    return distance

def nearest_neighbor_classification(test_hist, train_data, distance_metric='sad'):
    distances = []
    
    for _, row in train_data.iterrows():
        train_hist = row['histogram']
        distance = calculate_distance(test_hist, train_hist, distance_metric)
        distances.append((row['label'], distance))
    
    # Find the label of the nearest neighbor
    nearest_neighbor = min(distances, key=lambda x: x[1])
    
    return nearest_neighbor[0]

def process_folder(folder_path, is_train=True):
    data = pd.DataFrame(columns=['filename', 'label', 'histogram'])

    if is_train: #有兩層資料夾要處理
        # 逐一處理每個種類的資料夾
        for label in os.listdir(folder_path):
            label_folder = os.path.join(folder_path, label)
            # 遍歷資料夾中的每張圖片
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                # 提取特徵（顏色直方圖）
                hist = extract_color_histogram(img_path)
                # 將特徵和標籤添加到 DataFrame
                data = data.append({'filename': filename, 'label': label, 'histogram': hist}, ignore_index=True)
                print(filename, label, hist)

    else:  #只有一層資料夾要處理
        # 逐一處理每個種類的資料夾
        for label in os.listdir(folder_path):
            img_path = os.path.join(folder_path, label)
            
            # 提取特徵（顏色直方圖）
            hist = extract_color_histogram(img_path)
        
            # 將特徵和標籤添加到 DataFrame
            data = data.append({'filename': label, 'histogram': hist}, ignore_index=True)
            print(label, hist)

    return data

def main():

    # 處理訓練資料夾
    train_data = process_folder('./plant-seedlings-classification/train')
    # 處理測試資料夾
    test_data = process_folder('./plant-seedlings-classification/test', is_train=False)

    # 準備測試集
    X_test = np.array(test_data['histogram'].tolist())
    # 進行最近鄰分類 (使用 SAD)
    predictions_sad = [nearest_neighbor_classification(test_hist, train_data, distance_metric='sad') for test_hist in X_test]
    # 進行最近鄰分類 (使用 SSD)
    #predictions_ssd = [nearest_neighbor_classification(test_hist, train_data, distance_metric='ssd') for test_hist in X_test]

    # 創建包含預測結果的 DataFrame
    results_sad = pd.DataFrame({'file': test_data['filename'], 'species': predictions_sad})
    #results_ssd = pd.DataFrame({'file': test_data['filename'], 'species': predictions_ssd})

    # 將結果保存為 CSV 檔
    results_sad.to_csv('color_histogram_sad.csv', index=False)
    #results_ssd.to_csv('color_histogram_ssd.csv', index=False)

if __name__ == '__main__':
    main()



