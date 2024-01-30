import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from skimage import feature
'''
def extract_lbp_feature(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 讀取圖像並轉換為灰度
    img = cv2.resize(img, (256,256))
    radius = 1 # LBP算法中范围半径的取值
    n_points = 8 * radius # 领域像素点数
    hist = []
    # 計算局部二值模式
    lbp = feature.local_binary_pattern(img, P=radius, R= n_points, method="default")
    # 提取特定位置的 LBP 值
    hist = lbp.reshape(-1)
    # 計算每個像素點及其周圍 8 個元素的 LBP 值  
    return hist
'''

def get_pixel(img, center, x, y): 
      
    new_value = 0
    try: 
        if img[x][y] >= center: 
            new_value = 1   
    except: 
        pass

    return new_value 
   
# Function for calculating LBP 
def lbp_calculated_pixel(img, x, y): 
   
    center = img[x][y] 
    val_ar = [] 
    # top_left 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 
    # top 
    val_ar.append(get_pixel(img, center, x-1, y)) 
    # top_right 
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
    # right 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
    # bottom_right 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
    # bottom 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
    # bottom_left 
    val_ar.append(get_pixel(img, center, x + 1, y-1)) 
    # left 
    val_ar.append(get_pixel(img, center, x, y-1)) 
       
    # Now, we need to convert binary 
    # values to decimal 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
   
    val = 0
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
    return val

def extract_lbp_feature(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 讀取圖像並轉換為灰度
    img = cv2.resize(img, (256, 256))
    hist = []
    # Get the height and width of the image
    height, width = img.shape

    # Iterate over each row (height)
    for x in range(height):
        # Iterate over each column (width)
        for y in range(width):
            # Access pixel value at position (i, j)
            hist.append(lbp_calculated_pixel(img, x, y))
    # Convert the list of LBP values to a NumPy array
    lbp_values = np.array(hist)
    # Create a histogram of LBP values
    hist, bins = np.histogram(lbp_values, bins=256, range=[0, 256])
    #print(len(hist))
    #for i in hist:
       #print(i)
    #exit()
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
                hist = extract_lbp_feature(img_path)
                # 將特徵和標籤添加到 DataFrame
                data = data.append({'filename': filename, 'label': label, 'histogram': hist}, ignore_index=True)
                #print(filename, label, hist)
                print(filename, label)

    else:  #只有一層資料夾要處理
        # 逐一處理每個種類的資料夾
        for label in os.listdir(folder_path):
            img_path = os.path.join(folder_path, label)
            
            # 提取特徵（顏色直方圖）
            hist = extract_lbp_feature(img_path)
        
            # 將特徵和標籤添加到 DataFrame
            data = data.append({'filename': label, 'histogram': hist}, ignore_index=True)
            #print(label, hist)
            print(label)

    return data

def main():
    # 處理訓練資料夾
    train_data = process_folder('./plant-seedlings-classification/train')
    # 處理測試資料夾
    test_data = process_folder('./plant-seedlings-classification/test', is_train=False)
    # 準備測試集
    X_test = np.array(test_data['histogram'].tolist())
    # 進行最近鄰分類 (使用 SAD)
    predictions_sad = [nearest_neighbor_classification(test_feature, train_data, distance_metric='sad') for test_feature in X_test]

    # 創建包含預測結果的 DataFrame
    results_sad = pd.DataFrame({'file': test_data['filename'], 'species': predictions_sad})

    # 將結果保存為 CSV 檔
    results_sad.to_csv('LBP_sad.csv', index=False)

if __name__ == '__main__':
    main()


