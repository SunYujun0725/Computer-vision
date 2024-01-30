
import cv2
import numpy as np
from scipy.interpolate import Rbf

def align_images_Perspective_sift(source_path, target_path):
    
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)

    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

    # 使用SIFT找關鍵點和匹配
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(source_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(target_img, None)

    # 使用FLANN進行特徵點匹配
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 進行比率測試，以確保匹配的準確性
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 提取匹配點的位置
    source_points = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    target_points = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    # 使用cv2.findHomography進行透視變換估算
    M, _ = cv2.findHomography(source_points, target_points, cv2.RANSAC)

    # 使用cv2.warpPerspective進行透視變換
    aligned_image = cv2.warpPerspective(source_img, M, (target_img.shape[1], target_img.shape[0]))

    cv2.imwrite('./warpPerspective.jpg', aligned_image)


def align_images_affine_sift(source_path, target_path):
   
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

    # 使用SIFT找關鍵點和匹配
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(source_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(target_img, None)

    # 使用FLANN進行特徵點匹配
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 進行比率測試，以確保匹配的準確性
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 提取匹配點的位置
    source_points = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    target_points = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    # 使用cv2.estimateAffine2D進行affine變換估算
    M = cv2.estimateAffine2D(source_points, target_points)[0]

    # 使用cv2.warpAffine進行affine變換
    aligned_image = cv2.warpAffine(source_img, M, (target_img.shape[1], target_img.shape[0]))

    cv2.imwrite('./warpAffine.jpg', aligned_image)


def main():
    source_image_path = './IMG_9963.jpg'  # 斜拍的照片
    target_image_path = './IMG_7671.JPG'  # 俯拍的照片(目標)

    # 轉成俯拍照片
    align_images_affine_sift(source_image_path, target_image_path)
    align_images_Perspective_sift(source_image_path, target_image_path)
    

if __name__ == '__main__':
    main()
