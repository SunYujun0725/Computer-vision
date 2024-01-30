Computer Vision Final Project

1.請預先使用conda 建立環境

conda env create --name cvproj --file=env.yaml
2.下載tag內的yolov7.onnx放在model_data內

3.運行程式

Affine Method

python project.py Standard.jpg Test_1.jpg 0 

Perspective Method

python project.py Standard.jpg Test_1.jpg 1
