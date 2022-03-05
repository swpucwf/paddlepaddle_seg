# paddlepaddle_seg
记录飞桨作业
- 利用的数据集

## 一、项目背景介绍
遥感影像地块分割, 旨在对遥感影像进行像素级内容解析，对遥感影像中感兴趣的类别进行提取和分类，在城乡规划、防汛救灾等领域具有很高的实用价值，在工业界也受到了广泛关注。现有的遥感影像地块分割数据处理方法局限于特定的场景和特定的数据来源，且精度无法满足需求。因此在实际应用中，仍然大量依赖于人工处理，需要消耗大量的人力、物力、财力。本项目旨在衡量遥感影像地块分割模型在多个类别（如建筑、道路、林地等）上的效果，利用人工智能技术，对多来源、多场景的异构遥感影像数据进行充分挖掘，打造高效、实用的算法，提高遥感影像的分析提取能力。



##模型的预测，遥感图像的预测结果存放在work/result/文件中
!python /home/aistudio/PaddleSeg/predict.py --config /home/aistudio/PaddleSeg/configs/ocrnet/ocrnet_hrnetw48_remotesensing_256x256.yml --model_path /home/aistudio/work/ocrnet_check/iter_105000/model.pdparams --image_path /home/aistudio/data/data80164/img_testA --save_dir /home/aistudio/work/result



import matplotlib.pyplot as plt
%matplotlib inline

# 显示图像
def show_img(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)  # 归一化到0-255
    img += 50  # 增加亮度看起来方便点
    img[img > 255] = 255  # 避免超出255
    return img
img1_path = "./work/result/added_prediction/1000.jpg"
img1 = cv2.imread(img1_path)
# 显示图像
plt.figure(figsize=(15, 10))
plt.subplot(231);
plt.imshow(img1);
plt.title('img1_cv2')  # cv2读取为BGR

plt.show()



import matplotlib.pyplot as plt
%matplotlib inline

# 显示图像
def show_img(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)  # 归一化到0-255
    img += 50  # 增加亮度看起来方便点
    img[img > 255] = 255  # 避免超出255
    return img
img1_path = "./work/result/pseudo_color_prediction/1000.png"
img1 = cv2.imread(img1_path)
# 显示图像
plt.figure(figsize=(15, 10))
plt.subplot(231);
plt.imshow(img1);
plt.title('img1_cv2')  # cv2读取为BGR

plt.show()
