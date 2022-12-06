import sys
import os
import cv2
import dlib
import matplotlib.pyplot as plt


def getface(imgpath):
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(imgpath)
    # 转为灰度图像寻找面部
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取面部位置
    face_sets = detector(img_gray, 1)
    for face in face_sets:
        y1 = face.bottom()  # detect box bottom y value
        y2 = face.top()  # top y value
        x1 = face.left()  # left x value
        x2 = face.right()  # right x value
        # cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3) #添加面部框
        img = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2), :]  # 裁切面部
    return img


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


class cas_peal(object):
    # 初始化：File_path:总数据集文件夹路径 ， txt_path:txt文件名字
    def __init__(self, File_path):
        self.File_path = File_path
        self.number = 0

    def run(self):
        files = os.listdir(self.File_path)
        tru = []
        for file in files:
            if file[-3:] != 'txt':
                Label = file[17:21]
                label = 0
                if Label[0] == 'P':
                    if Label[1] == 'D':
                        label += 0 * 7
                    if Label[1] == 'M':
                        label += 1 * 7
                    if Label[1] == 'U':
                        label += 2 * 7
                    if Label[2] == '-':
                        label += 4
                    if Label[3] == '2':
                        label += 1
                    if Label[3] == '4':
                        label += 2
                    if Label[3] == '6':
                        label += 3
                    if Label[3] == '0':
                        label += 4
                else:
                    print('----------------There is a error-----------------')
                tru.append(label)
                # 存到各个类别对应的文件夹下
                member = file[3:9]
                new_dir = "CAS_PEAL\\class" + str(label)
                mkdir(new_dir)
                img = getface(self.File_path + "/" + file)
                cv2.imwrite(new_dir + "\\" + member + ".jpg", img)


if __name__ == '__main__':
    data = cas_peal("/content/cas_data")
    data.run()