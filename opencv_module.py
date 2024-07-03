import os
import cv2
import numpy as np
from ultralytics import YOLO
import cv2
project_dir=os.path.dirname(os.path.abspath(__file__))
class OpencvModule:
    def __init__(self,model_path='yolov8n.pt'):
        model_path=os.path.join(project_dir,model_path)
        self.detector=YOLO(model_path)

    def edge_det(
            self,
            img_path,#图片路径
            threshold1=50,#边缘检测阈值
            threshold2=150
        ):
        '''
            @param img_path:图片数据，也可以传入图片路径
            @param threshold1:边缘检测阈值1
            @param threshold2:边缘检测阈值2
            @return:返回边缘检测结果
        '''
        

        if isinstance(img_path,str):
            image = cv2.imread(img_path)
        else:
            image=img_path
        crop_h,crop_w=image.shape[:2]
        crop_area=image.shape[0]*image.shape[1]

        # 转换到灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用Canny算法检测边缘
        edges = cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)
        return edges
    def target_detect(
            self,img="#",#当前帧
            local_camera=False#是否开启本地摄像头，False则检测传入图片
            ):
        '''
        @param img:当前帧，也可以传入图片路径
        @param local_camera:是否开启本地摄像头，False则检测传入图片
        @return:返回检测结果
        '''
        if local_camera:
            cap = cv2.VideoCapture(0)
            while True:
                frame = cap.read()[1]
                results =self.detector.predict(frame)  # 检测
                im2 = results[0].plot()
                cv2.imshow("te",im2)
                if cv2.waitKey(1) == 27:
                    break
            cap.release()
        else:
            if isinstance(img,str):
                img = cv2.imread(img)

            return self.detector.predict(img)[0].plot()
    def diff_image(self,img1,img2):
        if isinstance(img1,str) and isinstance(img2,str):
            img1 = cv2.imread(img1)
            img2 = cv2.imread(img2)
        #获取两张图片不同的像素并绘制轮廓
        diff = cv2.absdiff(img1,img2)
        diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        ret,diff = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(diff,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)
        # cv2.imshow("diff",diff)
        return img1