from utils import datasets
from utils import utils,net
import torch
import os
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from ultralytics import YOLO
import cv2
import time
#当前文件路径的上一级目录
project_dir=os.path.dirname(os.path.abspath(__file__))

class Recognizer:
    def __init__(
            self,
            features_extractor_path=r"save_weights\CustomEfficientNet_b1-acc=0.47312-loss=0.009343-max_epochs=100-1100",
            yolo_path="yolov8n.pt"
        ):
        '''
        @param features_extractor_path: 特征向量提取器路径,可默认
        @param yolo_path: yolo模型路径，可默认
        '''
        features_extractor_path=os.path.join(project_dir,features_extractor_path)
        yolo_path=os.path.join(project_dir,yolo_path)
        self.features_extractor=self.load_features_extractor(
            features_extractor_path
        )#特征向量提取器
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.detector=YOLO(yolo_path)

        pass
    def get_users_face(self,frame,user_name='葛宇恒'):
        '''
        @param frame: 当前帧
        @param user_name: 需要登记的用户名
        @return: 返回字典，包含frame:当前帧，user:用户名，user_face:用户人脸图像
        用户人脸和特征向量默认保存到data/users
        '''
        flag=0
        rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result=self.detector.predict(rgb)#检测人脸位置
        faces=self.get_certain_face(frame,result)
        frame=result[0].plot()
        res=frame
        res_2=0
        for face in faces:
            if res_2<face.shape[0]*face.shape[1]:
                res_2=face.shape[0]*face.shape[1]
                res=face
                break
        
        # 保存用户数据
        cv2.imwrite(os.path.join(project_dir,'data/users',user_name+".jpg"),res)
        self.up_user(res,name=user_name)
        return {
            "frame":frame,#当前帧
            "user":user_name,
            "user_face":res,
        }

    def recognition_target(self,frame,threshold=0.8):
        '''
        @param frame: 当前帧
        @param threshold: 阈值，默认0.8
        @return: 返回当前帧的识别结果，字典，包含frame:当前帧，user:用户名，conf:置信度，用户为None则没有识别成功
        '''
        # rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb=frame
        result=self.detector.predict(rgb)
        faces=self.get_certain_face(rgb,result)
        res=(None,None)
        flag=0
        for face in faces:
            # res_2=self.match_users(face,threshold=threshold)
            try:
                res_2=self.match_users(face,threshold=threshold)
            except Exception as e:
                res_2=(None,None)
                print(e)
            if res_2[0] is not None:
                res=res_2
                break
        frame=result[0].plot()
        if res[0] is not None:
            final_res=res
        if flag!=0 or res[0] is not None:
            text=f'识别成功，欢迎 用户{final_res[0] } 登入,\n置信度 :'+str(final_res[1])
            frame=self.cv2ImgAddText(frame, text, 10, 50, (	0, 0, 188), 28)
            return {
                "frame":frame,#显示图像
                "user":res[0],#用户
                "conf":res[1]#置信度
            }
        else:
            text="Not a user in the database"
            cv2.putText(frame, text, (int((frame.shape[1] - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]) / 2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return {
                "frame":frame,#显示图像
                "user":None,#用户
                "conf":None#置信度
            }
    def check_users_data(self,data_dir=os.path.join(project_dir,'data/users')):
        return [os.path.join(data_dir,x) for x in os.listdir(data_dir) if x.find("npy")!=-1]
         
    def up_user(self,img_path,name=None,output_dirs=os.path.join(project_dir,'data/users')):#上传用户的图像
        # output_dir=os.path.join(output_dirs,name)
        output_dir=output_dirs
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir, f'None.npy')
        if not name  is None:
            output_file = os.path.join(output_dir, f'{name}.npy')
        print(output_file)
        # 处理图像
        image_tensor = self.process_image(img_path)

        # 提取特征
        features = self.extract_features(image_tensor,self.features_extractor)

        # 保存特征到文件
        self.save_features_to_file(features, output_file)

    


    def match_users(self,img_path, user_features_dir=os.path.join(project_dir,'data/users'),threshold=0.85):#低于阈值的不是用户
        
        image_tensor = self.process_image(img_path)
        uploaded_features = self.extract_features(image_tensor,self.features_extractor)
        user_features=self.load_user_features(user_features_dir)
        similarities = self.calculate_similarity(uploaded_features, user_features)
      
        best_match, best_similarity = self.find_best_match(similarities, threshold)

        return best_match, best_similarity#返回匹配对象和相似度，match是None则不是用户
    def get_certain_face(self,image,res):
    
        results = res
        boxes=results[0].boxes.cls.cpu().numpy()
        #根据boxes的大小裁剪出框内图像
        faces=[]
        for i in range(len(boxes)):
            if boxes[i]==0:
                x1,y1,x2,y2=results[0].boxes.xyxy[i].cpu().numpy()
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                crop_img = image[y1:y2, x1:x2]
                faces.append(crop_img)
        return faces
    def load_features_extractor(
        self,
        model_dir=os.path.join(project_dir,r"save_weights\CustomEfficientNet_b1-acc=0.47312-loss=0.009343-max_epochs=100-1100")
    ):
        dir=model_dir
        config=utils.Config(os.path.join(dir,'config.json'))
        model=utils.BestSelector(os.path.join(dir,'metrics.json'))
        fe=model['model'].to('cpu')
        return nn.Sequential(*list(fe.children())[:-1])
    def process_image(self,img_path):
        if isinstance(img_path,str):
            image = Image.open(img_path).convert('RGB')
        else:
            image=img_path
            image = self.preprocess(image)
            image = image.unsqueeze(0)  # 添加批次维度
            return image.to("cpu")
    def extract_features(self,image_tensor,fe):#获取特征向量
        with torch.no_grad():
            features = fe(image_tensor)
            features = features.view(features.size(0), -1)  # 展平特征
        return features.cpu().numpy()  # 将特征转换为numpy数组
    def save_features_to_file(self,features, output_file):
        np.save(output_file, features)  # 保存为.npy文件
    def load_user_features(self,feature_dir):
        user_features = {}
        for filename in os.listdir(feature_dir):
            if filename.endswith('.npy'):
                user_id = os.path.splitext(filename)[0]
                user_features[user_id] = np.load(os.path.join(feature_dir, filename))
        return user_features


    def calculate_similarity(self,uploaded_features, users_features):#上传的图像特征和数据库的对比
        similarities = {}
        for user_id, features in users_features.items():
            similarity = cosine_similarity(uploaded_features, features)
            similarities[user_id] = similarity[0][0]
        return similarities

    def find_best_match(self,similarities, threshold=0.1):
        best_match = max(similarities, key=similarities.get)
        if similarities[best_match] < threshold:
            return None, None
        return best_match, similarities[best_match]
    def cv2ImgAddText(self,img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "STSONG.TTF", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
