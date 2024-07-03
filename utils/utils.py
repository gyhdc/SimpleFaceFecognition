import json
import os

import torch


class Config:
    def __init__(self,*l_params,**params):
        if len(params) == 0 and len(l_params) == 1 and isinstance(l_params[-1], str):
            self.read(l_params[-1])
        else:
            self.config = params
            self.set_attr()
    def update(self,**params):
        self.config.update(params)
        self.set_attr()
    def read(self,path):
        with open(path, "r") as f:
            self.config=json.load(f)
        self.set_attr()
    def save(self,path):
        if os.path.isdir(path):
            path=os.path.join(path,"config.json")
        with open(path,mode='w') as f:
            json.dump(self.config,f)
    def __str__(self):#展示config
        res=[]
        for key,val in self.config.items():
            res.append(f"{key} : {val} ")
        return "\n".join(res)
    def __repr__(self):#展示config
        res=[]
        for key,val in self.config.items():
            res.append(f"{key} : {val} ")
        return "\n".join(res)
    def set_attr(self,):
        for key, value in self.config.items():
            setattr(self, key, value)#设置字典为成员变量


class BestSelector:

    def __init__(self, *l_params,**params):
        self.metrics_path=None
        if len(params) == 0 and len(l_params) == 1 and isinstance(l_params[-1], str):
            self.metrics_path=l_params[-1]
            self.read(l_params[-1])
        else:
            self.bestMetrics = params
            self.set_attr()

    def __getitem__(self, key):
        return self.bestMetrics[key]

    def update(self, **params):
        self.bestMetrics = params
        self.set_attr()
    def read(self,path):
        with open(path) as f:
            data=json.load(f)
        new_data={}
        for key, val in data.items():
            if key=='model':
                if not os.path.exists(val):
                    val=os.path.join(os.path.dirname(self.metrics_path),"best.pth")
                
                try:
                    model=torch.load(val)
                except:
                    model=torch.load(val, map_location=torch.device('cpu'))
                new_data[key]=model
            else:
                new_data[key]=val
        self.bestMetrics=new_data
        self.set_attr()


    def save(self, dirPath):
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        metrics = {}
        for key, val in self.bestMetrics.items():
            if key == "model":
                pthSavePath = os.path.join(dirPath, "best.pth")
                torch.save(val, pthSavePath)
                metrics["model"] = pthSavePath
            else:
                metrics[key] = val
        with open(os.path.join(dirPath, "metrics.json"), mode='w') as f:
            json.dump(metrics, f)

    def __str__(self):
        res = []
        for key, val in self.bestMetrics.items():
            if len(str(val)) > 50:
                continue
            res.append(f"{key} : {val} ")
        return ",".join(res)
    def __repr__(self):
        res = []
        for key, val in self.bestMetrics.items():
            if len(str(val)) > 50:
                continue
            res.append(f"{key} : {val} ")
        return ",".join(res)

    def set_attr(self, ):
        for key, value in self.bestMetrics.items():
            setattr(self, key, value)  # 设置字典为成员变量


class Logs:
    def __init__(self,*l_params,**params):

        if len(params)==0 and len(l_params)==1 and isinstance(l_params[-1],str):

            self.read(l_params[-1])
        else:
            self.logs=params
            self.set_attr()
    def update(self,**params):
        self.logs.update(params)
        self.set_attr()
    def read(self,path):

        with open(path, "r") as f:
            self.logs = json.load(f)

        self.set_attr()

    def save(self,path):
        if os.path.isdir(path):
            path=os.path.join(path,"logs.json")
        with open(path,mode='w') as f:
            json.dump(self.logs,f)
    def set_attr(self,):
        for key, value in self.logs.items():
            setattr(self, key, value)#设置字典为成员变量
    def __str__(self):#展示config

        res=[]
        for key,val in self.logs.items():
            # print(key,val)
            res.append(f"{key} : {type(val)} ")

        return "\n".join(res)
    def __repr__(self):#展示config
        res=[]
        for key,val in self.logs.items():
            res.append(f"{key} : {type(val)} ")
        return "\n".join(res)
def saveProcess(saveDir,bestMod,train_log,config):
    print(saveDir)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    bestMod.save(saveDir)
    config.save(saveDir)
    train_log.save(saveDir)

if __name__ == '__main__':
    logs_path=r'D:\Desktop\深度学习\4.nlp-\生物信息\Test\models\exp_3_4_4\train_log.json'
    mod=BestSelector(r"D:\Desktop\深度学习\4.nlp-\生物信息\Test\models\exp_3_4_4\metrics.json")
    config=Config(r'D:\Desktop\深度学习\4.nlp-\生物信息\Test\models\exp_3_4_4\config.json')
    print(config)
    logs=Logs(logs_path)
