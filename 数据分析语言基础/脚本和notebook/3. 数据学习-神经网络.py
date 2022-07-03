# %% [markdown]
# # 说明
# * 本notebook配置如下
#     * python；3.7.0
#     * torch:1.10.1

# %%
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset
import torch.nn as nn
from 处理文件 import Process_data
from hyperopt import tpe,fmin,Trials,hp,rand,anneal,space_eval

# %%
res=Process_data("E:/python/数据分析语言基础/大作业/脚本和notebook/Beijing-result-2021-10-26.csv")
X_columns=res["X_columns"]
X_train,X_test,y_train,y_test=res["X_train"],res["X_test"],res["y_train"],res["y_test"]
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

# %% [markdown]
# ## 定义模型

# %%
class MyDNN(nn.Module):
    def __init__(self,num_features,hidden1=56,hidden2=20):
        super().__init__()
        self.hidden1=nn.Linear(num_features,hidden1)
        self.hidden2=nn.Linear(hidden1,hidden2)
        self.output=nn.Linear(hidden2,1)
        self.activation=nn.ReLU()
    def forward(self,x):
        x=self.hidden1(x)
        x=self.activation(x)
        x=self.hidden2(x)
        x=self.activation(x)
        x=self.output(x)
        output=x
        return output

# %%
# 定义损失函数
# 由于是回归任务，使用经典的MESLoss
"""
定义r2_loss
From https://en.wikipedia.org/wiki/Coefficient_of_determination
"""
def r2_loss(output, target):
    """这个越大越好"""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    # print(ss_tot)
    ss_res = torch.sum((target - output) ** 2)
    # print(ss_res)
    r2 = 1 - ss_res / ss_tot
    # print("ss_tot,ss_res,r2:",ss_tot,ss_res,r2)
    return r2
#也有nn.L1Loss

# %%
# 定义Loader,因为这里数据量较小，一个batch定义为1000
class MyDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X).float()
        if type(y)==np.ndarray:
            self.y=torch.tensor(y.reshape(len(y),1)).float()
        else:
            self.y=torch.tensor(y.to_numpy().reshape(len(y),1)).float()
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]

# %%
class Machine(object):
    """定义一个Machine的类，和前面3.数据学习的思路是一致的，只是是专门为神经网络设计。
    如果想象征性的跑一跑代码看看能不能跑通，把HyperoptTrain(self,max_evals=50)中的50改为较小的数字即可（比如3）"""
    def __init__(self,params:dict,
                 X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.params=params
        self.criterion=nn.MSELoss()
    def Report(self):
        """报告分类性能的函数,同时画出分类结果ROC函数"""
        train_mae=self.test(criterion=nn.L1Loss(),loader=self.trainloader)
        train_mse=self.test(criterion=nn.MSELoss(),loader=self.trainloader)
        train_r2=self.test(criterion=r2_loss,loader=self.trainloader)
        test_mae=self.test(criterion=nn.L1Loss(),loader=self.testloader)
        test_mse=self.test(criterion=nn.MSELoss(),loader=self.testloader)
        test_r2=self.test(criterion=r2_loss,loader=self.testloader)
        print("="*60)    
        print("train数据集上模型精度指标(MAE,MSE,R2):",[train_mae,train_mse,train_r2])
        print("test数据集上模型精度指标(MAE,MSE,R2):",[test_mae,test_mse,test_r2])
        print("="*60)
    def train(self,criterion,epoch,loader):
        self.clf.train()
        for _ in range(epoch):
            running_loss=0
            for x,label in loader:
                self.optimizer.zero_grad()
                pred=self.clf(x)
                loss=criterion(pred,label)
                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()
        return running_loss
    def test(self,criterion,loader):
        self.clf.eval()
        total_loss=0
        total_size=0
        with torch.no_grad():
            for x,label in loader:
                output=self.clf(x)
                # if criterion==r2_loss:
                temp=criterion(output,label).item()*x.shape[0]
                total_size+=x.shape[0]
                # else:
                # temp=criterion(output,label).item()
                total_loss+=temp
            # if criterion==r2_loss:
            return total_loss/total_size
            # else:
                # return total_loss
    def objective(self,params):
        self.clf=MyDNN(num_features=self.X_train.shape[1],hidden1=params["hidden1"],hidden2=params["hidden2"])
        cv5_index=[0,len(self.y_train)//5,2*len(self.y_train)//5,3*len(self.y_train)//5,4*len(self.y_train)//5,len(self.y_train)-1]
        res=0
        for i,index in enumerate(cv5_index[:-1]):
            self.trainset=MyDataset(self.X_train[[j for j in np.arange(len(self.y_train)) if j<index or j>cv5_index[i+1]], :],
                                    self.y_train[[j for j in np.arange(len(self.y_train)) if j<index or j>cv5_index[i+1]]])
            self.trainloader=torch.utils.data.DataLoader(self.trainset,batch_size=params["batch_size"],shuffle=True)
            # 这里的testset只是名字，其实是validset
            self.testset=MyDataset(self.X_train[index:cv5_index[i+1],:],self.y_train[index:cv5_index[i+1]])
            self.testloader=torch.utils.data.DataLoader(self.testset,batch_size=params["batch_size"],shuffle=True)
            self.optimizer=optim.Adam(self.clf.parameters(),lr=params["lr"])
            self.train(criterion=nn.MSELoss(),epoch=params["epoch"],loader=self.trainloader)
            res+=self.test(criterion=r2_loss,loader=self.testloader)*len(self.y_train[index:cv5_index[i+1]])
        return -res/len(self.y_train)
    def HyperoptTrain(self,max_evals=50):
        """使用tpe.suggest寻找最优参数"""
        trials=Trials()
        best_params=fmin(fn=self.objective,space=self.params,
                         algo=tpe.suggest,max_evals=max_evals,trials=trials)
        best_params=space_eval(self.params, best_params)
        print("best params:\n",best_params)
        self.clf=MyDNN(num_features=self.X_train.shape[1],hidden1=best_params["hidden1"],hidden2=best_params["hidden2"])
        self.set=MyDataset(self.X_train,self.y_train)
        self.trainset=MyDataset(self.X_train,self.y_train)
        self.testset=MyDataset(self.X_test,self.y_test)
        self.trainloader=torch.utils.data.DataLoader(self.trainset,batch_size=best_params["batch_size"],shuffle=True)
        self.testloader=torch.utils.data.DataLoader(self.testset,batch_size=best_params["batch_size"],shuffle=True)
        self.loader=torch.utils.data.DataLoader(self.set,batch_size=best_params["batch_size"],shuffle=True)
        self.optimizer=optim.Adam(self.clf.parameters(),lr=best_params["lr"])
        self.train(criterion=nn.MSELoss(),epoch=best_params["epoch"],loader=self.loader)
        return self.clf
    def headquarter(self,model_name):
        """中心调度器，完成从模型训练直到模型报告的所有工作"""
        self.clf=self.HyperoptTrain()
        print(model_name,"模型","训练完成，下面是模型报告：")
        self.Report()
        return self.clf

# %%
params={
    "batch_size":hp.choice("batch_size",[500,1000,5000,10000]),
    "epoch":hp.choice("epoch",np.arange(1,100)),
    "hidden1":hp.choice("hidden1",np.arange(1,len(X_columns))),
    "hidden2":hp.choice("hidden2",np.arange(1,len(X_columns))),
    "lr":hp.choice("lr",[1e-5,1e-4,4e-4,1e-3])
}
model=Machine(params)
clf=model.headquarter("Deeplearning model")

# %%



