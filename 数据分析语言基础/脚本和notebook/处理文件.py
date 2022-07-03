import pandas as pd
import numpy as np
import os

def Read(dir:str)->tuple:
    """读取dir的短租数据集，返回字典包装的pandas.Dataframe
    dir 格式:Airbnb/Beijing-2021-10-26/  """
    print("正在读取",dir)
    res={}
    if os.path.isdir(dir+"calendar.csv"):
        calender=pd.read_csv(dir+"calendar.csv/calendar.csv")
        res["calender"]=calender
    else:
        print("calender不存在，请检查格式是否错误...")
        res["calender"]=None
    if os.path.isdir(dir+"listings.csv (2)"):
        res["listings_2"]=pd.read_csv(dir+"listings.csv (2)/listings.csv")
    else:
        print("listings_2不存在，请检查格式是否错误...")
        res["calender"]=None
    if os.path.isdir(dir+"reviews.csv (2)"):
        res["reviews_2"]=pd.read_csv(dir+"reviews.csv (2)/reviews.csv")
    else:
        print("reviews_2不存在，请检查格式是否错误...")
        res["reviews_2"]=None
    if os.path.isfile(dir+"listings.csv"):
        res["listings"]=pd.read_csv(dir+"listings.csv")
    else:
        print("listing不存在，请检查格式是否错误...")
        res["listings"]=None
    if os.path.isfile(dir+"neighbourhoods.csv"):
        res["neighbourhoods"]=pd.read_csv(dir+"neighbourhoods.csv")
    else:
        print("neighbourhoods不存在，请检查格式是否错误...")
        res["neighbourhoods"]=None
    if os.path.isfile(dir+"reviews.csv"):
        res["reviews"]=pd.read_csv(dir+"reviews.csv")
    else:
        print("reviews不存在，请检查格式是否错误...")
        res["reviews"]=None
    print("读取结束....")
    return res

def Process_data(path:str,with_scale=False):
    """读取数据处理文件已经处理好的数据，
    并利用标准的函数将其转换为X_train,X_test,y_train,y_test"""
    print("正在处理:",path)
    data=pd.read_csv(path,index_col=0)
    # price取log
    data.adjusted_price=np.log(data.adjusted_price+1e-6)
    # 
    print(" 正在删除强相关变量...")
    print("  删除前:",data.shape)
    data=data.drop("host_listings_count",axis=1)
    data=data.drop("accommodates",axis=1)
    data=data.drop("beds",axis=1)
    data=data.drop("number_of_reviews_l30d",axis=1)
    data=data.drop("review_scores_communication",axis=1)
    print(" 删除完成",data.shape)
    # 分X y
    X=data.loc[:,[i for i in data.columns if i!="listing_id" and i!="notavailable"]].copy()
    X_columns=X.columns
    y=data.loc[:,"notavailable"].copy()
    from sklearn.preprocessing import StandardScaler
    clf=StandardScaler()
    X=clf.fit_transform(X)
    scales=clf.scale_
    means=clf.mean_
    # 进行特征选择 
    # from sklearn.linear_model import Lasso
    # from sklearn.feature_selection import SelectFromModel
    # lsvc=Lasso(alpha=0.001,random_state=0).fit(X, y)
    # model = SelectFromModel(lsvc, prefit=True)
    # X = model.transform(X)
    # X_columns=model.transform(X_columns.to_numpy().reshape(1,-1))
    # 分训练集和测试集
    from sklearn.model_selection import train_test_split
    clf=train_test_split(X,y,random_state=0)
    X_train,X_test,y_train,y_test=clf

    print("请核对训练集，测试集形状:")
    print("X_train.shape:",X_train.shape)
    print("y_train.shape:",y_train.shape)
    print("X_test.shape:",X_test.shape)
    print("y_test.shape:",y_test.shape)
    return {"X_columns":X_columns,"X_train":X_train,"X_test":X_test,
    "y_train":y_train,"y_test":y_test,"scale":scales,"mean":means}
    
