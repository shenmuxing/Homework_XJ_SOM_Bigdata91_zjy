# %%
import pandas as pd
import numpy as np
# 统计notavailable的情况
import matplotlib.pyplot as plt
plt.style.use("seaborn")
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False  
import seaborn as sns


# %%
# 在这里修改路径
data=pd.read_csv("E:/python/数据分析语言基础/大作业/脚本和notebook/Beijing-result-2021-10-26.csv",index_col=0)
print("展示表格")
print(data.shape)
print(data.head())


# %%
# 展示预定比率分布直方图
plt.hist(data.notavailable);
plt.xlabel("预定比率");
plt.ylabel("频数");
plt.title("预定比率频数分布直方图");
plt.show();


# %%
# 评论情感统计
from sklearn.preprocessing import StandardScaler
plt.hist(StandardScaler().fit_transform(data.sentiment.to_numpy().reshape(-1,1)).reshape(len(data.sentiment)));
plt.title("用户情感直方图")
plt.xlabel("情感分布")
plt.ylabel("频数")
plt.show();


# %%
# adjusted_price 价格分布直方图

plt.hist(data.adjusted_price.to_numpy());
plt.title("价格分布直方图")
plt.xlabel("价格分布")
plt.ylabel("频数")
plt.show();


# %%
plt.hist(np.log(data.adjusted_price+1e-6));
plt.title("log价格分布直方图")
plt.xlabel("log价格分布")
plt.ylabel("频数")
plt.show();


# %%
# 正式对数据进行变换
data.adjusted_price=np.log(data.adjusted_price+1e-6)




# %%
plt.figure(figsize=(20,20))
sns.heatmap(data.loc[:,"adjusted_price":].corr(), vmax=1, square=True,
            cmap="Blues")
plt.title("未删除强相关变量相关系数矩阵");
plt.show();


# %%
# 尝试删除强相关变量
data_corr=data.corr()
for i in data_corr.index:
    for j in data_corr.columns:
        if data_corr.loc[i,j]>0.8 and i!=j:
            print(i,j, data_corr.loc[i,j])
#  data_corr[data_corr>0.8]

# %%
plt.figure(figsize=(20,20))
sns.heatmap(data.loc[:,"adjusted_price":].corr(), vmax=1, square=True,
            cmap="Blues")
plt.title("删除强相关变量后相关系数矩阵");
plt.show();

# %%
# 对变量进行标准化
X_columns=[i for i in data.columns if i!="listing_id" and i!="notavailable"]
X=data.loc[:,X_columns].copy()
y=data.loc[:,"notavailable"].copy()
from sklearn.preprocessing import StandardScaler
clf=StandardScaler()
X=clf.fit_transform(X)

X_columns=[i for i in data.columns if i!="listing_id" and i!="notavailable"]
X=data.loc[:,X_columns].copy()
y=data.loc[:,"notavailable"].copy()


# %%
# 删掉强相关变量
print("正在删除强相关变量...")
print("删除前:",data.shape)
data=data.drop("host_listings_count",axis=1)
data=data.drop("accommodates",axis=1)
data=data.drop("beds",axis=1)
data=data.drop("number_of_reviews_l30d",axis=1)
data=data.drop("review_scores_communication",axis=1)
print("删除完成",data.shape)

# %%
neighbors=np.sum(data.loc[:,"neighbourhood_cleansed0":"neighbourhood_cleansed15"])
plt.bar([i for i in range(len(neighbors))],neighbors);
plt.xlabel("不同地区")
plt.ylabel("个数统计")
plt.title("北京市不同地区的总民宿个数统计");
plt.show();


# %%
# bathrooms 和 notavailable之间的关系
sns.set_theme(style="white")
data_copy=data.copy().sort_values(by="bathrooms")
g = sns.JointGrid(data=data_copy, x="bathrooms", y="notavailable", space=0)
g.plot_joint(sns.scatterplot, s=30, alpha=.5)
g.plot_marginals(sns.histplot,  alpha=1, bins=25)#color="blue",
plt.show();


# %%
sns.set_theme(style="ticks")
sns.pairplot(data.loc[:,["adjusted_price","notavailable","bathrooms","neighbourhood_cleansed6"]].sample(1000),hue="neighbourhood_cleansed6")



