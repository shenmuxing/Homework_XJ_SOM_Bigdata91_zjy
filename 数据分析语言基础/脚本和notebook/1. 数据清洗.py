# %% [markdown]
# ## 读取文件

# %%
from 处理文件 import Read
import numpy as np
import pandas as pd
# 读取的路径
read_file="E:/python/数据分析语言基础/大作业/Airbnb数据/Beijing-2021-10-26/"
# 整合出来的表格写入的路径
write_file="E:/python/数据分析语言基础/大作业/脚本和notebook/Beijing-result-2021-10-26.csv"
# %%
Beijing=Read(read_file)

# %%
reviews=Beijing["reviews_2"]
print("reviews.head():")
print(reviews.head())

# %%
calender=Beijing["calender"]
print("calender.head():")
print(calender.head())

# %%
listings=Beijing["listings"]
print("listings.head():")
print(listings.head())

# %%
listings_2=Beijing["listings_2"]
print("listings_2.head():")
print(listings_2.head())

# %% [markdown]
# * 结论一
#     * 经过上面一顿操作，发现目前对我们有用的表格好像是listings,reviews,listings_2
#     * 关键的主键有两种，一种是用户id,一种是房源的id
#     * 聚合时，应该用listings,reviews的模式来聚合，因为我们关注的是房源定价的高低。
#     * 进一步看看

# %% [markdown]
# ## 首先应该把reviews中的文本进行充分之挖掘，得到好评、差评的数据

# %%
import textblob
from snownlp import SnowNLP

# %% [markdown]
# * 情绪可以考虑不只是描述其正负面，可以考虑赋予其值。

# %% [markdown]
# ### 使用apply函数返回各个评论的情绪

# %%
def Process_review(reviews_2:str)->str:
    """对评论数据进行操作，计算其sentiment,返回情绪值"""
    if reviews_2 is np.nan or reviews_2 is None:
        return 0
    temp=reviews_2
    blob=textblob.TextBlob(temp)
    s=SnowNLP(temp)
    return s.sentiments*2-1+blob.sentiment.polarity*2

# %%
print("正在对reviews进行情绪评估，可能需要若干分钟时间...")
reviews.loc[:,"sentiment"]=reviews.loc[:,"comments"].apply(Process_review)

# %% [markdown]
# ## 分别探索各个数据的模式，挖掘可能的信息

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %%
def describe(data:pd.DataFrame):
    """给出基本的数据描述，包括是否含有缺失值等"""
    print("="*70)
    print("表格形状：",data.shape)
    temp_1=pd.Series([pd.unique(data.loc[:,i]).shape[0] for i in data.columns],index=data.columns,name="独立元素个数")
    temp_3=temp_1/data.shape[0];temp_3.name="独立元素比例"
    temp=np.sum(data.isna())
    temp.index=data.columns
    temp.name="缺失值个数"
    temp_2=pd.merge(left=temp_1,right=temp,how="inner",left_index=True,right_index=True)
    temp_2=pd.merge(left=temp_2,right=temp_3,left_index=True,right_index=True)
    print("结果：\n",temp_2.to_string())
    if np.any(temp)!=0:
        data_na=data.loc[np.any(data.isna(),axis=1),:].isna()
        plt.figure(figsize=(5,5))
        sns.heatmap(data_na)
        plt.show()
    print("="*70)

# %% [markdown]
# * reviews结果说明：
#     * sentiment: 按照listing_id进行聚合与别的listing进行合并

# %% [markdown]
# * calender是按照未来365天listing的日历进行排列的
#     * 主键：listing_id,date
#     * available：是否有房。
#     * price,adjusted_price:adjusted_price应该是根据季度进行的正儿八经的价格，和是否有房有着很大的关系，应该分时间、listing_id进一步探索
#     * minimum_nights,maximum_nights和房子的价格有很大关系。
# * 总体来说，应该按照calender进行计算，把如下公式作为房东$x$满意收益（在4.典型房东应用部分会用到）：
#     $$Adjusted\_price_x \cdot not_available_x-\frac{\sum_{i}{Adjust\_price_{i,j}\cdot \mathbf{1}}}{\sum_{i}1}$$
#     其中$i$是这个地区中的所有房子,$j$是365天每天的价格。

# %% [markdown]
# * 缺失元素之间有很强的关联

# %% [markdown]
# * listings_2应该是主要的参数来源
# * listings表是listings_2表的子集

# %% [markdown]
# ## 数据清洗-数据合并-数据再清洗

# %%
# 把reviews进行聚合，主要是进行情感聚合
group_reviews=reviews.groupby("listing_id",as_index=False)["sentiment"].mean()
print("对评论数据计算情绪值结束，每个房源的平均情绪展示前五条：")
print(group_reviews.head())

# %%
def To_number(str_price):
    """把price价格去掉$,变成number"""
    if type(str_price)==float:
        return str_price
    while str_price.find("$")!=-1:
        str_price=str_price[:str_price.find("$")]+str_price[str_price.find("$")+1:]
    while str_price.find(",")!=-1:
        str_price=str_price[:str_price.find(",")]+str_price[str_price.find(",")+1:]
    return float(str_price)
# 对calender进行清洗
calender.loc[calender.loc[:,"available"]=="f","notavailable"]=1
calender.loc[calender.loc[:,"available"]=="t","notavailable"]=0
calender.loc[:,"adjusted_price"]=calender.loc[:,"adjusted_price"].apply(To_number)
def compute_calender(data:pd.DataFrame):
    """对calender进行最后的计算，返回notavaiable比率和相应adjusted_price"""
    # 先把这里面全是notavailable的给删掉，返回空值
    if np.all(data.notavailable==1):
        print("在calender中删掉了一些异常数据，异常数据listing_id为:",pd.unique(data.listing_id))
        return None
    temp=data.groupby("adjusted_price",as_index=False)["notavailable"].mean()
    temp["listing_id"]=pd.unique(data["listing_id"])[0]
    return temp

print("正在聚合calender的数据...")
group_calender=calender.groupby("listing_id",as_index=False).apply(compute_calender)
group_calender.index=[i for i in range(0,group_calender.shape[0])]
print("聚合完成，展示如下：")
print(group_calender.head())

# %%
# 清洗listings_2，并确定入列元素
listings_2["is_description"]=1
listings_2.loc[listings_2.loc[:,"description"].isna(),"is_description"]=0
listings_2["is_neighborhood_overview"]=1
listings_2.loc[listings_2.loc[:,"neighborhood_overview"].isna(),"is_neighborhood_overview"]=0
listings_2["is_host_about"]=1
listings_2.loc[listings_2.loc[:,"host_about"].isna(),"is_host_about"]=0
#删除host_response_time,host_response_rate的缺失数据
listings_2=listings_2.loc[listings_2.loc[:,"host_response_time"].notna(),:]
listings_2=listings_2.loc[listings_2.loc[:,"host_response_rate"].notna(),:]
listings_2=listings_2.loc[listings_2.loc[:,"host_acceptance_rate"].notna(),:]
#reviews_per_month缺失值到时候直接置为0
listings_2.loc[listings_2.loc[:,"reviews_per_month"].isna(),"reviews_per_month"]=0

# %%
# 确定所有进入数据合并的列
group_calender_column=["listing_id","adjusted_price","notavailable"]
group_reviews_column=["listing_id","sentiment"]
listings_2_column=["id","is_description","is_neighborhood_overview","is_host_about",'host_since',"host_response_time",
                  'host_response_rate', 'host_acceptance_rate','host_is_superhost',"host_listings_count",
                 "host_total_listings_count","host_verifications","host_has_profile_pic",'host_identity_verified',
                   'neighbourhood_cleansed','latitude','longitude','property_type', 'room_type', 'accommodates','bathrooms_text',
                   'bedrooms', 'beds','minimum_nights', 'maximum_nights','number_of_reviews_ltm', 'number_of_reviews_l30d',
                 'review_scores_rating', 'review_scores_accuracy','review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location','review_scores_value', 'instant_bookable', 'reviews_per_month']


# %%
print("正在整合列表...")
result=pd.merge(group_calender.loc[:,group_calender_column],group_reviews.loc[:,group_reviews_column],how="left",left_on=["listing_id"],right_on=["listing_id"])
result=pd.merge(result,listings_2.loc[:,listings_2_column],how="left",left_on=["listing_id"],right_on=["id"])
result=result.loc[:,[i for i in result.columns if i!="id"]]
print("整合完成...")
print(result.head())

# %%
result=result.loc[result.loc[:,"is_description"].notna(),:]
result.loc[result.loc[:,"sentiment"].isna(),"sentiment"]=0

# %%
# bathrooms 按照个数来处理
def Process_bathrooms(text:str)->float:
    """处理厕所数量"""
    if text is None or text is np.nan:
        return np.nan
    if type(text)==str and "baths" in text:
        return float(text.split()[0])
    elif type(text)==str and text=="1 bath" or "1 private bath" or "1 shared bath":
        return float(1)
    elif type(text)==str and text=="Half-bath" or text=="Private half-bath" or text=="Shared half-bath":
        return float(0.5)
    elif text==0 or text=="0":
        return float(0)
    else:
        return np.nan

result.loc[:,"bathrooms"]=result.loc[:,"bathrooms_text"].apply(Process_bathrooms)
result=result.drop("bathrooms_text",axis=1)

# %%
def as_type(name):
    """把一些列变成float形式"""
    result.loc[:,name]=result.loc[:,name].astype("float")

l=["adjusted_price","notavailable","sentiment","is_description","is_neighborhood_overview","is_host_about"]
for i in l:
    as_type(i)

# 把str类型的时间变成float形式
def as_time(time):
    a=pd.Timestamp(listings_2.loc[1,"last_scraped"])-pd.Timestamp(time)
    return float(a.to_numpy()/10**9/3600/24)    
result.loc[:,"host_time"]=result.loc[:,"host_since"].apply(as_time)
result=result.drop("host_since",axis=1)


# %%
# 百分数变浮点数
def change_to_float(text:str):
    """把百分数转换为浮点数"""
    if text is np.nan or text is None:
        return text
    elif type(text)!=str:
        return text
    else:
        return float(text.strip("%"))/100

result.loc[:,"host_response_rate"]=result.loc[:,"host_response_rate"].apply(change_to_float)
result.loc[:,"host_acceptance_rate"]=result.loc[:,"host_acceptance_rate"].apply(change_to_float)
result.loc[result.loc[:,"host_is_superhost"]=="f","host_is_superhost"]=0
result.loc[result.loc[:,"host_is_superhost"]=="t","host_is_superhost"]=1
result.loc[result.loc[:,"instant_bookable"]=="f","instant_bookable"]=0
result.loc[result.loc[:,"instant_bookable"]=="t","instant_bookable"]=1
result.loc[result.loc[:,"host_has_profile_pic"]=="t","host_has_profile_pic"]=1
result.loc[result.loc[:,"host_has_profile_pic"]=="f","host_has_profile_pic"]=0
result.loc[result.loc[:,"host_identity_verified"]=="t","host_identity_verified"]=1
result.loc[result.loc[:,"host_identity_verified"]=="f","host_identity_verified"]=0


# %%
def Strip_verifications(text:str):
    """把verifications里面的文本长度存储起来"""
    if type(text)!=str:
        return text
    elif text[0]=="[" and text[-1]=="]":
        return len(list(eval(text)))
    else :
        return 0
result.loc[:,"host_verifications"]=result.loc[:,"host_verifications"].apply(Strip_verifications)

# %%
# 把不是float的全部转换成float
for i in result.columns:
    if type(result.loc[0,i])==str:
        print(i,type(result.loc[0,i]))

# %%
from sklearn.preprocessing import OneHotEncoder
def encoder(name,result):
    """对分类数据进行独热编码"""
    clf=OneHotEncoder()
    temp=result.loc[:,name].copy().to_numpy()
    temp.shape=(len(temp),1)
    temp[temp==0]="nan"
    temp=clf.fit_transform(temp)
    result.loc[:,[name+str(i) for i in range(0,len(pd.unique(result.loc[:,name])))]]=temp.todense()
    # result.loc[:,name]=temp
    return result.drop(name,axis=1)
    
result=encoder("host_response_time",result)
result=encoder("property_type",result)
result=encoder("room_type",result)
result=encoder("neighbourhood_cleansed",result)

# %%
# 在填补剩下的缺失值之前需要对变量进行其他操作
from sklearn.impute import KNNImputer
clf=KNNImputer()
result=pd.DataFrame(clf.fit_transform(result),columns=result.columns,index=result.index)
print("清洗数据结束，展示最终结果：")
describe(result)

# %%
result.to_csv(write_file)


