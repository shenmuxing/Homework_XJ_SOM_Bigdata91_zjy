# -*- coding: utf-8 -*-
"""
Created on Mon May 17 09:30:51 2021
@author: 赵敬业
"""

class mymatrix(object):
    """
    自定义矩阵类型，用list来实现
    """
    value=None#矩阵的值
    x=None#矩阵的行数
    y=None#矩阵的列数
    def __init__(self,a):
        self.value=a[:][:]
        self.x=len(a)
        self.y=len(a[0])
    def temp(self):
        #创立一个临时的、由0组成的，和self一样大的list
        c=[0]*self.x
        for i in range(0,self.x):
            c[i]=[0]*self.y
        return c
    def insert(self,i,j,val):
        #设定一个值的大小
        self.value[i][j]=val
    def addmat(self,self2):
        #矩阵相加的简单运算
        c=self.temp()
        for i in range(0,self.x):
            for j in range(0,self.y):
                c[i][j]=self.value[i][j]+self2.value[i][j]
        c=mymatrix(c)
        return c
    def diag(self):
        #求矩阵的diag
        c=self.temp()
        for i in range(0,self.x):
            c[i][i]=self.value[i][i]
        c=mymatrix(c)
        return c
    def upmat(self):
        #求矩阵的上三角阵，不包括对角元素
        c=self.temp()
        for i in range(0,self.x):
            for j in range(i+1,self.y):
                c[i][j]=self.value[i][j]
        c=mymatrix(c)
        return c
    def lowmat(self):
        #求矩阵的下三角矩阵，不包括对角元素
        c=self.temp()
        for i in range(0,self.x):
            for j in range(0,i):
                c[i][j]=self.value[i][j]
        c=mymatrix(c)
        return c
    def subtractmat(self,self2):
        #矩阵相减运算，self是被减矩阵
        c=self.temp()
        for i in range(0,self.x):
            for j in range(0,self.y):
                c[i][j]=self.value[i][j]-self2.value[i][j]
        c=mymatrix(c)
        return c
    def myprint(self):
        #打印矩阵的所有值
        print("xt是：")
        for i in range(0,self.x):
            print(self.value[i])
def jacobi(A,b,x0,tmax):
    """
    雅各比迭代函数
    传入的都是mymatrix型的矩阵,最后一个tmax设定迭代上限
    """
    D=A.diag() #得到矩阵A的对角元素
    x1=x0 #使用新的名称定义x0
    x=[None]*tmax #存储迭代过程中的所有xt
    x[0]=x1 #第一个xt
    t=0#设定迭代次数
    B=A.subtractmat(D)#B矩阵是A矩阵减去D矩阵
    x2=[0]*A.y #一个list用来暂时存储过程中产生的x(t+1)
    for i in range(0,len(x2)):
        #这一步初始化x2,使得x2为好看的列向量形式
        x2[i]=[0]
    while t<tmax :
        #主迭代函数
        for i in range(0,A.x):
            temp=0#temp存储argsum(a[i][j])*x
            for j in range(0,A.y):
                temp=temp+B.value[i][j]*x1.value[j][0]                
            x2[i][0]=1/(D.value[i][i])*(b.value[i][0]-temp)
            #上面这句话就是雅各比迭代的公式
        a=[0]*A.y
        """
        弱智的python会传递链表的指针，所以新建一个a,把x2中的元素利用下面的循环倒腾到a中
        """
        for i in range(0,len(x2)):
            a[i]=x2[i][:]
        x1=mymatrix(a)#这一轮结束了，给x1赋予新值，注意是mymatrix形式
        x1.myprint()#打印出来看看逼近x*了吗
        x[t]=x1#把x1存到数组里面，万一以后改进的时候用得到呢？
        t=t+1
    return x#把整个过程返回给调用它的函数
def Gauss(A,b,x0,tmax):
    """
    这是高斯-塞塔他迭代函数
    传入的都是mymatrix型的矩阵,最后一个tmax设定迭代上限
    """
    D=A.diag()#同上
    L=A.lowmat()#下三角阵，对角元素为0
    U=A.upmat()#上三角矩阵，对角元素为0
    t=0
    x1=x0
    x=[None]*tmax
    x[0]=x1#和上面一样，只是不需要x2临时存储型新x了
    while t<tmax:
        for i in range(0,x1.x):
            temp=0
            for j in range(0,A.y):
                temp=temp+L.value[i][j]*x1.value[j][0]+U.value[i][j]*x1.value[j][0]
            x1.value[i][0]=1/(D.value[i][i])*(b.value[i][0]-temp)
        x1.myprint()
        x[t]=x1
        t=t+1
    return x    

#第一题主函数
A1=mymatrix([[1,2,1],[3,8,1],[0,4,1]])
b1=mymatrix([[2],[12],[2]])
x0=mymatrix([[0],[0],[0]])
tmax=10#设定最大迭代次数为10
print("下面表演雅各比迭代求解第一问")
j1=jacobi(A1,b1,x0,10)
print("下面表演高斯迭代求解第一问")
g1=Gauss(A1,b1,x0,tmax)
A2=mymatrix([[8,-3,2],[4,11,-1],[2,1,4]])
b2=mymatrix([[20],[33],[12]])
x0=mymatrix([[10],[10],[10]])
tmax=10
print("下面表演雅各比迭代求解第二问")
j2=jacobi(A2,b2,x0,tmax)
print("下面表演高斯迭代求解第二问")
g2=Gauss(A2,b2,x0,tmax)