# 架构
1. Airbnb数据文件夹包括Airbnb数据，保证模型正常运行
2. 脚本和notebook文件夹是脚本文件和对应notebook以及生成的数据所在地。
    1. 所有的.py按照序号排列为整个工作流程，所有.ipynb按照序号排列为整个工作流程。如<a href="脚本和notebook/1. 数据清洗.py">1. 数据清洗.py</a>是.py文件的第一个流程。同序号代表不分先后顺序，流程级别相等。
    2. 其中.ipynb文件展示了运行过程的结果和图片，结果与<a href="main.pdf">pdf文件</a>完全一致。
    3. <a href="脚本和notebook/110000.json">110000.json</a> 文件是<a href="脚本和notebook/2. 数据探索专刊-地理信息展示.ipynb">地理信息展示</a>的专用地理文件，里面记载了北京市的地理信息。另外，这一部分需要geopandas系列的python包。
    4. <a href="脚本和notebook/处理文件.py">处理文件.py</a>放置了两个基本的处理文件函数，供其他文件调用。
    5. 其余文件均为运行的中间结果
        1. <a href="脚本和notebook/Beijing-result-2021-10-26.csv">Beijing-result-2021-10-26.csv</a>是流程<a href="脚本和notebook/1. 数据清洗.py">1. 数据清洗.py</a>生成的中间文件，供后面的流程使用。
        2.  <a href="脚本和notebook/Booking_map.svg">两个svg文件</a>是地理信息展示保存的结果
        3.  <a href="脚本和notebook/LGBMbest_params.npy">三个npy</a>文件是<a href="脚本和notebook/3. 数据学习.py">3. 数据学习.py</a>生成的中间文件，保存了相应模型的中间参数，只有LGBM的参数用在了流程4
3. <a href="main.pdf">main.pdf</a>是作业文本，里面包括所有的作业结果

# 使用到的环境
* python:3.7.
* pytorch:1.10.2
* hyperopt
* lightGBM
* sklearn
* geopandas:0.9.0