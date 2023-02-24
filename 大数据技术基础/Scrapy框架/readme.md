# 说明
## 概要
* 这是赵敬业2021年11月制作的scrapy爬虫框架联系程序，爬取了<a href="http://quotes.toscrape.com/">练习网站</a> 上面的箴言以及名人介绍
* 本爬虫使用了主要库及版本说明
    1. python: 3.8.8
    2. IPython:7.22.0 #非必要
    3. pandas:1.2.4 #用来存储csv
    4. scrapy:2.4.1
    
* 本爬虫的特色如下：
    1. 所有内容存储到一张表格中，表格名字命名为<a href="./quotetutorial/quotetutorial/quotes.csv">quotes.csv</a>，其排序是按照人名排序，方便阅读
    2. 在原有的程序的基础上增加了<a href="./quotetutorial/quotetutorial/start.py">start.py</a>和<a href="./quotetutorial/quotetutorial/start.ipynb">start.ipynb</a> 可以用来直接运行，省去了使用命令行的烦恼。
        * 其中<a href="./quotetutorial/quotetutorial/start.ipynb">start.ipynb</a>试运行成功，里面保存了最后一次运行的记录。
        * <a href="./quotetutorial/quotetutorial/start.py">start.py</a>理论上可以使用。
    3. 代码整体比较简洁。
    
## 主要思路
1. 使用<a href="./quotetutorial/quotetutorial/items.py">items,py</a>来完成主要的解析操作
    1. 考虑需要分别解析列表页和详情页，分别定义两个item类:<code>QuotetutorialItem</code>和<code>AuthorItem</code>分别用来承载列表中的箴言信息和作者的详情信息
2. 使用<a href="./quotetutorial/quotetutorial/spiders/quotes.py">quotes.py</a> 文件来完成主要的发出请求以及处理操作
    1. **发出request**的操作主要使用<code>parse</code>函数，包括发送下一个列表页的请求以及详情页的请求。
        1. 其中在处理箴言的循环中，每轮for循环都会在找到about的超链接之后发出获取网址的请求，并设定callback为<code>parse_detail</code>函数
        2. 在整个网页循环结束后，尊重教程的逻辑，发出获取下一页的请求，设定callback为<code>parse</code>
    2. **处理response**的操作使用两个函数来定义
        1. <code>parse</code>函数在for循环中处理列表页中的箴言，每一句箴言信息都会打包为<code>QuotetutorialItem</code>类并yield
        2. <code>parse_detail</code>处理详情页，打包为<code>AuthorItem</code>
    3. **存储文件**的功能由<a href="./quotetutorial/quotetutorial/pipelines.py">pipelines.py</a>完成，主要流程是：在开始阶段调用<code>open_spider</code>创建空的DATa Frame，在中间每次接收到Items，则根据情况将信息写入DaTaFrame，关闭程序前调用<code>close_spider</code>对DataFrame按照作者进行排序，填补缺失值，保存
        * 出现缺失值的原因是虽然之前parse的逻辑保证了pipeline中接收到相应作者的信息一定比作者箴言更加晚，但是可能有些作者不止出现了一次，而scrapy默认不重复爬取单一网页，导致了有些作者在后来出现时会出现缺失，主要是针对这种情况填补缺失值

## 其他说明
