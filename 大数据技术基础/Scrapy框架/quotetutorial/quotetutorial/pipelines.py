# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import pandas as pd
class QuotetutorialPipeline:
    def process_item(self, item, spider):
        item_temp=ItemAdapter(item).asdict()#转成dict形式
        if "text" in item_temp.keys():
            #这种情况是这个是列表页的一个item,即QuotetutorialItem,只需要append即可
            self.file=self.file.append(item_temp,ignore_index=True)
        else:
            #这种情况是这个是作者详情页的一个item
            self.file.loc[self.file["author"]==item_temp["author"],list(item_temp.keys())]=list(item_temp.values())
        return item

    def open_spider(self, spider):
        self.file = pd.DataFrame([],columns=["author","text","tags",'born_date','born_location','description'])

    def close_spider(self, spider):
        #填补可能的缺失值，并将作者进行排序
        self.file=self.file.sort_values(by='author').fillna(method="backfill")
        self.file.to_csv("quotes.csv")
        print("文件已保存!")

