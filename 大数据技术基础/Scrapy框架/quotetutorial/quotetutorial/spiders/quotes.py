from typing import Callable
import scrapy
from quotetutorial.items import QuotetutorialItem,AuthorItem


class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    allowed_domains = ['quotes.toscrape.com']
    start_urls = ['http://quotes.toscrape.com/']
    
    def parse(self, response):
        quotes = response.css('.quote')
        for quote in quotes:
            
            text = quote.css('.text::text').extract_first()
            author = quote.css('.author::text').extract_first()
            tags = quote.css('.tags .tag::text').extract()
            item = QuotetutorialItem(text=text,author=author,tags=tags)
            #解析详情页url
            detail_link=quote.css('span a::attr(href)').extract_first()
            detail_link=response.urljoin(detail_link)+"/"#省得他老是给我redirecting烦我
            yield item #把item给他
            yield scrapy.Request(url=detail_link,callback=self.parse_detail)#把详情页给他，因为是在yield item 之后，一定会比列表页更晚传送到
                 
        next = response.css('.pager .next a::attr(href)').extract_first()
        url = response.urljoin(next)
        yield scrapy.Request(url=url, callback=self.parse)
            
    def parse_detail(self,response):
        """本函数解析详情页"""
        author=response.css('.author-title::text').extract_first().strip()
        details=response.css('.author-details')
        born_date=details.css('.author-born-date::text').extract_first().strip()
        born_location=details.css('.author-born-location::text').extract_first().strip()
        description=details.css('.author-description::text').extract_first().strip()
        item=AuthorItem(author=author,born_date=born_date,born_location=born_location,description=description)
        yield item