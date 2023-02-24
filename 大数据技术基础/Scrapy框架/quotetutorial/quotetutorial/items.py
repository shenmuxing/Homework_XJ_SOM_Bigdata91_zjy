# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class QuotetutorialItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    text = scrapy.Field()
    author = scrapy.Field()
    tags = scrapy.Field()

class AuthorItem(scrapy.Item):
    """作者的细节的Item，
    内含borninformation,description"""
    author=scrapy.Field()
    born_date=scrapy.Field()
    born_location=scrapy.Field()
    description=scrapy.Field()