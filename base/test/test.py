

test_str = '''| 类型 | 名称 | 属性 | 属性值 | 关系 |
| --- | --- | --- | --- | --- |
| 实体 | 乔布斯 | 职业 | 创始人、CEO | - |
| 实体 | 苹果公司 | 创始人 | 乔布斯、沃兹尼亚克、韦恩·戴瓦尔德等人| - |
| 实体 | 苹果公司 | CEO | 乔布斯、蒂姆·库克等人| - |
| 概念/属性值 | 设计理念/哲学思想/经营理念/文化价值观等| 简洁、美观、易用、卓越的用户体验等| - |
| 概念/属性值 | 产品线/产品类型/产品特点等| iPhone、iPad、MacBook等产品线，高端设计和工艺，创新技术和功能等特点| - |
| 概念/属性值 | 市场份额/销售额/利润率等财务指标或市场表现指标| 全球最大的科技公司之一，市值超过1万亿美元，年度销售额和净利润均居于行业前列等指标| - |
| 关系类型1：拥有关系（一对多）| 苹果公司拥有多个产品线和产品类型，如iPhone、iPad和MacBook等。|
| 关系类型2：创始人关系（多对多）| 苹果公司的创始人包括乔布斯、沃兹尼亚克、韦恩·戴瓦尔德等人。|
| 关系类型3：CEO关系（多对一）| 苹果公司的CEO包括乔布斯、蒂姆·库克等人。|'''

if __name__ == "__main__":
    # split str test_str into lines
    lines = test_str.splitlines()
    # split each line with "|"
    lines = [line.split("|") for line in lines]
    print(lines)
