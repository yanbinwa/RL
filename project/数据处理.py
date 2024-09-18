import math
import random

import pandas as pd

INPUT_FILE = 'file/Table.xlsx'
OUTPUT_FILE = 'file/result{}.xlsx'
SCORE_SAMPLE_ARRAY = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
SAMPLE_SIZE = 60


def load_and_cal_data():
    ret = []
    df = pd.read_excel(INPUT_FILE)
    # 遍历df中每行数据，并打印出来
    # 读取excel
    for index in df.index:
        var_T = convert_float(df.loc[index, 'T (AJCC 8th)'])
        var_N = convert_float(df.loc[index, 'N'])
        var_CEA = convert_float(df.loc[index, 'CEA(μg/l)'])
        var_CA19 = convert_float(df.loc[index, 'CA19-9(U/ml)'])
        var_AFP = convert_float(df.loc[index, 'AFP(μg/l)'])
        var_PA = convert_float(df.loc[index, 'PA（mg/l）'])
        var_ret = 10 * (1.2 + 0.186 * var_T + 0.656 * var_N + 0.147 * math.log(var_CEA) + 0.12 * math.log(var_CA19)
                        + 0.055 * math.log(var_AFP) - 0.187 * math.log(var_PA))
        # var_ret 保留三位小数
        var_ret = round(var_ret, 3)
        var_ST = df.loc[index, 'Survival time（M）']
        ret.append([var_ret, var_ST])
    return ret


def convert_float(value):
    # 判断value是否为str类型的值
    if isinstance(value, str):
        if value == '1a' or value == '1b':
            return float(1)
        # 将字符串转换为浮点数
        return float(value.replace("＞", "").replace(">", "").replace(" ", ""))
    else:
        # 如果value不是字符串，则直接返回
        return float(value)


def store_data(data_all, data_5, data_95):
    # 将ret 写入到excel中，第一例名字为预测值，第二列名字为实际值
    df_ret = pd.DataFrame(data_all, columns=['分数', '生存时间'])
    df_ret.to_excel(OUTPUT_FILE.format('all'), index=False)
    df_ret = pd.DataFrame(data_5, columns=['分数', '生存时间'])
    df_ret.to_excel(OUTPUT_FILE.format('05'), index=False)
    df_ret = pd.DataFrame(data_95, columns=['分数', '生存时间'])
    df_ret.to_excel(OUTPUT_FILE.format('95'), index=False)


def data_filter(datas):
    # datas存放着分数和生存时间的列表，分数按照每2分进行分数，求出区间内数据的期望和方差，遍历数据，如果与期望的差值大于方差的5倍，将该点去掉
    new_data = []
    for i in range(len(SCORE_SAMPLE_ARRAY) - 1):
        score_start = SCORE_SAMPLE_ARRAY[i]
        score_end = SCORE_SAMPLE_ARRAY[i + 1]
        data_list = [data for data in datas if score_start <= data[0] < score_end]
        if len(data_list) > 3:
            data_list = sorted(data_list, key=lambda x: x[1])
            mean = sum([data[1] for data in data_list]) / len(data_list)
            std = math.sqrt(sum([(data[1] - mean) ** 2 for data in data_list]) / len(data_list))
            for data in data_list:
                if abs(data[1] - mean) > 5 * std:
                    data_list.remove(data)
        new_data.extend(data_list)
    return new_data


def data_clean(datas, sample_size, range_x, range_y):
    new_data = []
    # datas里存储的是分数和生存时间的列表，分数按照每2分进行分数，如果数量大于10，将对应的生存时间保留95分位分组后，如果数量还大于10，通过随机抽样取出10个
    for i in range(len(SCORE_SAMPLE_ARRAY) - 1):
        score_start = SCORE_SAMPLE_ARRAY[i]
        score_end = SCORE_SAMPLE_ARRAY[i + 1]
        data_list = [data for data in datas if score_start <= data[0] < score_end]
        if len(data_list) > sample_size:
            data_list = sorted(data_list, key=lambda x: x[1])
            data_list = data_list[int(len(data_list) * range_x):int(len(data_list) * range_y)]
            if len(data_list) > sample_size:
                # python 从列表中随机抽取指定数量的元素
                data_list = random.sample(data_list, sample_size)
        new_data.extend(data_list)
    return new_data


if __name__ == '__main__':
    ret = load_and_cal_data()
    ret = data_filter(ret)
    data_all = data_clean(ret, SAMPLE_SIZE, 0.05, 0.95)
    data_5 = data_clean(ret, 10, 0.00, 0.05)
    data_95 = data_clean(ret, 10, 0.95, 1.0)
    store_data(data_all, data_5, data_95)


