import csv


def save_pred(preds, file):
    '''将预测结果保存到指定文件中'''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)  # 创建一个CSV写入器对象
        writer.writerow(['id', 'tested_positive'])  # 写入CSV文件的表头
        for i, p in enumerate(preds):
            writer.writerow([i, p])  # 写入每个预测结果的行，包括id和预测值
