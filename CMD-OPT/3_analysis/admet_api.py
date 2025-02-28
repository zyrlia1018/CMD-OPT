import json
import time
import requests
import pandas as pd

baseUrl = 'https://admetlab3.scbdd.com'

def transform(data):
    resultList = []
    for mol in data['data']:
        if not mol:
            # 如果分子数据为空，说明 SMILES 无效
            tmp = {'smiles': 'Invalid SMILES'}
        else:
            # 如果分子数据不为空，则构建包含分子信息的临时字典
            tmp = {'smiles': mol['smiles']}  # 添加 SMILES 到临时字典

            # 遍历分子数据中的每个属性（除了 SMILES）
            for key, value in mol.items():
                if key != 'smiles':  # 跳过 SMILES 属性，因为已经单独处理了
                    # 将分子的其他属性添加到临时字典
                    tmp[key] = value

        # 将临时字典添加到结果列表
        resultList.append(tmp)

    # 将结果列表转换为 Pandas DataFrame
    result_df = pd.DataFrame(resultList)

    return result_df


def divide_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



if __name__ == '__main__':
    api = '/api/admet'
    url = baseUrl + api
    param = {
        'SMILES': []
    }
    n = 500
    # 读取 CSV 文件中的 SMILES 数据
    csv_file = 'src-train-nospace.csv'
    df = pd.read_csv(csv_file)

    # 提取 SMILES 列的数据
    smiles_list = df['SMILES'].tolist()

    for _, sublist in enumerate(divide_list(smiles_list, n)):
        param['SMILES'] = sublist
        print(_) 
        response = requests.post(url, json=param)

        if response.status_code == 200:  # If access is successful
            data = response.json()['data']
            # transform to csv file
            result = transform(data)
            result.to_csv('result' + str(_) + '.csv', index=False)
        time.sleep(1.0)
