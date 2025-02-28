from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import Draw
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os


# 读取CSV文件中的数据
def read_csv_file(csv_file):
    df = pd.read_csv(csv_file)
    return df


# 提取指定列的SMILES字符串
def extract_smiles(df, column_prefix, num_smiles=10):
    smiles_list = []
    for i in range(num_smiles):
        col_name = f"{column_prefix}{i + 1}"
        smiles = df[col_name].dropna().values
        smiles_list.extend(smiles)
    return smiles_list


from tqdm import tqdm


# 计算静电指纹
def calculate_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]  # Remove None values

    morgan_fps = []
    # 使用 tqdm 显示进度条
    for mol in tqdm(mols, desc="Calculating fingerprints", unit="molecule"):
        morgan_fps.append(GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

    return np.array(morgan_fps)  # 将指纹转换为 numpy 数组


# 进行TSNE降维
def perform_tsne(fingerprints):
    tsne = TSNE(n_components=2)
    embedded = tsne.fit_transform(fingerprints)
    return embedded


# 绘制TSNE图
def plot_tsne(embedded, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], alpha=0.8)
    plt.title(title)
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.grid(True)
    plt.show()


import seaborn as sns



if __name__ == "__main__":
    # 读取数据
    csv_file = 'generated_molecules_v2.csv'
    df = read_csv_file(csv_file)

    # 提取SMILES字符串
    source_mol_smiles = extract_smiles(df, 'Source_Mol', num_smiles=1)
    generated_mol_smiles = extract_smiles(df, 'Predicted_smi_', num_smiles=10)

    # 计算静电指纹
    source_mol_fps = calculate_fingerprints(source_mol_smiles)
    generated_mol_fps = calculate_fingerprints(generated_mol_smiles)

    # 进行TSNE降维
    source_mol_embedded = perform_tsne(source_mol_fps)
    generated_mol_embedded = perform_tsne(generated_mol_fps)

    # 合并数据
    all_embedded = np.concatenate((generated_mol_embedded, source_mol_embedded), axis=0)  # 调换顺序
    all_labels = ['Generated Mol'] * len(generated_mol_embedded) + ['Source Mol'] * len(source_mol_embedded)  # 调换顺序

    # 绘制TSNE图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=all_embedded[:, 0], y=all_embedded[:, 1], hue=all_labels, palette={'Generated Mol': '#82B0D2', 'Source Mol': '#FA7F6F'}, alpha=0.8)
    # 设置横纵坐标轴标签，加粗并设置字体大小为16
    plt.xlabel('tSNE Dimension 1', fontweight='bold', fontsize=12)
    plt.ylabel('tSNE Dimension 2', fontweight='bold', fontsize=12)
    # 设置横纵坐标刻度的字体大小为12
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best')

    plt.show()
    plt.savefig('v2_gen_analy.png', dpi=1200, bbox_inches='tight')

