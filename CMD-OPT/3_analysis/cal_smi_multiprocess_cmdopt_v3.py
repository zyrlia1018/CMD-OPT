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
import multiprocessing
from functools import partial
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolAlign, rdShapeHelpers
from tqdm import tqdm
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig

# from espsim import EmbedAlignConstrainedScore, EmbedAlignScore, ConstrainedEmbedMultipleConfs, GetEspSim, GetShapeSim


# Set up features to use in FeatureMap
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
fdef = AllChem.BuildFeatureFactory(fdefName)

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable',
        'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')


def get_FeatureMapScore(query_mol, ref_mol):
    featLists = []
    for m in [query_mol, ref_mol]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're interested in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep])
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode = FeatMaps.FeatMapScoreMode.Best
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))

    return fm_score


def calc_SC_RDKit_score(query_mol, ref_mol):
    fm_score = get_FeatureMapScore(query_mol, ref_mol)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(query_mol, ref_mol,
                                                     allowReordering=False)
    SC_RDKit_score = 0.5 * fm_score + 0.5 * (1 - protrude_dist)

    return SC_RDKit_score


def calc_2D_similarity(mol1, mol2):
    if mol1 is None or mol2 is None:
        return -1.0
    try:
        f1 = AllChem.GetMorganFingerprint(mol1, 3)
        f2 = AllChem.GetMorganFingerprint(mol2, 3)
        return DataStructs.TanimotoSimilarity(f1, f2)
    except Exception:
        return -1.0


def calc_3D_similarity(ref, gen):
    if ref is None or gen is None:
        return -1.0
    try:
        ref = Chem.AddHs(ref)
        Chem.AllChem.EmbedMolecule(ref)
        Chem.AllChem.UFFOptimizeMolecule(ref)

        gen = Chem.AddHs(gen)
        Chem.AllChem.EmbedMolecule(gen)
        Chem.AllChem.UFFOptimizeMolecule(gen)

        pyO3A = rdMolAlign.GetO3A(gen, ref).Align()
        return calc_SC_RDKit_score(gen, ref)
    except Exception:
        return -1.0


def prepare_conformer(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        Chem.AllChem.EmbedMolecule(mol)
        Chem.AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        return None
    return mol


def calc_3d_score(ref_mol, gen_mol):
    try:
        pyO3A = rdMolAlign.GetO3A(gen_mol, ref_mol).Align()
        return calc_SC_RDKit_score(gen_mol, ref_mol)
    except Exception:
        return -1.0


def standardize_smiles(smiles):
    """
    Standardize the given SMILES string by converting it to an RDKit Mol object,
    sanitizing the molecule, and converting it back to a standardized SMILES string.
    If the input SMILES is invalid or cannot be processed, return None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.SanitizeMol(mol)
        standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        return standardized_smiles
    else:
        return None


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

# 计算静电指纹
def calculate_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]  # Remove None values
    morgan_fps = [GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
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


if __name__ == "__main__":
    # 读取数据
    csv_file = 'generated_molecules_v3.csv'
    df = read_csv_file(csv_file)

    # 创建输出文件
    output_file = 'similarity_results.csv'
    with open(output_file, 'w') as file:
        file.write("Hit_ID,Source_Mol,Generated_Mol,Similarity_2D,Similarity_3D\n")  # 写入列名

    # 对于每一行数据，计算源分子与每个生成分子的相似性，并写入CSV文件
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        source_mol_smiles = row['Source_Mol1']
        generated_mol_smiles = [row[f'Predicted_smi_{i + 1}'] for i in range(10)]

        # 准备source和generated的conformer
        source_mol = prepare_conformer(source_mol_smiles)
        generated_mol = [prepare_conformer(smiles) for smiles in generated_mol_smiles]

        # 计算每个生成分子与源分子的相似性
        for gen_mol in generated_mols:
            similarity_2d = calc_2D_similarity(source_mol, gen_mol)
            similarity_3d = calc_3D_similarity(source_mol, gen_mol)

            # 将结果写入CSV文件
            with open(output_file, 'a') as file:
                file.write(f"{idx},{Chem.MolToSmiles(source_mol)},{Chem.MolToSmiles(gen_mol)},{similarity_2d},{similarity_3d}\n")

