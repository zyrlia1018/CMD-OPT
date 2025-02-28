from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def standardize_smiles(smiles, basic_clean=True, clear_charge=True, clear_fragment=True, canonicalize_tautomer=True,
                       isomeric=False):
    try:
        # Convert SMILES to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)

        # Basic cleaning: remove hydrogens and sanitize molecule
        if basic_clean:
            mol = rdMolStandardize.Cleanup(mol)

        # Fragmentation: keep only main fragment as the molecule
        if clear_fragment:
            mol = rdMolStandardize.FragmentParent(mol)

        # Remove charges from the molecule
        if clear_charge:
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)

        # Handle tautomer enumeration
        if canonicalize_tautomer:
            tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
            mol = tautomer_enumerator.Canonicalize(mol)

        # Convert the standardized molecule back to SMILES format
        standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)

    except Exception as e:
        # Print the error message along with the SMILES string causing the error
        print(f"Error occurred while standardizing SMILES: {e}\nSMILES: {smiles}")
        return None

    return standardized_smiles

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 设置 matplotlib 的默认字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# 从 SDF 文件中读取分子和对接得分信息
def read_sdf(filename):
    suppl = Chem.SDMolSupplier(filename)
    molecules = []
    for mol in suppl:
        if mol is not None:
            r_i_docking_score = float(mol.GetProp("r_i_docking_score"))
            molecules.append((mol, r_i_docking_score))
    return molecules

# 示例数据
sdf_file = "v3-docking.sdf"
molecules_with_scores = read_sdf(sdf_file)

# 计算分子的骨架指纹
def calculate_scaffold_fingerprint(mol):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = Chem.MolFromSmiles(standardize_smiles(Chem.MolToSmiles(scaffold)))
    scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(scaffold, 2, nBits=1024)
    return scaffold_fp

# 获取所有分子的骨架指纹
scaffold_fps = []
scores = []
for mol, score in molecules_with_scores:
    scaffold_fps.append(calculate_scaffold_fingerprint(mol))
    scores.append(score)

# 转换为 numpy 数组
scaffold_fps_array = np.array(scaffold_fps)

# 使用 T-SNE 将骨架指纹降维到二维空间
tsne = TSNE(n_components=2)
embedded = tsne.fit_transform(scaffold_fps_array)


n_clusters = 12
# 使用 K-means 聚类算法
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embedded)

# 找到每个聚类中打分最低的分子
lowest_score_molecules = []
for cluster_idx in range(n_clusters):
    cluster = [(mol, score, coord) for mol, label, score, coord in zip(molecules_with_scores, cluster_labels, scores, embedded) if label == cluster_idx]
    lowest_score_molecule = min(cluster, key=lambda x: x[1])
    lowest_score_molecules.append((cluster_idx, lowest_score_molecule))

# 绘制 t-SNE 聚类图，并根据类别上色
plt.figure(figsize=(8, 6))
plt.scatter(embedded[:, 0], embedded[:, 1], c=cluster_labels, cmap='Set3')

# 标记每个聚类中打分最低的分子
for cluster_idx, (mol, docking_score, (x, y)) in lowest_score_molecules:
    mol = Chem.MolFromSmiles(standardize_smiles(Chem.MolToSmiles(mol[0])))
    smiles = Chem.MolToSmiles(mol)
    plt.scatter(x, y, marker='*', s=75, color='black')
    print(f"Cluster {cluster_idx}: SMILES: {smiles}, Coordinates: ({x:.2f}, {y:.2f}), Docking Score: {docking_score:.2f}")

# 获取当前图的坐标轴对象
ax = plt.gca()

# 隐藏右边和上面的坐标轴线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置横纵坐标轴标签，加粗并设置字体大小为16
plt.xlabel('tSNE Dimension 1', fontweight='bold', fontsize=12)
plt.ylabel('tSNE Dimension 2', fontweight='bold', fontsize=12)
# 设置横纵坐标刻度的字体大小为12
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()
plt.savefig('v3-ctop1.png',dpi=1200, bbox_inches='tight')

