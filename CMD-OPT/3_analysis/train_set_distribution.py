import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 设置 matplotlib 的默认字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# 读取你的CSV文件
data = pd.read_csv("mmp_scaffold.csv")

# 设置Seaborn主题样式
sns.set_theme(style="ticks")

# 使用你的数据
x = data['2D_Similarity']
y = data['3D_Similarity']

# 画图
plot = sns.jointplot(x=x, y=y, kind="hex", color="#BEB8DC")

# 设置坐标范围
plot.ax_joint.set_xlim(0, 0.6)
plot.ax_joint.set_ylim(0, 1)

# 设置坐标轴标签的字体、字号和字重
plot.set_axis_labels('2D Similarity', '3D Similarity', fontsize=12, fontweight='bold')

# 显示图形
plt.show()
plt.savefig('train_set_Similarity.png',dpi=1200, bbox_inches='tight')