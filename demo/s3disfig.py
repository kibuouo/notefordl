
import matplotlib.pyplot as plt
import numpy as np

# 1. 类别名称
categories = [
    'ceiling', 'floor', 'wall', 'beam',
    'column', 'window', 'door', 'table',
    'chair', 'sofa', 'bookcase', 'board',
    'clutter', 'mIoU'
]

# 2. 实验数据（示例）
ours = [77, 98, 72, 0, 17, 42, 41,
        78, 84, 62, 54, 60, 40, 56]

psd = [90, 95, 75, 0, 19, 51, 18,
       60, 69, 32, 61, 50, 42, 51]

sqn = [87, 94, 72, 0, 10, 32, 35,
       61, 75, 41, 63, 37, 44, 50]

# 3. 创建画布
plt.figure(figsize=(12, 5))

# 4. 绘制折线
plt.plot(
    categories, ours,
    color='#4472C4',
    marker='s',
    markersize=4,
    linewidth=1.8,
    label='Ours'
)

plt.plot(
    categories, psd,
    color='#ED7D31',
    marker='D',
    markersize=4,
    linewidth=1.8,
    label='PSD'
)

plt.plot(
    categories, sqn,
    color='#FFC000',
    marker='o',
    markersize=4,
    linewidth=1.8,
    label='SQN'
)

# 5. 设置横纵坐标
plt.xticks(rotation=45, ha='right')
plt.yticks(np.arange(0, 101, 20))
plt.ylim(0, 100)

# 6. 设置水平网格线
plt.grid(axis='y', color='#E0E0E0', linewidth=0.8)
plt.gca().set_axisbelow(True)

# 7. 隐藏边框和刻度线
ax = plt.gca()

for side in ['top', 'right', 'left']:
    ax.spines[side].set_visible(False)

ax.spines['bottom'].set_color('#D9D9D9')
ax.tick_params(axis='both', length=0)

# 8. 设置图例
plt.legend(
    loc='upper right',
    ncol=3,
    frameon=False
)

# 9. 自动调整布局
plt.tight_layout()

# 10. 保存图片
#plt.savefig('line_01.png', dpi=600, bbox_inches='tight')
#plt.savefig('line_01.pdf', bbox_inches='tight')

# 11. 显示图表
plt.show()
