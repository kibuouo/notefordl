
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

# 3. 设置横坐标和柱子宽度
x = np.arange(len(categories))
width = 0.22

# 4. 创建画布
plt.figure(figsize=(12, 5))

# 5. 绘制柱状图
plt.bar(x - width, ours, width,
        label='Ours', color='#C9C9C9')

plt.bar(x, psd, width,
        label='PSD', color='#858585')

plt.bar(x + width, sqn, width,
        label='SQN', color='#D2D9E4')

# 6. 设置横纵坐标
plt.xticks(x, categories, rotation=45, ha='right')
plt.yticks(np.arange(0, 101, 25))
plt.ylim(0, 100)

# 7. 设置水平网格线
plt.grid(axis='y', color='#E0E0E0', linewidth=0.8)
plt.gca().set_axisbelow(True)

# 8. 隐藏边框和刻度线
ax = plt.gca()

for side in ['top', 'right', 'left']:
    ax.spines[side].set_visible(False)

ax.spines['bottom'].set_color('#D9D9D9')
ax.tick_params(axis='both', length=0)

# 9. 设置图例
plt.legend(
    loc='upper right',
    ncol=3,
    frameon=False
)

# 10. 自动调整布局
plt.tight_layout()

# 11. 保存图片
#plt.savefig('bar_01.png', dpi=600, bbox_inches='tight')
#plt.savefig('bar_01.pdf', bbox_inches='tight')

# 12. 显示图表
plt.show()
