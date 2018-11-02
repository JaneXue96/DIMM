import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.switch_backend('agg')


def draw_weights(index_w, medicine_w):
    f, (ax1, ax2) = plt.subplots(figsize=(48, 27), ncols=2)
    # cmap用cubehelix map颜色
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(index_w, linewidths=0.05, ax=ax1, vmax=np.max(index_w), vmin=np.min(index_w), cmap='rainbow')
    ax1.set_title('index weight map')
    ax1.set_xlabel('output')
    ax1.set_xticklabels([])  # 设置x轴图例为空值
    ax1.set_ylabel('input')

    # cmap用matplotlib colormap
    sns.heatmap(medicine_w, linewidths=0.05, ax=ax2, vmax=np.max(medicine_w), vmin=np.min(medicine_w), cmap='rainbow')
    # rainbow为 matplotlib 的colormap名称
    ax2.set_title('medicine weight colormap')
    ax2.set_xlabel('output')
    ax2.set_ylabel('input')

    plt.savefig('./pics/final/' + t + '_weights.jpg', format='jpg')


# def draw_pca():


if __name__ == '__main__':
    tasks = ['5849_720', '25000_720', '41401_720', '4019_720']
    # tasks = ['208']
    for t in tasks:
        index = np.loadtxt(os.path.join('data/results/', t, 'DIMM', t + '_index_W.txt'), delimiter=',')
        medicine = np.loadtxt(os.path.join('data/results/', t, 'DIMM', t + '_medicine_W.txt'), delimiter=',')
        draw_weights(index, medicine)
