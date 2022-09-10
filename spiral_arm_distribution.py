from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from skimage import io, color


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
# plt.style.use('default')


def cut_and_plot(x1=-16, x2=16, y1=-16, y2=16):
    path = 'spiral_arm/Milky_way_with_armname.jpg'
    img = io.imread(path)
    # 图片转成灰度图
    img_gray = color.rgb2gray(img)
    # 黑白色反转:
    img_gray = 1 - img_gray
    '''
    x1,x2,y1,y2 分别是x轴和y轴的取值范围, 单位是kpc.
    x轴和y轴的最大取值范围都是[-24,24]
    '''
    # 裁剪图片
    pixel_x1 = int((1 - (y2 - y1) / 48) / 2 * img_gray.shape[1])
    pixel_x2 = int(img_gray.shape[1] - pixel_x1)
    pixel_y1 = int((1 - (x2 - x1) / 48) / 2 * img_gray.shape[0])
    pixel_y2 = int(img_gray.shape[0] - pixel_y1)

    img_gray_cut = img_gray[pixel_x1:pixel_x2 + 1, pixel_y1:pixel_y2 + 1]

    # 画图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_gray_cut, extent=[x1, x2, y1, y2], cmap='gray')
    ax.scatter(x_1, -1 * (y_1 - 8.15), c='red', marker='.', s=5, alpha=0.6, zorder=5)
    # 标出太阳位置
    ax.scatter(0, 8.15, marker='$\odot$', s=40, c='lime', alpha=0.8, zorder=15)  # zorder控制图层

    ax.set_xlabel('x (kpc)', fontsize=12)
    ax.set_ylabel('y (kpc)', fontsize=12)
    ax.set_xticks(np.arange(x1 + abs(x1) % 5, x2 + 1, 5))
    ax.set_yticks(np.arange(y1 + abs(y1) % 5, y2 + 1, 5))
    ax.grid(alpha=0.1, c='gray', ls='--')
    ax.minorticks_on()
    # ax.set_axis_off()

    # 画极坐标
    polar_x = np.linspace(x1, x2, 10)
    for i in range(12):
        if (i != 6) & (i != 12):
            polar_y = np.tan(np.pi / 12 * i) * polar_x + 8.15
            plt.plot(polar_x, polar_y, c='gray', linestyle='--', alpha=0.2)
        else:
            plt.axvline(x=0, c='gray', linestyle='--', alpha=0.4)
            plt.axhline(y=0, c='gray', linestyle='--', alpha=0.4)
            plt.axhline(y=8.15, c='gray', linestyle='--', alpha=0.4)

    # 画同心圆,标示距离
    def plot_circle(r):
        c = plt.Circle((0, 8.15), radius=r, color='gray', alpha=0.2, fill=False, linestyle='--')
        plt.gca().add_artist(c)

    for i in np.arange(2.5, y2 + 8.15, 2.5):
        plot_circle(i)

    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)


# -----------函数：添加比例尺--------------
def add_scalebar(lon, lat, length):
    plt.hlines(y=lat, xmin=lon, xmax=lon + length, colors="black", ls="-", lw=1, label='%d kpc' % (length))
    plt.vlines(x=lon, ymin=lat - 0.1, ymax=lat + 0.4, colors="black", ls="-", lw=1)
    plt.vlines(x=lon + length, ymin=lat - 0.1, ymax=lat + 0.4, colors="black", ls="-", lw=1)
    plt.text(lon + length / 2, lat + 0.5, '2.5 kpc', horizontalalignment='center')


if __name__ == '__main__':
    df_raw = pd.read_csv(r'outcat_all_d.csv', sep='\t')
    print(len(df_raw))

    Galactic_Longitude = df_raw.Galactic_Longitude
    Galactic_Latitude = df_raw.Galactic_Latitude
    gl_rad_1 = np.deg2rad(Galactic_Longitude)
    gb_rad_1 = np.deg2rad(Galactic_Latitude)
    d1 = df_raw['Distance(kpc)'].values

    # 获得平面坐标系(银道面)的坐标
    dd1 = d1 * np.cos(gb_rad_1)  # dd 是 d 在银道面上的投影
    x_1 = dd1 * np.sin(gl_rad_1)
    y_1 = dd1 * np.cos(gl_rad_1)
    cut_and_plot()
    add_scalebar(-14, -14, 2.5)

    plt.tight_layout()
    plt.savefig('spiral_arm/spatial_distribution_cutted.png', dpi=200)
    # plt.savefig('spiral_arm/spatial_distribution_cutted.pdf', dpi=200)
    # plt.savefig('spiral_arm/spatial_distribution_cutted.jpg', dpi=200)
    # plt.savefig('spiral_arm/spatial_distribution_cutted.eps', dpi=200)
    # plt.show()


    # cut_and_plot(x1=-10, x2=10, y1=-12, y2=10)
    # add_scalebar(-8, -8, 2.5)

    # plt.tight_layout()
    # # plt.savefig('./spatial_distribution_cutted.png', dpi=200)
    # # plt.savefig('./spatial_distribution_cutted.pdf', dpi=200)
    # # plt.savefig('./spatial_distribution_cutted.jpg', dpi=200)
    # # plt.savefig('./spatial_distribution_cutted.eps', dpi=200)
    # plt.show()