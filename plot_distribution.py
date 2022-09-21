import numpy as np
import pandas as pd
import pylab as p
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


def m_vir(x, y, pixs, d):
    '''
    calculate the viral mass

     Parameters:
     ----------
     input:
    x: vlsr
    y: T_A
    d: distance
    pixs: number of pixels of cloud core area

    return:
     M_vir: viral mass
    '''
    gauss = lambda x, a, b, c: a * np.exp(-(x - b)**2 / c**2)
    popt, pcov = curve_fit(gauss, x, y, p0 = [0.7, 5, 5])
    FWHM = 2.238 * popt[2]
    A = np.sum(pixs) * (30 / 3600 * np.pi / 180)**2
    Reff = d * np.sqrt(A / np.pi - (50 / 3600 * np.pi / 180)**2 / 4)
    M_vir = 210 * Reff * FWHM**2
    return M_vir


def get_outcat_all():
    R2_data_params_path = r'F:\Parameter_reduction\calc_mass\R2_data_params_fwhm'
    file_list = [r'F:\Parameter_reduction\calc_mass\R2_data_params_fwhm\%s\MWISP_outcat_d_physical_again.csv' % item
                 for item in os.listdir(R2_data_params_path)]
    outcat_all = pd.DataFrame([])
    for item in file_list:

        print(item)
        outcat_item = pd.read_csv(item, sep='\t')
        outcat_all = pd.concat([outcat_all, outcat_item])
    outcat_all.to_csv('outcat_all_fwhm.csv', sep='\t', index=False)


def plot_n(data):
    """
    绘制数密度的统计直方图
    """
    bins = 25
    uint = 'cm${}^{-3}$'
    label = ''
    xlabel = r'average density (cm${}^{-3}$)'
    eps_name = 'n_cm-3'

    data_len = data.shape[0]
    fit_par_lognorm = scipy.stats.lognorm.fit(data, floc=0)
    lognorm_dist_fitted = scipy.stats.lognorm(*fit_par_lognorm)
    t = np.linspace(np.min(data), np.max(data), 100)
    data_hist = np.histogram(data, bins)
    Peak = data_hist[1][data_hist[0].argmax()] + 0.5 * (data_hist[1][1] - data_hist[1][0])
    data_fit = lognorm_dist_fitted.pdf(np.linspace(np.min(data), np.max(data), bins))
    correlation = scipy.stats.spearmanr(data_hist[0] / data_len, data_fit)[0]
    Median = np.median(data)

    plt.figure(figsize=(6, 4))
    plt.plot(t, lognorm_dist_fitted.pdf(t), lw=2, color='r', ls='-',
             label='Lognormal fit: $R^2$={}'.format(str(correlation)[:4]))

    sns.distplot(data, bins=bins, hist=True, kde=False, norm_hist=True, rug=False, vertical=False,
                 label='$\sigma$ = {: .1f}'.format(lognorm_dist_fitted.std()),
                 axlabel=xlabel, hist_kws={'color': 'y', 'edgecolor': 'k', 'histtype': 'step'})
    plt.hist(data, bins=25, histtype='step')
    plt.ylabel('N')
    plt.axvline(Median, label='Median={:1.1f} {}'.format(Median, uint), linestyle=':')
    plt.axvline(Peak, label='Peak $\sim${:1.1f} {}'.format(Peak, uint), linestyle='-.')
    plt.legend()

    plt.yticks(np.linspace(0, 0.0005, 6), np.linspace(0, 160, 6))
    plt.annotate(label, xy=(0.15, 0.8), xycoords='figure fraction')

    plt.savefig(eps_name + '.eps')
    plt.savefig(eps_name + '.png')
    plt.close()


def plot_item(data, d_min, d_max, fig_name=None, xlabel=r'Tex $[K]$', rect=None, bins=25, uint='K', xlims=None, fitting=True):
    """
    绘制参数的统计直方图
    """

    if rect is None:
        rect = [0.08, 0.1, 0.85, 0.87]
    label = ''
    # rect[left, bottom, width, height]
    data_len = data.shape[0]
    t = np.linspace(np.min(data), np.max(data), 100)
    data_hist = np.histogram(data, bins)

    Peak = data_hist[1][data_hist[0].argmax()] + 0.5 * (data_hist[1][1] - data_hist[1][0])
    Median = np.median(data)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes(rect)
    ax.hist(data, bins=bins, histtype='step')
    ax.set_ylabel('Number')
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, ax.get_yticks().max())
    if xlims is None:
        xlims = [ax.get_xticks().min(), ax.get_xticks().max()]
    ax.set_xlim(xlims)
    lns2 = ax.axvline(Median, label='Median={:1.2f} {}'.format(Median, uint), linestyle=':')
    lns3 = ax.axvline(Peak, label='Peak $\sim${:1.2f} {}'.format(Peak, uint), linestyle='-.')
    ax1 = ax.twinx()
    lns0 = ax1.plot([], [], '.', label='distance=[%.2fpc, %.2fpc]' % (d_min, d_max))
    if fitting:
        fit_par_lognorm = scipy.stats.lognorm.fit(data)
        lognorm_dist_fitted = scipy.stats.lognorm(*fit_par_lognorm)
        data_fit = lognorm_dist_fitted.pdf(np.linspace(np.min(data), np.max(data), bins))
        correlation = scipy.stats.spearmanr(data_hist[0] / data_len, data_fit)[0]
        sigma = lognorm_dist_fitted.std()

        lns1 = ax1.plot(t, lognorm_dist_fitted.pdf(t), lw=2, color='r', ls='-',
                        label='Lognormal fit\n$R^2$={:1.2f}\n$\sigma$ = {: .2f}'.format(correlation, sigma))
        ax1.set_ylabel('Frequency [%]')

        # added these three lines
        lns = lns0 + lns1 + [lns2] + [lns3]
    else:
        lns = lns0 + [lns2] + [lns3]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    ax1.set_ylim(0, ax1.get_yticks().max())
    y_ticks = ['%d' % item for item in ax1.get_yticks() * 100]
    ylabels = ax1.get_yticks().tolist()
    ax1.yaxis.set_major_locator(mticker.FixedLocator(ylabels))  # 定位到散点图的x轴
    ax1.set_yticklabels(y_ticks)  # 使用列表推导式循环将刻度转换成浮点数

    plt.annotate(label, xy=(0.15, 0.3), xycoords='figure fraction')
    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()


def plot_fwhm(data, fig_name=None, xlabel=r'Tex $[K]$'):
    """
    绘制数密度的统计直方图
    """
    bins = 25
    uint = 'K'
    label = ''

    [left, bottom, width, height] = [0.08, 0.1, 0.85, 0.87]

    data_len = data.shape[0]
    fit_par_lognorm = scipy.stats.lognorm.fit(data)
    lognorm_dist_fitted = scipy.stats.lognorm(*fit_par_lognorm)
    t = np.linspace(np.min(data), np.max(data), 100)
    data_hist = np.histogram(data, bins)
    Peak = data_hist[1][data_hist[0].argmax()] + 0.5 * (data_hist[1][1] - data_hist[1][0])
    data_fit = lognorm_dist_fitted.pdf(np.linspace(np.min(data), np.max(data), bins))
    correlation = scipy.stats.spearmanr(data_hist[0] / data_len, data_fit)[0]
    Median = np.median(data)
    sigma = lognorm_dist_fitted.std()

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([left, bottom, width, height])
    ax.hist(data, bins=bins, histtype='step')
    ax.set_ylabel('Number')
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, ax.get_yticks().max())
    lns2 = ax.axvline(Median, label='Median={:1.1f} {}'.format(Median, uint), linestyle=':')
    lns3 = ax.axvline(Peak, label='Peak $\sim${:1.1f} {}'.format(Peak, uint), linestyle='-.')
    ax1 = ax.twinx()
    lns1 = ax1.plot(t, lognorm_dist_fitted.pdf(t), lw=2, color='r', ls='-',
                    label='Lognormal fit\n$R^2$={:1.2f}\n$\sigma$ = {: .1f}'.format(correlation, sigma))
    ax1.set_ylabel('Frequency [%]')

    # added these three lines
    lns = lns1 + [lns2] + [lns3]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best', alpha=0.2)
    ax1.set_ylim(0, ax1.get_yticks().max())
    y_ticks = ['%d' % item for item in ax1.get_yticks() * 100]
    ax1.set_yticklabels(y_ticks)

    plt.annotate(label, xy=(0.15, 0.3), xycoords='figure fraction')
    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()


def plot_vir_a(M_sum, M_vir, fig_name=None, rect=None, ):
    x = np.arange(10, 10000, 1)
    y = 2 * x
    if rect is None:
        rect = [0.11, 0.14, 0.84, 0.85]
    fontsize = 15
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes(rect)
    ax.plot(x, y, 'r--', zorder=10)
    ax.plot(x, x, 'b--', zorder=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # M_sum = np.log10(M_sum)
    # M_vir = np.log10(M_vir)
    ax.scatter(M_sum, M_vir)
    ax.set_ylabel(r'$M_{vir}$ $[M_{\odot}]$',  fontsize=fontsize)
    ax.set_xlabel(r'$M_{LTE}$ $[M_{\odot}]$',  fontsize=fontsize)
    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()
    # ylims = [0, ax.get_yticks().max()]
    # ax.set_ylim(ylims)
    # ax.set_ylabel(r'$M_{vir}$ $[M_{\bigodot}]$')
    # ax.set_xlabel(r'$M_{LTE}$ $[M_{\bigodot}]$')


def plot_dis_vel(Velocity, Distance, rect=None, fig_name=None):
    if rect is None:
        rect = [0.08, 0.1, 0.85, 0.87]
    # rect[left, bottom, width, height]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes(rect)
    ax.plot(Velocity, Distance, '.')

    ax.set_ylabel('Distance [pc]')
    ax.set_xlabel('Velocity [km s$^{-1}$]')
    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()


def make_plot_l_v(Galactic_Longitude, Velocity, rect=None, fig_name=None):
    if rect is None:
        rect = [0.1, 0.1, 0.87, 0.87]

    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_axes(rect)
    ax.plot(Galactic_Longitude, Velocity, '.')
    ax.set_xlabel('Galactic Longitude [degree]', fontsize=12)
    ax.set_ylabel('Velocity [km s$^{-1}$]', fontsize=12)
    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()


def make_plot_l_b(Galactic_Longitude, Galactic_Latitude, Velocity, rect=None, fig_name=None):
    if rect is None:
        rect = [0.1, 0.1, 0.8, 0.81]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes(rect)
    plot0 = ax.scatter(Galactic_Longitude, Galactic_Latitude, color='coral')
    ax.set_xlabel('Galactic Longitude [degree]', fontsize=12)
    ax.set_ylabel('Galactic Latitude [degree]', fontsize=12)
    ax.invert_xaxis()

    pos = ax.get_position()
    pad = 0.05
    width = 0.02
    axes1 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])

    # cbar = fig.colorbar(plot0, cax=axes1)
    # cbar.set_label('K m s${}^{-1}$')
    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # get_outcat_all()
    outcat_all1 = pd.read_csv('outcat_all_fwhm.csv', sep='\t')
    d_min = 2.5
    d_max = 3.5
    outcat_all = outcat_all1.loc[outcat_all1['Distance(kpc)'] > d_min]
    outcat_all = outcat_all.loc[outcat_all['Distance(kpc)'] < d_max]
    Galactic_Longitude = outcat_all1['Galactic_Longitude'].values
    Galactic_Latitude = outcat_all1['Galactic_Latitude'].values
    Velocity = outcat_all1['Velocity'].values

    # make_plot_l_v(Galactic_Longitude, Velocity)
    Distance = outcat_all['Distance(kpc)'].values
    make_plot_l_b(Galactic_Longitude, Galactic_Latitude, Velocity, rect=None, fig_name=None)
