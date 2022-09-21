import numpy as np
import pandas as pd
import pylab as p
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
import plot_distribution as plt_d


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
    # ax.set_xscale('log')
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


if __name__ == '__main__':
    # get_outcat_all()
    outcat_all1 = pd.read_csv('data/outcat_all_fwhm_dloc.csv', sep='\t')
    d_min = 1.95
    d_max = 2.2
    save_path = 'Distance_%.2fto%.2f_3' % (d_min, d_max)
    dis_key = 'Distance(kpc)_new'
    outcat_all = outcat_all1.loc[outcat_all1[dis_key] > d_min]
    outcat_all = outcat_all.loc[outcat_all[dis_key] < d_max]

    Distance = outcat_all[dis_key].values

    outcat_all = outcat_all.loc[outcat_all1['reff(pc)'] < 3]
    outcat_all = outcat_all.loc[outcat_all1['v_fwhm(km/s)'] < 3]

    os.makedirs(save_path, exist_ok=True)
    outcat_all = outcat_all[outcat_all['reff(pc)'].notna()]

    tex_max = outcat_all['Tex_max(K)'].values
    tex_mean = outcat_all['Tex_mean(K)'].values

    plot_item(tex_max, d_min, d_max, xlabel=r'Tex max $[K]$', fig_name=os.path.join(save_path, 'tex_max.png'), bins=50)
    plot_item(tex_mean, d_min, d_max, xlabel=r'Tex mean $[K]$', fig_name=os.path.join(save_path, 'tex_mean.png'), bins=50)

    tao_mean = outcat_all['tao_max(-)'].values
    plot_item(tao_mean, d_min, d_max, xlabel=r'$\tau$', rect=[0.1, 0.1, 0.8, 0.88], uint='', bins=50, xlims=[0, 2],
              fig_name=os.path.join(save_path, 'tao_mean.png'))

    v_fwhm = outcat_all['v_fwhm(km/s)'].values
    plot_item(v_fwhm, d_min, d_max, xlabel=r'$\Delta v_{FWHM}$ [km s$^{-1}$]', uint='km s$^{-1}$', rect=[0.08, 0.11, 0.85, 0.87],
              fig_name=os.path.join(save_path, 'v_fwhm.png'))
    nh2_mean = outcat_all['nh2_mean(10**21*cm-2)'].values
    plot_item(nh2_mean, d_min, d_max, xlabel=r'N(H$_2$) [$\times$10$^{21}$ cm$^{-2}$]', uint=r'$\times$10$^{21}$ cm$^{-2}$',
              rect=[0.08, 0.11, 0.85, 0.87], xlims=[0, 10], bins=100,
              fig_name=os.path.join(save_path, 'nh2_mean.png'))

    nh2_max = outcat_all['nh2_max(10**21*cm-2)'].values
    plot_item(nh2_max, d_min, d_max, xlabel=r'N(H$_2$) [$\times$10$^{21}$ cm$^{-2}$]', uint=r'$\times$10$^{21}$ cm$^{-2}$',
              rect=[0.09, 0.11, 0.84, 0.87], bins=200, xlims=[0, 100],
              fig_name=os.path.join(save_path, 'nh2_max.png'))

    reff = outcat_all['reff(pc)'].values
    reff[np.isnan(reff)] = 0
    a = outcat_all.loc[np.isnan(reff)]
    plot_item(reff, d_min, d_max, xlabel=r'Reff [pc]', uint=r'pc', rect=[0.09, 0.11, 0.84, 0.87], bins=50, xlims=[0, 10],
              fig_name=os.path.join(save_path, 'reff.png'))

    M_sum = outcat_all['M_sum(Msun)'].values / 2
    M_sum[np.isnan(M_sum)] = 0
    plot_item(np.log10(M_sum), d_min, d_max, xlabel=r'$M_{LTE}$ $[M_{\odot}]$', uint=r'$M_{\odot}$', rect=[0.09, 0.11, 0.84, 0.87],
              bins=50, fig_name=os.path.join(save_path, 'M_lte.png')
              )

    M_vir = outcat_all['M_vir(Msun)'].values / (2.3548 ** 2) / 209 * 232.5
    plot_item(np.log10(M_vir), d_min, d_max, xlabel=r'$M_{vir}$ $[M_{\odot}]$', uint=r'$M_{\odot}$', rect=[0.09, 0.11, 0.84, 0.87],
              bins=50, fig_name=os.path.join(save_path, 'M_vir.png'))

    vir_a = outcat_all['vir_a(-)'].values
    vir_a[np.isnan(vir_a)] = 0
    plot_item(vir_a, d_min, d_max, xlabel=r'vir_a [-]', uint=r'-', rect=[0.09, 0.11, 0.84, 0.87], bins=100,
              xlims=[0, 10], fig_name=os.path.join(save_path, 'vir_a.png'), fitting=False)

    Velocity = outcat_all['Velocity'].values
    plot_item(Velocity, d_min, d_max, xlabel=r'Velocity', uint=r'km s$^{-1}$', rect=[0.09, 0.11, 0.84, 0.87], bins=50,
             fitting=False, fig_name=os.path.join(save_path, 'Velocity.png'))

    Distance = outcat_all['Distance(kpc)'].values
    plot_item(Distance, d_min, d_max, xlabel=r'Distance [kpc]', uint=r'kpc', rect=[0.09, 0.11, 0.84, 0.87], bins=30,
              fitting=False, fig_name=os.path.join(save_path, 'Distance.png'))

    plot_dis_vel(Velocity, Distance, rect=[0.09, 0.11, 0.84, 0.87],
                 fig_name=os.path.join(save_path, 'Velocity_Distance.png'))

    z = outcat_all['z(pc)'].values
    plot_item(z, d_min, d_max, xlabel=r'z [pc]', uint=r'pc', rect=[0.09, 0.11, 0.84, 0.87], bins=50, fitting=False,
              fig_name=os.path.join(save_path, 'z.png'))

    plot_vir_a(M_sum, M_vir, rect=[0.11, 0.14, 0.84, 0.85], fig_name=os.path.join(save_path, 'M_lte_to_M_vir.png'))
    Galactic_Longitude = outcat_all['Galactic_Longitude'].values
    Galactic_Latitude = outcat_all['Galactic_Latitude'].values
    Velocity = outcat_all['Velocity'].values
    # plot_vir_a(M_sum, M_vir, rect=[0.11, 0.14, 0.84, 0.85])
    plt_d.make_plot_l_v(Galactic_Longitude, Velocity, fig_name=os.path.join(save_path, 'l_v.png'))
    plt_d.make_plot_l_b(Galactic_Longitude, Galactic_Latitude, Velocity, fig_name=os.path.join(save_path, 'l_b.png'))

    # dinance_new = pd.read_csv(r'ccd_loc/outcat.txt', delim_whitespace=True)
    # dis_new = dinance_new['(kpc1)'].values
    #
    # dis_old = outcat_all1['Distance(kpc)'].values
    # plt.figure()
    # plt.plot(dis_new, dis_old, '.')
    #
    # import astropy.io.fits as fits
    # nh2 = fits.getdata(r'check_CDC/NH2_13CO.fits')
    #
