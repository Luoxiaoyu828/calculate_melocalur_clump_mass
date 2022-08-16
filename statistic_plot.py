import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def plot_n(data):
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
    plt.hist(data, bins=25, histtype='step', normed=False)
    plt.ylabel('N')
    plt.axvline(Median, label='Median={:1.1f} {}'.format(Median, uint), linestyle=':')
    plt.axvline(Peak, label='Peak $\sim${:1.1f} {}'.format(Peak, uint), linestyle='-.')
    plt.legend()

    plt.yticks(np.linspace(0, 0.0005, 6), np.linspace(0, 160, 6))
    plt.annotate(label, xy=(0.15, 0.8), xycoords='figure fraction')

    plt.savefig(eps_name + '.eps')
    plt.savefig(eps_name + '.png')
    plt.close()


if __name__ == '__main__':
    pass