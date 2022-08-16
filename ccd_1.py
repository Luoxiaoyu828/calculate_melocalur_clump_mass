from astropy.io import fits
import astropy.wcs as WCS
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename
import astropy.constants as constants
import astropy.units as u
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from scipy.ndimage import binary_dilation
from scipy.sparse import bsr_array
import tqdm
from fit_clump_function.multi_gauss_fitting_new import get_multi_gauss_func_by_params as get_func
from DensityClust.clustring_subfunc import get_xyz

warnings.filterwarnings("ignore")
"""
计算M16天区分子云核物理参数，并对其做统计分析！
2020/04/18
"""


class CO_Data:
    def __init__(self, co_12_path=None, co_13_path=None, co_18_path=None):
        self.co_13_path = co_13_path
        self.co_12_path = co_12_path
        self.co_18_path = co_18_path

        self.co_13_data, self.wcs13 = self.dd_header(self.co_13_path)
        self.co_12_data, self.wcs12 = self.dd_header(self.co_12_path)
        self.co_18_data, self.wcs18 = self.dd_header(self.co_18_path)

    def dd_header(self, path):
        """
        :param data_header: 数据原始头文件
        :return: 经过处理后的wcs坐标系下的信息
        """
        if path is None:
            return None, None

        filename = get_pkg_data_filename(path)
        hdu = fits.open(filename)[0]
        data = hdu.data
        data_header = hdu.header

        keys = data_header.keys()
        key = [k for k in keys if k.endswith('4')]
        [data_header.remove(k) for k in key]
        data_header.remove('VELREF')

        wcs = WCS.WCS(data_header)

        return data, wcs


class COData:

    def __init__(self, co_data_path=None, co='13'):
        k = constants.k_B
        h = constants.h
        v0_12 = 115.271204 * u.GHz
        v0_13 = 110.201353 * u.GHz
        v0_18 = 109.782183 * u.GHz
        self.co_data_path = co_data_path
        self.data, self.wcs = self.dd_header(self.co_data_path)
        self.co = co
        if co == '12':
            self.v0 = v0_12
            self.miu_mol = 28    # 分子量
            self.delta_v = 0.1666  # 速度分辨率
            self.ratio = None
        elif co == '13':
            self.v0 = v0_13
            self.miu_mol = 29
            self.delta_v = 0.166
            self.ratio = 3.9 * 10 ** 5   # N(H2) = ratio * N(CO)
        elif co == '18':
            self.v0 = v0_18
            self.miu_mol = 30
            self.delta_v = 0.1666
            self.ratio = 3.0 * 10 ** 6
        else:
            raise KeyError('v0 mast be 12 or 13 or 18!!')
        self.T0 = h * self.v0 / k

    def dd_header(self, path):
        """
        :param data_header: 数据原始头文件
        :return: 经过处理后的wcs坐标系下的信息
        """
        if path is None:
            return None, None

        filename = get_pkg_data_filename(path)
        hdu = fits.open(filename)[0]
        data = hdu.data
        data_header = hdu.header

        keys = data_header.keys()
        key = [k for k in keys if k.endswith('4')]
        [data_header.remove(k) for k in key]
        data_header.remove('VELREF')

        wcs = WCS.WCS(data_header)

        return data, wcs


def ln(x):
    return math.log(x, math.e)


def planck_function(T, miu):
    """
     k = 1.380649 * 10 ** (-23)# 玻尔兹曼常数, 单位: J/K
     h = 6.62607015 * 10 ** (-34)# h: 普朗克常数，单位: J*s
    """
    h = constants.h
    k = constants.k_B
    intensity = h * miu / k / (math.exp(h * miu / (k * T)) - 1)
    return intensity


class Calculate_Parameters:
    """
    计算云核物理参数
    # k = 1.380649 * 10 ** (-23) # 玻尔兹曼常数, 单位: J/K
    # h = 6.62607015 * 10 ** (-34)# h: 普朗克常数，单位: J*s
     # mh = 1.66 * 10 ** (-27 # 氢原子的质量,单位:kg)
      # 1GHz = <Quantity 1.e+09 1 / s>  # v0: 谱线频率，单位: Hz
    """

    k = constants.k_B
    h = constants.h
    mh = constants.m_p
    miu_h = 2.8
    T_bg = 2.73 * u.K

    def __init__(self, record, pdf_path, co_data12=COData(co='12'), co_data=COData(co='13')):
        self.mask = None
        self.pdf_path = pdf_path
        os.makedirs(self.pdf_path, exist_ok=True)
        self.core_num = record[0]
        self.co_data = co_data

        self.co_data12 = co_data12
        self.record = record
        self.pdf = PdfPages('{}/{}.pdf'.format(self.pdf_path, str(self.core_num).zfill(3)))

        self.data12 = self.get_data_cube(co_data12)
        self.data, self.local_wcs = self.get_data_cube(co_data)
        self.XX = get_xyz(self.data)

        self.tex, self.tr_12 = self.calculate_tex()
        self.v_fwhm, self.tr = self.calculate_vfwhm()
        self.tao = self.calculate_tao()

        self.vth = self.calculate_vth()

        self.vnth = self.calculate_vnth()

        self.n_co, self.n_h2 = self.get_n_h2()
        #
        self.mass_, self.mass_sum = self.calculate_m()
        #
        self.reff = self.calculate_reff_GC()
        #
        self.mass_vir = self.calcultate_M_vir()
        #
        self.vir_a = self.calculate_vir()

        # self.p_th, self.p_nth, self.p_tot, self.n = self.calculate_p_internal()
        # self.p_cloud = self.calculate_p_external()

        self.density_s = self.calculate_density_s()

        self.pdf.close()

    def get_data_cube(self, co_data):
        """
        get the clump data as described in the outcat table
        :return: data_cube
        """
        wcs = co_data.wcs
        data = co_data.data
        # outcat table core center
        cen_pt = self.record[['Galactic_Longitude', 'Galactic_Latitude', 'Velocity']].values
        cen_pt_pix = np.array(wcs.all_world2pix(cen_pt[0], cen_pt[1], cen_pt[2] * 1000, 0)).T
        size = self.record[['Size_major', 'Size_minor', 'Size_velocity']].values
        delta = np.array(wcs.pixel_scale_matrix.sum(axis=0)) * np.array([3600, 3600, 1 / 1000])
        fwhm = np.abs(size / delta)
        range_down = np.floor(cen_pt_pix - fwhm).astype(np.int32)
        range_up = np.ceil(cen_pt_pix + fwhm).astype(np.int32)

        data_cube = data[range_down[2]:range_up[2], range_down[1]:range_up[1], range_down[0]:range_up[0]]
        local_wcs = wcs[range_down[2]:range_up[2], range_down[1]:range_up[1], range_down[0]:range_up[0]]

        return data_cube, local_wcs

    def calculate_tex(self):
        """
        ok, 验证通过，2020/08/12
        计算公式：w37  公式(3.3)
        计算激发温度：通过12CO数据计算, each spectral-line calculates an excitation temperature
        第一维是速度轴,
        :return:
            激发温度Tex, tr
        """
        data = self.data12
        tr = data.max(axis=0) * u.K
        v0 = self.co_data12.v0
        T_bg = self.T_bg
        T0 = self.co_data12.T0

        [size_i, size_j] = tr.shape
        tex = np.zeros(tr.shape)
        for i in range(size_i):
            for j in range(size_j):
                temp = T0 / ln(1 + T0 / (tr[i, j] + planck_function(T_bg, v0)))
                tex[i, j] = temp.to(u.K).value
        tex = tex * u.K
        return tex, tr

    def calculate_vfwhm(self):
        """
        :return:
        线宽  单位：km/s  用MGM中的Size_velocity (FWHM)
        Tr   单位：K
        """
        data = self.data
        Tr = data.max(axis=0) * u.K
        fwhm = self.record['Size_velocity'] * u.km / u.second

        return fwhm, Tr

    def calculate_tao(self):
        """
        ok，验证通过，2020/08/212
        计算谱线的光学厚度
        :param tex: 激发温度，单位：K
        :param tr: 主波束温度，单位：K
        :param spectral_line: 谱线类型，有13CO，C18O两种
        :return: 谱线的光学厚度
        """
        tex = self.tex
        T_bg = self.T_bg
        tr = self.data.max(axis=0) * u.K
        v0 = self.co_data.v0
        tao = np.zeros_like(tex.value)
        [size_i, size_j] = tao.shape
        counts = 0
        for i in range(size_i):
            for j in range(size_j):
                try:
                    tao[i, j] = - ln(1 - tr[i, j] / (planck_function(tex[i, j], v0) - planck_function(T_bg, v0)))
                except ValueError:
                    tao[i, j] = -1
                    counts += 1

        # 处理tao=0的点，采用该点邻域中其他点的光深的均值替换
        idx_i_all, idx_j_all = np.where(tao == -1)
        for i in range(counts):
            idx_i, idx_j = np.where(tao == -1)
            mask_idx = bsr_array((np.ones_like(idx_i, np.int32), (idx_i, idx_j)), shape=(size_i, size_j)).toarray()
            mask = np.zeros([size_i, size_j], np.int32)
            mask[idx_i_all[i], idx_j_all[i]] = 1
            mask1 = binary_dilation(mask, structure=np.ones([3, 3])).astype(np.int32)
            mask_region = mask1 - mask_idx
            mask_region[mask_region == -1] = 0
            tao_region = mask_region * tao
            tao[idx_i_all[i], idx_j_all[i]] = tao_region.sum() / mask_region.sum()

        return tao

    def calculate_vth(self):
        """
        ok,验证通过，2020/03/25
        计算云核的线宽
        :param tex:激发温度，单位：K    公式应代入运动学温度，  在LTE条件下，Tex=TK
        :return:
        热线宽，单位： km/s
        """

        k = self.k         # k = 1.380649 * 10 ** (-23) # 玻尔兹曼常数, 单位: J/K
        mh = self.mh  # name='Proton mass' value=1.67262192369e-27 uncertainty=5.1e-37 unit='kg' reference='CODATA 2018'
        tex = self.tex
        miu_mol = self.co_data.miu_mol    # 分子量

        vth = (8 * ln(2) * k * tex / (miu_mol * mh)) ** 0.5
        vth = vth.decompose().to(u.km/u.s)

        return vth

    def calculate_vnth(self):
        """
        验证通过，2020/04/28
        计算非热线宽
        """
        v_fwhm = self.v_fwhm
        v_th = self.vth
        v_nth = (v_fwhm ** 2 - v_th ** 2) ** 0.5
        return v_nth

    def get_n_h2(self, use_fit=True, boundary_times=2.3548):
        """
        计算C18O的柱密度的公式
        算法：Corrigendum: How to Calculate Molecular Column Density 中 公式(90)
        :param tex: 激发温度
               data: 分子云核小立方体数据
               use_fit: 计算柱密度时 利用MGM的计算结果
               boundary_times: # YY_0 为v0 + boundary_times*sigma_v 处的值-->作为云核的边界值
        :return:
        柱密度，尺寸为一个m*n的矩阵
        """
        tex = self.tex
        T_bg = self.T_bg
        tao_ = np.ones_like(tex.value)
        [size_i, size_j] = tao_.shape
        data = self.data
        XX = self.XX
        loc_wcs = self.local_wcs

        size = self.record[['Size_major', 'Size_minor', 'Size_velocity']].values
        delta = np.array(loc_wcs.pixel_scale_matrix.sum(axis=0)) * np.array([3600, 3600, 1 / 1000])
        size_pt = np.abs(size / delta) / 2.3548
        angle = self.record['Theta']
        A = self.record['Peak']

        cen_pt = self.record[['Galactic_Longitude', 'Galactic_Latitude', 'Velocity']].values
        cen_pt_pix = np.array(loc_wcs.all_world2pix(cen_pt[0], cen_pt[1], cen_pt[2] * 1000, 0)).T   # 转换坐标时，坐标原点为0

        params = np.array([A, cen_pt_pix[0], cen_pt_pix[1], size_pt[0], size_pt[1], -angle, cen_pt_pix[2], size_pt[2]])
        params_0 = np.array([A, cen_pt_pix[0], cen_pt_pix[1], size_pt[0], size_pt[1], 0, cen_pt_pix[2], size_pt[2]])
        gauss_func_0 = get_func(params_init=params_0)
        YY_0 = gauss_func_0(cen_pt_pix[0], cen_pt_pix[1], cen_pt_pix[2] + size_pt[2] * boundary_times)

        gauss_func = get_func(params_init=params)
        YY = gauss_func(XX[:, 0]-1, XX[:, 1]-1, XX[:, 2]-1)
        YY[YY < YY_0] = 0
        data_Y = YY.reshape(data.shape)
        v0 = self.co_data.v0
        delta_v = self.co_data.delta_v
        ratio = self.co_data.ratio
        T0 = self.co_data.T0
        if use_fit:
            data_cube = data_Y  # 拟合的结果
        else:
            data_cube = data  # 原始数据

        if self.co_data.co == '13':
            coef, tao = 2.482 * 10 ** 14, self.tao
            for i in range(size_i):
                for j in range(size_j):
                    tao_[i, j] = tao[i, j] / (1 - math.exp(-tao[i, j]))
        else:
            coef, tao_ = 2.48 * 10 ** 14, tao_

        n_co = np.zeros(data.shape[1:], np.float)
        data_Y_0 = data_Y.sum(0)
        self.mask = (data_Y_0 > 0).astype(np.int32)
        [size_i, size_j] = np.where(data_Y_0 > 0)
        for i in size_i:
            for j in size_j:
                data_fwhm = data_cube[:, i, j]

                temp = coef * (tex[i, j] + 0.88 * u.K) * math.exp(T0 / tex[i, j]) / (
                        math.exp(T0 / tex[i, j]) - 1) * data_fwhm.sum() * delta_v / (
                               planck_function(tex[i, j], v0) - planck_function(T_bg, v0)) * tao_[i, j] / (
                               u.cm ** 2)
                n_co[i, j] = temp.to(u.cm**-2).value
        n_co = n_co*u.cm**-2
        nh2 = ratio * n_co
        self.save_fig(nh2.value)

        return n_co, nh2

    def calculate_reff_GC(self):
        """
        计算云核尺度因子, 应该不是做为云核的半径的
        :param data:
        :return:
        calc.reff/d/60/60/180*math.pi
        #  1PC(秒差距)=30835997962819660.8米  1pc ~= 206265AU ~= 3.26光年
        """
        # 观测角面积，单位: 平方角秒
        gc1 = self.record['Size_major'] * u.arcsec
        gc2 = self.record['Size_minor'] * u.arcsec
        area = math.pi * gc1 * gc2 / 4
        if 'Distance(kpc)' in self.record.keys():
            d = self.record['Distance(kpc)'] * constants.kpc.to(u.cm)
        else:
            d = 2000 * constants.pc.to(u.cm)

        pi = math.pi
        # 望远镜主波束宽度, 单位: 角秒(")
        sita_mb = 52 * u.arcsec

        reff = 0.5 * d * (4 / pi * area - sita_mb ** 2) ** 0.5
        reff = (reff / 60 / 60 / 180 * math.pi) / u.arcsec
        return reff.to(constants.pc)

    def calculate_m(self):
        """
        采用单个谱线进行计算
        """
        if 'Distance(kpc)' in self.record.keys():
            d = (self.record['Distance(kpc)'] * constants.kpc).to(u.cm)
        else:
            d = (2000 * constants.pc).to(u.cm)
        single_reff = d * 30 / 60 / 60 / 180 * math.pi  # 一个像素的长度(cm)
        area = (single_reff ** 2)   # 一个像素的面积
        n_h2 = self.n_h2
        mass = self.miu_h * (2 * self.mh) * n_h2 * area   # 2 * self.mh 表示氢分子质量

        mass = mass.decompose().to(u.Msun)
        mass_sum = mass.decompose().to(u.Msun).sum()
        return mass, mass_sum

    def calcultate_M_vir(self):

        R = self.reff.value
        FWHM = self.v_fwhm.value

        return 209 * R * FWHM ** 2 * u.Msun

    def calculate_vir(self):

        mass_vir = self.mass_vir
        mass = self.mass_

        m_v = mass_vir
        m = mass

        vir = m_v.sum() / m.sum()
        return vir

    def save_fig(self, data_cube):
        local_wcs = self.local_wcs

        cen_pt = self.record[['Galactic_Longitude', 'Galactic_Latitude', 'Velocity']].values
        cen_pt_pix = np.array(local_wcs.all_world2pix(cen_pt[0], cen_pt[1], cen_pt[2] * 1000, 0)).T
        markersize = 2
        width = self.record['Size_major'] / 30 / 2.3548 * 4
        height = self.record['Size_minor'] / 30 / 2.3548 * 4
        angle = self.record['Theta']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=local_wcs.celestial)
        plt.imshow(data_cube)

        ellipse = Ellipse(xy=(cen_pt_pix[0], cen_pt_pix[1]), width=width, height=height, alpha=0.3, angle=-1*angle)
        ax.add_patch(ellipse)

        outcat_wcs_c = SkyCoord(frame="galactic", l=cen_pt[0], b=cen_pt[1], unit="deg")
        ax.plot_coord(outcat_wcs_c, 'r*', markersize=markersize)
        plt.title(self.co_data.co + '_CO')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.xlabel('GLON')
        plt.ylabel('GLAT')

        self.pdf.savefig()
        plt.close()

    def calculate_p_internal(self):

        # 需要将柱密度为0的点去掉，现在只是简单的求均值
        n_h2 = self.n_h2.mean()
        radii = self.reff
        k = self.k
        m_H = self.mh
        miu = 2.8
        v_nth = self.vnth
        T = self.tex

        n = (3 / (4 * radii) * n_h2).to(u.cm**-3)

        p_th = k * n * T
        p_nth = miu * m_H * n * v_nth**2 / (8 * math.log(2, math.e))
        p_tot = p_nth + p_th

        return (p_th/k).to(u.K/u.cm**3), (p_nth/k).to(u.K/u.cm**3), p_tot, n

    def calculate_p_external(self):
        n = self.n_h2
        k = self.k
        phy_G = 1.6
        N = n.mean() / (10**21 * u.cm**-2)
        p_cloud = 4.5 * 10**3 * phy_G * k * N ** 2

        return (p_cloud / k) * (u.K * u.cm**-3)

    def calculate_density_s(self):

        R = self.reff
        M = self.mass_sum

        density_s = M / (math.pi * R**2)
        return density_s


def calculate_physics_parameter(outcat):
    # outcat = fits.getdata('gaussclumps_result/gauss_outcat_m16_13_ellipse.FIT')
    # file_list = os.listdir('data/core')
    # k = [int(k.split('.')[0]) for k in file_list]
    info = []
    for i, item in enumerate(outcat):
        calc = Calculate_Parameters(item, 'pdf_gaussclumps_control_1', '13')
        if i % 50 == 0:
            print('the {}-th record'.format(i))
        info.append([calc.tex, calc.v_fwhm, calc.tao, calc.n_h2, calc.mass_sum, calc.vir_a, calc.reff,
                     calc.mass_vir, calc.vth, calc.vnth])

    result = np.zeros((len(outcat), 10), np.float)
    for num, item in enumerate(info):
        for i, item1 in enumerate(item):
            if i == 0 or i == 3:
                result[num, i] = item1.value.max()
            elif i == 2:
                result[num, i] = item1.mean()
            elif i == 8 or i == 9:
                result[num, i] = item1.value.mean()
            else:
                result[num, i] = item1.value

    reault_pd = pd.DataFrame(
        {
            'Tex': result[:, 0], 'v_fwhm': result[:, 1], 'tao': result[:, 2], 'nh2': result[:, 3], 'M_sum': result[:, 4],
         'vir_a': result[:, 5], 'reff': result[:, 6], 'M_vir': result[:, 7], 'vth': result[:, 8], 'vnth': result[:, 9]
         })

    reault_pd1 = pd.DataFrame()
    for item in reault_pd:
        reault_pd1[item] = [reault_pd[item].max(), reault_pd[item].min(), reault_pd[item].std(), reault_pd[item].mean()]

    # save result into xlsx file, and draw pictures use it
    writer = pd.ExcelWriter('data/gaussclumps_parameter_3_m16_20201005.xlsx')
    reault_pd.to_excel(writer, 'Sheet1')
    reault_pd1.to_excel(writer, 'Sheet2')
    writer.close()


def calculate_pp(outcat_path, co_12_path, co_13_path, outcat_d_p_path=None):

    outcat = pd.read_csv(outcat_path, sep='\t')
    co_data12 = COData(co_12_path)
    co_data13 = COData(co_13_path)
    info = []
    for i in tqdm.tqdm(range(outcat.shape[0])):
        item = outcat.iloc[i]
        calc = Calculate_Parameters(item, pdf_path='pdf_gaussclumps_control_37', co_data12=co_data12, co_data=co_data13)
        info.append([calc.tex, calc.v_fwhm, calc.tao, calc.n_h2, calc.mass_sum, calc.vir_a, calc.reff,
                     calc.mass_vir, calc.vth, calc.vnth])

    result = np.zeros((len(outcat), 10), np.float32)
    for num, item in enumerate(info):
        for i, item1 in enumerate(item):
            if i == 0 or i == 3:
                result[num, i] = item1.value.max()
            elif i == 2:
                result[num, i] = item1.mean()
            elif i == 8 or i == 9:
                result[num, i] = item1.value.mean()
            else:
                result[num, i] = item1.value
    reault_pd = pd.DataFrame(
        {
            'Tex': result[:, 0], 'v_fwhm': result[:, 1], 'tao': result[:, 2], 'nh2': result[:, 3],
            'M_sum': result[:, 4],
            'vir_a': result[:, 5], 'reff': result[:, 6], 'M_vir': result[:, 7], 'vth': result[:, 8],
            'vnth': result[:, 9]
        })

    outcat_d_physical = pd.concat([outcat, reault_pd], axis=1)
    if outcat_d_p_path is None:
        outcat_d_p_path = outcat_path.replace('.csv', '_physical.csv')
    outcat_d_physical.to_csv(outcat_d_p_path, sep='\t', index=False)


if __name__ == '__main__':
    outcat_path = r'test_data/0155+005_L_MGM/MWISP_outcat_d.csv'
    co_12_path = r'test_data/0155+005_U/0155+005_U.fits'
    co_13_path = r'test_data/0155+005_L/0155+005_L.fits'
    calculate_pp(outcat_path, co_12_path, co_13_path, outcat_d_p_path='None1.csv')