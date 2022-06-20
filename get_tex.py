import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.constants as Constant
import astropy.units as u
from scipy.optimize import curve_fit
import calculate_core_parameters as ccp
#
# def calculate_tex():
#     """
#     ok, 验证通过，2020/08/12
#     计算公式：w37  公式(3.3)
#     计算激发温度：通过12CO数据计算, each spectral-line calculates an excitation temperature
#     第一维是速度轴,
#     :return:
#         激发温度Tex, tr
#     """
#     data = self.data12
#     tr = data.max(axis=0) * u.K
#     v0 = self.v0_12
#     h = self.h
#     k = self.k
#     T_bg = self.T_bg
#
#     [size_i, size_j] = tr.shape
#     tex = np.zeros_like(tr)
#     for i in range(size_i):
#         for j in range(size_j):
#             tex[i, j] = ( h *v 0 /k) / (ln(1 + ( h *v 0 /k) / (tr[i, j] + planck_function(T_bg, v0))))
#
#     return tex, tr
k = Constant.k_B
# h = 6.62607015 * 10 ** (-34)# h: 普朗克常数，单位: J*s
h = Constant.h
# mh = 1.66 * 10 ** (-27 # 氢原子的质量,单位:kg)
mh = Constant.m_p
miu_h = 2.8
# 1GHz = <Quantity 1.e+09 1 / s>  # v0: 谱线频率，单位: Hz
v0_12 = 115.271204 * 10 ** 9 * u.Hz
v0_13 = 110.201353 * 10 ** 9 * u.Hz
v0_18 = 109.782183 * 10 ** 9 * u.Hz
T_bg = 2.73 * u.K


def make_pic(data):
    data_12_x = data['vaxis12'].values
    data_13_x = data['vaxis13'].values
    data_18_x = data['vaxis18'].values
    data_12 = data['spec12'].values
    data_13 = data['spec13'].values
    data_18 = data['spec18'].values
    plt.plot(data_12_x, data_12, label='12CO')
    plt.plot(data_13_x, data_13, label='13CO')
    plt.plot(data_18_x, data_18, label='C18O')
    plt.legend()
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Temperature [K]')
    plt.show()


def calc_tex(h, v0, k, tr, T_bg):
    tex = (h * v0 / k) / (ccp.ln(1 + (h * v0 / k) / (tr + ccp.planck_function(T_bg, v0))))
    return tex


def f_gauss(x, A, B, sigma):
    """
    高斯拟合函数
    """
    return A * np.exp(-(x - B) ** 2 / (2 * sigma ** 2))


def calc_fwhm_tr(s_line):

    Tr = s_line.max() * u.K
    x = np.arange(1, s_line.shape[0] + 1, 1)
    # fit the average line through gauss_function
    popt, pcov = curve_fit(f_gauss, x, s_line, bounds=([0, 3, 0], [40, 50, 50]))
    # res_robust = least_squares(f_gauss, x, s_line, bounds=[[0, 3, 0], [40, 20, 10]])
    fwhm = (8 * ccp.ln(2)) ** 0.5 * popt[-1].__abs__() * 0.167 * u.km / u.second
    # Tr = popt[0] * u.K
    return fwhm, Tr


def calculate_tao(tex, T_bg, tr, v0):
    """
    ok，验证通过，2020/08/212
    计算谱线的光学厚度
    :param tex: 激发温度，单位：K
    :param tr: 主波束温度，单位：K
    :param spectral_line: 谱线类型，有13CO，C18O两种
    :return: 谱线的光学厚度
    """
    try:
        tao = - ccp.ln(1 - tr / (ccp.planck_function(tex, v0) - ccp.planck_function(T_bg, v0)))
    except ValueError:
        tao = 0
        print(1)

    return tao


def calculate_vth(k, mh, tex, co='13'):
    """
    ok,验证通过，2020/03/25
    计算云核的线宽
    :param tex:激发温度，单位：K    公式应代入运动学温度，  在LTE条件下，Tex=TK
    :return:
    热线宽，单位： km/s
    """

    # k = self.k         # 氢原子的质量,单位:kg
    # mh = self.mh       # mh = 1.66 * 10 ** (-24)
    # tex = self.tex

    if co == '13':
        miu_mol = 29
    else:
        miu_mol = 30

    vth = (8 * ccp.ln(2) * k * tex / (miu_mol * mh)) ** 0.5
    vth = vth.decompose().to(u.km/u.s)

    return vth


def calculate_vnth(v_fwhm, v_th):
    """
    验证通过，2020/04/28
    计算非热线宽
    """

    v_nth = (v_fwhm ** 2 - v_th ** 2) ** 0.5
    return v_nth


def get_n_h2(h, k, tex, T_bg, v0, tao, data_fwhm, co='13'):
    """
    计算C18O的柱密度的公式
    算法：Corrigendum: How to Calculate Molecular Column Density 中 公式(90)
    :param tex: 激发温度
           data: 分子云核小立方体数据
    :return:
    柱密度，尺寸为一个m*n的矩阵
    """
    # h = self.h
    # k = self.k
    # tex = self.tex
    # T_bg = self.T_bg

    if co == '13':
        coef, delta_v, ratio = 2.42 * 10 ** 14, 0.166, 7 * 10 ** 5

        tao_ = tao / (1 - math.exp(-tao))
    else:
        coef, delta_v, tao_, ratio = 2.48 * 10 ** 14, 0.1666, 1, 7 * 10 ** 6

    temp = coef * (tex + 0.88 * u.K) * math.exp((h*v0/k) / tex) / (
                math.exp((h*v0/k) / tex) - 1) * data_fwhm.sum() * delta_v / (
                       ccp.planck_function(tex, v0) - ccp.planck_function(T_bg, v0)) * tao_ / (
                       u.cm ** 2)
    n_co = temp.to(u.cm**-2).value
    n_co = n_co*u.cm**-2
    nh2 = ratio * n_co

    return n_co, nh2

if __name__ == '__main__':
    data = pd.read_csv('ColumnDensityCalcCheck-spectrum.csv')
    data_12 = data['spec12'].values
    data_13 = data['spec13'].values
    data_13 = data_13[np.where(np.isnan(data_13).astype(np.int)!=1)]
    data_18 = data['spec18'].values
    data_18 = data_18[np.where(np.isnan(data_18).astype(np.int) != 1)]
    tr = data_12.max()*u.K
    tex = calc_tex(h=h, v0=v0_12, k=k, tr=tr, T_bg=T_bg)
    fwhm_13, Tr_13 = calc_fwhm_tr(data_13)
    tao_13 = calculate_tao(tex=tex, T_bg=T_bg, tr=Tr_13, v0=v0_13)
    vth_13 = calculate_vth(k, mh, tex, co='13')
    vnth_13 = calculate_vnth(fwhm_13, vth_13)
    n_co_13, nh2_13 = get_n_h2(h, k, tex, T_bg, v0_13, tao_13, data_13[36:53], co='13')


    fwhm_18, Tr_18 = calc_fwhm_tr(data_18)
    tao_18 = calculate_tao(tex=tex, T_bg=T_bg, tr=Tr_18, v0=v0_18)
    vth_18 = calculate_vth(k, mh, tex, co='18')
    vnth_18 = calculate_vnth(fwhm_18, vth_18)
    n_co_18, nh2_18 = get_n_h2(h, k, tex, T_bg, v0_18, tao_18, data_18[36:53], co='18')

