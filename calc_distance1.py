from tools.calculate_distance import calc_dis_main as cdm
import os
import argparse
from multiprocessing import Pool


"""
根据MGM过后的核表，计算云核的距离，并将距离、距离误差、概率和云核高度附加在核表上，并保存到指定位置
"""
if __name__ == '__main__':

    ouctat_path = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_MGM_1/'
    save_path = r'/home/data/clumps_share/data_luoxy/R2_data_params/'
    file_list = os.listdir(ouctat_path)
    outcat_name_ = [ouctat_path + r'%s/MWISP_outcat.csv' % item for item in file_list]
    save_outcat_path = [save_path + r'%s' % item for item in file_list]
    save_outcat = [save_path + r'%s/MWISP_outcat.csv' % item for item in file_list]

    # outcat_name = outcat_name_[i]
    # save_outcat_name = save_outcat[i]

    outcat_name = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_MGM_1/0125-020_L/MWISP_outcat.csv'
    save_outcat_name = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_MGM_1/0125-020_L/MWISP_outcat_new/aa.csv'
    cdm(outcat_name, save_outcat_path=save_outcat_name)
