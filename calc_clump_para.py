from tools.calc_clump_params import calc_clump_params_main as ccp
import argparse
import os
import pandas as pd


def get_pp_outcat():
    outcat_path = r'test_data/0155+005_L_MGM/MWISP_outcat_d.csv'
    co_12_path = r'test_data/0155+005_U/0155+005_U.fits'
    co_13_path = r'test_data/0155+005_L/0155+005_L.fits'
    save_outcat_path = 'tets.csv'
    ccp(outcat_path, co_12_path, co_13_path, save_outcat_path=save_outcat_path)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('cell', type=str, default='', help='the cell name :0140+000_L')
    # parser.add_argument('co12_path', type=str, default='test_data/0155+005_U/0155+005_U.fits', help='the catlog of clumps')
    # parser.add_argument('co13_path', type=str, default='test_data/0155+005_L/0155+005_L.fits', help='the catlog of clumps')
    # parser.add_argument('physical_outcat_path', type=str, default='test.csv', help='the catlog of clumps')
    # args = parser.parse_args()
    # outcat_path = args.outcat_path
    # co_12_path = args.co12_path
    # co_13_path = args.co13_path
    # print(outcat_path, co_12_path, co_13_path)
    # physical_outcat_path = args.physical_outcat_path

    outcat_verify = r'/home/data/clumps_share/data_longc/云核认证/R2_LDC/second_verify/Yes_LDC_R2_09.csv'
    outcat_loc_path = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect'
    outcat_loc_path_again = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_detect_again'
    outcat_path = r'/home/data/clumps_share/data_luoxy/R2_data_params_fwhm/'
    outcat_path1 = r'/home/data/clumps_share/data_luoxy/R2_data_params/'
    outcat_mgm_path = r'/home/data/clumps_share/data_luoxy/detect_R2_LDC/R2_200_MGM_1/'

    outcat_all = pd.read_csv(outcat_verify, sep='\t')
    os.makedirs(outcat_path, exist_ok=True)
    file_list = os.listdir(r'/home/data/clumps_share/data_luoxy/R2_data_params/')
    parser = argparse.ArgumentParser()
    parser.add_argument('st', type=int, default=0, help='')
    parser.add_argument('end_', type=int, default=len(file_list), help='')

    args = parser.parse_args()
    st = args.st
    end_ = args.end_
    # st = 0
    # end_ = len(file_list)

    for i in range(st, end_, 1):
        item = file_list[i]
        print(i, item)
        os.makedirs(outcat_path + r'%s' % item, exist_ok=True)
        outcat_cell_verify_path = outcat_path1 + r'%s/MWISP_outcat.csv' % item   # 人工证认且有距离的核表
        physical_outcat_path = outcat_path + r'%s/MWISP_outcat_d_physical_again.csv' % item

        # if not os.path.exists(outcat_cell_verify_path):
        #     outcat_name = outcat_mgm_path + r'%s/MWISP_outcat.csv' % item  # 拟合核表
        #     outcat_cell_mgm = pd.read_csv(outcat_name, sep='\t')
        #     outcat_cell_mgm = outcat_cell_mgm.drop_duplicates('ID', keep='first')
        #
        #     oucat_ldc_loc_name = outcat_loc_path + r'/%s/LDC_auto_loc_outcat_wcs.csv' % item  # ldc检测局部核表(wcs)
        #     if not os.path.exists(oucat_ldc_loc_name):
        #         oucat_ldc_loc_name = outcat_loc_path_again + r'/%s/LDC_auto_loc_outcat_wcs.csv' % item
        #         # ldc将以前没有的数据再一次检测的局部核表(wcs)
        #     outcat_cell = pd.read_csv(oucat_ldc_loc_name, sep='\t')
        #     outcat_cell_intersected = pd.merge(outcat_all['ID'], outcat_cell['ID'], on=['ID'], how='right')
        #     idx_ = []
        #     for i in range(outcat_cell_intersected.shape[0]):
        #         if outcat_cell_intersected.iloc[i]['ID'] in outcat_cell['ID'].values:
        #             idx_.append(i)
        #
        #     outcat_cell_verify = outcat_cell_mgm.iloc[idx_]
        #     outcat_cell_verify.to_csv(outcat_cell_verify_path, sep='\t', index=False)
        if os.path.exists(physical_outcat_path):
            continue

        item_1 = item[:-2]
        co_12_path = r'/home/data/clumps_share/MWISP/R2_Real_Data/R2_200_U/%s/%s_U.fits' % (item_1, item_1)
        co_13_path = r'/home/data/clumps_share/MWISP/R2_Real_Data/R2_200_L/%s/%s_L.fits' % (item_1, item_1)

        ccp(outcat_cell_verify_path, co_12_path, co_13_path, save_outcat_path=physical_outcat_path)
