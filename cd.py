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
    p = Pool(1)
    outcat_name_ = [ouctat_path + r'%s/MWISP_outcat.csv' % item for item in file_list]
    save_outcat_path = [save_path + r'%s' % item for item in file_list]
    save_outcat = [save_path + r'%s/MWISP_outcat.csv' % item for item in file_list]

    parser = argparse.ArgumentParser()
    parser.add_argument('st', type=int, default=0, help='the start index')
    parser.add_argument('end_', type=int, default=len(file_list), help='the end index')
    args = parser.parse_args()

    for i in range(args.st, args.end_, 1):
        os.makedirs(save_outcat_path[i], exist_ok=True)

        outcat_name = outcat_name_[i]
        save_outcat_name = save_outcat[i]

        print(i)
        if os.path.exists(save_outcat_name):
            continue
        print(i, outcat_name)
        # cdm(outcat_name, save_outcat_path=save_outcat_name)
        p.apply_async(cdm, args=(outcat_name, save_outcat_name))

    p.close()
    p.join()