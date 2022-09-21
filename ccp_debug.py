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

    outcat_cell_verify_path = r'data_check_cpp/MWISP_outcat.csv'   # 人工证认且有距离的核表
    physical_outcat_path = r'data_check_cpp/MWISP_outcat_d_physical_again.csv'


    co_12_path = r'data_check_cpp/0110-005_U.fits'
    co_13_path = r'data_check_cpp/0110-005_L.fits'

    ccp(outcat_cell_verify_path, co_12_path, co_13_path, save_outcat_path=physical_outcat_path)
