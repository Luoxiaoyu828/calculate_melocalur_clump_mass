import requests
from lxml import etree
import re
import numpy as np
import os
import pandas as pd


def get_distance(l, b, vlsr, dvlsr, save_folder, file_name, prob=0.5, u_a=0, du_a=0, u_d=0, du_d=0):
    # 'http://www3.mpifr-bonn.mpg.de/staff/abrunthaler/bessel_calc2.0/bayesian.php?l=15.72&b=0.75&prob=0.5&vlsr=12.3&dvlsr=1.17&u_a=0.0&du_a=0.0&u_d=0.0&du_d=0.0'
    url_base = r'http://www3.mpifr-bonn.mpg.de/staff/abrunthaler/bessel_calc2.0/'
    r = requests.get(
        url_base + r'bayesian.php?l=%.2f&b=%.2f&prob=%.1f&vlsr=%.2f&dvlsr=%.2f&u_a=%.2f&du_a=%.2f&u_d=%.2f&du_d=%.2f' % (
            l, b, prob, vlsr, dvlsr, u_a, du_a, u_d, du_d))
    png_url = r'http://www3.mpifr-bonn.mpg.de/staff/abrunthaler/bessel_calc2.0/'
    distece_info = os.path.join(save_folder, '%s.txt' % file_name)
    distece_png = os.path.join(save_folder, '%s.png' % file_name)
    if r.status_code == 200:
        html = r.content
        tree = etree.HTML(html)
        name1 = tree.xpath("//h2/text()")
        name2 = tree.xpath("//h3/text()")
        name3 = tree.xpath("//a/text()")

        with open(distece_info, 'w') as f:
            f.write(str(name1[0]))
            f.write('\n')
            f.write(str(name2[0]))
        f.close()

        name1_num = re.findall("\d+\.?\d*", str(name1[0]))  # 正则表达式
        name2_num = re.findall("\d+\.?\d*", str(name2[0]).split(',')[0])  # 正则表达式
        dis_zf_prob = name1_num + name2_num
        dis_zf_prob = np.array(dis_zf_prob, np.float32).reshape([1, 3])

        response = requests.get(png_url + name3[0])
        if response.status_code == 200:
            with open(distece_png, 'wb') as f:
                f.write(response.content)
            f.close()
        else:
            raise FileNotFoundError('HTTP Error 404: Not Found')
        return dis_zf_prob
    else:
        return None


def calculate_distance_outcat(outcat_path):
    outcat = pd.read_csv(outcat_path, sep='\t')
    dis_zf_prob_all = np.zeros([outcat.shape[0], 3], np.float32)
    for i in range(outcat.shape[0]):
        l = outcat['Galactic_Longitude'].values[i]
        b = outcat['Galactic_Latitude'].values[i]
        vlsr = outcat['Velocity'].values[i]
        dvlsr = outcat['Size_velocity'].values[i]
        file_name = outcat['ID'].values[i]
        save_folder = outcat_path.replace('.csv', '')
        os.makedirs(save_folder, exist_ok=True)
        dis_zf_prob = get_distance(l, b, vlsr, dvlsr, save_folder, file_name)
        # print(dis_zf_prob[0,:])
        dis_zf_prob_all[i, :] = dis_zf_prob
    dis_zf_prob_all_pd = pd.DataFrame(dis_zf_prob_all, columns=['Distance(kpc)', 'd_Distance(kpc)', 'Prob'])
    outcat_all = pd.concat([outcat, dis_zf_prob_all_pd], axis=1)

    return outcat_all


if __name__ == '__main__':
    outcat_path = r'test_data/0155+005_L_MGM/MWISP_outcat.csv'
    outcat_all = calculate_distance_outcat(outcat_path)