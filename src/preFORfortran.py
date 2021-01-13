# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    preFORfortran.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: marvin <marvin@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/12/10 19:46:16 by marvin            #+#    #+#              #
#    Updated: 2020/12/10 19:46:16 by marvin           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# this is a script that preprocessing to procure the required files for the following fortran
# programmes

from post_analysis import extract_filenames
import numpy as np
import dask.dataframe as dd
import sys
import datetime
import time
import platform
import glob
from scipy import interpolate


class IoFile(object):
    # define a class named IoFile
    def __init__(self, id, year, month, date_from, date_to=''):
        """ Initiate the object

        Args:
            id (string): the id for the satellite, gracefo or taiji
            year (string): year of the data
            month (string): month of the data
            date (string): the date start
        """
        self.id = id
        # check if the inputs are standard
        date_to = date_from if date_to == '' else date_to
        temp_list = [year, month, date_from, date_to]
        attr_list = ['year', 'month', 'date_from', 'date_to']
        for _, (ele, attr) in enumerate(zip(temp_list, attr_list)):
            if isinstance(ele, str) is not True:
                raise ValueError('Error occurs in the initialisation')
            else:
                setattr(self, attr, ele)
        self.date_time_from = year + '-' + month + '-' + date_from
        self.date_time_to = year + '-' + month + '-' + date_to
        self.data_dict = {}
        if id == 'gracefo':
            self.flags = ['KBR1A', 'KBR1B', 'GNV1B', 'GNI1B', 'SCA1A', 'LRI1A', 'LRI1B', 'USO1B', 'CLK1B']
        elif id == 'taiji':
            self.flags = []


    def extract_filenames(self, file_flag, id):
        """ This method is designed to extract the required filenames from the directory

        Args:
            file_flag (string): data product
            id (string): gracefo_c or gracefo_d

        Returns:
            string_list: the data product file names list
        """
        self.gracefo_dataset_dir = ''
        if platform.system() == 'Linux':
            self.gracefo_dataset_dir = '..//..//..//gracefo_dataset/'
        elif platform.system() == 'Windows':
            self.gracefo_dataset_dir = 'E:/lhsProgrammes/gracefo_dataset'
        suffix = '.txt'
        temp = glob.glob(self.gracefo_dataset_dir + '/**' + '/' + file_flag + '*' + id + '*' + suffix)
        temp.sort()
        # obtain the desired date span
        start_index = temp.index([s for s in temp if self.date_time_from in s][0])
        end_index = temp.index([s for s in temp if self.date_time_to in s][0]) + 1
        temp = temp[start_index: end_index]
        return temp


    def load_data(self, data_flag, id='Y'):
        """ This method is designed to load data from files

        Args:
            data_flag (string): data product flags
            id (str, optional): Defaults to 'Y'.
        """
        if data_flag not in self.flags:
            raise ValueError('Error transcending data flag in loading date')
        else:
            self.urlpaths = self.extract_filenames(data_flag, id)
            # header dictionary for different data product
            header_dict = {
                'GNI1B': [
                    'gps_time', 'id', 'frame', 'xpos', 'ypos', 'zpos', 'zpos',
                    'xpos_err', 'ypos_err', 'zpos_err', 'xvel', 'yvel', 'zvel',
                    'xvel_err', 'yvel_err', 'zvel_err', 'qualflg'
                ],
                'USO1B': [
                    'gps_time', 'id', 'uso_id', 'uso_freq', 'k_freq',
                    'ka_freq', 'qualflg'
                ],
                'CLK1B': [
                    'kbr_time', 'id', 'clk_id', 'eps_time', 'eps_err',
                    'eps_drift', 'drift_err', 'qualflg'
                ],
                'KBR1A': [
                    'kbr_time_intg', 'kbr_time_frac', 'id', 'prn_id', 'ant_id',
                    'qualflg', 'k_phase', 'ka_phase', 'k_snr', 'ka_snr'
                ]
            }
            # the data type for different columns
            dtype_dict = {'GNI1B': {'xpos': np.longdouble, 'ypos': np.longdouble, 'zpos': np.longdouble,
                                    'xvel': np.longdouble, 'yvel': np.longdouble, 'zvel': np.longdouble},
                          'USO1B': {'k_freq': np.longdouble, 'ka_freq': np.longdouble},
                          'CLK1B': {'eps_time': np.longdouble},
                          'KBR1A': {'k_phase': np.longdouble, 'ka_phase': np.longdouble}}
            # skiprows
            skiprows_dict = {'GNI1B': 148, 'USO1B': 90, 'CLK1B': 118, 'KBR1A': 235}
            # using dask to load data
            self.data_dict[data_flag] = dd.read_csv(urlpath=self.urlpaths,
                                                    sep='\s+',
                                                    engine='c',
                                                    skiprows=skiprows_dict[data_flag],
                                                    storage_options=dict(auto_mkdir=False),
                                                    names=header_dict[data_flag],
                                                    dtype=dtype_dict[data_flag])
            if data_flag == 'KBR1A':
                self.gps_time = self.data_dict[
                    data_flag].kbr_time_intg.compute().to_numpy(
                    ) + self.data_dict[data_flag].kbr_time_frac.compute(
                    ).to_numpy() * 1.e-6


    def gracefo_interp_kbr_clk(self, id='Y'):
        """ This is the method for interpolating the kbr eps time for krb1a time tags

        Args:
            id (str, optional): the id of the satellite. Defaults to 'Y'.
        """
        self.load_data('CLK1B', id)
        self.load_data('KBR1A', id)
        # kbr time
        self.kbr_time = self.data_dict['KBR1A'].kbr_time_intg.compute(
        ).to_numpy() + self.data_dict['KBR1A'].kbr_time_frac.compute(
        ).to_numpy() * 1.e-6
        self.kbr_eps_time = self.data_dict['CLK1B'].eps_time.compute().to_numpy()[0: -1]
        clk1b_kbr_time = self.data_dict['CLK1B'].kbr_time.compute().to_numpy()[0: -1]
        #> linear interpolation
        temp = interpolate.interp1d(
            clk1b_kbr_time - np.floor(self.kbr_time[0]),
            self.kbr_eps_time,
            kind='linear')(self.kbr_time - np.floor(self.kbr_time[0]))
        self.kbr_eps_time = temp
        #> write interpolated data to files
        output_urlpaths = self.output_filename('CLK1A', id)
        np.savetxt(output_urlpaths, self.kbr_eps_time, newline='\n')


    def gracefo_interp_uso_freq(self, id='Y'):
        """ This method is to interpolate the k and ka microwave frequency to the gps time tags

        Args:
            id (str, optional): the satellite id. Defaults to 'Y'.
        """
        self.load_data('USO1B', id)
        self.k_freq = interpolate.interp1d(
            self.data_dict['USO1B'].gps_time.compute().to_numpy(),
            self.data_dict['USO1B'].k_freq.compute().to_numpy(),
            kind='linear')(self.gps_time)
        self.ka_freq = interpolate.interp1d(
            self.data_dict['USO1B'].gps_time.compute().to_numpy(),
            self.data_dict['USO1B'].ka_freq.compute().to_numpy(),
            kind='linear')(self.gps_time)
        #> write the k and ka frequency to files
        output_urlpaths = self.output_filename('USO1B', id)
        np.savetxt(output_urlpaths, np.c_[self.k_freq, self.ka_freq], fmt='%.4f', newline='\n')



    def output_filename(self, flag, id):
        """ This is the method for create output file names

        Args:
            flag (string): data product type
            id (string): satellite id
        """
        output_filenames = ''
        if self.urlpaths.__len__() == 1:
            directory = self.urlpaths[0][0: 67]
            output_filenames = directory + flag + '_' + self.date_time_from + '_' + id + '_05.txt'
        else:
            directory = self.urlpaths[0][0: 58]
            output_filenames = directory + 'gracefo_data_' + self.year + '-' + self.month + '/' + \
                                flag + '_fr' + self.date_time_from + 'to' + self.date_time_to + '_' \
                                    + id + '_05.txt'
        return output_filenames


def timer(func):
    def call_func(*args, **kwargs):
        begin_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - begin_time
        print(str(func.__name__) + "函数运行时间为" + str(run_time))
        return ret
    return call_func


def star(func):
    def inner(*args, **kwargs):
        print("*" * 88)
        func(*args, **kwargs)
        print("*" * 88)
    return inner


def percent(func):
    def inner(*args, **kwargs):
        print("%" * 88)
        func(*args, **kwargs)
        print("%" * 88)
    return inner


@star
@percent
@timer
def main(date_time):
    # initiate an object
    kbr_preprocess = IoFile('gracefo', date_time[0: 4], date_time[5: 7], date_time[8: 11])
    kbr_preprocess.gracefo_interp_kbr_clk('C')
    kbr_preprocess.gracefo_interp_kbr_clk('D')
    kbr_preprocess.gracefo_interp_uso_freq('C')
    kbr_preprocess.gracefo_interp_uso_freq('D')



if __name__ == '__main__':
    # display digital length
    np.set_printoptions(precision=15)
    try:
        date_time = sys.argv[1]
        year = int(date_time[0: 4])
        month = int(date_time[5: 7])
        date = int(date_time[8: 11])
        temp = datetime.datetime(year, month, date)
    except:
        raise ValueError("The input argument is false, please check.")
    main(date_time)
