import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from glob import glob
from os import path
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle
import cartopy.crs as crs
import cartopy.feature as cfeature
import matplotlib.tri as tri
import plotly.graph_objects as go
from scipy.signal import welch, kaiserord, firwin, filtfilt
from scipy.fft import fft, fftfreq
from pyquaternion import Quaternion
from plotly.subplots import make_subplots


def file_product4specific_days(start_date, end_date, product_flags, path_dir, subfolder_path_format):
    """ This is a function to find all files with certain product_flag from the start_date to end_date from path_dir
        Input Args:
            start_date: date from
            end_date: date to
            product_flags: certain files
            path_dir: files should be found in the path_dir
            subfolder_path_format: the sub-folder format
        Output Args:
            output_paths: filenames from start_date to end_date including product_flag
    """
    DateFormat = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, DateFormat).date()
    end_date = datetime.strptime(end_date, DateFormat).date()

    delta_one_day = timedelta(days=1)
    date = start_date
    output_paths = {}
    for _, product_flag in enumerate(product_flags):
        file_paths = []
        while date <= end_date:
            subfolder_path = date.strftime(subfolder_path_format)
            data_folder = path.join(path_dir, subfolder_path)
            if path.isdir(data_folder):
                for filename in glob(path.join(data_folder, '*.txt')):
                    if product_flag in filename:
                        file_paths.append(filename)
            date += delta_one_day
        date = start_date
        output_paths[product_flag] = file_paths

    return output_paths


def xyz2latlon(date, secofday, xyz):
    """ This is a function to transfer a set of coordinates in cartesian coordinate of ITRS to those
        corresponding latitude and longitude.

    Args:
        date (str): specifically the starting date of this formulism
        secofday (int): second of the day which is exactly the [date]. Note that [secofday] can be 
                        more than 86400
        xyz (np.float): cartesian coordinates for each epoch
    """
    # if (secofday.__len__() != xyz.__len__()):
    #     raise ValueError("The dimensions of secofday and xyz are not compatible")
    
    time_stamps = np.asarray([], dtype=str)
    for _, sec in enumerate(secofday):
        time_stamps = np.append(
            time_stamps, datetime(2000, 1, 1, 11, 59, 47) + timedelta(seconds=sec))

    xs = xyz[:, 0]
    ys = xyz[:, 1]
    zs = xyz[:, 2]
    ground_lat = np.asarray([])
    ground_lon = np.asarray([])
    secofday = secofday - secofday.min()
    secofday = secofday.astype(np.int)
    for index, sec in enumerate(secofday):
        now = Time(time_stamps[index])
        cartrep = coord.CartesianRepresentation(
            x=xs[sec],
            y=ys[sec],
            z=zs[sec],
            unit=u.m
        )
        itrs = coord.ITRS(cartrep, obstime=now)
        loc = coord.EarthLocation(*itrs.cartesian.xyz)
        ground_lat = np.append(ground_lat, Angle(loc.lat).degree)
        ground_lon = np.append(ground_lon, Angle(loc.lon).degree)
    
    return ground_lat, ground_lon


def map_cntr(lat, lon, data, lat_min=-90., lat_max=90., lon_min=-180., lon_max=180.):
    '''
    This is function to plot the contour image.
    '''
    ngrid_lat = 1801
    ngrid_lon = 3601

    lati = np.linspace(lat_min, lat_max, ngrid_lat)
    loni = np.linspace(lon_min, lon_max, ngrid_lon)

    triang = tri.Triangulation(lat, lon)
    interpolator = tri.LinearTriInterpolator(triang, data)
    Lati, Loni = np.meshgrid(lati, loni)
    data = interpolator(Lati, Loni)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())

    ax.contourf(Loni, Lati, data,
                transform=ccrs.PlateCarree(),
                levels=10,
                cmap='jet')
    ax.coastlines()
    ax.set_global()
    plt.show()


def main():
    # client = Client()
    start_date = '2019-01-01'
    end_date = '2019-01-07'

    files_20181201220181204 = file_product4specific_days(
        start_date, end_date, ['KBR1B', 'GNV1B'], r'..//..//..//..//gracefo_dataset', r"gracefo_1B_%Y-%m-%d_RL04.ascii.noLRI")

    # read in files
    kbr1b_x = pd.concat(map(lambda file: pd.read_csv(file,
                                                     skiprows=162,
                                                     names=['gps_time', 'biased_range', 'range_rate', 'range_accl', 'iono_corr', 'lighttime_err', 'lighttime_rate',
                                                            'lighttime_accl', 'ant_centr_corr', 'ant_centr_rate', 'ant_centr_accl', 'k_a_snr', 'ka_a_snr', 'k_b_snr',
                                                            'ka_b_snr', 'qualflg'],
                                                     dtype=np.float64,
                                                     sep='\s+'),
                            files_20181201220181204['KBR1B']),
                        ignore_index=True)
    gnv1b_c = pd.concat(map(lambda file: pd.read_csv(file,
                                                     skiprows=148,
                                                     names=['gps_time', 'id', 'coord_ref',
                                                            'xpos', 'ypos', 'zpos', 'xpos_err', 'ypos_err', 'zpos_err',
                                                            'xvel', 'yvel', 'zvel', 'xvel_err', 'yvel_err', 'zvel_err',
                                                            'qualflg'],
                                                     sep='\s+'),
                            [ctemp for ctemp in files_20181201220181204['GNV1B'] if '_C_' in ctemp]),
                        ignore_index=True)
    gnv1b_d = pd.concat(map(lambda file: pd.read_csv(file,
                                                     skiprows=148,
                                                     names=['gps_time', 'id', 'coord_ref',
                                                            'xpos', 'ypos', 'zpos', 'xpos_err', 'ypos_err', 'zpos_err',
                                                            'xvel', 'yvel', 'zvel', 'xvel_err', 'yvel_err', 'zvel_err',
                                                            'qualflg'],
                                                     sep='\s+'),
                            [ctemp for ctemp in files_20181201220181204['GNV1B'] if '_D_' in ctemp]),
                        ignore_index=True)

    # convert xyz to lat lon
    kbr1b_x['lat'], kbr1b_x['lon'] = xyz2latlon(start_date,
                                                gnv1b_c.gps_time.to_numpy().astype(np.float64)[::5],
                                                np.c_[gnv1b_c.xpos.to_numpy(),
                                                      gnv1b_c.ypos.to_numpy(),
                                                      gnv1b_c.zpos.to_numpy()])

    # ionosphere correction to TEC
    f_ka = 32e9  # unit in Hz
    # kbr1b_x['TEC'] = -kbr1b_x['iono_corr'] / 40.3 * f_ka**2
    # kbr1b_x['Ne'] = kbr1b_x['TEC'] / kbr1b_x.biased_range
    kbr1b_x["iono_corr_hf"] = kbr1b_x['iono_corr'].to_numpy(
    ) - kaiser(kbr1b_x['iono_corr'].to_numpy(), 0.1, 0.035)
    kbr1b_x["iono_corr_hf_abs"] = np.abs(kbr1b_x.iono_corr_hf.to_numpy())

    kbr1b_x = kbr1b_x[kbr1b_x.iono_corr_hf_abs > np.std(kbr1b_x.iono_corr_hf.to_numpy() * 2)]

    fig = plt.figure(figsize=(10,8))

    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree())

    ax.set_global()

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    gl = ax.gridlines(draw_labels=True, linewidth=2, linestyle="--")
    gl.xlabel_style = {'size': "20"}
    gl.ylabel_style = {'size': "20"}

    plt.scatter(x=kbr1b_x.lon, y=kbr1b_x.lat,
                c=kbr1b_x.iono_corr_hf,
                cmap='jet',
                s=25,
                alpha=0.9,
                transform=crs.PlateCarree())

    plt.show()

    # np.savetxt("iono_corr.txt", kbr1b_x['iono_corr'].to_numpy() - kaiser(kbr1b_x['iono_corr'].to_numpy(), 0.1, 0.035))

    # map_cntr(kbr1b_x.lat.to_numpy(), kbr1b_x.lon.to_numpy(), np.log10(-kbr1b_x.TEC.to_numpy()))
    # fig = go.Figure(data=go.Scattergeo(
    #     lon=kbr1b_x.lon,
    #     lat=kbr1b_x.lat,
    #     mode='markers',
    #     marker={
    #         "color": np.abs(kbr1b_x['iono_corr'].to_numpy() - kaiser(kbr1b_x['iono_corr'].to_numpy(), 0.1, 0.035)) * 1e6,
    #         "size": 4, #  np.abs(kbr1b_x['iono_corr'].to_numpy() - kaiser(kbr1b_x['iono_corr'].to_numpy(), 0.1, 0.035)) * 5e6,
    #         "colorbar_title": r"TECU $10^{14}$"
    #     },
    # ))
    # fig.update_layout(geo_scope='world')
    # fig.show()


def kaiser(x, fq, cutoff_hz, ripple_db=600.):
    # the desired width of the transition from pass to stop
    width = 0.12 / fq

    # the desired attenuation in the stop band in db: ripple_db

    # compute the kaiser parameter for the fir filter
    n, beta = kaiserord(ripple_db, width)
    print('The length of the lowpass filter is ', n, '.')

    # use firwin with a kaiser window
    taps = firwin(n,
                  cutoff_hz,
                  window=('kaiser', beta),
                  pass_zero='lowpass',
                  nyq=fq)

    # use filtfilt to filter x with the fir filter
    filtered_x = filtfilt(taps, 1.0, x)

    return filtered_x


if __name__ == "__main__":
    main()
