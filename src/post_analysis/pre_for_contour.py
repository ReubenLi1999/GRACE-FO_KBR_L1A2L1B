import numpy as np
import dask.dataframe as dd
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle
from datetime import timedelta


def main():

    header_position = [
        'gps_time', 'id', 'coord_ref',
        'xpos', 'ypos', 'zpos', 'xpos_err', 'ypos_err', 'zpos_err',
        'xvel', 'yvel', 'zvel', 'xvel_err', 'yvel_err', 'zvel_err',
        'qualflg'
    ]
    dtype_position = {
        'gps_time': np.float64, 'id': 'str', 'coord_ref': 'str',
        'xpos': np.float64, 'ypos': np.float64, 'zpos': np.float64,
        'xvel': np.float64, 'yvel': np.float64, 'zvel': np.float64
    }
    gnv_c_data = dd.read_csv(
        urlpath="..//..//..//..//gracefo_dataset//gracefo_1B_2018-12-07_RL04.ascii.noLRI//GNV1B_2018-12-07_C_04.txt",
        skiprows=148,
        engine='c',
        header=None,
        names=header_position,
        sep='\s+',
        dtype=dtype_position
    )
    iono_corr = np.loadtxt(fname='..//..//output//tec_2018-12-07.txt', dtype=np.longdouble)
    iono_corr = np.r_[iono_corr, np.zeros(1000)]

    secofday = np.floor(gnv_c_data.gps_time.compute().to_numpy() - gnv_c_data.gps_time.compute().to_numpy()[0]).astype(np.int64)
    secofday = secofday[::5]
    time_stamps = np.asarray([], dtype=str)
    for _, sec in enumerate(secofday):
        sec = np.float64(sec)
        time_stamps = np.append(time_stamps, "2018-12-07 " + str(timedelta(seconds=sec)))
    xs = gnv_c_data['xpos'].compute().to_numpy()
    ys = gnv_c_data['ypos'].compute().to_numpy()
    zs = gnv_c_data['zpos'].compute().to_numpy()
    ground_lat = np.asarray([])
    ground_lon = np.asarray([])
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
    np.savetxt(fname='..//..//output//tec_2018-12-07_with_coor.txt', X=np.c_[ground_lon, ground_lat, iono_corr[0: 17280]])


if __name__ == '__main__':
    main()