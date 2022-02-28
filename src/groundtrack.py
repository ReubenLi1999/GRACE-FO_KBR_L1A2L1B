import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
from datetime import timedelta
import pmagpy.pmag as pmag
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import time
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle


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
        print("*" * 80)
        func(*args, **kwargs)
        print("*" * 80)
    return inner


def percent(func):
    def inner(*args, **kwargs):
        print("%" * 80)
        func(*args, **kwargs)
        print("%" * 80)
    return inner


@star
@percent
@timer
def main():
    secofday = np.unique(np.loadtxt("..//output//groundtrack.txt", dtype=int))
    time_stamps = np.asarray([], dtype=str)
    for _, sec in enumerate(secofday):
        sec = np.float64(sec)
        time_stamps = np.append(time_stamps, "2018-12-01 " + str(timedelta(seconds=sec)))

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
        urlpath="..//..//..//gracefo_dataset//gracefo_1B_2018-12-01_RL04.ascii.noLRI//GNV1B_2018-12-01_C_04.txt",
        skiprows=148,
        engine='c',
        header=None,
        names=header_position,
        sep='\s+',
        dtype=dtype_position
    )
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


    plt.style.use(['science', 'no-latex', 'high-vis'])
    fig, ax = plt.subplots(figsize=(10, 5))

    date, mod, lon_0, alt, ghfile = 2018.1201, 'cals10k.2', 0, 0, ""  # only date is required
    Ds, Is, Bs, Brs, lons, lats = pmag.do_mag_map(date, mod=mod, lon_0=lon_0, alt=alt, file=ghfile, resolution='low')

    map = Basemap(
        llcrnrlat=-90., llcrnrlon=-180., urcrnrlat=90., urcrnrlon=180., projection='cyl',
        resolution='c', lon_0=0
    )
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries(linewidth=0.3)
    # draw parallels.
    parallels = np.arange(-90, 90, 10.)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10, linewidth=0.3)
    # draw meridians
    meridians = np.arange(-180, 180, 30.)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10, linewidth=0.3)

    clevs = np.arange(Is.min(), Is.max(), 15)
    lons, lats = np.meshgrid(lons, lats)
    lons, lats = map(lons, lats)
    cntr = map.contour(lons, lats, Is, clevs, cmap=cm.jet)
    map.scatter(ground_lon, ground_lat)
    cbar = map.colorbar(cntr, location='bottom', pad='10%')
    plt.show()


if __name__ == "__main__":
    main()
