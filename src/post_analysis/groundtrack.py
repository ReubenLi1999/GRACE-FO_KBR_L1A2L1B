import dask.dataframe
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
from datetime import timedelta
# from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import time
import matplotlib.tri as tri
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle
import plotly.graph_objects as go
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.io as pio


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
    tec_2018_12_01 = np.loadtxt(fname='..//..//output//tec_2018-12-01_with_coor.txt', dtype=np.float64)
    tec_2018_12_02 = np.loadtxt(fname='..//..//output//tec_2018-12-02_with_coor.txt', dtype=np.float64)
    tec_2018_12_03 = np.loadtxt(fname='..//..//output//tec_2018-12-03_with_coor.txt', dtype=np.float64)
    tec_2018_12_04 = np.loadtxt(fname='..//..//output//tec_2018-12-04_with_coor.txt', dtype=np.float64)
    tec_2018_12_05 = np.loadtxt(fname='..//..//output//tec_2018-12-05_with_coor.txt', dtype=np.float64)
    tec_2018_12_06 = np.loadtxt(fname='..//..//output//tec_2018-12-06_with_coor.txt', dtype=np.float64)
    tec_2018_12_07 = np.loadtxt(fname='..//..//output//tec_2018-12-07_with_coor.txt', dtype=np.float64)
    tec_2018_12_08 = np.loadtxt(fname='..//..//output//tec_2018-12-08_with_coor.txt', dtype=np.float64)
    tec_2018_12_09 = np.loadtxt(fname='..//..//output//tec_2018-12-09_with_coor.txt', dtype=np.float64)
    tec_2018_12_10 = np.loadtxt(fname='..//..//output//tec_2018-12-10_with_coor.txt', dtype=np.float64)
    tec_2018_12_11 = np.loadtxt(fname='..//..//output//tec_2018-12-11_with_coor.txt', dtype=np.float64)
    tec_2018_12_12 = np.loadtxt(fname='..//..//output//tec_2018-12-12_with_coor.txt', dtype=np.float64)
    tec_2018_12_13 = np.loadtxt(fname='..//..//output//tec_2018-12-13_with_coor.txt', dtype=np.float64)
    tec_2018_12_14 = np.loadtxt(fname='..//..//output//tec_2018-12-14_with_coor.txt', dtype=np.float64)
    tec_2018_12_15 = np.loadtxt(fname='..//..//output//tec_2018-12-15_with_coor.txt', dtype=np.float64)
    tec_2018_12_16 = np.loadtxt(fname='..//..//output//tec_2018-12-16_with_coor.txt', dtype=np.float64)
    tec_2018_12_17 = np.loadtxt(fname='..//..//output//tec_2018-12-17_with_coor.txt', dtype=np.float64)
    tec_2018_12_18 = np.loadtxt(fname='..//..//output//tec_2018-12-18_with_coor.txt', dtype=np.float64)
    tec_2018_12_19 = np.loadtxt(fname='..//..//output//tec_2018-12-19_with_coor.txt', dtype=np.float64)
    tec_2018_12_20 = np.loadtxt(fname='..//..//output//tec_2018-12-20_with_coor.txt', dtype=np.float64)
    tec_2018_12_21 = np.loadtxt(fname='..//..//output//tec_2018-12-21_with_coor.txt', dtype=np.float64)
    tec_2018_12_22 = np.loadtxt(fname='..//..//output//tec_2018-12-22_with_coor.txt', dtype=np.float64)
    tec_2018_12_23 = np.loadtxt(fname='..//..//output//tec_2018-12-23_with_coor.txt', dtype=np.float64)
    tec_2018_12_24 = np.loadtxt(fname='..//..//output//tec_2018-12-24_with_coor.txt', dtype=np.float64)
    tec_2018_12_25 = np.loadtxt(fname='..//..//output//tec_2018-12-25_with_coor.txt', dtype=np.float64)
    tec_2018_12_26 = np.loadtxt(fname='..//..//output//tec_2018-12-26_with_coor.txt', dtype=np.float64)
    tec_2018_12_27 = np.loadtxt(fname='..//..//output//tec_2018-12-27_with_coor.txt', dtype=np.float64)
    tec_2018_12_28 = np.loadtxt(fname='..//..//output//tec_2018-12-28_with_coor.txt', dtype=np.float64)


    tec = np.r_[
        tec_2018_12_01,
        tec_2018_12_02,
        tec_2018_12_03,
        tec_2018_12_04,
        tec_2018_12_05,
        tec_2018_12_06,
        tec_2018_12_07,
        tec_2018_12_08,
        tec_2018_12_09,
        tec_2018_12_10,
        tec_2018_12_11,
        tec_2018_12_12,
        tec_2018_12_13,
        tec_2018_12_14,
        tec_2018_12_15,
        tec_2018_12_16,
        tec_2018_12_17,
        tec_2018_12_18,
        tec_2018_12_19,
        tec_2018_12_20,
        tec_2018_12_21,
        tec_2018_12_22,
        tec_2018_12_23,
        tec_2018_12_24,
        tec_2018_12_25,
        tec_2018_12_26,
        tec_2018_12_27,
        tec_2018_12_28
    ]
    tec = tec[np.where(np.abs(tec[:, 2]) < 1e15)[0], :]
    tec = tec[0: 17280 * 3]
    df = pd.DataFrame({'lat': tec[:, 1], "lon": tec[:, 0], "data": tec[:, 2]})
    # ddf = dd.from_pandas(df)
    fig = go.Figure(data=go.Scattergeo(
        lon=df.lon,
        lat=df.lat,
        mode='markers+lines',
        marker={
            "color": df.data.to_numpy() / 1e14,
            "size": np.abs(df.data.to_numpy()) / df.data.to_numpy().max() * 15,
            'colorscale': 'jet',
            # "colorbar_title": r"TECU $10^{14}$"
        },
    ))
    fig.update_layout(geo_scope='world')
    fig.show()
    # app = dash.Dash()
    # app.layout = html.Div([dcc.Graph(figure=fig)])
    # app.run_server(debug=True, use_reloader=False)
    # cntr(tec[:, 1], tec[:, 0], tec[:, 2], filepath='./contour.png')


def cntr(lat, lon, data, filepath, lat_min=-90., lat_max=90., lon_min=-180., lon_max=180.):

    '''
    This is function to plot the contour image.
    '''
    ngrid_lat = 181
    ngrid_lon = 361
    data_in = data
    lati = np.linspace(lat_min, lat_max, ngrid_lat)
    loni = np.linspace(lon_min, lon_max, ngrid_lon)

    triang = tri.Triangulation(lat, lon)
    interpolator = tri.LinearTriInterpolator(triang, data)
    Lati, Loni = np.meshgrid(lati, loni)
    data = interpolator(Lati, Loni)

    map = Basemap(
        llcrnrlat=-90, llcrnrlon=-180, urcrnrlat=90, urcrnrlon=180, projection='cyl',
        resolution='c',
    )

    map.drawcoastlines()
    # map.drawcountries(linewidth=0.3)
    # draw parallels.
    parallels = np.arange(lat_min, lat_max, 30.)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10, linewidth=0.3)
    # draw meridians
    meridians = np.arange(lon_min, lon_max, 40.)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10, linewidth=0.3)

    # -- contour levels
    clevs = np.arange(data.min(), data.max(), 1e14)

    x, y = map(Lati, Loni)

    # plt.contour(y, x, data, levels=40, linewidths=0.5, colors='k')
    # cntr = map.contourf(y, x, data, clevs, cmap=cm.jet)
    # map.scatter(lon, lat, c=data_in, cmap=cm.jet, s=np.abs(data_in) / data_in.max())
    # cbar = map.colorbar(cntr, location='bottom', pad='10%')
    # plt.colorbar()
    # plt.savefig(filepath, dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
