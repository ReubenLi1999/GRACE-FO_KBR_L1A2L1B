import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm


def cntr(lat, lon, data, filepath, lat_min=-90., lat_max=90., lon_min=-180., lon_max=180.):
    
    '''
    This is function to plot the contour image.
    '''
    ngrid_lat = 181
    ngrid_lon = 361
    
    lati = np.linspace(lat_min, lat_max, ngrid_lat)
    loni = np.linspace(lon_min, lon_max, ngrid_lon)

    triang = tri.Triangulation(lat, lon)
    interpolator = tri.LinearTriInterpolator(triang, data)
    Lati, Loni = np.meshgrid(lati, loni)
    data = interpolator(Lati, Loni)

    fig = plt.figure(figsize=(10, 5))

    map = Basemap(
        llcrnrlat=lat_min, llcrnrlon=lon_min, urcrnrlat=lat_max, urcrnrlon=lon_max, projection='cyl',
        resolution='c', lon_0=0
    )

    map.drawcoastlines(linewidth=0.3)
    map.drawcountries(linewidth=0.3)

    # draw parallels.
    parallels = np.arange(lat_min, lat_max, 10.)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10, linewidth=0.3)
    # draw meridians
    meridians = np.arange(lon_min, lon_max, 30.)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10, linewidth=0.3)

    # -- contour levels
    clevs = np.arange(data.min(), data.max(), 15)

    x, y = map(Lati, Loni)

    # plt.contour(y, x, data, levels=40, linewidths=0.5, colors='k')
    cntr = map.contourf(y, x, data, clevs, cmap=cm.jet)
    cbar = map.colorbar(cntr, location='bottom', pad='10%')
    plt.savefig(filepath, dpi=500)
    plt.show()
