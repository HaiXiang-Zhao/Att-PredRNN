import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def visualize(ax, v):
    # 经纬度范围和间隔
    lon_range = [-180, 180]
    lat_range = [-90, 90]
    lon_interval = 5.625
    lat_interval = 5.625
    
    # 创建经纬度网格
    lon = np.arange(lon_range[0], lon_range[1] + lon_interval, lon_interval)
    lat = np.arange(lat_range[0], lat_range[1] + lat_interval, lat_interval)
    
    # 显示地图数据
    cdict = ['#A9F090', '#5DCE47', '#40B73F', '#63B7FF', '#3354D0', '#0000FE', '#C700FF', '#E399FF', '#FF00FC', '#FF3BFA', '#AF005E', '#D23D43', '#FF4500','#FF6666', '#4c0082', '#843F62']
    my_cmap = colors.ListedColormap(cdict)
    
    clevs = [0.1,5., 10.,20., 30., 40., 50.,60., 70., 80.,90.,100.,110.,120.,125.]

    norm = colors.BoundaryNorm(clevs, my_cmap.N)

    im = ax.imshow(v, origin='lower', extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]], interpolation='bilinear', cmap=my_cmap, norm=norm)
    
 

    # 添加海岸线
    ax.coastlines()
    
    # 添加网格线
    ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='dotted')
    
    return ax


fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

ax = visualize(ax, a1[18])


plt.savefig('pred.png', dpi=300, bbox_inches='tight')
plt.show()
