# conda activate SSElFaroPaper
import salem
from salem import GoogleVisibleMap, Map
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import seaborn as sns
import pandas as pd
import numpy as np
import xarray as xr
import importlib
import paper_utils as ut
import os
import sys


figdir = '/Volumes/SAMSUNG/WORK/POSTDOC_RSMAS_2016/PYTHON3/Publications/'\
'SSElFaro/figs/'
if not os.path.isdir(figdir):
    os.makedirs(figdir)


# Figures to create
region_map = False # Figure of domain with vessel and hurricane information

surface_currents_event = False # current map of event
surface_currents_clim = False # current climatology. NOT PAPER FIGURE
surface_currents_anom = False # HyCOM data minus climatology
vessel_speed_direction_along_track = False # Time seris of SOG and COG
current_speed_direction_along_track = False # Along track time series of current speed and direction

wind_event = False # wind map of event
wind_clim = False # Wind climatology. NOT PAPER FIGURE
wind_anom = False # ERA5 data minus climatology
wind_speed_direction_along_track = False # Along track time series of wind speed and direction

wave_event = False # hmax map of event
wave_clim = False # hmax climatology. NOT PAPER FIGURE
wave_anom = False # ERA5 data minus climatology
hmax_direction_along_track = False # Along track time series of hmax and mean wave direction
period_steepness_along_track = False # Along track time series of wave period

spectra_1d = True

if region_map:
    figname = 'domain'

    # Read in vessel location
    vessel_loc = ut.open_vessel_data()

    # Read in hurricane location
    hurricane = ut.open_hurricane_data()

    # create extent of map [W, E, S, N]
    mapextent = [-82, -65, 17, 31]

    # Get google map. Scale is for more details. Mapytype can have
    # 'terrain' or 'satellite'
    _maptype = 'satellite'
    g = GoogleVisibleMap(x=[mapextent[0], mapextent[1]],
                         y=[mapextent[2], mapextent[3]],
                         scale=4, maptype=_maptype)
    # Could look at size_x and size_y as well https://github.com/fmaussion/salem/blob/430b9c25a8ffb305fe115517d074b15322b6eafb/salem/datasets.py#L541
    #g = GoogleVisibleMap(x=[mapextent[0], mapextent[1]], y=[mapextent[2],
    #                        mapextent[3]], scale=4)
    ggl_img = g.get_vardata()

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    # make a map of the same size as the image (no country borders)
    sm = Map(g.grid, factor=1, countries=False)
    #sm = Map(g.grid, factor=1)
    # add the background rgb image
    sm.set_rgb(ggl_img)
    # add scale
    #sm.set_scale_bar(location=(0.88, 0.94))  # add scale

    # Plot the US states
    # Download the file from https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_1_states_provinces.zip
    gdf = salem.read_shapefile('data/ne_110m_admin_1_states_provinces.shp')
    names = gdf.name
    for _, name in enumerate(names):
        sm.set_shapefile(gdf.loc[gdf.name == name], linewidth=1)

    # plot it
    sm.visualize(ax=ax)

    # Plot vessel locations
    cmap = plt.get_cmap('cool')
    for i in range(len(vessel_loc)):
        x, y = sm.grid.transform(vessel_loc['longitude'][i],
                                 vessel_loc['latitude'][i])
        # Plot different color after flooding and after loss of propulstion
        # https://en.wikipedia.org/wiki/SS_El_Faro#Voyage_Data_Recorder_audio
        if i >= 67 and i <= 70:
            # Flooding
            ax.scatter(x, y, color=cmap(0.6), marker='d', label='SS El Faro: Flooding' if i == 67 else "")
        elif i >= 71:
            # Loss of propulstion
            ax.scatter(x, y, color=cmap(0.9), marker='d', label='SS El Faro: Lost propulsion' if i == 71 else "")
        else:
            ax.scatter(x, y, color=cmap(0), marker='d', label='SS El Faro' if i == 0 else "")

    # Plot hurricane locations
    cmap = plt.get_cmap('nipy_spectral')
    c1 = c2 = c3 = c4 = c5 = 0
    for i in range(len(vessel_loc)):
        x, y = sm.grid.transform(hurricane['Longitude'][i],
                                 hurricane['Latitude'][i])
        wind_speed = hurricane['Wind_WMO'][i]
        if wind_speed < 64.0:
            ax.scatter(x, y, color=cmap(0.1), marker='X', label='Joaquin: TS' if c1 == 0 else "")
            c1 += 1
        elif wind_speed >= 64.0 and wind_speed <= 82.0:
            ax.scatter(x, y, color=cmap(0.3), marker='X', label='Joaquin: Category 1' if c2 == 0 else "")
            c2 += 1
        elif wind_speed >= 83.0 and wind_speed <= 95.0:
            ax.scatter(x, y, color=cmap(0.5), marker='X', label='Joaquin: Category 2' if c3 == 0 else "")
            c3 += 1
        elif wind_speed >= 96.0 and wind_speed <= 112.0:
            ax.scatter(x, y, color=cmap(0.7), marker='X', label='Joaquin: Category 3' if c4 == 0 else "")
            c4 += 1
        else:
            ax.scatter(x, y, color=cmap(0.9), marker='X', label='Joaquin: Category 4' if c5 == 0 else "")
            c5 += 1

    # Plot Jacksonville, FL
    x, y = sm.grid.transform(-81.655651, 30.332184)
    ax.plot(x, y, 'ro')
    ax.text(x, y, 'Jacksonville, FL', color='white')
    # Plot San Juan, PR
    x, y = sm.grid.transform(-66.105735, 18.465539)
    ax.plot(x, y, 'ro')
    ax.text(x, y, 'San Juan, Puerto Rico', color='white')

    handles, labels = ax.get_legend_handles_labels()
    # Change the order put -1 to -3
    handles.insert(5, handles[-1])
    handles = handles[0:-1]
    labels.insert(5, labels[-1])
    labels = labels[0:-1]
    plt.legend(handles, labels, loc=3)

    plt.tight_layout()
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    #plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')
    plt.savefig(figdir+figname+'.pdf', bbox_inches='tight', format='pdf')

if surface_currents_event:
    figname = 'HyCOM_event'
    # Download the HyCOM data through a web browser using the netcdf subsetter
    # the region is download is (31.0/-84.0/15.0/-65.0) and time
    # starttime = '2015-09-28T00:00:00' endtime = '2015-10-04T00:00:00'
    # wget -O out.nc "http://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_53.X/2015/3hrly?var=water_u&north=31.0000&west=-84.0000&east=-65.0&south=15.0000&disableProjSubset=on&horizStride=1&time_start=2015-09-28T00%3A00%3A00Z&time_end=2015-10-04T00%3A00%3A00Z&timeStride=1&vertCoord=&accept=netcdf4"

    water_u, water_v = ut.open_HyCOM_data()
    water_u_2d = water_u.sel(time='2015-10-01T12:00:00')
    water_v_2d = water_v.sel(time='2015-10-01T12:00:00')

    cs_2d = ut.magnitude(water_u_2d, water_v_2d)
    cs_2d = cs_2d.rename('current_speed')

    # Plot
    #lattickspacing = 3.0
    #lontickspacing = 4.0
    #coastlineres = '10m'
    #maskland = 1
    #cbarmin = 0.0
    #cbarmax = 2.0
    #cbarstep = 0.2
    #cbarunits = r'm s$^{-1}$'
    #ccoltype = 'viridis'
    #plottitle = 'test'
    #figsavename = figdir+figname
    #cpl.da_2d_plot(cs_2d, lattickspacing, lontickspacing, cbarmin, cbarmax,
    #               cbarstep, cbarunits, ccoltype,
    #               coastlineres, plottitle, figsavename, maskland=1)
    #plt.figure(figsize=(5.12985642927, 3))
    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='black')
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-84.0, -65.0, 15.0, 31.0], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(15.0, 30.0+5.0, 5.0), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks(np.flip(np.arange(65.0, 80.0+5.0, 5.0)*-1, axis=0), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    p = ax.contourf(cs_2d.lon, cs_2d.lat, cs_2d, cmap=cmocean.cm.speed,
                    transform=ccrs.PlateCarree())
    q = ax.quiver(cs_2d.lon[::5], cs_2d.lat[::5],
                  water_u_2d[::5, ::5], water_v_2d[::5, ::5], transform=ccrs.PlateCarree())
    qk = plt.quiverkey(q, 0.05, 1.02, 1, 'velocity (1 m s$^{-1}$)', labelpos='E', transform=ax.transAxes)
    ax.coastlines(resolution='110m')
    #ax.add_feature(land_10m)

    # Plot vessel location
    vessel_loc = ut.open_vessel_data()
    x = vessel_loc['longitude'].iloc[-1]
    y = vessel_loc['latitude'].iloc[-1]
    _cmap = plt.get_cmap('cool')
    ax.scatter(x, y, color=_cmap(0.6), marker='d', s=100, transform=ccrs.PlateCarree())

    # Read in hurricane location
    hurricane = ut.open_hurricane_data()
    _loc = hurricane[hurricane['time_iso'] == '2015-10-01T12:00:00']
    x = _loc['Longitude']
    y = _loc['Latitude']
    _cmap = plt.get_cmap('nipy_spectral')
    ax.scatter(x, y, color=_cmap(0.9), marker='X', s=100, transform=ccrs.PlateCarree())

    # Plot San Juan, PR
    ax.plot(-66.105735, 18.465539, 'ro', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(p)
    cbar.set_label('ocean current speed (m s$^{-1}$)')
    plt.title('HyCOM 2015-10-01T12:00:00 UTC')
    plt.tight_layout()
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if surface_currents_clim:
    figname = 'HyCOM_climatology'
    # Use HyCOM climatology
    u, v = ut.open_HyCOM_clim_data()
    # http://ncss.hycom.org/thredds/ncss/grid/datasets/GLBv0.08/expt_53.X/meanstd/netcdf/hycom_GLBv0.08_53X_archMN.1994_01_2015_12_uv3z.nc/dataset.html
    # http://ncss.hycom.org/thredds/ncss/datasets/GLBv0.08/expt_53.X/meanstd/netcdf/hycom_GLBv0.08_53X_archMN.1994_01_2015_12_uv3z.nc?var=water_u&north=31.0000&west=-84.0000&east=-65.0&south=15.0000&disableProjSubset=on&horizStride=1&time_start=2004-11-26T18%3A43%3A12Z&time_end=2004-11-26T18%3A43%3A12Z&timeStride=1&vertCoord=&accept=netcdf4
    #ds = xr.open_dataset('/Volumes/SAMSUNG/WORK/POSTDOC_RSMAS_2016/DATA/GLD/gld_climatology.nc')
    cs = ut.magnitude(u, v)
    cs = cs.rename('current_speed')

    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-84.0, -65.0, 15.0, 31.0], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(15.0, 30.0+5.0, 5.0), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks(np.flip(np.arange(65.0, 80.0+5.0, 5.0)*-1, axis=0), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    p = ax.contourf(cs.lon, cs.lat, cs, cmap=cmocean.cm.speed,
                    transform=ccrs.PlateCarree())
    q = ax.quiver(cs.lon[::5], cs.lat[::5],
                  u[::5, ::5], v[::5, ::5], transform=ccrs.PlateCarree())
    qk = plt.quiverkey(q, 0.05, 1.02, 1, 'velocity (1 m s$^{-1}$)', labelpos='E', transform=ax.transAxes)
    ax.coastlines(resolution='110m')

    cbar = plt.colorbar(p)
    cbar.set_label('ocean current speed (m s$^{-1}$)')
    plt.title('HyCOM annual climatology')
    plt.tight_layout()
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if surface_currents_anom:
    figname = 'HyCOM_anomaly'

    water_u, water_v = ut.open_HyCOM_data()
    water_u_2d = water_u.sel(time='2015-10-01T12:00:00')
    water_v_2d = water_v.sel(time='2015-10-01T12:00:00')
    cs_2d = ut.magnitude(water_u_2d, water_v_2d)
    cs_2d = cs_2d.rename('current_speed')

    u, v = ut.open_HyCOM_clim_data()
    cs = ut.magnitude(u, v)
    cs = cs.rename('current_speed')

    u_anom = water_u_2d - u
    v_anom = water_v_2d - v
    cs_anom = cs_2d - cs
    colorbar_ticks = np.arange(-1.6, 1.8, 0.2)

    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-84.0, -65.0, 15.0, 31.0], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(15.0, 30.0+5.0, 5.0), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks(np.flip(np.arange(65.0, 80.0+5.0, 5.0)*-1, axis=0), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    p = ax.contourf(cs.lon, cs.lat, cs_anom, colorbar_ticks, cmap="RdBu_r",
                    transform=ccrs.PlateCarree())
    q = ax.quiver(cs_anom.lon[::5], cs_anom.lat[::5],
                  u_anom[::5, ::5], v_anom[::5, ::5], transform=ccrs.PlateCarree())
    qk = plt.quiverkey(q, 0.05, 1.02, 1.0, 'velocity anomaly (1 m s$^{-1}$)', labelpos='E', transform=ax.transAxes)
    ax.coastlines(resolution='110m')

    # Plot vessel location
    vessel_loc = ut.open_vessel_data()
    x = vessel_loc['longitude'].iloc[-1]
    y = vessel_loc['latitude'].iloc[-1]
    _cmap = plt.get_cmap('cool')
    ax.scatter(x, y, color=_cmap(0.6), marker='d', s=100, transform=ccrs.PlateCarree())

    # Read in hurricane location
    hurricane = ut.open_hurricane_data()
    _loc = hurricane[hurricane['time_iso'] == '2015-10-01T12:00:00']
    x = _loc['Longitude']
    y = _loc['Latitude']
    _cmap = plt.get_cmap('nipy_spectral')
    ax.scatter(x, y, color=_cmap(0.9), marker='X', s=100, transform=ccrs.PlateCarree())

    # Plot San Juan, PR
    ax.plot(-66.105735, 18.465539, 'ro', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(p, ticks=colorbar_ticks)
    cbar.set_label('ocean current speed anomaly (m s$^{-1}$)')
    plt.title('HyCOM anomaly 2015-10-01T12:00:00 UTC', loc='right')
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')


if vessel_speed_direction_along_track:
    figname = 'sog_cog_ts'
    vessel_data = ut.open_vessel_data()
    _sog = pd.Series(vessel_data['sog'].values, index=vessel_data['time_utc'])

    _xlabels = pd.date_range('2015-09-30 06:00:00', '2015-10-01 12:00:00',
                             freq='3H')

    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    ax.plot(vessel_data['time_utc'], vessel_data['sog'].values, 'b')
    ax.set_ylabel('Speed over ground (m s$^{-1}$)', color='b')
    ax.set_xlabel('Time (UTC; MM-DD HH)')
    ax.tick_params('y', colors='b')
    ax2 = ax.twinx()
    ax2.plot(vessel_data['time_utc'], vessel_data['cog'].values, 'r')
    ax2.set_ylabel('Course over ground (direction to ($^{\circ}$))', color='r')
    ax2.tick_params('y', colors='r')
    # Plot straights corresponding to time period of wave spectra plot
    ax2.plot([_xlabels[-4], _xlabels[-4]], [min(vessel_data['cog'].values), max(vessel_data['cog'].values)], 'black', linewidth=0.5)
    ax2.plot([_xlabels[-1], _xlabels[-1]], [min(vessel_data['cog'].values), max(vessel_data['cog'].values)], 'black', linewidth=0.5)
    locs, labels = plt.xticks()
    locs += 1./24
    plt.xticks(locs, _xlabels[1:].strftime('%m-%d %H'))
    #print(locs)
    #print(labels)
    #plt.xticks(np.arange(len(_xlabels)), _xlabels)
    fig.autofmt_xdate()
    plt.xlabel('Time (UTC; MM-DD HH)')

    plt.title('SS El Faro speed over ground and course over ground')
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')


if current_speed_direction_along_track:
    figname = 'HyCOM_ts'
    water_u, water_v = ut.open_HyCOM_data()
    cs = ut.magnitude(water_u, water_v).rename('current_speed')
    dir = ut.direction(water_u, water_v).rename('current_direction')
    conv_points = dir.where(dir < 0) + 360
    old_points = dir.where(dir >= 0)
    dir = xr.merge([old_points, conv_points])['current_direction']

    vessel_data = ut.open_vessel_data()
    vessel_times = vessel_data['time_utc']

    # Interpolate ocean data to vessel time
    cs_vessel_times = cs.interp(time=vessel_times)
    dir_vessel_times = dir.interp(time=vessel_times)

    # Index where the vessel is at each time...
    # Do in a loop to start with
    cs_along_track = xr.DataArray(np.empty(len(cs_vessel_times.time)),
                                  [('time', cs_vessel_times.time)])
    dir_along_track = cs_along_track.copy()
    for i, _time in enumerate(cs_vessel_times.time):
        _lat = vessel_data.iloc[i]['latitude']
        _lon = vessel_data.iloc[i]['longitude']

        cs_along_track.values[i] = cs_vessel_times.sel(time=_time).interp(lat=_lat, lon=_lon)
        dir_along_track.values[i] = dir_vessel_times.sel(time=_time).interp(lat=_lat, lon=_lon)

    _xlabels = pd.date_range('2015-09-30 06:00:00', '2015-10-01 12:00:00',
                             freq='3H')

    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    ax.plot(vessel_data['time_utc'], cs_along_track.values, 'b')
    ax.set_ylabel('ocean current speed (m s$^{-1}$)', color='b')
    ax.set_xlabel('Time (UTC; MM-DD HH)')
    ax.tick_params('y', colors='b')
    ax2 = ax.twinx()
    ax2.plot(vessel_data['time_utc'], dir_along_track.values, 'r')
    ax2.set_ylabel('ocean current direction (direction to ($^{\circ}$))', color='r')
    ax2.tick_params('y', colors='r')
    # Plot straights corresponding to time period of wave spectra plot
    ax2.plot([_xlabels[-4], _xlabels[-4]], [min(dir_along_track.values), max(dir_along_track.values)], 'black', linewidth=0.5)
    ax2.plot([_xlabels[-1], _xlabels[-1]], [min(dir_along_track.values), max(dir_along_track.values)], 'black', linewidth=0.5)
    locs, labels = plt.xticks()
    locs += 1./24
    plt.xticks(locs, _xlabels[1:].strftime('%m-%d %H'))
    #print(locs)
    #print(labels)
    #plt.xticks(np.arange(len(_xlabels)), _xlabels)
    fig.autofmt_xdate()
    plt.xlabel('Time (UTC; MM-DD HH)')

    plt.title('HyCOM current speed and direction along SS El Faro track')
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')


if wind_event:
    figname = 'wind_event'

    wind_u, wind_v, mslp = ut.open_ERA5_sfc_data()
    wind_u_2d = wind_u.sel(time='2015-10-01T12:00:00')
    wind_v_2d = wind_v.sel(time='2015-10-01T12:00:00')

    ws_2d = ut.magnitude(wind_u_2d, wind_v_2d).rename('wind_speed')

    land_cf = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='white')
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-84.0, -65.0, 15.0, 31.0], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(15.0, 30.0+5.0, 5.0), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks(np.flip(np.arange(65.0, 80.0+5.0, 5.0)*-1, axis=0), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    p = ax.contourf(wind_u_2d.longitude, wind_u_2d.latitude, ws_2d,
                    transform=ccrs.PlateCarree())
    q = ax.quiver(wind_u_2d.longitude[::3], wind_u_2d.latitude[::3],
                  wind_u_2d[::3, ::3], wind_v_2d[::3, ::3], transform=ccrs.PlateCarree())
    qk = plt.quiverkey(q, 0.05, 1.02, 10, 'velocity (10 m s$^{-1}$)', labelpos='E', transform=ax.transAxes)
    ax.coastlines(resolution='110m')
    ax.add_feature(land_cf)

    # Plot vessel location
    vessel_loc = ut.open_vessel_data()
    x = vessel_loc['longitude'].iloc[-1]
    y = vessel_loc['latitude'].iloc[-1]
    _cmap = plt.get_cmap('cool')
    ax.scatter(x, y, color=_cmap(0.6), marker='d', s=100, transform=ccrs.PlateCarree())

    # Read in hurricane location
    hurricane = ut.open_hurricane_data()
    _loc = hurricane[hurricane['time_iso'] == '2015-10-01T12:00:00']
    x = _loc['Longitude']
    y = _loc['Latitude']
    _cmap = plt.get_cmap('nipy_spectral')
    ax.scatter(x, y, color=_cmap(0.9), marker='X', s=100, transform=ccrs.PlateCarree())

    # Plot San Juan, PR
    ax.plot(-66.105735, 18.465539, 'ro', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(p)
    cbar.set_label('10-m wind speed (m s$^{-1}$)')
    plt.title('ERA5 2015-10-01T12:00:00 UTC')
    plt.tight_layout()
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if wind_clim:
    figname = 'ERA5_climatology'

    wind_u, wind_v, mslp = ut.open_ERA5_sfc_clim_data()

    ws = ut.magnitude(wind_u, wind_v).rename('wind_speed')

    land_cf = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='white')
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-84.0, -65.0, 15.0, 31.0], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(15.0, 30.0+5.0, 5.0), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks(np.flip(np.arange(65.0, 80.0+5.0, 5.0)*-1, axis=0), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    p = ax.contourf(wind_u.longitude, wind_u.latitude, ws,
                    transform=ccrs.PlateCarree())
    q = ax.quiver(wind_u.longitude[::3], wind_u.latitude[::3],
                  wind_u[::3, ::3], wind_v[::3, ::3], transform=ccrs.PlateCarree())
    qk = plt.quiverkey(q, 0.05, 1.02, 10, 'velocity (10 m s$^{-1}$)', labelpos='E', transform=ax.transAxes)
    ax.coastlines(resolution='110m')
    ax.add_feature(land_cf)

    # Plot vessel location
    vessel_loc = ut.open_vessel_data()
    x = vessel_loc['longitude'].iloc[-1]
    y = vessel_loc['latitude'].iloc[-1]
    _cmap = plt.get_cmap('cool')
    ax.scatter(x, y, color=_cmap(0.6), marker='d', s=100, transform=ccrs.PlateCarree())

    # Read in hurricane location
    hurricane = ut.open_hurricane_data()
    _loc = hurricane[hurricane['time_iso'] == '2015-10-01T12:00:00']
    x = _loc['Longitude']
    y = _loc['Latitude']
    _cmap = plt.get_cmap('nipy_spectral')
    ax.scatter(x, y, color=_cmap(0.9), marker='X', s=100, transform=ccrs.PlateCarree())

    # Plot San Juan, PR
    ax.plot(-66.105735, 18.465539, 'ro', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(p)
    cbar.set_label('10-m wind speed (m s$^{-1}$)')
    plt.title('ERA5 annual climatology')
    plt.tight_layout()
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if wind_anom:
    figname = 'ERA5_anomaly'

    wind_u, wind_v, mslp = ut.open_ERA5_sfc_data()
    wind_u_2d = wind_u.sel(time='2015-10-01T12:00:00')
    wind_v_2d = wind_v.sel(time='2015-10-01T12:00:00')
    ws_2d = ut.magnitude(wind_u_2d, wind_v_2d).rename('wind_speed')

    u, v, mslp = ut.open_ERA5_sfc_clim_data()
    ws = ut.magnitude(u, v).rename('wind_speed')

    wind_u, wind_v, mslp = ut.open_ERA5_sfc_data()
    wind_u_2d = wind_u.sel(time='2015-10-01T12:00:00')
    wind_v_2d = wind_v.sel(time='2015-10-01T12:00:00')

    ws_2d = ut.magnitude(wind_u_2d, wind_v_2d).rename('wind_speed')

    u_anom = wind_u_2d - u
    v_anom = wind_v_2d - v
    ws_anom = ws_2d - ws

    colorbar_ticks = np.arange(-32.0, 36.0, 4.0)

    land_cf = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='white')
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-84.0, -65.0, 15.0, 31.0], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(15.0, 30.0+5.0, 5.0), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks(np.flip(np.arange(65.0, 80.0+5.0, 5.0)*-1, axis=0), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    p = ax.contourf(ws_anom.longitude, ws_anom.latitude, ws_anom, colorbar_ticks, cmap="RdBu_r",
                    transform=ccrs.PlateCarree())
    q = ax.quiver(u_anom.longitude[::3], u_anom.latitude[::3],
                  u_anom[::3, ::3], v_anom[::3, ::3], transform=ccrs.PlateCarree())
    qk = plt.quiverkey(q, 0.05, 1.02, 10, 'velocity anomaly (10 m s$^{-1}$)', labelpos='E', transform=ax.transAxes)
    ax.coastlines(resolution='110m')
    ax.add_feature(land_cf)

    # Plot vessel location
    vessel_loc = ut.open_vessel_data()
    x = vessel_loc['longitude'].iloc[-1]
    y = vessel_loc['latitude'].iloc[-1]
    _cmap = plt.get_cmap('cool')
    ax.scatter(x, y, color=_cmap(0.6), marker='d', s=100, transform=ccrs.PlateCarree())

    # Read in hurricane location
    hurricane = ut.open_hurricane_data()
    _loc = hurricane[hurricane['time_iso'] == '2015-10-01T12:00:00']
    x = _loc['Longitude']
    y = _loc['Latitude']
    _cmap = plt.get_cmap('nipy_spectral')
    ax.scatter(x, y, color=_cmap(0.9), marker='X', s=100, transform=ccrs.PlateCarree())

    # Plot San Juan, PR
    ax.plot(-66.105735, 18.465539, 'ro', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(p)
    cbar.set_label('10-m wind speed anomaly (m s$^{-1}$)')
    plt.title('ERA5 anomaly 2015-10-01T12:00:00 UTC', loc='right')
    plt.tight_layout()
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if wind_speed_direction_along_track:
    figname = 'ERA5_wind_ts'

    wind_u, wind_v, mslp = ut.open_ERA5_sfc_data()
    ws = ut.magnitude(wind_u, wind_v).rename('wind_speed')
    dir = ut.direction_from(wind_v, wind_u).rename('wind_direction')
    conv_points = dir.where(dir < 0) + 360
    old_points = dir.where(dir >= 0)
    dir = xr.merge([old_points, conv_points])['wind_direction']

    vessel_data = ut.open_vessel_data()
    vessel_times = vessel_data['time_utc']

    ws_vessel_times = ws.interp(time=vessel_times)
    dir_vessel_times = dir.interp(time=vessel_times)

    ws_along_track = xr.DataArray(np.empty(len(ws_vessel_times.time)),
                                  [('time', ws_vessel_times.time)])
    dir_along_track = ws_along_track.copy()
    for i, _time in enumerate(ws_vessel_times.time):
        _lat = vessel_data.iloc[i]['latitude']
        _lon = vessel_data.iloc[i]['longitude']

        ws_along_track.values[i] = ws_vessel_times.sel(time=_time).interp(latitude=_lat, longitude=_lon)
        dir_along_track.values[i] = dir_vessel_times.sel(time=_time).interp(latitude=_lat, longitude=_lon)

    _xlabels = pd.date_range('2015-09-30 06:00:00', '2015-10-01 12:00:00',
                             freq='3H')

    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    ax.plot(vessel_data['time_utc'], ws_along_track.values, 'b')
    ax.set_ylabel('10-m wind speed (m s$^{-1}$)', color='b')
    ax.set_xlabel('Time (UTC; MM-DD HH)')
    ax.tick_params('y', colors='b')
    ax2 = ax.twinx()
    ax2.plot(vessel_data['time_utc'], dir_along_track.values, 'r')
    ax2.set_ylabel('10-m wind direction (direction from ($^{\circ}$))', color='r')
    ax2.tick_params('y', colors='r')
    # Plot straights corresponding to time period of wave spectra plot
    ax2.plot([_xlabels[-4], _xlabels[-4]], [min(dir_along_track.values), max(dir_along_track.values)], 'black', linewidth=0.5)
    ax2.plot([_xlabels[-1], _xlabels[-1]], [min(dir_along_track.values), max(dir_along_track.values)], 'black', linewidth=0.5)
    locs, labels = plt.xticks()
    locs += 1./24
    plt.xticks(locs, _xlabels[1:].strftime('%m-%d %H'))
    #print(locs)
    #print(labels)
    #plt.xticks(np.arange(len(_xlabels)), _xlabels)
    fig.autofmt_xdate()
    plt.xlabel('Time (UTC; MM-DD HH)')

    plt.title('ERA5 10-m wind speed and direction along SS El Faro track')
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if wave_event:
    figname = 'hmax_event'

    hmax, mp2, swh, mwd, pp1d, mwp, bfi, tmax, wss, wdw, wsk = ut.open_ERA5_wave_data()
    hmax_2d = hmax.sel(time='2015-10-01T12:00:00')
    mwd_2d = mwd.sel(time='2015-10-01T12:00:00')

    u, v = ut.direction_to_magnitude(mwd_2d)

    land_cf = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='white')
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-84.0, -65.0, 15.0, 31.0], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(15.0, 30.0+5.0, 5.0), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks(np.flip(np.arange(65.0, 80.0+5.0, 5.0)*-1, axis=0), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    p = ax.contourf(hmax_2d.longitude, hmax_2d.latitude, hmax_2d,
                    transform=ccrs.PlateCarree())
    q = ax.quiver(hmax_2d.longitude[::3], hmax_2d.latitude[::3],
                  u[::3, ::3], v[::3, ::3], transform=ccrs.PlateCarree())
    qk = plt.quiverkey(q, 0.05, 1.02, 1.0, 'mean wave direction', labelpos='E', transform=ax.transAxes)
    ax.coastlines(resolution='110m')
    ax.add_feature(land_cf)

    # Plot vessel location
    vessel_loc = ut.open_vessel_data()
    x = vessel_loc['longitude'].iloc[-1]
    y = vessel_loc['latitude'].iloc[-1]
    _cmap = plt.get_cmap('cool')
    ax.scatter(x, y, color=_cmap(0.6), marker='d', s=100, transform=ccrs.PlateCarree())

    # Read in hurricane location
    hurricane = ut.open_hurricane_data()
    _loc = hurricane[hurricane['time_iso'] == '2015-10-01T12:00:00']
    x = _loc['Longitude']
    y = _loc['Latitude']
    _cmap = plt.get_cmap('nipy_spectral')
    ax.scatter(x, y, color=_cmap(0.9), marker='X', s=100, transform=ccrs.PlateCarree())

    # Plot San Juan, PR
    ax.plot(-66.105735, 18.465539, 'ro', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(p)
    cbar.set_label('Maximum individual wave height (m)')
    plt.title('ERA5 2015-10-01T12:00:00 UTC')
    plt.tight_layout()
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if wave_clim:
    figname = 'hmax_climatology'

    hmax, u, v = ut.open_ERA5_wave_clim_data()

    land_cf = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='white')
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-84.0, -65.0, 15.0, 31.0], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(15.0, 30.0+5.0, 5.0), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks(np.flip(np.arange(65.0, 80.0+5.0, 5.0)*-1, axis=0), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    p = ax.contourf(hmax.longitude, hmax.latitude, hmax,
                    transform=ccrs.PlateCarree())
    q = ax.quiver(hmax.longitude[::3], hmax.latitude[::3],
                  u[::3, ::3], v[::3, ::3], transform=ccrs.PlateCarree())
    #qk = plt.quiverkey(q, 0.05, 1.02, 10, 'velocity (10 m s$^{-1}$)', labelpos='E', transform=ax.transAxes)
    ax.coastlines(resolution='110m')
    ax.add_feature(land_cf)

    # Plot vessel location
    vessel_loc = ut.open_vessel_data()
    x = vessel_loc['longitude'].iloc[-1]
    y = vessel_loc['latitude'].iloc[-1]
    _cmap = plt.get_cmap('cool')
    ax.scatter(x, y, color=_cmap(0.6), marker='d', s=100, transform=ccrs.PlateCarree())

    # Read in hurricane location
    hurricane = ut.open_hurricane_data()
    _loc = hurricane[hurricane['time_iso'] == '2015-10-01T12:00:00']
    x = _loc['Longitude']
    y = _loc['Latitude']
    _cmap = plt.get_cmap('nipy_spectral')
    ax.scatter(x, y, color=_cmap(0.9), marker='X', s=100, transform=ccrs.PlateCarree())

    # Plot San Juan, PR
    ax.plot(-66.105735, 18.465539, 'ro', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(p)
    cbar.set_label('Hmax (m)')
    plt.title('ERA5 annual climatology')
    plt.tight_layout()
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if wave_anom:
    figname = 'hmax_anomaly'

    hmax, mp2, swh, mwd, pp1d, mwp, bfi, tmax, wss, wdw, wsk = ut.open_ERA5_wave_data()
    hmax_2d = hmax.sel(time='2015-10-01T12:00:00')
    mwd_2d = mwd.sel(time='2015-10-01T12:00:00')
    u_2d, v_2d = ut.direction_to_magnitude(mwd_2d)

    hmax, u, v = ut.open_ERA5_wave_clim_data()

    u_anom = u_2d - u
    v_anom = v_2d - v
    hmax_anom = hmax_2d - hmax

    colorbar_ticks = np.arange(-16.0, 18.0, 2.0)

    land_cf = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor='white')
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-84.0, -65.0, 15.0, 31.0], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(15.0, 30.0+5.0, 5.0), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticks(np.flip(np.arange(65.0, 80.0+5.0, 5.0)*-1, axis=0), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    p = ax.contourf(hmax_anom.longitude, hmax_anom.latitude, hmax_anom, colorbar_ticks, cmap="RdBu_r",
                    transform=ccrs.PlateCarree())
    q = ax.quiver(hmax_anom.longitude[::3], hmax_anom.latitude[::3],
                  u_anom[::3, ::3], v_anom[::3, ::3], transform=ccrs.PlateCarree())
    qk = plt.quiverkey(q, 0.05, 1.02, 1, 'mean wave direction anomaly', labelpos='E', transform=ax.transAxes)
    ax.coastlines(resolution='110m')
    ax.add_feature(land_cf)

    # Plot vessel location
    vessel_loc = ut.open_vessel_data()
    x = vessel_loc['longitude'].iloc[-1]
    y = vessel_loc['latitude'].iloc[-1]
    _cmap = plt.get_cmap('cool')
    ax.scatter(x, y, color=_cmap(0.6), marker='d', s=100, transform=ccrs.PlateCarree())

    # Read in hurricane location
    hurricane = ut.open_hurricane_data()
    _loc = hurricane[hurricane['time_iso'] == '2015-10-01T12:00:00']
    x = _loc['Longitude']
    y = _loc['Latitude']
    _cmap = plt.get_cmap('nipy_spectral')
    ax.scatter(x, y, color=_cmap(0.9), marker='X', s=100, transform=ccrs.PlateCarree())

    # Plot San Juan, PR
    ax.plot(-66.105735, 18.465539, 'ro', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(p)
    cbar.set_label('Maximum individual wave height anomaly (m)')
    plt.title('ERA5 anomaly 2015-10-01T12:00:00 UTC', loc='right')
    plt.tight_layout()
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if hmax_direction_along_track:
    figname = 'ERA5_hmax_mwd_ts'

    hmax, mp2, swh, mwd, pp1d, mwp, bfi, tmax, wss, wdw, wsk = ut.open_ERA5_wave_data()
    u, v = ut.direction_to_magnitude(mwd)

    vessel_data = ut.open_vessel_data()
    vessel_times = vessel_data['time_utc']

    hmax_vessel_times = hmax.interp(time=vessel_times)
    u_vessel_times = u.interp(time=vessel_times)
    v_vessel_times = v.interp(time=vessel_times)

    hmax_along_track = xr.DataArray(np.empty(len(hmax_vessel_times.time)),
                                    [('time', hmax_vessel_times.time)])
    u_along_track = hmax_along_track.copy()
    v_along_track = hmax_along_track.copy()
    for i, _time in enumerate(hmax_vessel_times.time):
        _lat = vessel_data.iloc[i]['latitude']
        _lon = vessel_data.iloc[i]['longitude']

        hmax_along_track.values[i] = hmax_vessel_times.sel(time=_time).interp(latitude=_lat, longitude=_lon)
        u_along_track.values[i] = u_vessel_times.sel(time=_time).interp(latitude=_lat, longitude=_lon)
        v_along_track.values[i] = v_vessel_times.sel(time=_time).interp(latitude=_lat, longitude=_lon)

    mwd_along_track = ut.direction(u_along_track, v_along_track).rename('mwd')
    conv_points = mwd_along_track.where(mwd_along_track < 0) + 360
    old_points = mwd_along_track.where(mwd_along_track >= 0)
    mwd_along_track = xr.merge([old_points, conv_points])['mwd']

    _xlabels = pd.date_range('2015-09-30 06:00:00', '2015-10-01 12:00:00',
                             freq='3H')

    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    ax.plot(vessel_data['time_utc'], hmax_along_track.values, 'b')
    ax.set_ylabel('Maximum individual wave height (m)', color='b')
    ax.set_xlabel('Time (UTC; MM-DD HH)')
    ax.tick_params('y', colors='b')
    ax2 = ax.twinx()
    ax2.plot(vessel_data['time_utc'], mwd_along_track.values, 'r')
    ax2.set_ylabel('Mean wave direction (direction to ($^{\circ}$))', color='r')
    ax2.tick_params('y', colors='r')
    # Plot straights corresponding to time period of wave spectra plot
    ax2.plot([_xlabels[-4], _xlabels[-4]], [min(mwd_along_track.values), max(mwd_along_track.values)], 'black', linewidth=0.5)
    ax2.plot([_xlabels[-1], _xlabels[-1]], [min(mwd_along_track.values), max(mwd_along_track.values)], 'black', linewidth=0.5)
    locs, labels = plt.xticks()
    locs += 1./24
    plt.xticks(locs, _xlabels[1:].strftime('%m-%d %H'))
    fig.autofmt_xdate()
    plt.xlabel('Time (UTC; MM-DD HH)')

    plt.title('ERA5 maximum individual wave height and mean wave direction along SS El Faro track')
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if period_steepness_along_track:
    figname = 'ERA5_tmax_steepness_ts'

    hmax, mp2, swh, mwd, pp1d, mwp, bfi, tmax, wss, wdw, wsk, steepness = ut.open_ERA5_wave_data()

    vessel_data = ut.open_vessel_data()
    vessel_times = vessel_data['time_utc']

    tmax_vessel_times = tmax.interp(time=vessel_times)
    steepness_vessel_times = steepness.interp(time=vessel_times)
    tmax_along_track = xr.DataArray(np.empty(len(tmax_vessel_times.time)),
                                  [('time', tmax_vessel_times.time)])
    steepness_along_track = tmax_along_track.copy()
    for i, _time in enumerate(tmax_vessel_times.time):
        _lat = vessel_data.iloc[i]['latitude']
        _lon = vessel_data.iloc[i]['longitude']

        tmax_along_track.values[i] = tmax_vessel_times.sel(time=_time).interp(latitude=_lat, longitude=_lon)
        steepness_along_track.values[i] = steepness_vessel_times.sel(time=_time).interp(latitude=_lat, longitude=_lon)

    _xlabels = pd.date_range('2015-09-30 06:00:00', '2015-10-01 12:00:00',
                             freq='3H')

    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    ax.plot(vessel_data['time_utc'], tmax_along_track.values, 'b')
    ax.set_ylabel('Period corresponding to maximum individual wave height (s)', color='b')
    ax.set_xlabel('Time (UTC; MM-DD HH)')
    ax.tick_params('y', colors='b')
    ax2 = ax.twinx()
    ax2.plot(vessel_data['time_utc'], steepness_along_track.values, 'r')
    ax2.set_ylabel('Maximum wave steepness', color='r')
    ax2.tick_params('y', colors='r')
    # Plot straights corresponding to time period of wave spectra plot
    ax.plot([_xlabels[-4], _xlabels[-4]], [min(tmax_along_track.values), max(tmax_along_track.values)], 'black', linewidth=0.5)
    ax.plot([_xlabels[-1], _xlabels[-1]], [min(tmax_along_track.values), max(tmax_along_track.values)], 'black', linewidth=0.5)
    locs, labels = plt.xticks()
    locs += 1./24
    plt.xticks(locs, _xlabels[1:].strftime('%m-%d %H'))
    fig.autofmt_xdate()
    plt.xlabel('Time (UTC; MM-DD HH)')

    plt.title('ERA5 period corresponding to maximum individual wave height and wave steepness along SS El Faro track')
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')

if spectra_1d:
    figname = 'ERA5_1d_spectra'

    # Read in spectra for time and location of interest
    hh = '03' # 03, 06, 09, 12
    _time = '20151001_' + hh
    _lat, _lon = ut.vessel_loc_at_time(_time)
    # Update fig name
    figname = figname + '_' + _time + '_' + str(_lat) + '_' + str(_lon)
    spec = ut.open_ERA5_wave_spectra(_time, _lat, _lon)

    _title_meta = 'time:'+str(spec.time.values)[0:13] + ', ' + 'lat:'+str(spec.latitude.values) + ', ' + 'lon:'+str(spec.longitude.values)
    _title = 'ERA5 1d wave spectra at ' + _title_meta

    # Integrate spectral energy over the directions.
    # https://github.com/openearth/oceanwaves-python/blob/00237ff6c74cd3b7a76aee7faa9550d3ef614cb3/oceanwaves/oceanwaves.py#L920
    spec_1d = np.trapz(spec, x=spec.direction.values * (np.pi / 180), axis=1)


    hmax, mp2, swh, mwd, pp1d, mwp, bfi, tmax, wss, wdw, wsk, steepness = ut.open_ERA5_wave_data()
    # Convert mwd to direction to
    mwd = ut.conv_dir_from_to(mwd)

    wind_u, wind_v = ut.open_ERA5_sfc_data()
    ws = ut.magnitude(wind_u, wind_v).rename('wind_speed')
    dir = ut.direction_from(wind_v, wind_u).rename('wind_direction')
    conv_points = dir.where(dir < 0) + 360
    old_points = dir.where(dir >= 0)
    wind_dir = xr.merge([old_points, conv_points])['wind_direction']

    water_u, water_v = ut.open_HyCOM_data()
    cs = ut.magnitude(water_u, water_v).rename('current_speed')
    dir = ut.direction(water_u, water_v).rename('current_direction')
    conv_points = dir.where(dir < 0) + 360
    old_points = dir.where(dir >= 0)
    current_dir = xr.merge([old_points, conv_points])['current_direction']

    # Get bulk spectral parameters
    hmax = str(np.around(hmax.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=1))
    mp2 = str(np.around(mp2.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=1))
    swh = str(np.around(swh.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=1))
    mwd = str(np.around(mwd.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=1))
    pp1d = str(np.around(pp1d.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=1))
    mwp = str(np.around(mwp.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=1))
    bfi = str(np.around(bfi.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=2))
    tmax = str(np.around(tmax.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=1))
    wss = str(np.around(wss.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=2))
    wdw = str(np.around(wdw.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=2))
    wsk = str(np.around(wsk.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=2))
    steepness = str(np.around(steepness.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=2))

    wind_dir = str(np.around(wind_dir.sel(time=spec.time, latitude=spec.latitude, longitude=spec.longitude).values, decimals=2))

    current_dir = str(np.around(current_dir.sel(time=spec.time, lat=spec.latitude, lon=spec.longitude, method='nearest').values, decimals=2))

    # Print directions
    print('wind direction is ' + wind_dir)
    print('ocean direction is ' + current_dir)
    print('mean wave direction is ' + mwd)

    fig, ax = plt.subplots(1, 1, figsize=(14,8))
    plt.plot(spec.frequency, spec_1d)
    plt.axis([0.0, 0.6, 0, 30])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Frequency (s radian$^{-1}$)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.set_ylabel('Spectral density (m$^{2}$ s radian$^{-1}$)', fontsize=20)
    plt.title(_title, fontsize=20)
    # Add text of spectral parameters
    plt.text(0.3, 25, 'H$_{s}$ = ' + swh + ', ', fontsize=20)
    plt.text(0.38, 25, 'H$_{max}$ = ' + hmax + ' (m)', fontsize=20)
    plt.text(0.2, 22, 'T$_{m-1}$ = ' + mwp + ', ', fontsize=20)
    plt.text(0.3, 22, 'T$_{m2}$ = ' + mp2 + ', ', fontsize=20)
    plt.text(0.39, 22, 'T$_{p}$ = ' + pp1d + ', ', fontsize=20)
    plt.text(0.47, 22, 'T$_{max}$ = ' + tmax + ' (s)', fontsize=20)
    plt.text(0.35, 19, 'S$_{max}$' + ' = ' + steepness, fontsize=20)
    plt.text(0.2, 16, r'$\sigma_{\theta}$' + ' = ' + wdw + ', ', fontsize=20)
    plt.text(0.3, 16, 'C$_{4}$' + ' = ' + wsk + ', ', fontsize=20)
    plt.text(0.39, 16, '$BFI$' + ' = ' + bfi + ', ', fontsize=20)
    plt.text(0.49, 16, 'C$_{3}$' + ' = ' + wss, fontsize=20)
    # from ($^{\circ}$)
    plt.savefig(figdir+figname+'.png', bbox_inches='tight')
    plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps', dpi=600)
