import salem
from salem import GoogleVisibleMap, Map
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import numpy as np
import xarray as xr
import importlib
import os
import sys


def open_vessel_data():
    cols = ['date_unknown', 'time_unknown', 'latitude', 'longitude', 'sog', 'cog',
            'heading', 'timestamp', 'date_edt', '_time_edt']
    data_type = [np.object_, np.object_, np.float64, np.float64, np.float64, np.float64,
                 np.float64, np.int, np.object_, np.object_]
    cols_to_keep = ['latitude', 'longitude', 'sog', 'cog', 'heading', 'date_edt',
                    '_time_edt']
    data_type_dic = dict(zip(cols, data_type))
    vessel_loc = pd.read_csv('/Volumes/SAMSUNG/WORK/POSTDOC_RSMAS_2016/DATA/SSElFaro/598655.csv',
                             header=8, names=cols, usecols=cols_to_keep,
                             dtype=data_type_dic,
                             parse_dates={'time_edt': ['date_edt', '_time_edt']})
    time_utc = vessel_loc['time_edt'].dt.tz_localize('US/Eastern').dt.tz_convert('UTC')
    vessel_loc.update(time_utc)
    vessel_loc = vessel_loc.rename(columns={'time_edt': 'time_utc'})
    # Data doesn't quite look right at start so start at index 13
    vessel_loc = vessel_loc.iloc[13:].reset_index()
    # Believe sog is in knots. Convert to m/s
    vessel_loc['sog'] = vessel_loc['sog'] * 0.514444
    return vessel_loc

def open_hurricane_data():
    # Download the data ftp://eclipse.ncdc.noaa.gov/pub/ibtracs/v03r10/wmo/csv/year/Year.2015.ibtracs_wmo.v03r10.csv
    cols = ['Serial_Num', 'Season', 'Num', 'Basin', 'Sub_basin', 'Name', 'ISO_time',
            'Nature', 'Latitude', 'Longitude', 'Wind_WMO', 'Pres_WMO', 'Center',
            'Wind_WMO_Percentile', 'Pres_WMO_Percentile', 'Track_type']
    data_type = [np.int, np.int, np.int, np.object_, np.object_,
                 np.object_, np.object_, np.object_, np.float64, np.float64,
                 np.float64, np.float64, np.object_, np.float64, np.float64,
                 np.object_]
    data_type_dic = dict(zip(cols, data_type))
    cols_to_keep = ['Name', 'ISO_time', 'Nature', 'Latitude', 'Longitude',
                    'Wind_WMO', 'Pres_WMO', 'Wind_WMO_Percentile',
                    'Pres_WMO_Percentile']
    hurricane = pd.read_csv('/Volumes/SAMSUNG/WORK/POSTDOC_RSMAS_2016/DATA/ibtracs/Year.2015.ibtracs_wmo.v03r10.csv',
                            skiprows=[0, 2])
    hurricane.columns = cols
    hurricane = hurricane[cols_to_keep]
    hurricane['ISO_time'] = pd.to_datetime(hurricane['ISO_time'])
    hurricane = hurricane.rename(columns={'ISO_time': 'time_iso'})
    hurricane = hurricane[hurricane['Name'] == 'JOAQUIN'].reset_index()
    return hurricane

def magnitude(a, b):
    func = lambda x, y: np.sqrt(x ** 2 + y ** 2)
    return xr.apply_ufunc(func, a, b)

def direction(a, b):
    func = lambda x, y: np.degrees(np.arctan2(x, y))
    return xr.apply_ufunc(func, a, b)

def direction_from(a, b):
    func = lambda y, x: np.degrees(np.arctan2(-y, -x))
    return xr.apply_ufunc(func, a, b)

def conv_dir_from_to(a):
    a = a - 180.0
    conv_points = a.where(a < 0) + 360
    old_points = a.where(a >= 0)
    return xr.merge([old_points, conv_points])['mwd']

def direction_to_magnitude(a):
    a = conv_dir_from_to(a)
    conv_points = a.where(a < 0) + 360
    old_points = a.where(a >= 0)
    a = xr.merge([old_points, conv_points])['mwd']
    u = 1.0 * np.sin(np.deg2rad(a))
    v = 1.0 * np.cos(np.deg2rad(a))
    return u, v

def open_HyCOM_data():
    water_u = xr.open_dataarray('/Volumes/SAMSUNG/WORK/POSTDOC_RSMAS_2016/DATA/HYCOM_Reanalysis/SSElFaro/u_53.X.nc4')
    water_u = water_u.sel(depth=slice(0.0, 12.0)).mean(dim='depth')
    water_v = xr.open_dataarray('/Volumes/SAMSUNG/WORK/POSTDOC_RSMAS_2016/DATA/HYCOM_Reanalysis/SSElFaro/v_53.X.nc4')
    water_v = water_v.sel(depth=slice(0.0, 12.0)).mean(dim='depth')
    # Interpolate missing times
    missing_times = np.array(['2015-09-30T12:00:00', '2015-10-01T12:00:00'], dtype='datetime64[ns]')
    water_u = xr.merge([water_u, water_u.interp(time=missing_times)])['water_u'] # works but gives a Dataset
    water_v = xr.merge([water_v, water_v.interp(time=missing_times)])['water_v']
    return water_u, water_v

def open_HyCOM_clim_data():
    water_u = xr.open_dataarray('/Volumes/SAMSUNG/WORK/POSTDOC_RSMAS_2016/DATA/HYCOM_Reanalysis/SSElFaro/climatology/hycom_GLBv0.08_53X_archMN.1994_01_2015_12_u.nc4').squeeze()
    water_u = water_u.sel(depth=slice(0.0, 12.0)).mean(dim='depth')
    water_u = water_u.assign_coords(lon=(((water_u.lon + 180) % 360) - 180))
    water_v = xr.open_dataarray('/Volumes/SAMSUNG/WORK/POSTDOC_RSMAS_2016/DATA/HYCOM_Reanalysis/SSElFaro/climatology/hycom_GLBv0.08_53X_archMN.1994_01_2015_12_v.nc4').squeeze()
    water_v = water_v.sel(depth=slice(0.0, 12.0)).mean(dim='depth')
    water_v = water_v.assign_coords(lon=(((water_v.lon + 180) % 360) - 180))
    return water_u, water_v

def open_ERA5_sfc_data():
    ds = xr.open_dataset('/Users/Ray/Volumes/Pegasus_data/DATA/ERA5/sfc/SSElFaro_sfc_new.nc')
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    u10 = ds['u10']
    v10 = ds['v10']
    return u10, v10

def open_ERA5_sfc_clim_data():
    ds = xr.open_mfdataset('/Users/Ray/Volumes/Pegasus_data/DATA/ERA5/monthly_means/winds/*')
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    u10 = ds['u10'].mean(dim='time').compute()
    v10 = ds['v10'].mean(dim='time').compute()
    _ = ds['msl'] / 10
    #msl = _.mean(dim='time').compute()
    return u10, v10

def open_ERA5_wave_data():
    ds = xr.open_dataset('/Users/Ray/Volumes/Pegasus_data/DATA/ERA5/waves/SSElFaro_waves_full.nc')
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    hmax = ds['hmax'] # Maximum individual wave height (m). 218.140
    mp2 = ds['mp2'] # Mean zero-crossing wave period (s). 221.140. DONE
    swh = ds['swh'] # Significant height of combined wind waves and swell (m). 229.140. DONE
    mwd = ds['mwd'] # Mean wave direction (degrees). 230.140. DONE
    pp1d = ds['pp1d'] # Peak wave period (s). 231.140. DONE
    mwp = ds['mwp'] # Mean wave period (s). 232.140. DONE
    wdw = ds['wdw'] # Wave spectral directional width (dimensionless). 222.140. DONE
    bfi = np.sqrt(ds['bfi'])  # Benjamin-Feir index (dimensionless). 253.140. DONE. Square of the BFI
    tmax = ds['tmax'] # Period corresponding to maximum individual wave height (s). 217.140
    wss = ds['wss'] # Wave spectral skewness (dimensionless). 207.140. DONE
    wsk = ds['wsk'] # Wave spectral kurtosis (dimensionless). 252.140. DONE
    steepness = (2 * np.pi * hmax) / (9.81 * (tmax ** 2))
    return hmax, mp2, swh, mwd, pp1d, mwp, bfi, tmax, wss, wdw, wsk, steepness

def open_ERA5_wave_clim_data():
    ds = xr.open_mfdataset('/Users/Ray/Volumes/Pegasus_data/DATA/ERA5/monthly_means/waves/*.nc')
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    hmax = ds['hmax'].mean(dim='time').compute()
    mwd = ds['mwd'] # Mean wave direction (degrees). 230.140
    u, v = direction_to_magnitude(mwd)
    u = u.mean(dim='time').compute()
    v = v.mean(dim='time').compute()
    return hmax, u, v

def open_ERA5_wave_spectra(_time, _latitude, _longitude):
    # _time: str. e.g. '20151001_03'
    # _latitude: float e.g. 25.5305
    # _longitde: float e.g. -75.732
    da = xr.open_dataarray('/Users/Ray/Volumes/Pegasus_data/DATA/ERA5/waves/SSElFaro_waves_spectra_' + _time + '.nc').squeeze()
    da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    da = da.assign_coords(direction=np.arange(7.5, 352.5 + 15, 15))
    da = da.assign_coords(frequency=np.full(30, 0.03453, dtype=np.float) * (1.1 ** np.arange(0, 30)))
    # Find location of interest
    da = da.sel(latitude=_latitude, longitude=_longitude, method='nearest')
    # Convert from log10 to spectral density
    da = 10 ** da
    # Replace nan with 0
    da = da.fillna(0)
    return da

def vessel_loc_at_time(_time):
    # _time: str. e.g. '20151001_03'
    # I know these already. Could do something more clever by reading from the dataframe
    if _time == '20151001_03':
        _lat = 25.5305
        _lon = -75.732
    if _time == '20151001_06':
        _lat = 24.7714
        _lon = -75.2525
    if _time == '20151001_09':
        _lat = 23.9981
        _lon = -74.7768
    if _time == '20151001_12':
        _lat = 23.5204
        _lon = -74.0195
    return _lat, _lon
