# conda activate spec_plot
import xarray as xr
import numpy as np
from mpl_toolkits.basemap import addcyclic
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys

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

def open_ERA5_wave_spectra(_time, _latitude, _longitude):
    # _time: str. e.g. '20151001_03'
    # _latitude: float e.g. 25.5305
    # _longitde: float e.g. -75.732
    da = xr.open_dataarray('data/SSElFaro_waves_spectra_' + _time + '.nc').squeeze()
    da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    da = da.assign_coords(direction=np.arange(7.5, 352.5 + 15, 15))
    da = da.assign_coords(frequency=np.full(30, 0.03453, dtype=np.float) * (1.1 ** np.arange(0, 30)))
    # Find location of interest
    da = da.sel(latitude=_latitude, longitude=_longitude, method='nearest')
    # Convert from log10 to spectral density
    da = 10 ** da
    # Replace zero values with very small values
    da = da.fillna(1e-4)
    # Reverse direction dimension
    da = da.sel(**{'direction': slice(None, None, -1)})
    return da

figdir = '/Volumes/SAMSUNG/WORK/POSTDOC_RSMAS_2016/PYTHON3/Publications/SSElFaro/figs/'
figname = 'ERA5_2d_spectra'
hh = '09' # 03, 06, 09, 12
_time = '20151001_' + hh
_lat, _lon = vessel_loc_at_time(_time)
# Update fig name
figname = figname + '_' + _time + '_' + str(_lat) + '_' + str(_lon)

spec = open_ERA5_wave_spectra(_time, _lat, _lon)

# Code developed by Todd Spindler at NOAA
freq = spec.frequency.values
theta = spec.direction.values * (np.pi / 180)
spec = spec.values.T

# Add cyclic point to make polar plot
spec, theta = addcyclic(spec.T, theta)
spec = spec.T

freq2d, theta2d = np.meshgrid(freq, theta)

# Define levels in log space
levels = np.logspace(-4, 2, num=13)
levels = np.hstack((spec.min(), levels))
vmin = 1e-4
vmax = 20.0

# Plot
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.grid(color='w')

cc=ax.contourf(theta2d, freq2d, spec, levels=levels, norm=LogNorm())
cc.set_clim(vmin, vmax)
cb = fig.colorbar(cc, ticks=levels)
cb.set_label('Spectral density (m$^{2}$ s radian$^{-1}$)')
plt.savefig(figdir+figname+'.png', bbox_inches='tight')
plt.savefig(figdir+figname+'.eps', bbox_inches='tight', format='eps')
