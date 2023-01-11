#!/usr/bin/env python
# coding: utf-8

from datetime import date, datetime, timedelta 
import os, re, getopt, sys
import netCDF4 as nc4
from nco import Nco
nco = Nco()
import pyart
import numpy as np
import numpy.ma as ma
import shutil
import glob
import gzip

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import cmocean
import getpass, socket

import cftime

import getopt, sys

try:
    opts, args = getopt.getopt(sys.argv[1:], "d:", ["date="])
except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized
        sys.exit(2)

data_date = datetime.now()

dt1 = data_date;
dt0 = data_date - timedelta(minutes=30);

datestr = data_date.strftime('%Y%m%d')

for o, a in opts:
    if o == "-d":
        datestr = a;
    else:
        assert False, "unhandled option"

mpl.use('Agg');


print(datestr);


# In[1]:




version = 0.1


# In[4]:


'''
NAME
    Custom Colormaps for Matplotlib
PURPOSE
    This program shows how to implement make_cmap which is a function that
    generates a colorbar.  If you want to look at different color schemes,
    check out https://kuler.adobe.com/create.
PROGRAMMER(S)
    Chris Slocum
REVISION HISTORY
    20130411 -- Initial version created
    20140313 -- Small changes made and code posted online
    20140320 -- Added the ability to set the position of each color
'''

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

with open('/home/cw66/python/hoganjet.csv') as f:
    hoganjet = [tuple(map(int, i.split(','))) for i in f]

cmap_hoganjet = make_cmap(hoganjet,bit=True)
print(cmap_hoganjet)




# LOCATE COPERNICUS FILES FOR SELECTED DATE

dateyr  = datestr[0:4];
dateym  = datestr[0:6];

user = getpass.getuser()

raw_path35 = os.path.join('/radar/radar-copernicus/raw',datestr);

os.chdir(raw_path35);
rawfiles35 = [os.path.join(raw_path35,f) for f in glob.glob('*{}*raw.nc'.format(datestr))]

figpath = os.path.join('/home/cw66/public_html/cloud-radars',dateyr,dateym,datestr);

istartfile = 0
iendfile   = -1

DS0 = nc4.Dataset(rawfiles35[istartfile], 'r',format='NETCDF4_CLASSIC');
dtime_start = cftime.num2pydate(DS0['time'][:],DS0['time'].units)

DS1 = nc4.Dataset(rawfiles35[iendfile], 'r',format='NETCDF4_CLASSIC');
dtime_end = cftime.num2pydate(DS1['time'][:],DS1['time'].units)

if dtime_end[-1]<dtime_end[-2]:
    dtime_end[-1]=dtime_end[-1]+timedelta(days=1)

dt_min = dt0;
dt_max = dt1;


#dt_min = dt_min.replace(second=0, microsecond=0)


DS0.close()
DS1.close()


print(dt_min, dt_max);

fig, axs = plt.subplots(4,1, figsize = (12, 15), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=2 / 72, h_pad=2 / 72, hspace=0.2,wspace=0.2)

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib import colors

hmax = 12;

axs[0].set_xlim(dt_min,dt_max);
axs[0].set_ylim(0,hmax);
axs[1].set_xlim(dt_min,dt_max);
axs[1].set_ylim(0,hmax);
axs[2].set_xlim(dt_min,dt_max);
axs[2].set_ylim(0,hmax);
axs[3].set_xlim(dt_min,dt_max);
axs[3].set_ylim(0,hmax);

DS0 = nc4.Dataset(rawfiles35[istartfile], 'r',format='NETCDF4_CLASSIC');
DS0.set_auto_mask(True);

print(DS0);

dtime0 = cftime.num2pydate(DS0['time'][:],DS0['time'].units)
if dtime0[-1]<dtime0[-2]:
    dtime0[-1]=dtime0[-1]+timedelta(days=1)

myFmt = mdates.DateFormatter('%H:%M')

nray = DS0['time'].shape[0];

print(nray);

rng = np.tile(DS0['range'][:],(nray,1));

ngate = DS0['range'].shape[0];

print(ngate)
print(rng.shape)
ZED_HCnew = DS0['ZED_HC'][:,:];
print(ZED_HCnew.shape)
ZED_HCnew = ZED_HCnew+20.*np.log10(rng)-20.*np.log10(rng-630.0)-165.;
ZED_HCPnew = DS0['ZED_HCP'][:,:]+20.*np.log10(rng)-20.*np.log10(rng-630.0)-146.8;
rng = rng - 630.0

drng = 630.0;

spw_plotmin = np.sqrt(1e-3);
spw_plotmax = np.sqrt(10.)

axs[0].xaxis.set_major_formatter(myFmt);
h0=axs[0].pcolormesh(dtime0[:],(DS0['range'][:]-drng)/1000.,ZED_HCPnew.transpose(),vmin=-40,vmax=40,cmap=cmap_hoganjet,shading='auto');
titlestr = "Chilbolton Ka-band Copernicus Radar: "+datestr;
axs[0].set_title(titlestr)
cb0=plt.colorbar(h0,ax=axs[0],orientation='vertical');
cb0.ax.set_ylabel("ZED_HCP (dB)");
axs[0].grid(True);
axs[0].set_xlabel('Time (UTC)');
axs[0].set_ylabel('Height (km)');

axs[1].xaxis.set_major_formatter(myFmt);
h1=axs[1].pcolormesh(dtime0[:],(DS0['range'][:]-drng)/1000.,DS0['VEL_HCP'][:,:].transpose(),vmin=-5,vmax=5,cmap=cmap_hoganjet,shading='auto');
cb1=plt.colorbar(h1,ax=axs[1],orientation='vertical')
cb1.ax.set_ylabel("VEL_HCP (m$s^{-1}$)");
axs[1].grid(True)
axs[1].set_xlabel('Time (UTC)')
axs[1].set_ylabel('Height (km)')

axs[2].xaxis.set_major_formatter(myFmt);
h2=axs[2].pcolormesh(dtime0[:],(DS0['range'][:]-drng)/1000.,DS0['LDR_CP'][:,:].transpose(),vmin=-35,vmax=5,cmap=cmap_hoganjet,shading='auto');
cb2=plt.colorbar(h2,ax=axs[2],orientation='vertical');
cb2.ax.set_ylabel("LDR_CP (dB)");
axs[2].grid(True);
axs[2].set_xlabel('Time (UTC)');
axs[2].set_ylabel('Height (km)');

axs[3].xaxis.set_major_formatter(myFmt);
h3=axs[3].pcolormesh(dtime0[:],(DS0['range'][:]-drng)/1000.,DS0['SPW_HCP'][:,:].transpose(),norm=colors.LogNorm(vmin=spw_plotmin,vmax=spw_plotmax),cmap=cmap_hoganjet,shading='auto');
cb3=plt.colorbar(h3,ax=axs[3],orientation='vertical');
cb3.ax.set_ylabel("SPW_HCP (m$s^{-1}$)");
axs[3].grid(True);
axs[3].set_xlabel('Time (UTC)');
axs[3].set_ylabel('Height (km)');




DS0.close();

for file in rawfiles35[istartfile+1:]:
    DS0 = nc4.Dataset(file, 'r',format='NETCDF4_CLASSIC');
    DS0.set_auto_mask(True);
            
    nray = DS0['time'].shape[0];

    if nray>0:
   
        dtime0 = cftime.num2pydate(DS0['time'][:],DS0['time'].units);
        if dtime0[-1]<dtime0[-2]:
            dtime0[-1]=dtime0[-1]+timedelta(days=1)
   
        rng = np.tile(DS0['range'][:],(nray,1));
        ZED_HCnew = DS0['ZED_HC'][:,:]+20.*np.log10(rng)-20.*np.log10(rng-630.0)-165.0;
        ZED_HCPnew = DS0['ZED_HCP'][:,:]+20.*np.log10(rng)-20.*np.log10(rng-630.0)-146.8;
        rng = rng - 630.0

        axs[1].pcolormesh(dtime0[:],(DS0['range'][:]-drng)/1000.,DS0['VEL_HCP'][:,:].transpose(),vmin=-5,vmax=5,cmap=cmap_hoganjet,shading='auto');
        axs[2].pcolormesh(dtime0[:],(DS0['range'][:]-drng)/1000.,DS0['LDR_CP'][:,:].transpose(),vmin=-35,vmax=5,cmap=cmap_hoganjet,shading='auto');
        axs[0].pcolormesh(dtime0[:],(DS0['range'][:]-drng)/1000.,ZED_HCPnew.transpose(),vmin=-40,vmax=40,cmap=cmap_hoganjet,shading='auto');
        axs[3].pcolormesh(dtime0[:],(DS0['range'][:]-drng)/1000.,DS0['SPW_HCP'][:,:].transpose(),norm=colors.LogNorm(vmin=spw_plotmin,vmax=spw_plotmax),cmap=cmap_hoganjet,shading='auto');
    
    DS0.close()

axs[0].grid(True);
axs[1].grid(True);
axs[2].grid(True);
axs[3].grid(True);

figfile = "radar-copernicus_{}_last30min.png".format(datestr);

plt.savefig(os.path.join(figpath,figfile),dpi=200)


# In[9]:




