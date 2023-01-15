#!/usr/bin/env python
# coding: utf-8

from os.path import expanduser
import sys
import getopt
import socket
import getpass
from matplotlib import colors
import cftime
import cmocean
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import gzip
import glob
import shutil
import numpy.ma as ma
import numpy as np
import pyart
from datetime import date, datetime, timedelta
import os
import re
import getopt
import sys
import netCDF4 as nc4
from nco import Nco
nco = Nco()

# ----------------------------
# Set up some path definitions
# ----------------------------
home_path = expanduser("~")
quicklook_base_path = os.path.join(home_path, "public_html/cloud-radars")
galileo_raw_path = '/radar/radar-galileo/raw'


mpl.use('Agg')


version = 0.1

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
    bit_rgb = np.linspace(0, 1, 256)
    if position == None:
        position = np.linspace(0, 1, len(colors))
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
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap


def main():

    data_datetime = datetime.now()

    hour_end = data_datetime.hour;

    print(hour_end);

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:h:", ["day=", "hour="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized
        sys.exit(2)


    for o, a in opts:
        if o == "-d":
            day = a
            data_datetime = datetime.strptime(day, "%Y%m%d")
        elif o in ("-h"):
            hour_end = int(a)
            print(hour_end)
            data_datetime = data_datetime.replace(hour=hour_end, minute=0, second=0, microsecond=0)
        else:
            assert False, "unhandled option"


    dt1 = data_datetime
    dt0 = data_datetime - timedelta(hours=4)
    datestr1 = dt1.strftime('%Y%m%d')
    datestr0 = dt0.strftime('%Y%m%d')

    print(dt1)
    print(dt0)

    with open('hoganjet.csv') as f:
        hoganjet = [tuple(map(int, i.split(','))) for i in f]

    cmap_hoganjet = make_cmap(hoganjet, bit=True)

    # LOCATE GALILEO FILES FOR SELECTED DATE

    dateyr = datestr1[0:4]
    dateym = datestr1[0:6]
    user = getpass.getuser()

    os.chdir(os.path.join(galileo_raw_path, datestr0))
    rawfiles94 = [os.path.join(galileo_raw_path, datestr0, f)
                for f in glob.glob('*{}*raw.nc'.format(datestr0))]

    os.chdir(os.path.join(galileo_raw_path, datestr1))
    rawfiles94 += [os.path.join(galileo_raw_path, datestr1, f)
                for f in glob.glob('*{}*raw.nc'.format(datestr1))]


    print(rawfiles94)

    output = set()
    for x in rawfiles94:
        output.add(x)
    print(output)

    rawfiles94 = list(output)

    print(rawfiles94)

    figpath = os.path.join(quicklook_base_path, dateyr, dateym, datestr1)

    istartfile = 0
    iendfile = -1

    DS0 = nc4.Dataset(rawfiles94[istartfile], 'r')
    dtime_start = cftime.num2pydate(DS0['time'][:], DS0['time'].units)

    DS1 = nc4.Dataset(rawfiles94[iendfile], 'r')
    dtime_end = cftime.num2pydate(DS1['time'][:], DS1['time'].units)

    # Fix day rollover problem
    if dtime_end[-1] < dtime_end[-2]:
        dtime_end[-1] = dtime_end[-1]+timedelta(days=1)

    dt_min = dt0
    dt_max = dt1

    # dt_min = dt_min.replace(second=0, microsecond=0)

    DS0.close()
    DS1.close()

    print(dt_min, dt_max)

    fig, axs = plt.subplots(5, 1, figsize=(12, 15), constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=2 / 72, h_pad=2 / 72, hspace=0.2, wspace=0.2)

    hmax = 12

    axs[0].set_xlim(dt_min, dt_max)
    axs[0].set_ylim(0, hmax)
    axs[1].set_xlim(dt_min, dt_max)
    axs[1].set_ylim(0, hmax)
    axs[2].set_xlim(dt_min, dt_max)
    axs[2].set_ylim(0, hmax)
    axs[3].set_xlim(dt_min, dt_max)
    axs[3].set_ylim(0, hmax)
    axs[4].set_xlim(dt_min, dt_max)
    axs[4].set_ylim(0, hmax)

    DS0 = nc4.Dataset(rawfiles94[istartfile], 'r', format='NETCDF4_CLASSIC')
    DS0.set_auto_mask(True)

    print(DS0)

    dtime0 = cftime.num2pydate(DS0['time'][:], DS0['time'].units)
    if dtime0[-1] < dtime0[-2]:
        dtime0[-1] = dtime0[-1]+timedelta(days=1)

    myFmt = mdates.DateFormatter('%H:%M')

    nray = DS0['time'].shape[0]

    print(nray)

    rng = DS0['range'][:]

    ngate = DS0['range'].shape[0]

    print(ngate)
    print(rng.shape)
    ZED_HCnew = DS0['ZED_HC'][:, :]
    print(ZED_HCnew.shape)

    rng = DS0['range'][:]

    ngate = DS0['range'].shape[0]

    range_offset = -389.7301954
    drng = range_offset-DS0['range'].range_offset


    ZED_HCnew = ZED_HCnew+20. * \
        np.log10(rng[None, :])-20.*np.log10(rng[None, :]+drng)

    range_km = (rng+drng)/1000.

    gate_edges = range_km - 29.9792458/1000.
    gate_edges = np.append(gate_edges, gate_edges[-1]+2*29.9792458/1000.)
    ray_duration = dtime0[-1]-dtime0[-2];
    ray_edges = dtime0;
    ray_edges = np.append(ray_edges,ray_edges[-1]+ray_duration)

    spw_plotmin = np.sqrt(1e-3)
    spw_plotmax = np.sqrt(10.)

    axs[0].xaxis.set_major_formatter(myFmt)
    h0 = axs[0].pcolormesh(ray_edges, gate_edges, ZED_HCnew.transpose(
    ), vmin=-40, vmax=40, cmap=cmap_hoganjet, shading='auto')

    titlestr = "Chilbolton W-band Galileo Radar: "+datestr1
    axs[0].set_title(titlestr)
    cb0 = plt.colorbar(h0, ax=axs[0], orientation='vertical')
    cb0.ax.set_ylabel("ZED_HC (dB)")
    axs[0].grid(True)
    axs[0].set_xlabel('Time (UTC)')
    axs[0].set_ylabel('Height (km)')

    axs[1].xaxis.set_major_formatter(myFmt)
    h1 = axs[1].pcolormesh(ray_edges, gate_edges, DS0['VEL_HC'][:, :].transpose(
    ), vmin=-5, vmax=5, cmap=cmap_hoganjet, shading='auto')
    cb1 = plt.colorbar(h1, ax=axs[1], orientation='vertical')
    cb1.ax.set_ylabel("VEL_HC (m$s^{-1}$)")
    axs[1].grid(True)
    axs[1].set_xlabel('Time (UTC)')
    axs[1].set_ylabel('Height (km)')

    try:
        axs[2].xaxis.set_major_formatter(myFmt)
        h2 = axs[2].pcolormesh(ray_edges, gate_edges, DS0['LDR_HC'][:, :].transpose(
        ), vmin=-35, vmax=5, cmap=cmap_hoganjet, shading='auto')
        cb2 = plt.colorbar(h2, ax=axs[2], orientation='vertical')
        cb2.ax.set_ylabel("LDR_HC (dB)")
        axs[2].grid(True)
        axs[2].set_xlabel('Time (UTC)')
        axs[2].set_ylabel('Height (km)')
    except:
        print("No LDR_HC")

    try:
        axs[3].xaxis.set_major_formatter(myFmt)
        h3 = axs[3].pcolormesh(ray_edges, gate_edges, DS0['SPW_HC'][:, :].transpose(
        ), norm=colors.LogNorm(vmin=spw_plotmin, vmax=spw_plotmax), cmap=cmap_hoganjet, shading='auto')
        cb3 = plt.colorbar(h3, ax=axs[3], orientation='vertical')
        cb3.ax.set_ylabel("SPW_HC (m$s^{-1}$)")
        axs[3].grid(True)
        axs[3].set_xlabel('Time (UTC)')
        axs[3].set_ylabel('Height (km)')
    except:
        print("No SPW_HC")

    try:
        ZED_XHCnew = DS0['ZED_XHC'][:, :]
        ZED_XHCnew = ZED_XHCnew+20. * \
            np.log10(rng[None, :])-20.*np.log10(rng[None, :]-drng)
    except:
        print("No ZED_XHC")
        ZED_XHCnew = np.empty([nray, ngate])

    axs[4].xaxis.set_major_formatter(myFmt)
    h4 = axs[4].pcolormesh(ray_edges, gate_edges, ZED_XHCnew[:, :].transpose(
    ), vmin=-40, vmax=40, cmap=cmap_hoganjet, shading='auto')
    cb4 = plt.colorbar(h4, ax=axs[4], orientation='vertical')
    cb4.ax.set_ylabel("ZED_XHC (dBZ)")
    axs[4].grid(True)
    axs[4].set_xlabel('Time (UTC)')
    axs[4].set_ylabel('Height (km)')


    DS0.close()

    for file in rawfiles94[istartfile+1:]:
        DS0 = nc4.Dataset(file, 'r', format='NETCDF4_CLASSIC')
        DS0.set_auto_mask(True)

        if nray > 0:

            dtime0 = cftime.num2pydate(DS0['time'][:], DS0['time'].units)
            if dtime0[-1] < dtime0[-2]:
                dtime0[-1] = dtime0[-1]+timedelta(days=1)

            rng = DS0['range'][:]

            drng = range_offset-DS0['range'].range_offset

            range_km = (rng+drng)/1000.

            gate_edges = range_km - 29.9792458/1000.
            gate_edges = np.append(gate_edges, gate_edges[-1]+2*29.9792458/1000.)
            ray_duration = dtime0[-1]-dtime0[-2];
            ray_edges = dtime0;
            ray_edges = np.append(ray_edges,ray_edges[-1]+ray_duration)

            ZED_HCnew = DS0['ZED_HC'][:, :]+20. * \
                np.log10(rng[None, :])-20.*np.log10(rng[None, :]+drng)

            try:
                axs[1].pcolormesh(ray_edges, gate_edges, DS0['VEL_HC'][:, :].transpose(
                ), vmin=-5, vmax=5, cmap=cmap_hoganjet, shading='auto')
                axs[2].pcolormesh(ray_edges, gate_edges, DS0['LDR_HC'][:, :].transpose(
                ), vmin=-35, vmax=5, cmap=cmap_hoganjet, shading='auto')
                axs[0].pcolormesh(ray_edges, gate_edges, ZED_HCnew.transpose(
                ), vmin=-40, vmax=40, cmap=cmap_hoganjet, shading='auto')
                axs[3].pcolormesh(ray_edges, gate_edges, DS0['SPW_HC'][:, :].transpose(), norm=colors.LogNorm(
                    vmin=spw_plotmin, vmax=spw_plotmax), cmap=cmap_hoganjet, shading='auto')

                ZED_XHCnew = DS0['ZED_XHC'][:, :]
                ZED_XHCnew = ZED_XHCnew+20. * \
                    np.log10(rng[None, :])-20.*np.log10(rng[None, :]+drng)
                axs[4].pcolormesh(ray_edges, gate_edges, ZED_XHCnew[:, :].transpose(
                ), vmin=-40, vmax=40, cmap=cmap_hoganjet, shading='auto')
            except:
                print("Problem with plot")

            DS0.close()

    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    axs[3].grid(True)
    axs[4].grid(True)

    figfile = "radar-galileo_last4hr.png"

    plt.savefig(os.path.join(quicklook_base_path, figfile), dpi=200)


if __name__ == "__main__":
    main()
