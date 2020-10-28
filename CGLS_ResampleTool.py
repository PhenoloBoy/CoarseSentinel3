# coding: utf-8
import datetime as dt
import os
import re
import sys

import dask_image.ndfilters
from distributed import Client
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm


def _param(ds):
    if 'LAI' in ds.data_vars:
        param = {'product': 'LAI',
                 'short_name': 'leaf_area_index',
                 'long_name': 'Leaf Area Index Resampled 1 Km',
                 'grid_mapping': 'crs',
                 'flag_meanings': 'Missing',
                 'flag_values': '255',
                 'units': '',
                 'PHYSICAL_MIN': 0,
                 'PHYSICAL_MAX': 7,
                 'DIGITAL_MAX': 210,
                 'SCALING': 1./30,
                 'OFFSET': 0}
        da = ds.LAI

    elif 'FCOVER' in ds.data_vars:
        param = {'product': 'FCOVER',
                 'short_name': 'vegetation_area_fraction',
                 'long_name': 'Fraction of green Vegetation Cover Resampled 1 Km',
                 'grid_mapping': 'crs',
                 'flag_meanings': 'Missing',
                 'flag_values': '255',
                 'units': '',
                 'valid_range': '',
                 'PHYSICAL_MIN': 0,
                 'PHYSICAL_MAX': 1.,
                 'DIGITAL_MAX': 250,
                 'SCALING': 1./250,
                 'OFFSET': 0}
        da = ds.FCOVER

    elif 'FAPAR' in ds.data_vars:
        param = {'product': 'FAPAR',
                 'short_name': 'Fraction_of_Absorbed_Photosynthetically_Active_Radiation',
                 'long_name': 'Fraction of Absorbed Photosynthetically Active Radiation Resampled 1 KM',
                 'grid_mapping': 'crs',
                 'flag_meanings': 'Missing',
                 'flag_values': '255',
                 'units': '',
                 'valid_range': '',
                 'PHYSICAL_MIN': 0,
                 'PHYSICAL_MAX': 0.94,
                 'DIGITAL_MAX': 235,
                 'SCALING': 1./250,
                 'OFFSET': 0}
        da = ds.FAPAR

    elif 'NDVI' in ds.data_vars:
        param = {'product': 'NDVI',
                 'short_name': 'Normalized_difference_vegetation_index',
                 'long_name': 'Normalized Difference Vegetation Index Resampled 1 Km',
                 'grid_mapping': 'crs',
                 'flag_meanings': 'Missing cloud snow sea background',
                 'flag_values': '[251 252 253 254 255]',
                 'units': '',
                 'PHYSICAL_MIN': -0.08,
                 'PHYSICAL_MAX': 0.92,
                 'DIGITAL_MAX': 250,
                 'SCALING': 1./250,
                 'OFFSET': -0.08}
        da = ds.NDVI

    elif 'DMP' in ds.data_vars:
        param = {'product': 'DMP',
                 'short_name': 'dry_matter_productivity',
                 'long_name': 'Dry matter productivity Resampled 1KM',
                 'grid_mapping': 'crs',
                 'flag_meanings': 'sea',
                 'flag_values': '-2',
                 'units': 'kg / ha / day',
                 'PHYSICAL_MIN': 0,
                 'PHYSICAL_MAX': 327.67,
                 'DIGITAL_MAX': 32767,
                 'SCALING': 1./100,
                 'OFFSET': 0}
        da = ds.DMP

    elif 'GDMP' in ds.data_vars:
        param = {'product': 'GDMP',
                 'short_name': 'Gross_dry_matter_productivity',
                 'long_name': 'Gross dry matter productivity Resampled 1KM',
                 'grid_mapping': 'crs',
                 'flag_meanings': 'sea',
                 'flag_values': '-2',
                 'units': 'kg / hectare / day',
                 'PHYSICAL_MIN': 0,
                 'PHYSICAL_MAX': 655.34,
                 'DIGITAL_MAX': 32767,
                 'SCALING': 1./50,
                 'OFFSET': 0}
        da = ds.GDMP

    else:
        sys.exit('GLC product not found please chek')

    return da, param


def _downloader(user, psw, folder):
    url = 'https://land.copernicus.vgt.vito.be/manifest/'

    session = requests.Session()
    session.auth = (user, psw)

    manifest = session.get(url, allow_redirects=True)
    products = pd.read_html(manifest.text)[0][2:-1]['Name']
    products = products[products.str.contains('300_')].reset_index(drop=True)
    print(products)
    val = input('Please select the product from the list:')
    url = f'{url}{products[int(val)]}'

    manifest = session.get(url, allow_redirects=True)
    product = pd.read_html(manifest.text)[0][-2:-1]['Name'].values[0]
    purl = f'{url}{product}'
    r = session.get(purl, stream=True)
    rows = r.text.split('\n')
    dates = pd.DataFrame()
    for line in rows[:-1]:
        r = re.search(r"\d\d\d\d/\d\d/\d\d", line)
        dates = dates.append(pd.DataFrame([line], index=[pd.to_datetime(r[0], format="%Y/%m/%d")]))

    val = input('Please insert the date in teh format YYYY/MM/DD:')

    dates = dates.sort_index()
    i = dates.index.searchsorted(dt.datetime.strptime(val, "%Y/%m/%d"))
    link = dates.iloc[i][0]
    filename = os.path.basename(link)
    if folder != '':
        path = sys.path.join(folder, filename)
    else:
        path = filename

    r = session.get(link, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(path, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

    return path


def _aoi(da, ds, AOI):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def bnd_box_adj(my_ext):
        lat_1k = np.round(np.arange(80., -60., -1. / 112), 8)
        lon_1k = np.round(np.arange(-180., 180., 1. / 112), 8)

        lat_300 = ds.lat.values
        lon_300 = ds.lon.values
        ext_1k = np.zeros(4)

        # UPL Long 1K
        ext_1k[0] = find_nearest(lon_1k, my_ext[0]) - 1. / 336
        # UPL Lat 1K
        ext_1k[1] = find_nearest(lat_1k, my_ext[1]) + 1. / 336

        # LOWR Long 1K
        ext_1k[2] = find_nearest(lon_1k, my_ext[2]) + 1. / 336
        # LOWR Lat 1K
        ext_1k[3] = find_nearest(lat_1k, my_ext[3]) - 1. / 336

        # UPL
        my_ext[0] = find_nearest(lon_300, ext_1k[0])
        my_ext[1] = find_nearest(lat_300, ext_1k[1])

        # LOWR
        my_ext[2] = find_nearest(lon_300, ext_1k[2])
        my_ext[3] = find_nearest(lat_300, ext_1k[3])
        return my_ext

    if len(AOI):
        assert AOI[0] <= AOI[2], 'min Longitude is bigger than correspond Max, ' \
                                       'pls change position or check values.'
        assert AOI[1] >= AOI[3], 'min Latitude is bigger than correspond Max, ' \
                                       'pls change position or check values.'
        assert ds.lon[0] <= AOI[0] <= ds.lon[-1], 'min Longitudinal value out of original dataset Max ext.'
        assert ds.lat[-1] <= AOI[1] <= ds.lat[0], 'Max Latitudinal value out of original dataset Max ext.'

        assert ds.lon[0] <= AOI[2] <= ds.lon[-1], 'Max Longitudinal value out of original dataset Max ext.'
        assert ds.lat[-1] <= AOI[3] <= ds.lat[0], 'min Latitudinal value out of original dataset Max ext.'

        adj_ext = bnd_box_adj(AOI)
        try:
            return da.sel(lon=slice(adj_ext[0], adj_ext[2]), lat=slice(adj_ext[1], adj_ext[3]))
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            raise sys.exit(1)
    else:
        return da
        # da = da.shift(lat=1, lon=1)


def _date_extr(path):
    _, tail = os.path.split(path)
    pos = [pos for pos, char in enumerate(tail) if char == '_'][2]
    date = tail[pos + 1: pos + 9]
    date_h = pd.to_datetime(date, format='%Y%m%d')
    if date_h.day <= 10:
        date_h = pd.datetime(date_h.year, date_h.month, 1)
    elif 10 < date_h.day <= 20:
        date_h = pd.datetime(date_h.year, date_h.month, 11)
    else:
        date_h = pd.datetime(date_h.year, date_h.month, 21)

    return date, date_h


def _kernel_sel(k_type):
    # Kernels

    if '3' in k_type:
        sum_k = np.repeat(1.0, 3 ** 2).reshape(3, 3)
        val_thresh = 5
    elif '5' in k_type:
        sum_k = np.repeat(1.0, 5 ** 2).reshape(5, 5)
        val_thresh = 15
    elif '7' in k_type:
        sum_k = np.repeat(1.0, 7 ** 2).reshape(7, 7)
        val_thresh = 30

    if not any([substring in k_type.lower() for substring in ['t', 'p', 'g']]):
        if k_type == '3x3':
            size = 3
        elif k_type == '5x5':
            size = 5
        elif k_type == '7x7':
            size = 7
        return np.repeat(1.0, size**2).reshape(size, size) / size**2, sum_k, val_thresh
    elif k_type.lower() == '3x3_g':
        return np.array([[1,  2, 1],
                         [2,  4, 2],
                         [1,  2, 1]]) * 1/16, sum_k, val_thresh
    elif k_type.lower() == '3x3_p':
        return np.array([[1,  1, 1],
                         [1,  2, 1],
                         [1,  1, 1]]) * 1/10, sum_k, val_thresh
    elif k_type.lower() == '5x5_g':
        return np.array([[1,  4,  6,  4, 1],
                         [4, 16, 24, 16, 4],
                         [6, 24, 36, 24, 6],
                         [4, 16, 24, 16, 4],
                         [1, 4, 6, 4, 1]]) * 1/256, sum_k, val_thresh
    elif k_type.lower() == '5x5_p':
        return np.array([[1, 1, 1, 1, 1],
                         [1, 2, 2, 2, 1],
                         [1, 2, 3, 2, 1],
                         [1, 2, 2, 2, 1],
                         [1, 1, 1, 1, 1]]) * 1 / 35, sum_k, val_thresh
    elif k_type.lower() == '5x5_t':
        return np.array([[1, 1, 1, 1, 1],
                         [1, 2, 2, 2, 1],
                         [1, 2, 2, 2, 1],
                         [1, 2, 2, 2, 1],
                         [1, 1, 1, 1, 1]]) * 1 / 34, sum_k, val_thresh
    elif k_type.lower() == '5x5_r':
        return np.array([[.0025, .0125, .0200, .0125, .0025],
                         [.0125, .0625, .1000, .0625, .0125],
                         [.0200, .1000, .1600, .1000, .0200],
                         [.0125, .0625, .1000, .0625, .0125],
                         [.0025, .0125, .0200, .0125, .0025]]), sum_k, val_thresh
    elif k_type.lower() == '7x7_g':
        return np.array([[0.00000067, 0.00002292,  0.00019117,  0.00038771, 0.00019117, 0.00002292, 0.00000067],
                         [0.00002292, 0.00078633,  0.00655965,  0.01330373, 0.00655965, 0.00078633, 0.00002292],
                         [0.00019117, 0.00655965,  0.05472157,  0.11098164, 0.05472157, 0.00655965, 0.00019117],
                         [0.00038771, 0.01330373,  0.11098164,  0.22508352, 0.11098164, 0.01330373, 0.00038771],
                         [0.00019117, 0.00655965,  0.05472157,  0.11098164, 0.05472157, 0.00655965, 0.00019117],
                         [0.00002292, 0.00078633,  0.00655965,  0.01330373, 0.00655965, 0.00078633, 0.00002292],
                         [0.00000067, 0.00002292,  0.00019117,  0.00038771, 0.00019117, 0.00002292, 0.00000067]]), \
                         sum_k, val_thresh
    elif k_type.lower() == '7x7_p':
        return np.array([[1, 1, 1, 1, 1, 1, 1],
                         [1, 2, 2, 2, 2, 2, 1],
                         [1, 2, 3, 3, 3, 2, 1],
                         [1, 2, 3, 4, 3, 2, 1],
                         [1, 2, 3, 3, 3, 2, 1],
                         [1, 2, 2, 2, 2, 2, 1],
                         [1, 1, 1, 1, 1, 1, 1]]) * 1 / 84, sum_k, val_thresh
    elif k_type.lower() == '7x7_t':
        return np.array([[1, 1, 1, 1, 1, 1, 1],
                         [1, 2, 2, 2, 2, 2, 1],
                         [1, 2, 3, 3, 3, 2, 1],
                         [1, 2, 3, 3, 3, 2, 1],
                         [1, 2, 3, 3, 3, 2, 1],
                         [1, 2, 2, 2, 2, 2, 1],
                         [1, 1, 1, 1, 1, 1, 1]]) * 1 / 83, sum_k, val_thresh


def _resampler(path, my_ext, plot, out_folder, kernel):
    # Load the dataset
    ds = xr.open_dataset(path, mask_and_scale=False, chunks={'lat': 5000, 'lon': 5000})

    # select parameters according to the product.
    da, param = _param(ds)
    date, date_h = _date_extr(path)

    # AOI
    da = _aoi(da, ds, my_ext)

    # Algorithm core
    try:
        # create the valid values mask according to the fixed values
        da_msk = da.where(da <= param['DIGITAL_MAX'])

        # create the valid values mask to be used in the minimum validity threshold
        vo = xr.where(da <= param['DIGITAL_MAX'], 1, 0)

        # create the Lat/Lon values array of the output
        if any(my_ext):
            frst_val = 1
            lat_res = da.lat[frst_val:-1:3]
            lon_res = da.lon[frst_val:-1:3]
        else:
            frst_val = 0
            lat_res = da.lat[frst_val:-1:3]
            lon_res = da.lon[frst_val:-1:3]

        if kernel == '3x3' and my_ext == []:
            # calculate the number of valid values
            vo_cnt = vo.coarsen(lat=3, lon=3, boundary='trim', keep_attrs=False).sum()
            val_thresh = 5

            weights = np.repeat(1.0, 3 ** 2).reshape(3, 3) / 3 ** 2

            if hasattr(da_msk, 'time'):
                data = da_msk.data[0]
            else:
                data = da_msk.data

            da_g = dask_image.ndfilters.convolve(data, weights, mode='nearest')
            coarsen = xr.DataArray(da_g[frst_val:-1:3, frst_val:-1:3], coords=[('lat', vo_cnt.lat),
                                                                               ('lon', vo_cnt.lon)])
        elif kernel == '3x3':
            vo_cnt = vo.coarsen(lat=3, lon=3, boundary='trim', keep_attrs=False).sum()
            val_thresh = 5
            coarsen = da_msk.coarsen(lat=3, lon=3, boundary='trim', keep_attrs=False).mean()

        else:
            weights, sum_k, val_thresh = _kernel_sel(kernel)

            # Valid values per cell
            vo_cnt = dask_image.ndfilters.convolve(vo.data, sum_k, mode='nearest')
            vo_cnt = xr.DataArray(vo_cnt[frst_val:-1:3, frst_val:-1:3], coords=[('lat', lat_res), ('lon', lon_res)])

            # Coarsen values
            da_g = dask_image.ndfilters.convolve(da_msk.data, weights, mode='nearest')
            coarsen = xr.DataArray(da_g[frst_val:-1:3, frst_val:-1:3], coords=[('lat', lat_res), ('lon', lon_res)])

        # force results to integer
        coarsen_int = np.rint(coarsen)

        # impose the minimum number of observations
        da_r = coarsen_int.where(vo_cnt >= val_thresh)

        # force nan to int
        da_r = xr.where(np.isnan(da_r), 255, coarsen_int)

        if not hasattr(da_r, 'time'):
            da_r = da_r.assign_coords({'time': date_h})
            da_r = da_r.expand_dims(dim='time', axis=0)

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        raise sys.exit(1)

    # Output write
    try:
        da_r.name = param['product']
        da_r.attrs['short_name'] = param['short_name']
        da_r.attrs['long_name'] = param['long_name']
        da_r.attrs['_FillValue'] = int(255)
        da_r.attrs['scale_factor'] = np.float32(param['SCALING'])
        da_r.attrs['add_offset'] = np.float32(param['OFFSET'])

        prmts = dict({param['product']: {'dtype': 'i4', 'zlib': 'True', 'complevel': 4}})

        name = param['product']
        if not kernel == 'Gauss':
            if len(my_ext) != 0:
                file_name = f'CGLS_{name}_{date}_1KM_Resampled_{kernel}_AOI.nc'
            else:
                file_name = f'CGLS_{name}_{date}_1KM_Resampled_{kernel}.nc'
        else:
            if len(my_ext) != 0:
                file_name = f'CGLS_{name}_{date}_1KM_Resampled_G_AOI.nc'
            else:
                file_name = f'CGLS_{name}_{date}_1KM_Resampled_G.nc'
        out_file = os.path.join(out_folder, file_name)

        da_r.to_netcdf(out_file, encoding=prmts)
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        raise sys.exit(1)

    print(f'{file_name} resampled')

    # Plot
    if plot:
        da_r.plot(robust=True, cmap='YlGn', figsize=(15, 10))
        plt.title(f'Copernicus Global Land\n Resampled {name} to 1K over Europe\n date: {date_h.date()}')
        plt.ylabel('latitude')
        plt.xlabel('longitude')
        plt.draw()
        plt.show()


def main():
    """ Copernics Global Land Resampler

        The aim of this tool is to facilitate the resampling of the 333m ProbaV Copernicus Global Land Service products
        [1] (i.e. NDVI, FaPAR LAI, ... ) to the coarsen resolution of 1km.
        With the present release only the main indexes per products can be resampled. Other indexes, like the RMSE,
        can't be resampled.
        More info a about quality assessment can be found in the report create for the R version of this tool [2]

        [1] https://land.copernicus.eu/global/themes/vegetation
        [2] https://github.com/xavi-rp/ResampleTool_notebook/blob/master/Resample_Report_v2.5.pdf
    """
    '''
    Instructions:
    The tool is able to process act in different way according to the necessity
    
    - Single file: fill the path with the exact position and the file name
    - Bach processing: Define the folder without any further info about the extension of files
    - SemiAutomatic download (single file): Leave the path empty, a wizard will help in the selection and download.
      If the semiautomatic download is selected as option user ID and password needs to be defined. 
      Credential can be obtained here https://land.copernicus.vgt.vito.be/PDF/portal/Application.html#Home 
      through the Register form (on the upper right part of the page)
      
    The default resampling method is an aggregation over a matrix of 3x3. This follows the ratio between the nominal 
    products at 333m (1/336) and the 1 km (1/112) grid. Other kernels are selectable so users can choose which best suits 
    their purpose keeping in mind that, even if the pyramidal 5x5 (5x5_P) has achieved better results, only the 3x3 is
    considered the official one.
    '''

    path = r'D:\data\Resample\c_gls_LAI300_201901100000_GLOBE_PROBAV_V1.0.1.nc'

    '''
    Kernels: 
    # iXi -> Kernel's shape 
    # _G -> Gauss
    # _P -> Pyramidal
    # _T -> Truncated pyramidal 
    # Available Kernels ['3x3', '3x3_G', '3x3_P', '5x5', '5x5_G', '5x5_P', '5x5_T', '7x7', '7x7_G', '7x7_P', '7x7_T']
    '''

    i = '3x3'

    # define the output folder
    out_folder = r'D:\tmp'

    # Define the credential for the Copernicus Global Land repository
    user = ''
    psw = ''

    # Define the AOI
    # Coordinates are expressed in Decimal degrees (DD)
    # expressed according to [Upper left long, lat, Lower right long, lat] schema
    # es: AOI = [-18.58, 62.95, 51.57, 28.5]  # Europe
    AOI = []

    # define covariance kernel or coarsen aggregation resolution
    kernel = i

    # Define if plot results or not
    plot = False

    with Client() as client:
        # Processing
        if path == '':
            # Download and process
            assert user, 'User ID is empty'
            assert psw, 'Password is empty'

            path = _downloader(user, psw, out_folder)
            _resampler(path, AOI, plot, out_folder, kernel)
        elif os.path.isfile(path):
            # Single file process
            _resampler(path, AOI, plot, out_folder, kernel)
        elif os.path.isdir(path):
            # Multiprocessing for local files
            if not os.listdir(path):
                print("Directory is empty")
            else:
                for filename in os.listdir(path):
                    if filename.endswith(".nc"):
                        path_ = os.path.join(path, filename)
                        _resampler(path_, AOI, plot, out_folder, kernel)
        print('Conversion complete')
    print('Workers stopped')


if __name__ == '__main__':
    try:
        print('\nCopernics Global Land Resampler started')
        main()
    except KeyboardInterrupt:
        print('Process killed by user')
        raise sys.exit(1)

