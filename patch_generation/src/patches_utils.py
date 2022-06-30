#!/usr/bin/env python
# coding: utf-8

### Utils Doc for creating patches from Ida NOAA data, using RMS footprints and labels 
### Added for completeness - patches data can be downloaded from  https://drive.google.com/drive/folders/1b9qMhMblYRnJHzZOqJeFnltUX9jpfLTo?usp=sharing

### Due to proprietary nature of RMS labels, no point turning this into publically usable notebooks


# Imports
import fiona 
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rioxarray as rxr
import xarray as xr 
import glob

from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd
from shapely.geometry import Polygon
from geopandas.tools import sjoin
import shapely.geometry as sg
import shapely.ops as so
from shapely.ops import unary_union
from pathlib import Path
from datetime import datetime
from shapely.geometry import Point
from math import sqrt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import rioxarray
from rioxarray import merge
from rasterio.plot import show  
import os
import random
    



def open_tile_index(tile_index_dir= '/Users/gracecolverd/mres_proj/IDA_shapefiles' ):
    #Load tile index with damaged numbers 
    shape_path= tile_index_dir + '/IDA_Tile_Index.shp'
    return gpd.read_file(shape_path)

def open_gdb_damage():
    #Load GDB with all polygons
    path = '/Users/gracecolverd/mres_proj/damage_assesment/HU_Ida_Damage_Assessment_20210903.shp'
    return gpd.read_file(path)


def return_buildings_within_polygon_damageInfo(polygon):
    """
    returns all buildings within polygon, returns a geodataframe 
    """
    #Load GDB with all polygons
    path = '/Users/gracecolverd/mres_proj/damage_assesment/HU_Ida_Damage_Assessment_20210903.shp'
    gdf = gpd.read_file(path)
    return sjoin(gdf, polygon, how='inner')


def default_patch_test(polygon):
    """ Generate square patch of default size around polygon
    """
    med_patch_area  =  3.312011e-08
    
    minx, miny, maxx, maxy = polygon.bounds
    
    # get the centroid
    centroid = [(maxx+minx)/2, (maxy+miny)/2]
    
    # get the diagonal
    l = sqrt(med_patch_area)
    diagonal = l * sqrt(2) 
    
    return Point(centroid).buffer(diagonal/sqrt(2.)/1.5, cap_style=3)

    
def to_df(polygon):
    """convert shapely polygon to GeoDataFrame
    """
    patch_df = gpd.GeoDataFrame(geometry=gpd.GeoSeries( polygon) ).set_crs( crs = 'EPSG:4326')
    return patch_df.to_crs( crs = 'EPSG:4326')


def find_overlapping_shapes(df1, df2):
    """ Find overlapping geoshapes between two GeoDataFrames
    """ 
    import geopandas as gpd
    return gpd.sjoin(df1, df2, how='inner')
            
    
def create_sample_list_with_boundary(boundary = 0.998):
    """
    Create list of equal numbers of damaged samples and non-damaged samples.
    Take only polygons within the boundary of the tif data we have based on the boundary value 
    """
    tif_df = open_tile_index()

    #Create outside boundary of all tifs 
    tile_boundary = gpd.GeoSeries(unary_union( tif_df.geometry))
    boundary_df = gpd.GeoDataFrame(geometry=gpd.GeoSeries( tile_boundary) )
    buff_df = gpd.GeoDataFrame(geometry=gpd.GeoSeries( boundary_df.scale(boundary, boundary) ) )
    
    #Find polys and attributes within 
    houses = return_buildings_within_polygon_damageInfo(buff_df)

    #Create index of damaged buildings, and find the same number of non damaged ones with ratio factor. both indexes shuffled.  
    damged_poly_df = houses[ houses['Damage'] == 'Yes' ]
    nond_poly_df = houses[ houses['Damage'] == 'No' ]

    d_list = damged_poly_df.index.to_list()
    non_d_list_long = nond_poly_df.index.to_list() 

    ratio = len(d_list) / len(non_d_list_long)
    
    random.Random(4).shuffle(non_d_list_long  )
    non_d_list= non_d_list_long[:(int(ratio *len(non_d_list_long)) ) ]
    
    indx_list = non_d_list + d_list

    df = houses.loc[ houses.index.isin(indx_list), : ]
    df.to_file('sample_list_houses.shp')
    
    return df

    
def create_TT_file_structure(output_dir):
    """ Create the file structure needed for download of patches in outputdir
    """
    #create file structure
    test_path = output_dir + '/test'
    train_path = output_dir + '/train'

    folders_str = ['/damaged_N/', '/damaged_Y/']
    test_list = [ test_path + x for x in folders_str ]
    train_list = [ train_path + x for x in folders_str]
    folders_list = test_list +train_list
    

    for f in folders_list:
        isExist = os.path.exists(f)

        if not isExist:
            os.makedirs(f)
            print("The new directory is created!")
        else:
            print('Folders already ready')


def download_clipped_image_max(poly, index, output_file_path, local, patch_method= default_patch_test ):
    """
    Download clipped tif image, based around the polygon house provided, saved to the output file path. 
    Inputs:
    poly: polygon to clip around
    output_file_path: location of 
    index: index of poly, to be used in label 
    
    patch_method: how we take the patch. default_patch_test uses the median shape for these buildings 
    """
    
    if local == True:
        base = '~/*RGB/' 
        overlapping_tiles = open_tile_index()

    if local == False:
        base = '/work/scratch-nopw/gbc/Data/*RGB/'
        base = '~/raw_data/*RGB/' 
        overlapping_tiles = open_tile_index('~/mred_proj/tile_index')

    patch = patch_method(poly)
    patch_df = gpd.GeoDataFrame(geometry=gpd.GeoSeries( patch) )
    patch_df = patch_df.set_crs( crs = 'EPSG:4326')
    
    #find relevant tiles    
    tiles = find_overlapping_shapes(patch_df, overlapping_tiles) 
    print('Num of tifs to merge: {}'.format( len(tiles) ) )
 
    if len(tiles) > 1: 
        if len(tiles) > 3:
            tiles = tiles.sort_values(by = ['NB_nDamage', 'NB_Damaged'], ascending = False )[:3]
            print('The reduced num is {}'.format( len(tiles) ) )
        
        tif_list = [x for x in tiles.location ] 
        tif_file_paths=[ base + str(tn) for tn in tif_list ]
        tif_file_list = [ glob.glob(t)[0] for t in tif_file_paths ]
        
        elements=[]

        for file in tif_file_list:
            try:
                clip_array =  rioxarray.open_rasterio(file).rio.clip([patch] )
                elements.append(clip_array)
            except:
                print('File {} failed clip'.format(file) )
        tif = merge.merge_arrays(elements, nodata=0.0) # this is the time part 
       

    if len(tiles) == 1:
        tif_path = [base + str(x) for x in tiles.location ] 
        tif_file_path = glob.glob(tif_path[0])[0]
        opened_tif =  rxr.open_rasterio(tif_file_path)
        tif = opened_tif.rio.clip([patch])

    if len(tiles) == 0:
        print('No tifs here for shape')

    #If tif exists then download it 
    try:
        tif
    except NameError:
        var_exists = False
        print('No clipped tif downloaded')
    else:
        var_exists = True
        tif.rio.to_raster(output_file_path)
        print('Downloaded {}'.format(index) ) 

        
def download_polys(poly_df, output_dir, local,  atr_indx=0):
    """
    Download all the patches based for each polygons given, based on 80% train test split 
    
    Inputs
    poly_df: dataframe of polygons to be clipped
    output_dir: location to download to
    local: is this happening locally or on Jasmin
    atr_indx: where to start download from (used for manual multi threading 
    
    """
    tt_split = 0.8
    
    #extract damage, index, and house polygon
    list_polys_attributes = [ x for x in zip(poly_df.geometry, poly_df.Damage, poly_df.geometry.index) ] 

    #Create train / test split based on index
    ind_list = [x for x in poly_df.index] 
    random.Random(4).shuffle(ind_list)
    train_index, test_index = ind_list[:int(tt_split*len(ind_list))], ind_list[int(tt_split*len(ind_list)):] 
    
    print('Num of train: {} \n Num of test: {}'.format(len(train_index), len(test_index) ) ) 
    
    #Download 
    for poly, damage, index in list_polys_attributes[atr_indx:]:
        print(index)
        #check if poly downloaded at all
        glob_file = output_dir +   '/**/*/' + str(index) + '.tif' 
        name = glob.glob( glob_file)

        if len(name) == 0:
            #Train or test , damaged / non damage
            if index in train_index:
                file_ext = output_dir + '/train'
            elif index in test_index:
                file_ext = output_dir + '/test'

            if damage == 'Yes':
                file_ext = file_ext + '/damaged_Y/'
            elif damage =='No':
                file_ext = file_ext + '/damaged_N/'

            output_file_path = file_ext + str(index) + '.tif'
            file_exists = os.path.exists(output_file_path)

            if file_exists is False:
                    download_clipped_image_max(poly, index, output_file_path, local) 

 

#
def download_all_clipped_polys(output_dir, atr_indx=0 , local = True , house_dir = None):
    """" Function to call to download all the patches for Ida imagery  

    """
    print('Creating file structure..')
    #Create folders needed 
    create_TT_file_structure(output_dir)
    
    print('Finding 100k samples..')
    if local == True:
        poly_df = create_sample_list_with_boundary()
        # poly_df = open_houses_sample('/Users/gracecolverd/jasmin/mred_proj/mred_proj/houses_final')
    if local == False:
        poly_df = open_houses_sample(house_dir)
    
    print('Downloading samples..')
    download_polys(poly_df, output_dir, local, atr_indx)


def open_houses_sample(base):
    return gpd.read_file(base + '/sample_list_houses.shp')

