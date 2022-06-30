#!/usr/bin/env python
# coding: utf-8


import urllib.request
import os 


### This scrip Downloads high res NOAA imagery

###Use https://storms.ngs.noaa.gov/ to find storm of choice, and update the values for dates and runs dept on the data available. This script downloads the Delta Imagery 

dates = [ 20210830 , 20210831 , 20210901, 20210902 ] 
runs = [ 'a' ,'b' ]

huricaneIDA_files = ['https://stormscdn.ngs.noaa.gov/downloads/{0}{1}_RGB.tar'.format(date, run) for date in dates for run in runs]


out_path = '/Users/gracecolverd/mres_proj/raw_data/'


for f in huricaneIDA_files:
    print('Starting file {}'.format(f[-17:])  )
    output_path = out_path + f[-17:]
    if  os.path.exists(output_path) is False:
        urllib.request.urlretrieve(f, output_path)
    print('Finished!')





