#!/usr/bin/env python
# coding: utf-8



### Download all patches in train/test folder structure. Added for completeness.
#####Data available from 
#https://drive.google.com/drive/folders/1b9qMhMblYRnJHzZOqJeFnltUX9jpfLTo?usp=sharing 

#Due to proprietary RMS footprints and labels, not turned into public notebook


#Use geo env 
import sys
sys.path.append('~/HurricaneDamage/patch_generation/src')
from patches_utils import download_all_clipped_polys


#Set output dir
output_dir =  '/Users/gracecolverd/mres_proj/full_data'

#Download patches 
download_all_clipped_polys(output_dir = output_dir, atr_indx=0 , local = True)






