#!/usr/bin/env python
# coding: utf-8



#Download patches 


#Use geo env 
import sys
sys.path.append('~/HurricaneDamage/patch_generation/src')
from patches_utils import download_all_clipped_polys



output_dir =  '/Users/gracecolverd/mres_proj/full_data'


download_all_clipped_polys(output_dir = output_dir, atr_indx=0 , local = True)






