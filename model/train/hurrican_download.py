#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import os 


# In[16]:


dates = [ 20210830 , 20210831 , 20210901, 20210902 ] 
runs = [ 'a' ,'b' ]

huricaneIDA_files = ['https://stormscdn.ngs.noaa.gov/downloads/{0}{1}_RGB.tar'.format(date, run) for date in dates for run in runs]



# In[18]:

out_path = '/Users/gracecolverd/mres_proj/raw_data/'




for f in huricaneIDA_files:
    print('Starting file {}'.format(f[-17:])  )
    output_path = out_path + f[-17:]
    if  os.path.exists(output_path) is False:
        urllib.request.urlretrieve(f, output_path)
    print('Finished!')







# In[ ]:




