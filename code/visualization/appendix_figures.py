#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------Appendix Figures----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import os
import re
import numpy as np
from datetime import datetime
import json
from ast import literal_eval

# Plotting Packages
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"

from arch.unitroot import ADF
from arch.unitroot import PhillipsPerron
from arch.unitroot import KPSS

from statsmodels.tsa.stattools import grangercausalitytests

#%%
from wordcloud import WordCloud
from PIL import Image

from sklearn.decomposition import PCA
import scipy.stats

import spacy
nlp = spacy.load("en_core_web_sm")

#%%
# Common variables
colors=['#033C5A','#AA9868','#0190DB','#FFC72C','#A75523','#008364','#78BE20','#C9102F',
        '#033C5A','#AA9868','#0190DB','#FFC72C','#A75523','#008364','#78BE20','#C9102F']

area_range=15   # Number of areas + 1
dict_area={'1': 'consumer safety and health',
    '2': 'national and homeland security',
    '3': 'transportation',
    '4': 'labor and workplace',
    '5': 'environment and natural resources',
    '6': 'energy',
    '7': 'finance and banking',
    '8': 'general business and trade',
    '9': 'agriculture and rural development',
    '10': 'education and culture',
    '11': 'communications',
    '12': 'criminal justice',
    '13': 'housing, urban development, and social security',
    '14': 'international relations'}

# %%
# Set directory
# directory=os.path.dirname(os.path.realpath(__file__))
directory='code/visualization'

# # Create an output directory if it does not exist
output_folder=f'{directory}/../../figures/appendix_figures'
os.makedirs(output_folder, exist_ok=True)




