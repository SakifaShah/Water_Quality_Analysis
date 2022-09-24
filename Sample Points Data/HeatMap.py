# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:29:20 2022

@author: Eutech
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_excel('Sample points data.xls', index_col=0)

print(data.corr())

# plotting correlation heatmap
dataplot = sb.heatmap(data.corr(), cmap="YlGnBu", annot=True)
  
# displaying heatmap
plt.show()

