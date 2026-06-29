import os
import json
import random
import torch
import math
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter
import matplotlib

import numpy as np
from typing import List, Tuple, Set

def generate_distinct_colors(n):
   colors = []
   for i in range(n):
       # Add slight variations to prevent too similar colors
       h = np.random.rand()
       s = 0.5 + np.random.rand() * 0.5  # 0.5-1.0
       v = 0.8 + np.random.rand() * 0.2  # 0.8-1.0
       colors.append(plt.cm.hsv(h))
   return np.array(colors)

print(generate_distinct_colors(69))