
# -*- coding: utf-8 -*-
from scipy.stats import friedmanchisquare
import numpy as np
#%%



data = np.array([[4, 3, 6, 1.5, 5, 1.5],
                 [3.5, 3.5, 6,  1.5, 5, 1.5],
                 [3, 2, 4, 5, 6, 1]]).T

# Perform the Friedman test
stat, p = friedmanchisquare(data[0,:],data[1,:],data[2,:],data[3,:],
                            data[4,:],data[5,:])

# Output the results
print("Friedman test statistic:", stat)
print("p-value:", p)