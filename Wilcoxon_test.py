#%%
# -*- coding: utf-8 -*-
from scipy.stats import wilcoxon
import numpy as np


#%%

ERR_ASGCN = np.load('1-Fed-traffic/ERR_ASGCN.npy')
ERR_LSTM = np.load('1-Fed-traffic/ERR_LSTM.npy')
ERR_TGCN = np.load('1-Fed-traffic/ERR_TGCN.npy')
ERR_MLP =np.load('1-Fed-traffic/ERR_MLP.npy')
ERR_DCRNN =np.load('1-Fed-traffic/ERR_DCRNN.npy')
ERR_FeSTGCN =np.load('1-Fed-traffic/ERR_FeSTGCN.npy')
#%%
statistic, p_value = wilcoxon(ERR_FeSTGCN, ERR_MLP)
# 输出检验结果
print("Wilcoxon statistic:", statistic)
print("p-value:", p_value)

#%%
M=[]
ERR=[ERR_MLP,ERR_LSTM,ERR_ASGCN,ERR_DCRNN,ERR_TGCN,ERR_FeSTGCN]
for i in range(5):
    statistic, p_value = wilcoxon(np.abs(ERR[-1]), np.abs(ERR[i]),method = 'approx',alternative='two-sided')
    M.append(np.array([statistic, p_value]))

MM=np.vstack(M)
