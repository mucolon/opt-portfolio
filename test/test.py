import numpy as np

num = 6
allocate_guess = 100 / num
etfs = np.array(["HDV", "SPHD", "SPYD", "DVY", "VYM", "SCHD"])
beta = np.array([1.02, 1.06, 1.11, 1.06, 1.01, .99])
beta_inv = 1 / beta

# =============================================================================
# allocate = beta_inv*allocate_guess
# sum = np.sum(allocate)
#
# allocate_edit = np.ceil(allocate)
# sum_edit = np.sum(allocate_edit)
# print(etfs)
# print(allocate_edit)
# =============================================================================

max_beta = np.max(beta)
min_beta = np.min(beta)
flag = False
i = 0
while flag == False:
    if beta[i] < max_beta:
        i += 1
    else:
        max_beta_index = i
        flag = True
flag = False
i = 0
while flag == False:
    if beta[i] > min_beta:
        i += 1
    else:
        min_beta_index = i
        flag = True
        
beta_mod = max_beta / beta
beta_mod_1 = 5 * beta_mod

