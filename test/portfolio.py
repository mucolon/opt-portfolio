import numpy as np
from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='RFNKYQY0BFFR03I9')


class Fund:

    def __init__(self, name, div, expr):
        # name: fund name
        # div: fund dividend rate
        # expr: fund expense ratio
        if type(name) != str:
            name = str(name)



        self.name = name
        self.div = div
        self.expr = expr



'''
# bonds
print("Bonds\n")

lqd = 3.34
lqd_e = .15

vcit = 3.4
vcit_e = .07

hyg = 5.21
hyg_e = .49

emb = 5.37
emb_e = .39

igib = 3.63
igib_e = .1

cwb = 5
cwb_e = .4

vclt = 3.86
vclt_e = .07

lqd_r = lqd - lqd_e     # 3.19
vcit_r = vcit - vcit_e  # 3.33
hyg_r = hyg - hyg_e     # 4.72
emb_r = emb - emb_e     # 4.98
igib_r = igib - igib_e  # 3.53
cwb_r = cwb - cwb_e     # 4.6
vclt_r = vclt - vclt_e  # 3.79

print("Actual Yields (After Expense Ratio Deduction)")
print("LQD %6.3f\nVCIT %6.3f\nHYG %6.3f\nEMB %6.3f\nIGIB %6.3f\nCWB %6.3f\nVCLT %6.3f" % (lqd_r,
                                                                                          vcit_r, hyg_r, emb_r, igib_r, cwb_r, vclt_r))
# print(lqd_r,
#       vcit_r,
#       hyg_r,
#       emb_r,
#       igib_r,
#       cwb_r,
#       vclt_r)

bonds = np.array([lqd_r, vcit_r, hyg_r, emb_r, igib_r, cwb_r, vclt_r])
bonds_ratio = np.array(np.size(bonds) * [None])
bonds_dis = np.array(np.size(bonds) * [None])
bonds_per = 11.4

for i in range(np.size(bonds)):
  bonds_ratio[i] = bonds[i] / lqd_r
  bonds_dis[i] = bonds_ratio[i] * bonds_per

print("\nDistributions")
print(bonds_ratio * bonds_per)
print("Total Distributions")
print(np.sum(bonds_ratio) * bonds_per)
print("\n")


# dividend funds

spyd = 4.57
spyd_e = .08

hdv = 3.29
hdv_e = .08

vnq = 3.69
vnq_e = .12

sphd = 4.18
sphd_e = .3

spyd_r = spyd - spyd_e
hdv_r = hdv - hdv_e
vnq_r = vnq - vnq_e
sphd_r = sphd - sphd_e
icahn_r = 7.418

print(spyd_r,
      hdv_r,
      vnq_r,
      sphd_r,
      icahn_r)
print("\n")

div = np.array([spyd_r,
                hdv_r,
                vnq_r,
                sphd_r,
                icahn_r])

div_ratio = np.array(np.size(div) * [None])

for i in range(np.size(div)):
  div_ratio[i] = div[i] / np.min(div)

div_per = 14.3
print(div_ratio * div_per)
print(np.sum(div_ratio) * div_per)
'''
