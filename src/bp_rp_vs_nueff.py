import numpy as np
from matplotlib import pyplot as plt

def remove_nan(x, y):
    x = x.tolist()
    y = y.tolist()
    x_without_nan = []
    y_without_nan = []
    for i in range(len(x)):
        if 'nan' not in str(x[i]).strip().lower() and 'nan' not in str(y[i]).strip().lower():
            x_without_nan.append(x[i])
            y_without_nan.append(y[i])

    return x_without_nan, y_without_nan

def return_y(x, coeff):
    y = 0
    for i in range(len(coeff)):
        y += coeff[i] * (x**((len(coeff)-1)-i))
    return y

filename = "../analysis/all.zeropoint.txt"
# 0    1     2        3    4     5   6   7       8  9  10     11
# gmag bp-rp zpoffset name nueff psc ecl soltype gl gb zpvals parallax
gmag, bp_rp, zp, zp0, nueff, psc, ecl, soltype, gl, gb, par = np.loadtxt(filename, usecols=(0,1,2,10,4,5,6,7,8,9,11), unpack=True, dtype=float)

bp_rp, nueff = remove_nan(bp_rp, nueff)
bp_rp = np.array(bp_rp, dtype=np.float)
nueff = np.array(nueff, dtype=np.float)
polynomial = np.polyfit(nueff, bp_rp, deg=4)
x = np.linspace(nueff.min(),nueff.max(),1000)
y = return_y(x, polynomial)
print(polynomial)
plt.plot(nueff, bp_rp, 'b.', alpha=0.1, label="Cluster stars")
plt.plot(x, y, 'g-', alpha = 1, label="Best-fit polynomial curve")
plt.ylabel('BP-RP')
plt.xlabel('Nueff')
plt.legend()
plt.show()

x = 1.3
y = return_y(x, polynomial)
print(x, y)
x = 1.6
y = return_y(x, polynomial)
print(x, y)