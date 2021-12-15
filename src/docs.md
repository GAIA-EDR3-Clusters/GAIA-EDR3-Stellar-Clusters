# Understanding the Code

This documentation aims to explain the programs used in this study.

### `cluster_clean.py`

`cluster_clean.py` aims to clean the data so that only stars that are considered to be in the cluster are selected and further analysed. The approach we have used is a direct, yet conservative approach and uses data from the SIMBAD database on Gaia observations to plot the relevant properties of the cluster stars.

### How to Run the Program

`cluster_clean.py` runs as so in the command line using Python 3 - 

```Bash

python cluster_clean.py cluster_name

```
### How the Program Works
The program first starts off by importing the relevant python libraries and plugins (follow the instructions [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/README.md) on this documentation to learn how you can install these libraries) - 

```Python

import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import pyvo as vo
import pylab

```

It then also loads the zeropoint correction tables attained by Lindegren et. al. 2020 from the Gaia Early Data Release 3, as shown below.

```Python

from zero_point import zpt
zpt.load_tables()

```

It then checks if a cluster name has been provided and then locates the relevant `.csv` file, as shown below - 

```Python

nargs = len(sys.argv) - 1
if nargs == 1:
    cluster = sys.argv[1]
else:
    print("Needs a cluster name, Rsky and Vlim")
    print("eg.")
    print("python cluster_clean.py M67 20 18")
    sys.exit()

filename = cluster+".csv"
print("reading "+filename)
try:
    data = pd.read_csv(filename)
except:
    print("Error reading data file / no such cluster")
    sys.exit()
    
```

It then tries to load a generated .stats file. If the `.stats` file is not located, then it will generate the `.stats` file itself later in the program. The `.stats` file contains information as to which stars to select. This selection process is done by checking if the stars fall within a certain range around the cluster (both in propermotion and in physical location) and then selects the star. To put it more simply, the program draws an oval-shape around the cluster and then selects stars that are within the oval.

This process of generating that oval shape can be seen below -

```Python

def draw_pm_circle(pmra0,pmdec0,rad,colour):
    t = np.linspace(0,2*np.pi,100)
    x = pmra0 + rad*np.sin(t)
    y = pmdec0 + rad*np.cos(t)
    plt.plot(x,y,colour,alpha=0.5)
    return

def draw_sky_circle(ra0,dec0,rad,colour):
    t = np.linspace(0,2*np.pi,100)
    x = ra0 + rad*np.sin(t)/np.cos(dec0*np.pi/180.0)
    y = dec0 + rad*np.cos(t)
    plt.plot(x,y,colour,alpha=0.5)
    return

```

And then all the relevant plots are then created. You can see the final result in [Figure 1](#figure-1-generated-plot-for-ngc7789) below.

Finally, the programs generates a `.zeropoint.txt`, which contains the following data for each star selected - 
1. Absolute magnitude of the stars (Gmag)
2. Blue light to red light ratio of these stars (BP-RP)
3. Offset between the parallaxes of the stars and the median parallax for the entire cluster (parallax offset)
4. Name of the cluster
5. Nueffused
6. Cluster PSC
7. The ecliptic latitude of the stars within the cluster
8. Soltype
9. The galactic latitude (gl) and the galactic longitude (gb) of the stars in the cluster
10. The zeropoint values for each of the stars
11. The parallax of the star

```Python

outputfilname = cluster+".zeropoint.txt"

for i in range(len(cluster_parallaxes)):
    if cluster_gmag[i]<18:
        f.write(str(cluster_gmag[i])+' ' 
            +str(cluster_bp_rp[i])+' '
            +str(cluster_parallaxes[i]-median_parallax_2)+' '
            +str(cluster)+' '
            +str(cluster_nueffused[i])+' '
            +str(cluster_psc[i])+' '
            +str(cluster_ecl_lat[i])+' '
            +str(cluster_soltype[i])+' '
            +str(cluster_gl[i])+' '
            +str(cluster_gb[i])+' '
            +str(cluster_zpvals[i])+' '
            +str(cluster_parallaxes[i])+' '
            +"\n")
            
 ```

## Figure 1: Generated Plot for NGC7789
![Generated Plot for NGC7789](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/plots/NGC7789.png)

### `zp.py`

`zp.py` aims to compare the stars of several clusters to each other and identify 'tip of the red giant branch' stars, also known as TRGB stars. With this program, we can gain insights into the properties of the stars of several clusters.

### How to Run the Program

The program can be run as follows using Python 3 - 

```Bash

python zp.py type_of_cluster

```

There are three options for the type of cluster. There is - 

1. all
2. open
3. globs

### How the program works

The program first starts off by importing the relevant python libraries and plugins (follow the instructions [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/README.md) on this documentation to learn how you can install these libraries) - 

```Python


import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as patches

```

After that, it defines the globular clusters - 

```Python

def glob(cluster_name):

    globs = ["NGC6397","M4","NGC6752","NGC104","M22","M55","M71","NGC5139",
             "M12","M10","NGC3201","M13","NGC5904","M92","NGC7099","NGC6352",
             "NGC6544","NGC6362","NGC6541","NGC288","NGC362","NGC6723","NGC4372"]

    return cluster_name in globs
    
```

It then gets the cluster type that has been selected - 

```Python

nargs = len(sys.argv) - 1
if (nargs==1):
    argument = sys.argv[1]    
    corr = sys.argv[1]
else:
    print("Useage : ")
    print("python zp.py all/glob/open")
    print("eg python zp.py all")
    sys.exit()

```

`zp.py` then collects the data from the `zeropoint.txt` files. To learn more about the `zeropoint.txt`, visit the link provided [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/analysis/docs.md) - 

```Python

filename = "../analysis/all.zeropoint.txt"
gmag, bp_rp, zp, zp0, nueff, psc, ecl, soltype, gl, gb, par = np.loadtxt(filename, usecols=(0,1,2,10,4,5,6,7,8,9,11), unpack=True, dtype=float)

```

Finally, the program plots the data. One of the plots generated by the program is shown in [Figure 2](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/docs.md#figure-2-the-compiled-collection-of-stars-selected) below.

## Figure 2. The Compiled Collection of Stars Selected
![The Compiled Collection of Stars Selected](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/plots/Gaia-EDR3-cluster-sample.png)

### `bp_rp_vs_nueff.py`

This program aims to find a relationship between nueff and the blue light to red light ratio (BP-RP). It is then used to confirm that our study is using data from similar stars that were used in other, independent studies such as the one by Zinn 2021.

### How to Run the Program

Simply - 

```Bash

python3 bp_rp_vs_nueff.py

```

### How the Program Works

The program first starts off by importing the relevant python libraries and plugins (follow the instructions [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/README.md) on this documentation to learn how you can install these libraries) - 

```Python

import numpy as np
from matplotlib import pyplot as plt

```

It then gets the data in the same way `zp.py` gets its data. `bp_rp_vs_nueff.py` then collects the data from the `zeropoint.txt` files. To learn more about the `zeropoint.txt`, visit the link provided [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/analysis/docs.md) -  - 

```Python

filename = "../analysis/all.zeropoint.txt"
# 0    1     2        3    4     5   6   7       8  9  10     11
# gmag bp-rp zpoffset name nueff psc ecl soltype gl gb zpvals parallax
gmag, bp_rp, zp, zp0, nueff, psc, ecl, soltype, gl, gb, par = np.loadtxt(filename, usecols=(0,1,2,10,4,5,6,7,8,9,11), unpack=True, dtype=float)

```


Next, the program needed to remove `NaN`, or None values from the arrays. The function does that for 2 arrays that can be displayed later - 

```Python

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
    
```

The program then applies a polynomial with the best fit - 

```Python

polynomial = np.polyfit(bp_rp, nueff, deg=4)
x = np.linspace(bp_rp.min(),bp_rp.max(),1000)
y = (polynomial[0] * (x**4)) + (polynomial[1] * (x**3)) + (polynomial[2] * (x**2)) + (polynomial[3] * x) + polynomial[4]

```

And then plots the curve along with the rest of the data as a scatter plot - 

```Python

plt.plot(bp_rp, nueff, 'b.', alpha=0.1, label="Cluster stars")
plt.plot(x, y, 'g-', alpha = 1, label="Best-fit polynomial curve")
plt.xlabel('BP-RP')
plt.ylabel('Nueff')
plt.legend()
plt.show()

```

The result is shown in [Figure 3](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/docs.md#figure-3-bp-rp-vs-nueff) below. The trend turned out to be - 

f(x) = -0.00359974x^4 +0.0291391x^3 -0.03846824x^2 -0.21949758x +1.75909619

## Figure 3. BP-RP vs Nueff
![BP-RP Vs. Nueff](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/plots/bp_rp_vs_nueff.png)
