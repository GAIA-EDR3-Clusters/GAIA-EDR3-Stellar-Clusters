# Understanding the Stats Files

The stats files contain information as to how to select the stellar cluster from the `.csv` file and how to clean the data so that only stars that are considered to be in the cluster are selected.

## How was this generated?

The stats file was generated using the following code from `cluster_clean.py` - 

```Python

if sfileloaded==False:
    stats_file = '{}.stats'.format(cluster)
    print("Results saved to "+stats_file)
    with open(stats_file, "w") as file:
        file.write('X, Y, Xlow, Xhigh, Ylow, Yhigh, PM_Rad, sky_rad\n{} {} {} {} {} {} {} {}\n'.format(x0,y0,xlo,xhi,ylo,yhi,rad,pos_rmax))
        file.close()

```

However, it is only generated if there is no already existing stats file for the specific stellar cluster the user has selected.

The link to the program is [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/cluster_clean.py) and the link to the documentation for the program is [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/docs.md#cluster_cleanpy)

## What does it contain?

The stats file contains the following from left to right - 

1. The RA (right ascension) for the centre of the stellar cluster
2. The DEC (declination) for the centre of the stellar cluster
3. A range around the stellar cluster from which stars can be considered part of the cluster (Xhigh, Xlow, Yhigh, Ylow)
4. The range around the propermotion of the stellar cluster from which stars can be considered part of the cluster (PMrad)

## How can the data be analysed?

This data in these `stats` files are used as a method to select stars based on their relative position to the stellar cluster and relative propermotion to the stellar cluster and judge if a particular star in the star should be considered a part of the stellar cluster chosen. It is primarily used by `cluster_clean.py` which cleans the data, leaving only stars to be considered in the stellar cluster.
