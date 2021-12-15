# Understanding the Zeropoint Data
## How was it Generated?

It was generated through `cluster_clean.py` using the following code - 

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

The link to the program is [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/cluster_clean.py) and the link to the documentation for the program is [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/docs.md#cluster_cleanpy)

## What does it contain?

The data is split in the following columns from left to right - 

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

## How can this Data be Analysed?

This data can be analysed through the program `zp.py`. The link to the program is [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/zp.py) and the link to the documentation is [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/docs.md#zppy)
