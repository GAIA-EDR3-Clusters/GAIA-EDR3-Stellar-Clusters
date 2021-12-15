# Understanding the Generated Plots

This documentation aims to provide a better picture as to what the plots in this directory for all the clusters represent.

## Figure 1. Generated Plot for NGC7789
![Generated Plot for NGC7789](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/plots/NGC7789.png)

[Figure 1](#figure-1-generated-plot-for-ngc7789) above shows the generated plots for NGC7789 by `cluster_clean.py`, link to the code [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/cluster_clean.py). 

The plot can be split into two parts - 

1. The six plots to the left show the program filtering through the data to obtain stars that are within the cluster. Stars in the field of view that are in the selected cluster are determined based on the Propermotion in the right ascension (PMRA) and the propermotion in the declination, or PMDEC (this is indicated by the bottom left scatter plot). The stars are also determined based of their physical right ascension (RA) and declination, or DEC (this is shown by the scatter plot in the top row, to the right of two histogram plots). The histogram plots simply show the distribution of stars in their respective scatter plots. The green dots show the stars which were selecetd and the blue stars were the ones that were dropped because the program `cluster_clean.py` deemed them not located within the cluster.
2. The three plots to the right show the details of those selected stars and their respective parallaxes based on their absolute magnitude (Gmag). The vertical plot in the middle of the image shows the absolute magnitude of the stars (Gmag) based on the ratio between red and blue light received from those stars (BP-RP). The top-right plot in the figure above shows the original parallaxes of those stars relative to their Gmag and the bottom right plot shows an attempted correction of the parallaxes by the program. The green dots in the far right plots show the stars that were selected based on their parallaxes while the red dots show stars that were dropped because their parallaxes were deemed to no fit within the standard deviation range calculated by the program.

The same figure has been created but with data from different clusters.

To learn more about how `cluster_clean.py` works, visit the following documentation [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/docs.md#cluster_cleanpy).

## Figure 2. The Compiled Collection of Stars Selected

![The Compiled Collection of Stars Selected](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/plots/Gaia-EDR3-cluster-sample.png)

[Figure 2](#figure-2-the-compiled-collection-of-stars-selected) above shows all the stars that are in all the clusters we have studied. Specifically, this plot shows the absolute magnitude of these stars (Gmag) based on their blue light to red light ratio (BP-RP). This plot was generated using `zp.py`, link to the code is [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/zp.py) and the link to the documentation for this code is [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/docs.md#zppy)

## Figure 3. BP-RP vs Nueff
![BP-RP Vs. Nueff](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/plots/bp_rp_vs_nueff.png)

[Figure 3](#figure-3-bp-rp-vs-nueff) above shows the relationship derived from the blue light to red light ratio (BP-RP) and the variable Nueff. The dots represent the stars that were being analysed in this study and the line shows the best-fit polynomial curve for the data. The best fit is - 

f(x) = -0.00359974x^4 +0.0291391x^3 -0.03846824x^2 -0.21949758x +1.75909619

This plot was generated using `bp_rp_vs_nueff.py`, ink to the code is [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/bp_rp_vs_nueff.py) and the link to the documentation for this code is [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/docs.md#bp_rp_vs_nueffpy)
