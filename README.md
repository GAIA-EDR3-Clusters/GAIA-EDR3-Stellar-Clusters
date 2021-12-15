# Gaia EDR3 Stellar Clusters
Gaia EDR3 Parallax Zero Point

This repository contains data from the Gaia Early Data Release 3 Data Set. This data set includes observations of stellar clusters made by the Gaia Telescope. This data is then analysed and then displayed in the form of figures, which are located in the `plots` folder.

## Aim of Study

The aim of the study is to analyse stellar clusters and determine if there is any systematic error in the GAIA EDR3 stellar cluster parallaxes. If so, how much is this error and what is causing the error.

## Cloning Project

To clone the project, just copy and paste this (make sure you have git installed on your computer) - 

```Bash

git clone https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters.git .

```

## Installing Relevant Libraries

To make sure that you have the relevant python libraries for the source code to run, download `requirements.txt` and then run the following command in the terminal - 

```Bash
pip3 install -r requirements.txt
```
If that does not work, then maybe try the following - 

```Bash
pip install -r requirements.txt
```
This repository will be using Python 3. There is currently no Python 2 equivalent.

## How the Data Was Analysed

Please refer to the documentation [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/src/docs.md) to learn how the programs work.

## Where can this Data be Accessed

Data was used from the .csv files located [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/tree/main/cluster_data). To learn more about the `.csv` files, click [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/cluster_data/docs.md). Stars were selected based on propermotion among other factors and their data was stored in the `.zeropoint.txt` files located [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/tree/main/analysis). To learn more about what is inside the `.zeropoint.txt` files and how to analyse them, visit the link [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/analysis/docs.md)

## Results

To see what the results look like, refer to figures 1 and 2 below. These figures are explained in more detail [here](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/plots/docs.md). Figures similar to figure 1 have been created for each of the stellar clusters analysed.

What we find is that there is a small, systematic error in Gaia's parallax data which is dependent on the colour of the stars.

## Figure 1: Generated Plot for NGC7789
![Generated Plot for NGC7789](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/plots/NGC7789.png)

## Figure 2. The Compiled Collection of Stars Selected
![The Compiled Collection of Stars Selected](https://github.com/rudrathegreat/Gaia-EDR3-Stellar-Clusters/blob/main/plots/Gaia-EDR3-cluster-sample.png)

## List of Clusters Analysed

IC4665,
M10,
M10,
M11,
M12,
M13,
M16,
M2,
M21,
M22,
M23,
M25,
M26,
M36,
M37,
M38,
M4,
M41,
M45,
M46,
M47,
M48,
M50,
M52,
M55,
M67,
M71,
M92,
M93,
NGC104,
NGC129,
NGC1502,
NGC188,
NGC1980,
NGC2204,
NGC2232,
NGC2301,
NGC2360,
NGC2362,
NGC2423,
NGC2477,
NGC2509,
NGC2516,
NGC288,
NGC3114,
NGC3201,
NGC3293,
NGC3532,
NGC362,
NGC3766,
NGC4372,
NGC457,
NGC4590,
NGC4755,
NGC5139,
NGC5460,
NGC5662,
NGC5822,
NGC5904,
NGC6067,
NGC6087,
NGC6093,
NGC6200,
NGC6204,
NGC6231,
NGC6242,
NGC6281,
NGC6352,
NGC6362,
NGC6397,
NGC6541,
NGC6544,
NGC663,
NGC6649,
NGC6723,
NGC6752,
NGC6791,
NGC6834,
NGC6939,
NGC6940,
NGC7099,
NGC7142,
NGC7419,
NGC752,
NGC7789,
NGC869,
NGC884

## Reporting Bugs?

If you do notice a bug, make sure to post it in the issues section of the repository and use the bug report template provided. That way, we can quickly and effectively remove the bug from our programs.
