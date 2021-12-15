import matplotlib.pyplot as plt
import numpy as np
import sys
import mpl_scatter_density 
from matplotlib.colors import LinearSegmentedColormap

def doit(y,x,name,Npoints,N,patch=False,singlesubplot=False):
    mask = np.isnan(x)
    x = x[~mask]
    y = y[~mask]
    if not singlesubplot:
        ax = plt.subplot(4, 2, N)
        symbol = 'b.'
    else:
        ax = plt.subplot(1, 1, 1)
        symbol = 'bo'
    plt.plot(x,y,symbol,alpha=0.2,ms=2)
    plt.axhline(0.0,alpha=0.5)
    plt.ylim(-0.1,0.1)
    plt.xlabel(name)
    xe, ye, xm, ym, sigma = running_med(x,y,Npoints)
    sigma = np.array(sigma)*1
    plt.ylabel("zero point offset [mas]")

    if patch:
        import matplotlib.patches as patches
        rect2 = patches.Rectangle((1.0,-0.2), width=1.2, height=0.4, alpha=0.1, facecolor='blue',label='Cepheid zone')
        rect3 = patches.Rectangle((0.5,-0.2), width=0.5, height=0.4, alpha=0.3, facecolor='green',label='QSO zone')
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        #plt.legend()


    return
    
def running_med(x,y,N):
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    # compute number of bins needed
    Nbins = np.int(len(x)/N)
    # if not an exact number of N sized bins, add one more
    if np.abs(np.int(len(x)/N)-len(x)/N)>1e-10:
        Nbins += 1
    print("Nbins=",Nbins)
    xmean = []
    xedges = []
    ymedian = []
    sigma = []
    for i in range(Nbins):
        ilo = i*N; ihi = ilo + N
        if ihi>len(x):
            ihi = len(x)
        xmean.append(np.mean(x[ilo:ihi]))
        xedges.append(np.min(x[ilo:ihi]))
        ymedian.append(np.median(y[ilo:ihi]))
        #ymedian.append(np.mean(y[ilo:ihi]))
        sigma.append(np.std(y[ilo:ihi])/np.sqrt(len(y[ilo:ihi])))
    xedges.append(np.max(x[ilo:ihi]))
    yedges = ymedian.copy()
    yedges.append(yedges[-1])
    return xedges, yedges, xmean, ymedian, sigma

gmag, par, zpoff, psc, el, soltype, bp_rp, gl, gb = np.loadtxt("trgb.range.dat", usecols=(0,1,2,3,4,5,6,7,8), unpack=True, dtype=float)

#gmag, par, zpoff, psc, el, soltype, bp_rp, gl, gb = np.loadtxt("b.range.dat", usecols=(0,1,2,3,4,5,6,7,8), unpack=True, dtype=float)

gmask = gmag < 11

#gmask = np.logical_and(gmag>11,gmag<14)

gmag = gmag[gmask]
par = par[gmask]
zpoff = zpoff[gmask]
psc = psc[gmask]
el = el[gmask]
soltype = soltype[gmask]
bp_rp = bp_rp[gmask]
gl = gl[gmask]
gb = gb[gmask]

soltype += np.random.random(len(soltype))*10
el += np.random.random(len(el))*5.0

fig = plt.figure(figsize=(8,6))
                
print("median zp offset : ",np.median(zpoff)*1000.0," uas")

doit(zpoff,bp_rp,"BP-RP",150,1,patch=True,singlesubplot=True)

def draw_median(bp_rp_lo,bp_rp_hi):
    bp_rp_mask = np.logical_and(bp_rp > bp_rp_lo,bp_rp < bp_rp_hi)
    zpoff_median = np.median(zpoff[bp_rp_mask])
    plt.plot((bp_rp_lo,bp_rp_hi),(zpoff_median,zpoff_median),'k-')
    bp_rp_average = np.mean(bp_rp[bp_rp_mask])
    sigma = np.std(zpoff[bp_rp_mask])/np.sqrt(np.sum(bp_rp_mask))
    plt.errorbar(bp_rp_average, zpoff_median, yerr=sigma,fmt='k+',ms=0,elinewidth=2)

    print(bp_rp_average, zpoff_median*1000, sigma*1000)
    
# colour ranges over which to compute the medians    

# this is for TRGB

trgb = True

if trgb:

    #draw_median(-0.3,-0.1)
    draw_median(-0.1,0.1)
    draw_median(0.1,0.3)
    draw_median(0.3,0.5)
    draw_median(0.5,1.0)
    draw_median(1.0,2.2)
    draw_median(2.2,4.1)
    plt.axhline(0.015,c='g',alpha=0.5,label='Riess et al 2021 (Cepheids)')
    plt.title("TRGB magnitude range")

# this is for stars down to 14th mag (bright limit 11th mag)

else:
    
    #draw_median(-0.3,-0.1)
    draw_median(0.1,0.3)
    draw_median(0.3,0.5)
    draw_median(0.5,0.75)
    draw_median(0.75,1.0)
    draw_median(1.0,1.2)
    draw_median(1.2,1.4)
    draw_median(1.4,1.8)
    draw_median(1.8,2.0)
    draw_median(2.0,3.0)
    plt.title("14 > G > 11")

plt.ylabel("zero point offset [mas]")
#plt.tight_layout()

plt.legend()

plt.show()
