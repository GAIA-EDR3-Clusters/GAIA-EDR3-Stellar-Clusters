import matplotlib.pyplot as plt
import numpy as np
import sys
import mpl_scatter_density 
from matplotlib.colors import LinearSegmentedColormap

def doit(y,x,name,Npoints,N,patch=False):
    mask = np.isnan(x)
    x = x[~mask]
    y = y[~mask]
    ax = plt.subplot(4, 2, N)
    plt.plot(x,y,'b.',alpha=0.2,ms=2)
    plt.axhline(0.0,alpha=0.5)
    plt.ylim(-0.1,0.1)
    plt.xlabel(name)
    xe, ye, xm, ym, sigma = running_med(x,y,Npoints)
    for i in range(len(xe)):
        plt.axvline(xe[i],color='k',alpha=0.2)
    plt.step(xe,ye,'k-',lw=2,where='post')
    sigma = np.array(sigma)*1
    plt.errorbar(xm, ym, yerr=sigma,fmt='k+',ms=0,elinewidth=2)
    plt.ylabel("zero point offset")
    plt.axhline(0.015,c='g',alpha=0.5,label='Riess et al 2021')

    if patch:
        import matplotlib.patches as patches
        rect2 = patches.Rectangle((1.0,-0.2), width=1.2, height=0.4, alpha=0.1, facecolor='blue',label='Cepheid zone')
        rect3 = patches.Rectangle((0.5,-0.2), width=0.5, height=0.4, alpha=0.3, facecolor='green',label='QSO zone')
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        plt.legend()


    panel = chr(N+96)
    plt.text(0.03, 0.90, '('+panel+')', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
        
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

#filename = "zp.finalsample.txt"
#gmag, par, zpoff, psc, el, soltype, bp_rp, gl, gb = np.loadtxt(filename, usecols=(0,1,2,3,4,5,6,7,8), unpack=True, dtype=float)


gmask = gmag < 11

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

fig = plt.figure(figsize=(16,8))
                
doit(zpoff,gmag,"Gmag",100,1)    

doit(zpoff,par,"parallax [mas]",100,2)

doit(zpoff,psc,"pseudo-colour",30,3)

doit(zpoff,el,"ecliptic latitude [deg]",100,4)

mask = soltype<50
doit(zpoff,soltype,"soltype",np.int(sum(mask)),5)
plt.xlim(20,120)

doit(zpoff,bp_rp,"BP_RP",200,6,patch=True)

gl += np.random.random(len(gl))*2
doit(zpoff,gl,"Gl [deg]",200,7)

gb += np.random.random(len(gb))*2
doit(zpoff,gb,"Gb [deg]",200,8)

plt.tight_layout()

plt.show()

print("median zp offset : ",np.median(zpoff)*1000.0," uas")
