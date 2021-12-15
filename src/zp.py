import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as patches

def glob(cluster_name):

# list is from /home/cflynn/alan/gaiaedr3/globs.MVsort.txt
# all clusters with M_V<15.5 for horizontal branch

    globs = ["NGC6397","M4","NGC6752","NGC104","M22","M55","M71","NGC5139",
             "M12","M10","NGC3201","M13","NGC5904","M92","NGC7099","NGC6352",
             "NGC6544","NGC6362","NGC6541","NGC288","NGC362","NGC6723","NGC4372"]

    print("Nglobs = "+str(len(globs)))
    
    return cluster_name in globs


def plotpoints(x,y,cluster_name,cluster,argument,label=True):
    for i in range(len(cluster)):
        alpha = 1
        cmask = cluster_name==cluster[i]
        open_clusters = False
        globs = False
        if argument == "all":
            globs = True
            open_clusters = True
        if argument == "globs":
            globs = True
        if argument == "open":
            open_clusters = True
            
        if label==False:    
            if globs :
                if glob(cluster[i]) == True:
                    plt.plot(x[cmask],y[cmask],'.',ms=2,alpha=alpha,zorder=0)
            if open_clusters:
                if glob(cluster[i]) == False:
                    plt.plot(x[cmask],y[cmask],'.',ms=2,alpha=alpha,zorder=0) 
        if label==True:                   
            if globs :
                if glob(cluster[i]) == True:
                    plt.plot(x[cmask],y[cmask],'.',ms=2,alpha=alpha,label=str(cluster[i]).split(" ")[0],zorder=0)
            if open_clusters:
                if glob(cluster[i]) == False:
                    plt.plot(x[cmask],y[cmask],'.',ms=2,alpha=alpha,label=str(cluster[i]).split(" ")[0],zorder=0)
    return

############################### main #################################

nargs = len(sys.argv) - 1
if (nargs==1):
    argument = sys.argv[1]    
else:
    print("Useage : ")
    print("python zp.py all/glob/open")
    print("eg python zp.py all")
    sys.exit()
    
filename = "../zeropoint.txt"
#filename = "zp.finalsample.txt"
#filename = "allstars.2trimmedclusters.dat"
filename="/home/cflynn/gaiaedr3/Gaia-EDR3-Stellar-Clusters/src/allstars.2trimmedclusters.dat"

# 0    1     2        3    4     5   6   7       8  9  10     11
# gmag bp-rp zpoffset name nueff psc ecl soltype gl gb zpvals parallax
gmag, bp_rp, zp, zp0, nueff, psc, ecl, soltype, gl, gb, par = np.loadtxt(filename, usecols=(0,1,2,10,4,5,6,7,8,9,11), unpack=True, dtype=float)

# TRGB test
#trgb_mask = np.logical_and(gmag < 13, gmag > 9)

# Zinn test
#trgb_mask = gmag < 11

# Riess et al test
trgb_mask = np.logical_and(gmag < 11.0, gmag > 6)

trgb_gmag = gmag[trgb_mask]
trgb_zpoff = zp[trgb_mask]
trgb_bp_rp = bp_rp[trgb_mask]
trgb_par = par[trgb_mask]
trgb_psc = psc[trgb_mask]
trgb_ecl = ecl[trgb_mask]
trgb_gl = gl[trgb_mask]
trgb_gb = gb[trgb_mask]
trgb_soltype = soltype[trgb_mask]
f = open("trgb.range.dat","w")
for i in range(len(trgb_gmag)):
    f.write(str(trgb_gmag[i])+" "+str(trgb_par[i])+" "+str(trgb_zpoff[i])+" "+str(trgb_psc[i])+" "+str(trgb_ecl[i])+" "+str(trgb_soltype[i])+" ")
    f.write(str(trgb_bp_rp[i])+" "+str(trgb_gl[i])+" "+str(trgb_gb[i])+"\n")
f.close()

# All bright stars mask
b_mask = np.logical_and(gmag < 14.0, gmag > 11)

b_gmag = gmag[b_mask]
b_zpoff = zp[b_mask]
b_bp_rp = bp_rp[b_mask]
b_par = par[b_mask]
b_psc = psc[b_mask]
b_ecl = ecl[b_mask]
b_gl = gl[b_mask]
b_gb = gb[b_mask]
b_soltype = soltype[b_mask]
f = open("b.range.dat","w")
for i in range(len(b_gmag)):
    f.write(str(b_gmag[i])+" "+str(b_par[i])+" "+str(b_zpoff[i])+" "+str(b_psc[i])+" "+str(b_ecl[i])+" "+str(b_soltype[i])+" ")
    f.write(str(b_bp_rp[i])+" "+str(b_gl[i])+" "+str(b_gb[i])+"\n")
f.close()



cluster_name = np.loadtxt(filename, usecols=(3), unpack=True, dtype=str)
cluster_list = list(set(cluster_name))

for cluster in cluster_list:
    print(cluster)

gmask = gmag < 14
print("Median parallax offset for G<14 ",np.median(zp[gmask]))

gmag_max = 18.2
gmag_min = 5

plt.figure(figsize=(20,10))

ax = plt.subplot(111)
plotpoints(bp_rp,gmag,cluster_name,cluster_list,argument)
plt.axhline(14.0,alpha=0.5)


# indicate positions of TRGB, Cepheids and QSOs

#                          colour start, magstart, colour width, mag height
#rect1 = patches.Rectangle((0.5,9), width=2.0, height=2, alpha=0.1, facecolor='black',label='TRGB zone')
#rect2 = patches.Rectangle((1.0,6), width=1.0, height=5, alpha=0.1, facecolor='blue',label='Cepheid zone')
#rect3 = patches.Rectangle((0.3,16), width=0.7, height=2, alpha=0.3, facecolor='green',label='QSO zone')

rect3 = patches.Rectangle((0.3,16), width=0.7, height=2, alpha=0.3,
                          edgecolor='green',facecolor='white',linewidth='10',label='QSO zone')

rect1 = patches.Rectangle((1.4,9), width=1.1, height=4, alpha=0.2,
                          edgecolor='black',facecolor='white',label='TRGB zone',linewidth='10')

rect2 = patches.Rectangle((1.0,6), width=1.2, height=5.5, alpha=0.2,
                          edgecolor='blue',facecolor='white',linewidth='10',label='Cepheid zone (Riess et al 2021)\n offset : $15\pm7$ uas')

rect4 = patches.Rectangle((1.1,6+0.2), width=0.5, height=(10.8-6-0.6), alpha=0.2,
                          edgecolor='red',facecolor='white',linewidth='10',label='Zinn et al 2021 astroseismology\n offset : $15\pm3$ uas')

rect5 = patches.Rectangle((1.15,6+0.4), width=1.0-0.05, height=(10.8-6-1), alpha=0.2,
                          edgecolor='yellow',facecolor='white',linewidth='10',label='Huang et al 2021 bright RC stars\n offset : $10\pm1$ uas')

#rect6 = patches.Rectangle((1.15,14), width=1.0-0.05, height=4, alpha=0.2,
#                          edgecolor='yellow',facecolor='white',linewidth='10',label='Huang et al 2021 faint RC stars\n offset : $10\pm1$ uas')

#rect7 = patches.Rectangle((0.69,14), width=(2.37-0.69), height=4, alpha=0.2,
#                          edgecolor='brown',facecolor='white',linewidth='10',label='Ren et al 2021 Contact binaries')


# draw the rectangles
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)

#ax.add_patch(rect6)
#ax.add_patch(rect7)

plt.ylim(gmag_max,gmag_min)
plt.xlabel('BP-RP')
plt.ylabel('Gmag')
plt.legend(ncol=4)

plt.tight_layout()

plt.show()

mask = gmag<20
fmask = gmag<12

plt.figure(figsize=(20,10))

parlim = 0.10

ax = plt.subplot(511)
plotpoints(gmag[mask],zp[mask]+zp0[mask],cluster_name[mask],cluster_list,argument)
plt.axvline(14.0,c='k',alpha=0.5)
plt.axhline(0.0,c='b',alpha=0.5)
plt.ylabel('zp offset [mas]')
#plt.xlabel('Gmag')
plt.xlim(6.0,18.0)
plt.ylim(-parlim,parlim)
plt.text(0.01, 0.1, '(a)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

ax = plt.subplot(513)
plotpoints(gmag[mask],zp[mask],cluster_name[mask],cluster_list,argument,label=False)
plt.axvline(14.0,c='k',alpha=0.5)
plt.axhline(0.0,c='b',alpha=0.5)
plt.ylabel('zp offset [mas]')
plt.xlabel('Gmag')
plt.xlim(6.0,18.0)
plt.ylim(-parlim,parlim)
riess_corr_x = np.array((6.0,10.0,18.0))
riess_corr_y = np.array((0.015,0.015,0.0)) 
plt.step(riess_corr_x,riess_corr_y,c='g',alpha=1,label="Cepheids -15 uas correction")
plt.text(0.01, 0.1, '(c)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.legend()

ax = plt.subplot(512)
plotpoints(gmag[mask],zp0[mask],cluster_name[mask],cluster_list,argument)
plt.axhline(0.0,c='b',alpha=0.5)
plt.axvline(14.0,c='k',alpha=0.5)
plt.ylabel('zp0 correction [mas]')
#plt.xlabel('Gmag')
plt.text(0.01, 0.1, '(b)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
plt.xlim(6.0,18.0)
plt.ylim(-parlim,parlim)

corr_mask = gmag < 10
print("median correction for gmag<10 : ",np.median(zp0[corr_mask]))

gmag_orig = np.copy(gmag)
zp_orig = np.copy(zp)
zp0_orig = np.copy(zp0)
bp_rp_orig = np.copy(bp_rp)

gmag = gmag_orig[mask]
zp = zp_orig[mask]
zp0 = zp0_orig[mask]
bp_rp = bp_rp_orig[mask]
medians = []
gbin = []
mscatt = []
glo = 6.00
dg = 0.50

colour_cut = 100.0
    
cmask = bp_rp<colour_cut
gmag = gmag[cmask]
zp = zp[cmask]
zp0 = zp0[cmask]
bp_rp = bp_rp[cmask]
for i in range(24):
    ax1 = plt.subplot(6,12,i+1+48)
    gmask = np.logical_and(gmag>glo,gmag<glo+dg)
    bins = np.arange(-0.20,0.20,0.02)
    Nzp = np.sum(gmask)
    medians.append(np.median(zp[gmask]))
    gbin.append(glo+dg/2.0)
    mscatt.append(np.std(zp[gmask])/np.sqrt(Nzp))
    print(glo,glo+dg,Nzp,np.std(zp[gmask]),mscatt[-1])
    plt.hist(zp[gmask],bins=bins,alpha=0.5)
    plt.axvline(0.0,c='k',alpha=0.5)
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    plt.xlim(-0.2,0.2)
    plt.title(str(glo)+"<G<"+str(glo+dg))
    glo += dg 

plt.subplot(513)
plt.step(np.array(gbin)+dg/2,medians,'k-',lw=1)
mscatt = np.array(mscatt)*2
plt.errorbar(gbin,medians,yerr=mscatt,fmt='k+',ms=0,elinewidth=2)

medians = []
gbin = []
mscatt = []
cmask = bp_rp<colour_cut
gmag = gmag[cmask]
glo = 6.00
bp_rp = bp_rp[cmask]
for i in range(24):
    gmask = np.logical_and(gmag>glo,gmag<glo+dg)
    Nzp = np.sum(gmask)
    raw_zp = zp[gmask]+zp0[gmask]
    medians.append(np.median(raw_zp))
    gbin.append(glo+dg/2.0)
    mscatt.append(np.std(zp[gmask])/np.sqrt(Nzp))
    print(glo,glo+dg,Nzp,np.std(zp[gmask]),mscatt[-1])
    glo += dg 

plt.subplot(511)
plt.step(np.array(gbin)+dg/2,medians,'k-',lw=1)
mscatt = np.array(mscatt)*2
plt.errorbar(gbin,medians,yerr=mscatt,fmt='k+',ms=0,elinewidth=2)

plt.show()

