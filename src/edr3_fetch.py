import numpy as np
import sys
import csv
import pyvo as vo
import matplotlib.pyplot as plt
import pandas as pd

def draw_pm_circle(pmra0,pmdec0,rad,colour):
    t = np.linspace(0,2*np.pi,100)
    x = pmra0 + rad*np.sin(t)
    y = pmdec0 + rad*np.cos(t)
    plt.plot(x,y,colour)
    return

def draw_sky_circle(ra0,dec0,rad,colour):
    t = np.linspace(0,2*np.pi,100)
    x = ra0 + rad*np.sin(t)/np.cos(dec0*np.pi/180.0)
    y = dec0 + rad*np.cos(t)
    plt.plot(x,y,colour)
    return

nargs = len(sys.argv) - 1
if nargs == 3:
    cluster = sys.argv[1]
    srad = np.float(sys.argv[2])
    gmag_lim = sys.argv[3]
else:
    print("Example use: give cluster name, search radius in arcmin and Gmag limit")
    print("python edr3_fetch.py M67 30 18")
    sys.exit()

print('Connecting to SIMBAD TAP')
service = vo.dal.TAPService("https://simbad.u-strasbg.fr/simbad/sim-tap")
command = "SELECT RA, DEC, PMRA, PMDEC FROM basic JOIN ident ON oidref = oid WHERE id = '"+str(cluster)+"'"
print(command)
print('Sending request')
data = service.search(command)

try:
    ra0 = np.float(data['ra'])
    dec0 = np.float(data['dec'])
    pmra0 = np.float(data['pmra'])
    pmdec0 = np.float(data['pmdec'])
except:
    print("Error getting RA and DEC of target from SIMBAD")
    sys.exit()

print("======================")
print("Data from Simbad")
print("Coords : ",ra0,dec0)
print("PMs : ",pmra0,pmdec0)
print("======================")

if np.isnan(pmra0):
    pmra0 = 0.0

if np.isnan(pmdec0):
    pmdec0 = 0.0

# generate square region to search around target RA and DEC    
ra_lo = (ra0 - srad/60.0/np.cos(dec0*np.pi/180.0))
ra_hi = (ra0 + srad/60.0/np.cos(dec0*np.pi/180.0))
dec_lo = dec0 - srad/60.0
dec_hi = dec0 + srad/60.0

ra_lo = np.around(ra_lo,2)
ra_hi = np.around(ra_hi,2)
dec_lo = np.around(dec_lo,2)
dec_hi = np.around(dec_hi,2)

# generate command to send to GAIA
command = 'SELECT * FROM gaiaedr3.gaia_source\n'
command += 'where ra>'+str(ra_lo)+' and ra<'+str(ra_hi)+' and '
command += 'dec>'+str(dec_lo)+' and dec<'+str(dec_hi)+'\nand '
command += 'phot_g_mean_mag < '+gmag_lim

# special cases for some clusters due to large background/foreground rates
if cluster == 'M21':
    command += ' and parallax > 0.5'
if cluster == 'M23':
    command += ' and parallax > 1.0'
if cluster == 'M45':
    command += ' and parallax > 6.0'
if cluster == 'M25':
    command += ' and parallax > 1.0'
if cluster == 'M48':
    command += ' and parallax > 0.7'
if cluster == 'NGC2423':
    command += ' and parallax > 0.9 and parallax < 1.3'
if cluster == 'NGC2516':
    command += ' and parallax > 1.0'
if cluster == 'NGC3114':
    command += ' and parallax > 0.7'
if cluster == 'NGC3532':
    command += ' and parallax > 1.0'
if cluster == 'NGC5460':
    command += ' and parallax > 1.0'
if cluster == 'NGC5822':
    command += ' and parallax > 0.7'
if cluster == 'NGC6025':
    command += ' and parallax > 0.7'
if cluster == 'NGC6087':
    command += ' and parallax > 0.7'
if cluster == 'NGC6281':
    command += ' and parallax > 0.7'
if cluster == 'NGC6633':
    command += ' and parallax > 1.8'
if cluster == 'NGC7142':
    command += ' and parallax > 0.20'
if cluster == 'NGC752':
    command += ' and parallax > 1.50'
if cluster == 'IC4665':
    command += ' and parallax > 2.2'
if cluster == 'IC5146':
    command += ' and parallax > 1.25'
if cluster == "NGC2627":
    command += " and parallax > 0.3"
if cluster == "NGC3960":
    command += " and parallax > 0.3"
if cluster == "IC 4651":
    command += " and parallax > 0.75"
if cluster == "IC 4756":
    command += " and parallax > 1.5"
if cluster == "NGC225":
    command += " and parallax > 1"
if cluster == "NGC3680":
    command += " and parallax > 0.7"
if cluster == "NGC7654":
    command += " and parallax > 0.3"
# print the command
print()
print(command)
print()

# set up the TAP address
service = vo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")

# make the data request
data = service.search(command)

# convert the dataset to a pandas frame
frame = pd.DataFrame(data)

# write the data set out as a CSV
cluster = cluster.replace(' ', '_') 
print("Writing data set to disk as "+cluster+".csv")
frame.to_csv("../cluster_data/"+cluster+".csv")

# extract relevant items from the data set
source_id = data['source_id']
ra = data['ra']
dec = data['dec']
pmra = data['pmra']
pmdec = data['pmdec']
parallax = data['parallax']
parallax_error = data['parallax_error']
gmag = data['phot_g_mean_mag']
bp_rp = data['bp_rp']
nueffused = data['nu_eff_used_in_astrometry']
psc = data['pseudocolour']
ecl_lat = data['ecl_lat']
soltype = data['astrometric_params_solved']
#gl = data['l']
#gb = data['b']

alpha_g = 0.8
alpha_b = 0.2

marker_b = 'b.'
marker_g = 'g.'

plt.figure(figsize=(20,10))

pm_rmax = 1.0
pos_rmax = srad/60.0

try:
    pmra0
    cluster_pms = True
    pm_rad = np.sqrt((pmra-pmra0)**2+(pmdec-pmdec0)**2)
    pm_mask = pm_rad < pm_rmax
except:
    pm_mask = ra>-1
    cluster_pms = False
    
try:
    ra0
    cluster_position = True
    pos_rad = np.sqrt(((ra-ra0)*np.cos(dec0*np.pi/180.0))**2+(dec-dec0)**2)
    pos_mask = pos_rad < pos_rmax 
    mask = np.logical_and(pm_mask, pos_mask)
except:
    mask = pm_mask
    cluster_position = False
    
# plot proper motions
plt.subplot(231)
plt.plot(pmra[~mask],pmdec[~mask],marker_b,alpha=alpha_b)
plt.plot(pmra[mask],pmdec[mask],marker_g,alpha=alpha_g) #,markeredgecolor='k')
if cluster_pms:
    draw_pm_circle(pmra0,pmdec0,pm_rmax,'r')
    xs = pm_rmax*10; ys = pm_rmax*10  # convert to mas/yr
    plt.xlim(pmra0-xs,pmra0+xs)
    plt.ylim(pmdec0-ys,pmdec0+ys)
plt.xlabel("PMRA [mas/yr]")
plt.ylabel("PMDEC [mas/yr]")

# plot parallaxes versus G magnitude
plt.subplot(133)
plt.plot(parallax[~mask],gmag[~mask],marker_b,alpha=alpha_b)
plt.plot(parallax[mask],gmag[mask],marker_g,alpha=alpha_g)
plt.xlabel("parallax [mas]")
plt.ylabel("Gmag")
plt.ylim(19.5,5)

# plot CMD
plt.subplot(132)
plt.title(cluster)
plt.plot(bp_rp[~mask],gmag[~mask],marker_b,alpha=alpha_b)
plt.plot(bp_rp[mask],gmag[mask],marker_g,alpha=alpha_g)
plt.xlabel("BP-RP")
plt.ylabel("Gmag")
plt.ylim(19.5,5)

# sky positions
plt.subplot(234)
plt.plot(ra[~mask],dec[~mask],marker_b,alpha=alpha_b)
plt.plot(ra[mask],dec[mask],marker_g,alpha=alpha_g)
if cluster_position:
    draw_sky_circle(ra0,dec0,pos_rmax,'r')
plt.xlabel("RA [deg]")
plt.ylabel("DEC [deg]")

plt.show()

sys.exit()


