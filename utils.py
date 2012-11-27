""" Routines useful for dealing with cos spectra.
"""

from barak.interp import interp_spline
from barak.io import readtxt, writetxt

import pyfits
import numpy as np

import os

datapath = os.path.split(os.path.abspath(__file__))[0] + '/'

# cache the LSF values
cacheLSF = {}

__all__ = ['readx1d', 'readLSF', 'writeLSF_vpfit']

def readx1d(filename):
    """ Read an x1d format spectrum from calcos.

    For the output spectra:
    
    wa = wavelengths (Angstroms)
    fl = flux
    er = 1 sigma error in flux
    dq = data quality, > 0 means bad for some reason
    gr = gross count rate, gross counts / (exposure time in s)
    bg = background count rate
    net = (gr - bg) / eps, where eps is the flat fielding.
    """
    vnum = int(filename.split('/')[-1][4:6])
    fh = pyfits.open(filename)
    hd = fh[0].header
    optelem = hd['OPT_ELEM']
    cwa = hd['CENTRWV']
    exptime = fh[1].header['EXPTIME']

    keys = 'WAVELENGTH FLUX ERROR DQ GROSS NET BACKGROUND'.split()
    names = 'wa,fl,er,dq,gr,net,bg'
    r = fh['SCI'].data
    fh.close()
    vals = [r[k][0] for k in keys]
    isort = vals[0].argsort()
    for i in range(1, len(vals)):
        vals[i] = vals[i][isort]
    sp1 = np.rec.fromarrays(vals, names=names)

    vals = [r[k][1] for k in keys]
    isort = vals[0].argsort()
    for i in range(1, len(vals)):
        vals[i] = vals[i][isort]
    sp2 = np.rec.fromarrays(vals, names=names)
    if sp1.wa[0] < sp2.wa[0]:
        info1 = vnum, optelem, cwa, 'FUVB', exptime
        info2 = vnum, optelem, cwa, 'FUVA', exptime
    else:
        info2 = vnum, optelem, cwa, 'FUVB', exptime
        info1 = vnum, optelem, cwa, 'FUVA', exptime
        
    return [sp1, sp2], [info1, info2]


def readLSF(grating, dw_new=None):
    """ Read the COS line spread function, optionally interpolated to
    a new pixel width.

    Parameters
    ----------
    grating : str, {'G130M', 'G160M', 'NUV'}
      Either 'NUV' for all near UV gratings, or the name of the far UV
      COS grating.
    dw_new : float
      The new pixel width in Angstroms.  Default is `None`, which
      returns the original LSF.

    Returns
    -------
    LSF, wa : ndarrays of shape (N,)
      The new LSF and the wavelength offsets from the line centre.
    """

    # see if we've already calculated it
    try:
        #print grating, dw, 'cached!'
        return cacheLSF[grating, dw_new]
    except KeyError:
        pass
    
    oldLSF = readtxt(datapath + '/LSF/%s.txt' % grating, readnames=1)

    dw_orig = dict(G130M=0.00997, G160M=0.01223, NUV=0.390)

    wa0 = oldLSF.relpix * dw_orig[grating]
    if dw_new is None:
        return oldLSF, wa0

    t = np.arange(0, wa0[-1], dw_new)
    wa1 = np.concatenate([-t[::-1][:-1], t])

    wavs = oldLSF.dtype.names[1:]
    newLSF = []
    for w in wavs:
        newLSF.append(interp_spline(wa1, wa0, oldLSF[w]))
        
    t = np.arange(len(wa1)//2 + 1)
    newpix = np.concatenate([-t[::-1][:-1], t])
    #import pdb; pdb.set_trace()
    newLSF = np.rec.fromarrays([newpix] + newLSF, names=oldLSF.dtype.names)

    cacheLSF[grating, dw_new] = newLSF, wa1
    return newLSF, wa1

def writeLSF_vpfit(wa, dw):
    """ Write a file giving the COS line spread function at wavelength
    wa for constant pixel width dw (both in Angstroms), suitable for
    input to VPFIT.
    """ 
    outname = 'LSF/LSF_%.1f.txt' % wa
    if wa < 1450:
        lsf, _ = readLSF('G130M', dw)
    else:
        lsf, _ = readLSF('G160M', dw)
 
    wavs = [float(n[1:]) for n in lsf.dtype.names[1:]]
    lsf1 = np.array([lsf[n] for n in lsf.dtype.names[1:]])
 
    newLSF = []
    for ipix in range(lsf1.shape[1]):
        newLSF.append(np.interp(wa, wavs, lsf1[:, ipix]))

    writetxt(outname, [lsf.relpix, newLSF], overwrite=1)
