""" Routines useful for dealing with COS spectra.
"""

from barak.interp import interp_spline
from barak.spec import find_wa_edges
from barak.io import readtxt, writetxt
from barak.utilities import between

import pyfits
import numpy as np

import os

datapath = os.path.split(os.path.abspath(__file__))[0] + '/'

# cache the LSF values
cacheLSF = {}

__all__ = ['readx1d', 'rebinx1d', 'readLSF', 'writeLSF_vpfit']


def readx1d(filename):
    """ Read an x1d format spectrum from calcos.

    For the output spectra:

    wa = wavelengths (Angstroms)
    fl = flux
    er = 1 sigma error in flux
    dq = data quality, > 0 means bad for some reason
    dq_wgt = data quality weight, 1 for good flags, 0 for bad
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

    keys = 'WAVELENGTH FLUX ERROR DQ DQ_WGT GROSS NET BACKGROUND'.split()
    names = 'wa,fl,er,dq,dq_wgt,gr,net,bg'
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


def rebinx1d(sp, wa_new):
    """ Rebins x1d spectrum from CalCOS pipeline to a new linear
    wavelength scale.

    Accepts x1d spectrum as numpy rec array in the same format as
    that from the readx1d output.

    Returns the rebinned spectrum.

    Will probably get the flux and errors for the first and last
    pixel of the rebinned spectrum wrong.

    """

    # Find pixel edges - used when rebinning:
    edges_old = find_wa_edges(sp.wa)
    edges_new = find_wa_edges(wa_new)
    widths_old = edges_old[1:] - edges_old[:-1]
    npts_old = len(sp.wa)
    npts_new = len(wa_new)
    fl_new = np.zeros(npts_new)
    er_new = np.empty(npts_new) * np.nan
    dq_new = np.empty(npts_new) * np.nan
    dq_wgt_new = np.empty(npts_new) * np.nan
    gr_new = np.zeros(npts_new)
    net_new = np.zeros(npts_new)
    bg_new = np.zeros(npts_new)
    df = 0.0
    de2 = 0.0
    ddq = 0.0
    ddqw = 0.0
    dg = 0.0
    dn = 0.0
    db = 0.0
    npix = 0    # Number of old pixels contributing to rebinned pixel
    j = 0       # Index of rebinned array
    i = 0       # Index of old array

    # Sanity check:
    if edges_old[-1] < edges_new[0] or edges_new[-1] < edges_old[0]:
        raise ValueError('Wavelength scales do not overlap!')

    # Find the first contributing old pixel to the rebinned spectrum:
    if edges_old[i + 1] < edges_new[0]:
        # Old wavelength scale extends lower than the rebinned scale.
        # Find the first old pixel that overlaps with the rebinned scale.
        while edges_old[i + 1] < edges_new[0]:
            i += 1
        i -= 1
    elif edges_old[0] > edges_new[j + 1]:
        # New wavelength scale extends lower than the old scale.
        # Find the first rebinned pixel that overlaps with the old spectrum.
        while edges_old[0] > edges_new[j + 1]:
            fl_new[j] = 0.0
            er_new[j] = 0.0
            dq_new[j] = 128
            dq_wgt_new[j] = 0.0
            gr_new[j] = 0.0
            net_new[j] = 0.0
            bg_new[j] = 0.0
            j += 1
        j -= 1
    lo_old = edges_old[i]          # Lower edge of contr. pixel in old scale.

    while True:
        hi_old = edges_old[i + 1]  # Upper edge of contr. pixel in old scale.
        hi_new = edges_new[j + 1]  # Upper edge of jth pixel in rebinned scale.

        if hi_old < hi_new:
            if sp.er[i] > 0:
                dpix = (hi_old - lo_old) / widths_old[i]
                df += sp.fl[i] * dpix
                de2 += sp.er[i] ** 2 * dpix
                ddq += sp.dq[i] * dpix
                ddqw += sp.dq_wgt[i] * dpix
                dg += sp.gr[i] * dpix
                dn += sp.net[i] * dpix
                db += sp.bg[i] * dpix
                npix += dpix
            lo_old = hi_old
            i += 1
            if i == npts_old:
                break
        else:
            if sp.er[i] > 0:
                dpix = (hi_new - lo_old) / widths_old[i]
                df += sp.fl[i] * dpix
                de2 += sp.er[i] ** 2 * dpix
                ddq += sp.dq[i] * dpix
                ddqw += sp.dq_wgt[i] * dpix
                dg += sp.gr[i] * dpix
                dn += sp.net[i] * dpix
                db += sp.bg[i] * dpix
                npix += dpix

            if npix > 0:
                # Find total flux and error, then divide by number of pixels.
                # Ensures flux density is conserved.
                fl_new[j] = df / npix
                er_new[j] = np.sqrt(de2) / npix
                dq_new[j] = ddq / npix
                dq_wgt_new[j] = ddqw / npix
                gr_new[j] = dg / npix
                net_new[j] = dn / npix
                bg_new[j] = db / npix
            else:
                fl_new[j] = 0.0
                er_new[j] = 0.0
                dq_new[j] = 128
                dq_wgt_new[j] = 0.0
                gr_new[j] = 0.0
                net_new[j] = 0.0
                bg_new[j] = 0.0
            df = 0.0
            de2 = 0.0
            ddq = 0.0
            ddqw = 0.0
            dg = 0.0
            dn = 0.0
            db = 0.0
            npix = 0.0
            lo_old = hi_new
            j += 1
            if j == npts_new:
                break

    sp = np.rec.fromarrays([wa_new, fl_new, er_new, dq_new, dq_wgt_new, \
        gr_new, net_new, bg_new], names='wa, fl, er, dq, dq_wgt, gr, net, bg')
    return sp


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

    t = np.arange(len(wa1) // 2 + 1)
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
    elif between(wa, 1450, 1800):
        lsf, _ = readLSF('G160M', dw)
    else:
        lsf, _ = readLSF('NUV', dw)

    wavs = [float(n[1:]) for n in lsf.dtype.names[1:]]
    lsf1 = np.array([lsf[n] for n in lsf.dtype.names[1:]])

    newLSF = []
    for ipix in range(lsf1.shape[1]):
        newLSF.append(np.interp(wa, wavs, lsf1[:, ipix]))

    writetxt(outname, [lsf.relpix, newLSF], overwrite=1)
