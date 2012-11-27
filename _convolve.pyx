# routine to convolve a model flux array with the
# wavelength-dependent, non-gaussian COS LSF.
"""
old compilation commands:

cython -a _convolve.pyx
gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -lm -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -o _convolve.so _convolve.c
gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -lm -I/loc/nca/nhmc/installed/epd/epd-5.1.0-rh3-x86_64/include/python2.5  -I/loc/nca/nhmc/installed/epd/epd-5.1.0-rh3-x86_64/lib/python2.5/site-packages/numpy/core/include -o _convolve.so _convolve.c
"""

from utils import readLSF
import numpy as np
cimport numpy as nc

cdef extern from "math.h":
    double exp(double)

cdef double gaussian(double x, double sigma):
    return exp(-0.5*(x / sigma)**2)

cimport cython

@cython.boundscheck(False)
def convolve_with_COS_FOS(
        nc.ndarray[nc.float64_t] a,
        nc.ndarray[nc.float64_t] wa,
        use_COS_nuv=False,
    ):
    """ convolve_with_COS_FOS(a, wa)

    convolve array `a` at wavelengths `wa` with either the COS LSF or
    a gaussian LSF for FOS, depending on the wavelength.

    a : array of floats with shape (N,)
        array to be convolved.
    wa : array of floats with shape (N,)
        wavelength at each point
    use_COS_nuv : False
        If True, use the COS NUV line spread function instead of
        convolving with FOS resolution.
    """
    cdef int i, m, n, imax, N, n_g130m, n_g160m, n_nuv
    cdef double result, wtot, wval, sigma, wup, dw
    cdef nc.ndarray[nc.float64_t] lsf, results=np.empty_like(a)

    ind = wa.searchsorted(1800)
    dw = np.median(np.diff(wa[:ind]))
    print np.diff(wa)[0], np.diff(wa)[-1]
    print 'COS FUV pixel width %.4f Ang' % dw

    profile,_ = readLSF('G130M', dw)
    keys = 'w1150 w1200 w1250 w1300 w1350 w1400 w1450'.split()
    g130m = [np.array(profile[k]) for k in keys]

    profile,_ = readLSF('G160M', dw)
    keys = 'w1450 w1500 w1550 w1600 w1650 w1700 w1750'.split()
    g160m = [np.array(profile[k]) for k in keys]

    g130uplim = [1175, 1225, 1275, 1325, 1375, 1425, 1450]
    g160uplim = [1475, 1525, 1575, 1625, 1675, 1725, 1800]

    n_g130m = len(g130m[0])
    n_g160m = len(g160m[0])    

    N = len(a)

    n = 0
    # G130M
    for k in range(len(g130m)):
        lsf = g130m[k]
        wup = g130uplim[k]
        while n < N and wa[n] < wup:
            result = 0
            wtot = 0
            for m in range(-(n_g130m//2), n_g130m//2 + 1):
                i = n + m
                if i < 0 or i >= N:  continue
                wval = lsf[m+n_g130m//2]
                result += wval*a[i]
                wtot += wval
            results[n] = result / wtot
            n += 1

    # G160M
    for k in range(len(g160m)):
        lsf = g160m[k]
        wup = g160uplim[k]
        while n < N and wa[n] < wup:
            result = 0
            wtot = 0
            for m in range(-(n_g160m//2), n_g160m//2 + 1):
                i = n + m
                if i < 0 or i >= N:  continue
                wval = lsf[m+n_g160m//2]
                result += wval*a[i]
                wtot += wval
            results[n] = result / wtot
            n += 1

    # G230L
    if use_COS_nuv and N - n > 5:
        dw = np.median(np.diff(wa[n:]))
        print 'new COS NUV pixel width %.4f Ang' % dw
        print '(from %.6f Ang)' % (wa[n])
        profile, _ = readLSF('NUV', dw)
        keys = 'w1700 w1800 w1900 w2000 w2100 w2200 w2300 w2400 w2500 w2600 \
        w2700 w2800 w2900 w3000 w3100 w3200'.split()
        nuv = [np.array(profile[k]) for k in keys]
        nuv_uplim = [1750.5, 1850.5, 1950.5, 2050.5, 2150.5, 2250.5, 2350.5,
                     2450.5, 2550.5, 2650.5, 2750.5, 2850.5, 2950.5, 3050.5,
                     3150.5, 3200.5]
        
        n_nuv = len(nuv[0])

        for k in range(len(nuv)):
            lsf = nuv[k]
            wup = nuv_uplim[k]
            while n < N and wa[n] < wup:
                result = 0
                wtot = 0
                for m in range(-(n_nuv // 2), n_nuv // 2 + 1):
                    i = n + m
                    if i < 0 or i >= N:
                        continue
                    wval = lsf[m + n_nuv // 2]
                    result += wval * a[i]
                    wtot += wval
                results[n] = result / wtot
                n += 1
    else:
        # use FOS resolution
        while n < N and wa[n] < 2300:        
            dw = wa[n] - wa[n-1]
            sigma = 1.39 / 2.35
            # determine how far away we need to go from the Gaussian
            # centre.
            imax = int(5*sigma / dw)
            result = 0
            wtot = 0
            for m in range(-imax, imax+1):
                i = n + m
                if i < 0 or i >= N:
                    continue
                wval = gaussian(-m*dw, sigma)
                result += wval*a[i]
                wtot += wval
     
            results[n] = result / wtot
            n += 1
     
        while n < N and wa[n] < 3300:        
            sigma = 1.97 / 2.35
            # determine how far away we need to go from the Gaussian
            # centre.
            imax = int(5*sigma / dw)
            result = 0
            wtot = 0
            for m in range(-imax, imax+1):
                i = n + m
                if i < 0 or i >= N:
                    continue
                wval = gaussian(-m*dw, sigma)
                result += wval*a[i]
                wtot += wval
     
            results[n] = result / wtot
            n += 1

    return results
