
import math
import numpy as np
from scipy import signal
from skimage import color as col
from scipy import ndimage as ndi


def G_carrier_real(t, freq, phase):
    '''
    Real part of the carrier.
    '''
    topass = 2 * math.pi * freq * t + phase
    out = np.cos(topass)
    return out

def G_carrier_imag(t, freq, phase):
    '''
    Imaginary part of the carrier.
    '''
    topass = 2 * math.pi * freq * t + phase
    out = np.sin(topass)
    return out


def G_envelope(t, amp, sdev):
    '''
    The impact of the filter is controlled by a Gaussian function.
    '''
    out = amp * np.exp( (-(t/sdev)**2) )
    return out


def G_fil_real(t, paras):
    '''
    Custom-built filter response (real part).
    Assumes that t is an array of temporal inputs.
    '''
    carrier = G_carrier_real(t=t, freq=paras["freq"], phase=paras["phase"])
    envelope = G_envelope(t=t, amp=paras["amp"], sdev=paras["sdev"])
    out = carrier * envelope
    return out

def G_fil_imag(t, paras):
    '''
    Custom-built filter response (imaginary part).
    Assumes that t is an array of temporal inputs.
    '''
    carrier = G_carrier_imag(t=t, freq=paras["freq"], phase=paras["phase"])
    envelope = G_envelope(t=t, amp=paras["amp"], sdev=paras["sdev"])
    out = carrier * envelope
    return out


def my_signal(t):
    
    highfreq = 0.25 * np.sin((2*math.pi*para_HIGHFREQ*t))
    lowfreq = 2 * np.sin((2*math.pi*para_LOWFREQ*t))
    
    cond = (np.abs(t) <= 5)
    signal = highfreq + lowfreq
    out = np.select([cond], [signal])
    
    return out


## Frequency response of the Gabor filter is obtained analytically as the (complex) Fourier transform.
def G_ft_real(f, amp, sdev, freq, phase):
    '''
    Real part of the complex Gabor filter's frequency response.
    '''
    topass = (f - freq) * sdev
    env = G_envelope(t=topass, amp=1, sdev=1)
    out = math.cos(phase) * amp * sdev * env
    return out

def G_ft_imag(f, amp, sdev, freq, phase):
    '''
    Imaginary part of the complex Gabor filter's frequency response.
    '''
    topass = (f - freq) * sdev
    env = G_envelope(t=topass, amp=1, sdev=1)
    out = math.sin(phase) * amp * sdev * env
    return out


def G_fr_real(f, paras):
    '''
    Frequency response for our custom-built filter (real part).
    Assumes that f is an array of frequency settings.
    '''
    out = G_ft_real(f=f, amp=paras["amp"], sdev=paras["sdev"], freq=paras["freq"], phase=paras["phase"])
    return out

def G_fr_imag(f, paras):
    '''
    Frequency response for our custom-built filter (real part).
    Assumes that f is an array of frequency settings.
    '''
    out = G_ft_imag(f=f, amp=paras["amp"], sdev=paras["sdev"], freq=paras["freq"], phase=paras["phase"])
    return out


def G2_carrier_real(x, y, freqx, freqy, phase):
    '''
    Real part of the 2-D Gabor carrier.
    '''
    topass = 2 * math.pi * (freqx*x + freqy*y) + phase
    out = np.cos(topass)
    return out


def G2_carrier_imag(x, y, freqx, freqy, phase):
    '''
    Imaginary part of the 2-D Gabor carrier.
    '''
    topass = 2 * math.pi * (freqx*x + freqy*y) + phase
    out = np.sin(topass)
    return out


def G2_envelope(x, y, amp, sdev):
    '''
    Gaussian envelope for a 2-D Gabor filter.
    We assume that it is circular (same decrease in x/y directions).
    '''
    out = amp * np.exp(-(x**2+y**2)/(sdev**2))
    return out



def G2_fil_real(x, y, paras):
    '''
    Custom-built filter response (real part).
    '''
    # Spatial frequency in polar coordinates.
    u = paras["freqs"] * math.cos(paras["dir"])
    v = paras["freqs"] * math.sin(paras["dir"])
    # Computations.
    carrier = G2_carrier_real(x=x, y=y, freqx=u, freqy=v, phase=paras["phase"])
    envelope = G2_envelope(x=x, y=y, amp=paras["amp"], sdev=paras["sdev"])
    out = carrier * envelope
    return out

def G2_fil_imag(x, y, paras):
    '''
    Custom-built filter response (imaginary part).
    '''
    # Spatial frequency in polar coordinates.
    u = paras["freqs"] * math.cos(paras["dir"])
    v = paras["freqs"] * math.sin(paras["dir"])
    # Computations.
    carrier = G2_carrier_imag(x=x, y=y, freqx=u, freqy=v, phase=paras["phase"])
    envelope = G2_envelope(x=x, y=y, amp=paras["amp"], sdev=paras["sdev"])
    out = carrier * envelope
    return out


def fil_kernel(paras, n_stds=3):
    '''
    Complex values of 2D Gabor filter, for use in convolution.
    When applied to images, this is typically called
    The linear size of the filter is determined as a multiple
    of the standard deviation of the Gaussian envelope, and the
    values passed to the filter are symmetric about zero.
    
    USAGE: pass the parameters only; the size of the grid of
    response values generated depends on these.
    '''
    
    pixnum = 2*math.ceil(n_stds*paras["sdev"])
    
    y0 = pixnum/2
    x0 = pixnum/2
    y, x = np.mgrid[-y0:(y0+1), -x0:(x0+1)]
    
    # Spatial frequency in polar coordinates.
    u = paras["freqs"] * math.cos(paras["dir"])
    v = paras["freqs"] * math.sin(paras["dir"])
    # Computations.
    envelope = G2_envelope(x=x, y=y,
                           amp=paras["amp"]/(2*math.pi*paras["sdev"]**2),
                           sdev=paras["sdev"])
    out = {"real": None, "imag": None}
    out["real"] = envelope * G2_carrier_real(x=x, y=y, freqx=u, freqy=v, phase=paras["phase"])
    out["imag"] = envelope * G2_carrier_imag(x=x, y=y, freqx=u, freqy=v, phase=paras["phase"])
    
    return out



def patch_stats(image, grid_w, grid_h):
    '''
    A simple function which takes an image, divides it
    into a (grid_y x grid_x) grid of patches, and iterates
    over the patches, computing per-patch statistics.
    
    In the special case of grid_x=grid_y=1, stats are for the whole image.
    '''
    
    pix_h = image.shape[0] # number of pixels
    pix_w = image.shape[1]
    
    gridsize = grid_w*grid_h
    
    dh = math.floor(pix_h/grid_h) # typical (non-edge) patch sizes
    dw = math.floor(pix_w/grid_w)
    
    meanvec = np.zeros(gridsize, dtype=np.float32) # initialize vectors to hold the stats
    medvec = np.zeros(gridsize, dtype=np.float32)
    maxvec = np.zeros(gridsize, dtype=np.float32)
    minvec = np.zeros(gridsize, dtype=np.float32)
    
    # Loop over the patches, starting at the top-left, and doing one grid row at a time.
    idx = 0
    for i in range(grid_h):
        
        start_h = i * dh
        if (i+1 == grid_h):
            stop_h = pix_h
        else:
            stop_h = start_h + dh
        
        for j in range(grid_w):
            
            start_w = j * dw
            if (j+1 == grid_w):
                stop_w = pix_w
            else:
                stop_w = start_w + dw
            
            patch = image[start_h:stop_h, start_w:stop_w]
            meanvec[idx] = np.mean(patch) # patch mean
            medvec[idx] = np.median(patch) # patch median
            maxvec[idx] = np.max(patch) # patch maximum
            minvec[idx] = np.min(patch) # patch minimum
            idx += 1
    
    return {"mean": meanvec, "med": medvec, "max": maxvec, "min": minvec}


def nonlin(u):
    '''
    A non-linear function to pass per-patch magnitude statistics through.
    '''
    
    return np.log(1+u)



def nonlin(u):
    '''
    A non-linear function to pass per-patch magnitude statistics through.
    '''
    
    return np.log(1+u)



def G2_getfeatures(ims, fil_paras, gridshape, mode="reflect", cval=0):
    '''
    A routine which takes an array of images with 4 coords.
    Dim 1 and 2: pixel position.
    Dim 3: RGB channel index.
    Dim 4: Time index.
    '''
    
    num_ims = ims.shape[3]
    num_feats = gridshape[0] * gridshape[1]
    
    out = np.zeros(num_ims*num_feats, dtype=np.float32).reshape((num_ims,num_feats))
    
    # Generate the kernel prior to loop over images.
    fil_values = fil_kernel(paras=fil_paras, n_stds=2)
    
    # Iterate over images.
    for i in range(num_ims):
        
        featvec = np.arange(0, dtype=np.float32)
        
        # Slice -> XYZ -> CIE Lab -> Take only Luminance channel.
        im = col.xyz2lab(col.rgb2xyz(ims[:,:,:,i]))[:,:,0]
        
        # Convolution.
        fil_response_real = ndi.convolve(input=im,
                                         weights=fil_values["real"],
                                         mode=mode, cval=cval)
        fil_response_imag = ndi.convolve(input=im,
                                         weights=fil_values["imag"],
                                         mode=mode, cval=cval)
        fil_response_magnitude = np.sqrt((fil_response_real**2 + fil_response_imag**2))
        
        # Per-patch statistics.
        imstats = patch_stats(image=fil_response_magnitude, grid_h=gridshape[0], grid_w=gridshape[1])
            
        # Pass per-patch statistics through non-linearity to compute final feature vector.
        imfeats = nonlin(imstats["mean"])
        
        # Store the feature vector for this image.
        out[i,:] = imfeats
    
    # Output is the array of feature vectors, one feature vector for each image.
    return out
