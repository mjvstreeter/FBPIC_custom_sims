import zipfile
import numpy as np
import os



def analyseSPIDERData(filePath):
    ''' Function to be called by dataRun.py to correctly extract
    both the spectral phase orders and the temporal pulse shape
    
    When performing SPIDER analysis, we know the device sometimes fails
    We need to ensure that if we're averaging results from the SPIDER,
    that we don't include measurements from times that it fails.

    This is quite difficult, so for the moment we will simply save the individual file
    In future, one way to see if the result is bull is to check if the intensity profile in time
    is zero at the temporal limits of the diagnostic
    '''
    
    timeProfile = readSPIDER_temporal_profile(filePath)

    specPhaseOrders = getSpecPhaseOrders(filePath)

    return timeProfile, specPhaseOrders


def readSPIDER_temporal_profile(path):
    
    z = zipfile.ZipFile(path)
    for filename in z.namelist():
        if 'time.dat' in filename:
            d = []
            for i, line in enumerate(z.open(filename)):
                 if not i == 0:
                     # The data is in binary, so we need to decode it to append to arrays
                     d.append([float(line.split(b"\t")[0].decode()), float(line.split(b"\t")[2].decode())])
            d = np.array(d)
            return d[:,0], d[:,1]
    # If it hasn't found the data return arrays of zero
    return np.zeros(10)

def readSPIDER_values(path):
    z = zipfile.ZipFile(path)
    for filename in z.namelist():
        if filename.endswith('values.dat'):
            with z.open(filename) as f:
                data = f.read().decode()
    return data

def extract_file_info(data, key):
    for line in data.splitlines():
        if key in line:
            return float(line.strip().split('\t')[-1])
    raise Exception()

def getSpecPhaseOrders(path):
    data = readSPIDER_values(path)
    GDD = float(extract_file_info(data, 'GDD'))
    TOD = float(extract_file_info(data, 'TOD'))
    FOD = float(extract_file_info(data, 'FOD'))
    return GDD, TOD, FOD

def polyOrders(filePathList):
    
    pOrders = []
    P0_TW_per_J=[]
    for file in filePathList:
        if file.endswith(".zip"):
            
            t,f = readSPIDER_temporal_profile(file)
            # a threshold to throw away junk files (i.e. where signal was too low to get a pulse)
            if np.mean(f)<0.1:
                pOrders.append(getSpecPhaseOrders(file))
            else:
                pOrders.append([np.nan]*3)
            
    return pOrders

def readSPIDER_spectral_domain(path):
    
    z = zipfile.ZipFile(path)
    for filename in z.namelist():
        if 'freq.dat' in filename:
            d = []
            for i, line in enumerate(z.open(filename)):
                 if not i == 0:
                     # The data is in binary, so we need to decode it to append to arrays
                     d.append([float(line.split(b"\t")[0].decode()),
                          float(line.split(b"\t")[1].decode()),
                          float(line.split(b"\t")[2].decode()),
                          float(line.split(b"\t")[3].decode())])
            d = np.array(d)
            return d
    # If it hasn't found the data return arrays of zero
    return np.zeros(10)

def readSPIDER_temporal_domain(path):
    
    z = zipfile.ZipFile(path)
    for filename in z.namelist():
        if 'time.dat' in filename:
            d = []
            for i, line in enumerate(z.open(filename)):
                 if not i == 0:
                     # The data is in binary, so we need to decode it to append to arrays
                     d.append([float(line.split(b"\t")[0].decode()),
                          float(line.split(b"\t")[1].decode()),
                          float(line.split(b"\t")[2].decode()),
                          float(line.split(b"\t")[3].decode())])
            d = np.array(d)
            return d
    # If it hasn't found the data return arrays of zero
    return np.zeros(10)

def E_w2E_t(w,E_w):
    N=len(w)
    E_t = np.fft.fftshift(np.fft.ifft((E_w)))
    dw = np.mean(np.diff(w))
    dt = 2*np.pi/((N-1)*dw)
    t = (np.arange(N)-N/2)*dt
    return t, E_t


def chirpSpiderPulse(filePath,beta,w_0=2.3503):
    """ adds polynomial orders in spectral phase given by beta 
    beta (array of floats) polynomial coefficients starting at beta^(0) (CEP)
    should be 3 elements to specify chirp i.e. np.array([0,0,100]) in units of fs
    """
    z = zipfile.ZipFile(filePath)
    for filename in z.namelist():
        if 'freq.dat' in filename:
            d = []
            for i, line in enumerate(z.open(filename)):
                 if not i == 0:
                     # The data is in binary, so we need to decode it to append to arrays
                     d.append([float(line.split(b"\t")[n].decode()) for n in range(4)])
            d = np.array(d)
    w = d[:,0]*2*np.pi/1e3
    wAxis = np.linspace(0,15,num=2**16)
    from scipy.interpolate import interp1d
    phi_wAxis = interp1d(w,d[:,3],bounds_error=False,fill_value=0)(wAxis)
    S_wAxis = interp1d(w,d[:,2],bounds_error=False,fill_value=0)(wAxis)

    for n,b in enumerate(beta):
        phi_wAxis = phi_wAxis + (wAxis-w_0)**(n)/np.math.factorial(n)*b

    E_w = np.sqrt(S_wAxis)*np.exp(-1j*phi_wAxis)
    t,E_t = E_w2E_t(wAxis,E_w)
    E_t = E_t
    P_t = np.abs(E_t**2)
    P_t=P_t/np.trapz(P_t,x=t)
    phi_t = -(np.unwrap(np.angle(E_t)) -w_0*t)
    phi_t =phi_t-phi_t[int(len(phi_t)/2)]
    return t, P_t, phi_t

