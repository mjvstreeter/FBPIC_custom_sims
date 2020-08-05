# Written 2019 by Matthew Streeter

"""
This file adds some extra custrom laser profiles for FBPIC
Includes loading I(t) and phi(t) (temporal intensity and phase) from a table or a .mat file (matlab)
"""
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.constants import c, m_e, e, pi, epsilon_0
from scipy.special import factorial, genlaguerre, binom
import scipy.io as sio
from scipy.signal import medfilt2d as mf
import copy

class LaserProfile( object ):
    """
    Base class for all laser profiles.

    Any new laser profile should inherit from this class, and define its own
    `E_field` method, using the same signature as the method below.

    Profiles that inherit from this base class can be summed,
    using the overloaded + operator.
    """
    def __init__( self, propagation_direction ):
        """
        Initialize the propagation direction of the laser.
        (Each subclass should call this method at initialization.)

        Parameter
        ---------
        propagation_direction: int
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        assert propagation_direction in [-1, 1]
        self.propag_direction = float(propagation_direction)

    def E_field( self, x, y, z, t ):
        """
        Return the electric field of the laser

        Parameters
        -----------
        x, y, z: ndarrays (meters)
            The positions at which to calculate the profile (in the lab frame)
        t: ndarray or float (seconds)
            The time at which to calculate the profile (in the lab frame)

        Returns:
        --------
        Ex, Ey: ndarrays (V/m)
            Arrays of the same shape as x, y, z, containing the fields
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return( np.zeros_like(x), np.zeros_like(x) )

    def __add__( self, other ):
        """
        Overload the + operations for laser profiles
        """
        return( SummedLaserProfile( self, other ) )


class SummedLaserProfile( LaserProfile ):
    """
    Class that represents the sum of two instances of LaserProfile
    """
    def __init__( self, profile1, profile2 ):
        """
        Initialize the sum of two instances of LaserProfile

        Parameters
        -----------
        profile1, profile2: instances of LaserProfile
        """
        # Check that both profiles propagate in the same direction
        assert profile1.propag_direction == profile2.propag_direction
        LaserProfile.__init__(self, profile1.propag_direction)

        # Register the profiles from which the sum should be calculated
        self.profile1 = profile1
        self.profile2 = profile2

    def E_field( self, x, y, z, t ):
        """
        See the docstring of LaserProfile.E_field
        """
        Ex1, Ey1 = self.profile1.E_field( x, y, z, t )
        Ex2, Ey2 = self.profile2.E_field( x, y, z, t )
        return( Ex1+Ex2, Ey1+Ey2 )


class TemporalProfile(object):
    def __init__(self, t = 0, I_t=0,phi_t=0):
        self.t = t
        self.I_t = I_t
        self.phi_t = phi_t
    
    def loadProfile(self,filePath):
        A = sio.loadmat(filePath)
        t = A['t']
        self.t = t.flatten()
        I_t = A['I_t']
        I_t = I_t/np.max(I_t)
        self.I_t = I_t.flatten()
        phi_t = A['phi_t']
        self.phi_t = phi_t.flatten()


class GaussXYlongFromTable( LaserProfile ):
    """Class that specifies a 2D gaussian trasnverse profile and a longitudinal intensity and phase from a table."""
    def __init__( self, a0, w0_x, w0_y, z0, zf=None, theta_pol=0.,
                    lambda0=0.8e-6, tProf = TemporalProfile(),
                    propagation_direction=1 ):
        
        """
        Define a linearly-polarized Gaussian laser profile.

        is given by:

        .. math::

            E(\\boldsymbol{x},t) = a_0\\times E_0\,
            \exp\left( -\\frac{r^2}{w_0^2} - \\frac{(z-z_0-ct)^2}{c^2\\tau^2} \\right)
            \cos[ k_0( z - z_0 - ct ) - \phi_{cep} ]

        where :math:`k_0 = 2\pi/\\lambda_0` is the wavevector and where
        :math:`E_0 = m_e c^2 k_0 / q_e` is the field amplitude for :math:`a_0=1`.

        .. note::

            The additional terms that arise **far from the focal plane**
            (Gouy phase, wavefront curvature, ...) are not included in the above
            formula for simplicity, but are of course taken into account by
            the code, when initializing the laser pulse away from the focal plane.

        Parameters
        ----------

        a0: float (dimensionless)
            The peak normalized vector potential at the focal plane, defined
            as :math:`a_0` in the above formula.

        waist: float (in meter)
            Laser waist at the focal plane, defined as :math:`w_0` in the
            above formula.

        tProf: TemporalProfile object
            an object containing the t, I_t and phi_t temporal intensity and phase. I_t should have a maximum of 1

        z0: float (in meter)
            The initial position of the centroid of the laser
            (in the lab frame), defined as :math:`z_0` in the above formula.

        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=z0``, i.e.
            that the laser pulse is at the focal plane initially.

        theta_pol: float (in radian), optional
           The angle of polarization with respect to the x axis.

        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame), defined as
            :math:`\\lambda_0` in the above formula.
            Default: 0.8 microns (Ti:Sapph laser).

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction
        LaserProfile.__init__(self, propagation_direction)

        # Set a number of parameters for the laser
        k0 = 2*np.pi/lambda0
        E0 = a0*m_e*c**2*k0/e
        zr_x = 0.5*k0*w0_x**2
        zr_y = 0.5*k0*w0_y**2

        # If no focal plane position is given, use z0
        if zf is None:
            zf = z0

        # Store the parameters
        self.k0 = k0
        self.inv_zr_x = 1./zr_x
        self.inv_zr_y = 1./zr_y
        self.zf = zf
        self.z0 = z0
        self.E0x = E0 * np.cos(theta_pol)
        self.E0y = E0 * np.sin(theta_pol)
        self.w0_x = w0_x
        self.w0_y = w0_y
        self.tProf = tProf

    def E_field( self, x, y, z, t ):
        """
        See the docstring of LaserProfile.E_field
        """
        # Note: this formula is expressed with complex numbers for compactness
        # and simplicity, but only the real part is used in the end
        # (see final return statement)
        # The formula for the laser (in complex numbers) is obtained by
        # multiplying the Fourier transform of the laser at focus
        # E(k_x,k_y,\omega) = exp( -(\omega-\omega_0)^2(\tau^2/4 + \phi^(2)/2)
        # - (k_x^2 + k_y^2)w_0^2/4 ) by the paraxial propagator
        # e^(-i(\omega/c - (k_x^2 +k_y^2)/2k0)(z-z_foc))
        # and then by taking the inverse Fourier transform in x, y, and t
        

        # Diffraction and stretch_factor
        prop_dir = self.propag_direction
        diffract_factor = 1. + 1j * prop_dir*(z - self.zf) * np.sqrt(self.inv_zr_x*self.inv_zr_y)


        # interpolate table values onto sim grid

        I_func = interp1d(self.tProf.t*c, self.tProf.I_t,axis=0,bounds_error=False,fill_value=0)
        I_z = I_func( -prop_dir*(z - self.z0) - c*t )

        phi_func = interp1d(self.tProf.t*c, self.tProf.phi_t,axis=0,bounds_error=False,fill_value=0)
        phi_z = phi_func( -prop_dir*(z - self.z0) - c*t )

        # Calculate the argument of the complex exponential
        exp_argument = 1j*self.k0*( prop_dir*(z - self.z0) - c*t ) \
            - ((x**2/self.w0_x**2)  + (y**2/self.w0_y**2)) / (diffract_factor) \
            + 1j*phi_z


        # Get the transverse profile
        profile = (np.exp(exp_argument)/diffract_factor)*np.sqrt(I_z)

        # Get the projection along x and y, with the correct polarization
        Ex = self.E0x * profile
        Ey = self.E0y * profile

        return( Ex.real, Ey.real)



class PulseFromData( LaserProfile ):
    """Class that specifies the laser pulse from saved experimental data
    ."""
    def __init__( self, expData,lambda0 = None, a0=1, z0=0.0, zf=None, theta_pol=0.,
                    propagation_direction=1 ):
        
        """

        Parameters
        ----------
        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame)
            If not given then it will look for it in the expData dictionary            

        expData: dictionary containing experimental data
            fields:
            t: time axis (in seconds)
            P_t: temporal power should be normalise to 1
            phi_t:  temporal phase 

            x and y: spatial axis (in meter) with values of 0 at center of pulse (will become r=0)
            F_xy: Fluence on spatial grid, should be normalised to 1
            phi_xy: spatial phase values on grid, should not include focusing term

            lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame)
            Will only look for this if not given to class init

        a0: float (dimensionless)
            The peak normalized vector potential at the focal plane

        z0: float (in meter)
            The initial position of the centroid of the laser

        zf: float (in meter), optional (not implemented yet)
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=z0``, i.e.
            that the laser pulse is at the focal plane initially.

        theta_pol: float (in radian), optional
           The angle of polarization with respect to the x axis.

        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction
        LaserProfile.__init__(self, propagation_direction)

        if lambda0 is None:
            if lambda0 in expData.keys():
                lambda0 = expData['lambda0']
            else:
                raise RuntimeError(
                    'You need to define lambda0 either when initialising this class'
                    ' or in the expData dictionary')
        if not all(x in expData.keys() for x in ['t','phi_t','P_t','x','y','F_xy','phi_xy'] ):
            raise RuntimeError(
                'expData must contain the keys [''t'',''phi_t'',''P_t'',''x'',''y'',''F_xy'',''phi_xy'']')
        # If no focal plane position is given, use z0
        if zf is None:
            zf = z0

        # Set a number of parameters for the laser
        k0 = 2*pi/lambda0
        E0 = a0*m_e*c**2*k0/e
        print(np.shape(expData['F_xy']))
        # shift for correct focal plane
        x,y,F_xy, phi_xy = shift_focal_plane(expData['x'],expData['y'],
                                         expData['F_xy'],expData['phi_xy'],zf,
                      lambda_0=lambda0,N_pad=2000)
        print(np.shape(F_xy))
        # make interpolators for obtaining values on sim x,y,z,t
        self.P_z = interp1d(expData['t']*c, expData['P_t'],bounds_error=False,fill_value=0)
        self.phi_z = interp1d(expData['t']*c, expData['phi_t'],bounds_error=False,fill_value=0)
        self.F_yx = RectBivariateSpline(y,x,F_xy)
        self.phi_yx = RectBivariateSpline(y,x,phi_xy)

        # Store the parameters
        self.k0 = k0
        self.zf = zf
        self.z0 = z0
        self.E0x = E0 * np.cos(theta_pol)
        self.E0y = E0 * np.sin(theta_pol)
        self.expData = expData

    def E_field( self, x, y, z, t ):
        """
        See the docstring of LaserProfile.E_field
        """

        # Diffraction and stretch_factor
        prop_dir = self.propag_direction
     
     
        # interpolate table values onto sim grid
        I_xyz = self.P_z( -prop_dir*(z - self.z0) + c*t )
        I_xyz *= self.F_yx.ev(y,x)
        I_xyz = np.clip(I_xyz,0,None)
        
        phi_xyz = self.phi_z( -prop_dir*(z - self.z0) + c*t )
        phi_xyz*= self.phi_yx.ev(y,x)

        # Calculate the argument of the complex exponential
        exp_argument = 1j*self.k0*( -prop_dir*(z - self.z0) + c*t ) + 1j*phi_xyz


        # Get the transverse profile
        E_xyz = np.exp(exp_argument)*np.sqrt(I_xyz)

        # Get the projection along x and y, with the correct polarization
        Ex = self.E0x * E_xyz
        Ey = self.E0y * E_xyz

        return( Ex.real, Ey.real)

def pulseToData(spiderFile,focalSpotFile,um_per_pixel,laser_energy=1,lambda0=800e-9,beta=None):
    import SPIDERAnalysis
    from PIL import Image
    from scipy.interpolate import splrep, splev
    if beta is None:
        F_t = SPIDERAnalysis.readSPIDER_temporal_domain(spiderFile)
        expData = {'t': F_t[:,0]*1e-15,'P_t_raw': F_t[:,2]/np.max(F_t[:,2]),'phi_t':F_t[:,3] }
        tck  = splrep(expData['t']*1e15, np.sqrt(expData['P_t_raw']), s=.025)
        P_t = splev(expData['t']*1e15, tck, der=0)**2
        expData['P_t'] = P_t/np.max(P_t)
    else:
        t, P_t, phi_t = SPIDERAnalysis.chirpSpiderPulse(spiderFile,beta)
        expData = {'t': t*1e-15,'P_t': P_t/np.max(P_t),'phi_t':phi_t }
   
    
    P0_TWperJ = np.max(expData['P_t']/trapz(expData['P_t'],x=expData['t'])/1e12)
    img = np.array(Image.open(focalSpotFile)).astype(float)

    y0,x0 =calcCOW(img,0.5*np.max(img))

    Ny,Nx = np.shape(img)
    x1 = (np.arange(Nx)-x0)*um_per_pixel
    y1 = (np.arange(Ny)-y0)*um_per_pixel
    dx1 = np.mean(np.diff(x1))
    dy1 = np.mean(np.diff(y1))
    expData['x'] = x1*1e-6
    expData['y'] = y1*1e-6
    expData['F_xy'] = img/np.max(img)
    expData['phi_xy'] = np.zeros_like(img)

    omega_0 = 2*pi*c/lambda0
    I_xy =  img/(dx1*1e-6*dy1*1e-6*np.sum(img))*laser_energy*P0_TWperJ*1e12
    E_xy = np.sqrt(2*I_xy/(c*epsilon_0))
    a_xy = e*E_xy/(m_e*c*omega_0)
    a0 = np.max(a_xy)
    return expData, a0

def smoothExpData(expData,rmax,Nr):
    x1 = x=expData['x']*1e6
    y1=expData['y']*1e6
    
    a_xy = np.sqrt(expData['F_xy'])
    
    # interpolator from experimental data
    F = RectBivariateSpline(y1,x1,a_xy)
    phi_func = RectBivariateSpline(y1,x1,expData['phi_xy'])
    # build polar coords
    Nm = 180
    r = np.linspace(0,rmax,num=Nr)
    theta = np.linspace(-pi,pi,num=2*Nm)
    [R,T] = np.meshgrid(r,theta)
    # and cartesian versions
    X = R*np.cos(T)
    Y = R*np.sin(T)
    F_rt = F(Y*1e6,X*1e6,grid=False)

    # reduced cartesian coords
    x2 = np.linspace(-rmax,rmax,num=Nr)
    y2 = np.linspace(-rmax,rmax,num=Nr)
    [X2,Y2] = np.meshgrid(x2,y2)
    R2 = np.sqrt(X2**2+Y2**2)
    T2 = np.arctan2(Y2,X2)


    F_rt = mf(F_rt,(1,11))*(1-np.tanh((R-rmax*0.5)/(rmax*0.1)))/2
    # interpolate from r,t to x,y reduced
    G = RectBivariateSpline(theta,r,F_rt)
    F_reduced = G(T2,R2,grid=False)
    expData2 = copy.deepcopy(expData)
    expData2['x'] = x2
    expData2['y'] = y2
    expData2['F_xy'] = (F_reduced/np.max(F_reduced))**2
    expData2['phi_xy'] = phi_func(y2*1e6,x2*1e6)
    return expData2


def calcCOW(A,aThresh=0):
    A[~np.isfinite(A)] = 0
    A = A*(A>=aThresh)
    Ny,Nx = np.shape(A)
    x0 = np.sum(np.sum(A,axis=0)*np.arange(Nx))/np.sum(A)
    y0 = np.sum(np.sum(A,axis=1)*np.arange(Ny))/np.sum(A)
    return y0,x0






def pad_data(x,y,I_xy,N_pad = 2000):
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    x0 = np.mean(x)
    y0 = np.mean(y)
    xL = (np.arange(N_pad)-N_pad/2)*dx + x0
    yL = (np.arange(N_pad)-N_pad/2)*dy + y0
    
    I_xyL = RectBivariateSpline(y,x,I_xy)(yL,xL)
    
    return xL,yL,I_xyL
    
    
def shift_focal_plane(x,y,F_xy,phi_xy,z_foc,
                      lambda_0=800e-9,N_pad=2000):
    from LightPipes import Begin, Forvard
    N = len(x)
    
    # pad
    xL,yL,F_xy_L = pad_data(x,y,F_xy,N_pad = N_pad)
    xL,yL,phi_xy_L = pad_data(x,y,phi_xy,N_pad = N_pad)
    size= np.max(xL)-np.min(xL)
    F_xy_L[np.isnan(F_xy_L)] = 0
    F_xy_L = np.clip(F_xy_L,0,None)
    
    # shift
    F=Begin(size,lambda_0,N_pad)
    F.field = np.sqrt(np.sqrt(F_xy_L))*np.exp(1j*phi_xy_L)
    F=Forvard(-z_foc,F)
    
    
    F_xy_L = np.abs(F.field**2)
    phi_xy_L = np.angle(F.field)
    phi_xy_L[np.isnan(phi_xy_L)] = 0
    F_xy_L = np.clip(F_xy_L,0,None)
    F_xy_L[np.isnan(F_xy_L)] = 0
    
    return xL,yL,F_xy_L, phi_xy_L


