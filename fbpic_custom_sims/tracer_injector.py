## injector for initialising tracer evenly space in x,y,z

import warnings
import numpy as np
from scipy.constants import c

import os
import h5py

from scipy import constants
from fbpic.openpmd_diag.data_dict import macro_weighted_dict, weighting_power_dict, unit_dimension_dict


import datetime
from dateutil.tz import tzlocal
from fbpic import __version__ as fbpic_version

from fbpic.openpmd_diag.generic_diag import OpenPMDDiagnostic

class TracerInjector( object ):
    """
    Class that stores a number of attributes that are needed for
    continuous injection by a moving window.
    """

    def __init__(self, Npx, xmin, xmax, Npy, ymin, ymax, Npz, zmin, zmax, 
            n,dens_func, ux_m, uy_m, uz_m, ux_th, uy_th, uz_th ):
        """
        Initialize continuous injection

        Parameters
        ----------
        See the docstring of the `Particles` object
        """
        # Register properties of the injected plasma
        self.Npx = Npx
        self.Npy = Npy
        self.Npz = Npz
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

        self.n = n
        self.dens_func = dens_func

        self.ux_m = ux_m
        self.uy_m = uy_m
        self.uz_m = uz_m
        self.ux_th = ux_th
        self.uy_th = uy_th
        self.uz_th = uz_th

        self.p_trace_data = None

        # Register spacing between evenly-spaced particles in z
        if Npz != 0:
            self.dz_particles = (zmax - zmin)/Npz
        else:
            # Fall back to the user-provided `dz_particles`.
            # Note: this is an optional argument of `Particles` and so
            # it is not always available.
            self.dz_particles = dz_particles

        # Register variables that define the positions
        # where the plasma is injected.
        self.v_end_plasma = c * uz_m / np.sqrt(1 + ux_m**2 + uy_m**2 + uz_m**2)
        # These variables are set by `initialize_injection_positions`
        self.nz_inject = None
        self.z_inject = None
        self.z_end_plasma = None


    def initialize_injection_positions( self, comm, v_moving_window,
                                        species_z, dt ):
        """
        Initialize the positions that keep track of the injection of particles.
        This is automatically called at the beginning of `step`.

        Parameters
        ----------
        comm: a BoundaryCommunicator object
            Contains information about grid MPI decomposition
        v_moving_window: float (in m/s)
            The speed of the moving window
        species_z: 1darray of float (in m)
            (One element per macroparticle)
            Used in order to infer the position of the end of the plasma
        dt: float (in s)
            Timestep of the simulation
        """
        # The injection position is only initialized for the last proc
        if comm.rank != comm.size-1:
            return
        # Initialize the injection position only if it has not be initialized
        if self.z_inject is not None:
            return

        # Initialize plasma *ahead* of the right *physical*
        # boundary of the box in the damping region (including the
        # injection area) so that after `exchange_period` iterations
        # (without adding new plasma), there will still be plasma
        # inside the physical domain and the damping region (without the
        # injection area). This ensures that there are never particles in the
        # rightmost guard region and that there are always particles inside
        # the damped region, where the field can be non-zero. New particles,
        # which are injected in the Injection region, do not see any fields.
        _, zmax_global_domain_with_damp = comm.get_zmin_zmax( local=False,
                                    with_damp=True, with_guard=False )
        self.z_inject = zmax_global_domain_with_damp \
                + (3-comm.n_inject)*comm.dz \
                + comm.exchange_period*dt*(v_moving_window-self.v_end_plasma)
        self.nz_inject = 0
        # Try to detect the position of the end of the plasma:
        # Find the maximal position of the continously-injected particles
        if len( species_z ) > 0:
            # Add half of the spacing between particles (the
            # injection function itself will add a half-spacing again)
            self.z_end_plasma = species_z.max() + 0.5*self.dz_particles
        else:
            # Default value for empty species
            _, zmax_global_physical_domain = comm.get_zmin_zmax( local=False,
                                    with_damp=False, with_guard=False )
            self.z_end_plasma = zmax_global_physical_domain

        # Check that the particle spacing has been properly calculated
        if self.dz_particles is None:
            raise ValueError(
                'The simulation uses continuous injection of particles, \n'
                'but was unable to calculate the spacing between particles.\n'
                'This may be because you used the `Particles` API directly.\n'
                'In this case, please pass the argument `dz_particles` \n'
                'initializing the `Particles` object.')

    def reset_injection_positions( self ):
        """
        Reset the variables that keep track of continuous injection to `None`
        This is typically called when restarting a simulation from a checkpoint
        """
        self.nz_inject = None
        self.z_inject = None
        self.z_end_plasma = None

    def increment_injection_positions( self, v_moving_window, duration ):
        """
        Update the positions between which the new particles will be generated,
        the next time when `generate_particles` is called.
        This function is automatically called when the moving window moves.

        Parameters
        ----------
        v_moving_window: float (in m/s)
            The speed of the moving window

        duration: float (in seconds)
            The duration since the last time that the moving window moved.
        """
        # Move the injection position
        self.z_inject += v_moving_window * duration
        # Take into account the motion of the end of the plasma
        self.z_end_plasma += self.v_end_plasma * duration

        # Increment the number of particle to add along z
        nz_new = int( (self.z_inject - self.z_end_plasma)/self.dz_particles )
        self.nz_inject += nz_new
        # Increment the virtual position of the end of the plasma
        # (When `generate_particles` is called, then the plasma
        # is injected between z_end_plasma - nz_inject*dz_particles
        # and z_end_plasma, and afterwards nz_inject is set to 0.)
        self.z_end_plasma += nz_new * self.dz_particles


    def generate_particles( self, time ):
        """
        Generate new particles at the right end of the plasma
        (i.e. between z_end_plasma - nz_inject*dz and z_end_plasma)

        Parameters
        ----------
        time: float (in second)
            The current physical time of the simulation
        """
        # Create a temporary density function that takes into
        # account the fact that the plasma has moved
        if self.dens_func is not None:
            def dens_func( z, x,y):
                return( self.dens_func( z-self.v_end_plasma*time, x,y ) )
            dens_func = None

        # Create new particle cells
        # Determine the positions between which new particles will be created
        Npz = self.nz_inject
        zmax = self.z_end_plasma
        zmin = self.z_end_plasma - self.nz_inject*self.dz_particles
        # Create the particles
        Ntot, x, y, z, ux, uy, uz, inv_gamma, w = generate_evenly_spaced_xyz(
                Npz, zmin, zmax, self.Npx, self.xmin, self.xmax, self.Npy, self.ymin, self.ymax,
                self.n, self.dens_func,
                self.ux_m, self.uy_m, self.uz_m,
                self.ux_th, self.uy_th, self.uz_th )

        # Reset the number of particle cells to be created
        self.nz_inject = 0

        return( Ntot, x, y, z, ux, uy, uz, inv_gamma, w )

def generate_evenly_spaced_xyz( Npz, zmin, zmax, Npx, xmin, xmax, Npy, ymin, ymax,
     n, dens_func,ux_m, uy_m, uz_m, ux_th, uy_th, uz_th ):
    """
    Generate evenly-spaced particles, with uniform density function evenly distributed in cartesian space
    and with the momenta given by the `ux/y/z` arguments.

    Parameters
    ----------
    See the docstring of the `Particles` object
    """
    # Generate the particles and eliminate the ones that have zero weight ;
    # infer the number of particles Ntot
    if Npz*Npx*Npy > 0:
        # Get the 1d arrays of evenly-spaced positions for the particles
        dz = (zmax-zmin)*1./Npz
        z_reg =  zmin + dz*( np.arange(Npz) + 0.5 )
        dx = (xmax-xmin)*1./Npx
        x_reg =  xmin + dx*( np.arange(Npx) + 0.5 )
        dy = (ymax-ymin)*1./Npy
        y_reg =  ymin + dy*( np.arange(Npy) + 0.5 )

        # Get the corresponding particles positions
        # (copy=True is important here, since it allows to
        # change the angles individually)
        zp, xp, yp = np.meshgrid( z_reg, x_reg, y_reg,
                                    copy=True, indexing='ij' )
        
        
        # Flatten them (This performs a memory copy)
        #r = rp.flatten()
        x = xp.flatten()
        y = yp.flatten()
        z = zp.flatten()
        # Get the weights (i.e. charge of each macroparticle), which
        # are equal to the density times the volume r d\theta dr dz
        w = n * dx*dy*dz*np.ones_like(x)
        # Modulate it by the density profile
        if dens_func is not None :
            w *= dens_func( z, x,y )

        # Select the particles that have a non-zero weight
        selected = (w > 0)
        if np.any(w < 0):
            warnings.warn(
            'The specified particle density returned negative densities.\n'
            'No particles were generated in areas of negative density.\n'
            'Please check the validity of the `dens_func`.')

        # Infer the number of particles and select them
        Ntot = int(selected.sum())
        x = x[ selected ]
        y = y[ selected ]
        z = z[ selected ]
        w = w[ selected ]
        # Initialize the corresponding momenta
        uz = uz_m * np.ones(Ntot) + uz_th * np.random.normal(size=Ntot)
        ux = ux_m * np.ones(Ntot) + ux_th * np.random.normal(size=Ntot)
        uy = uy_m * np.ones(Ntot) + uy_th * np.random.normal(size=Ntot)
        inv_gamma = 1./np.sqrt( 1 + ux**2 + uy**2 + uz**2 )
        # Return the particle arrays
        return( Ntot, x, y, z, ux, uy, uz, inv_gamma, w )
    else:
        # No particles are initialized ; the arrays are still created
        Ntot = 0
        return( Ntot, np.empty(0), np.empty(0), np.empty(0), np.empty(0),
                      np.empty(0), np.empty(0), np.empty(0), np.empty(0) )


class TracksOpenPMDDiagnostic(object) :
    """
    Generic class that contains methods which are common
    to both FieldDiagnostic and ParticleDiagnostic
    """

    def __init__(self, period, comm, write_dir=None,
                iteration_min=0, iteration_max=np.inf,
                dt_period=None, dt_sim=None ):
        """
        General setup of the diagnostic

        Parameters
        ----------
        period : int, optional
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`). Specify either this or
            `dt_period`.

        comm : an fbpic BoundaryCommunicator object or None
            If this is not None, the data is gathered on the first proc
            Otherwise, each proc writes its own data.
            (Make sure to use different write_dir in this case.)

        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory

        iteration_min, iteration_max: ints
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)

        dt_period : float (in seconds), optional
            The period of the diagnostics, in physical time of the simulation.
            Specify either this or `period`

        dt_sim : float (in seconds), optional
            The timestep of the simulation.
            Only needed if `dt_period` is not None.
        """
        # Get the rank of this processor
        if comm is not None :
            self.rank = comm.rank
        else :
            self.rank = 0

        # Check period argument
        if ((period is None) and (dt_period is None)):
            raise ValueError("You need to pass either `period` or `dt_period`"
                "to the diagnostics.")
        if ((period is not None) and (dt_period is not None)):
            raise ValueError("You need to pass either `period` or `dt_period`"
                "to the diagnostics, \nbut do not pass both.")

        # Get the diagnostic period
        if period is None:
            period = dt_period/dt_sim  # Extract it from `dt_period`
        self.period = max(1, int(round(period))) # Impose non-zero integer

        # Register the arguments
        self.iteration_min = iteration_min
        self.iteration_max = iteration_max
        self.comm = comm

        # Get the directory in which to write the data
        if write_dir is None:
            self.write_dir = os.path.join( os.getcwd(), 'diags' )
        else:
            self.write_dir = os.path.abspath(write_dir)

        # Create a few addiditional directories within self.write_dir
        self.create_dir("")
        self.create_dir("tracks")

    def open_file( self, fullpath ):
        """
        Open a file on either several processors or a single processor
        (For the moment, only single-processor is enabled, but this
        routine is a placeholder for future multi-proc implementation)

        If a processor does not participate in the opening of
        the file, this returns None, for that processor

        Parameter
        ---------
        fullpath: string
            The absolute path to the openPMD file

        Returns
        -------
        An h5py.File object, or None
        """
        # In gathering mode, only the first proc opens/creates the file.
        if self.rank == 0 :
            # Create the filename and open hdf5 file
            f = h5py.File( fullpath, mode="a" )
        else:
            f = None

        return(f)


    def write( self, iteration ) :
        """
        Check if the data should be written at this iteration
        (based on iteration) and if yes, write it.

        Parameter
        ---------
        iteration : int
            The current iteration number of the simulation.
        """
        self.store_data(iteration)
        # Check if the fields should be written at this iteration
        if iteration % self.period == 0 \
            and iteration >= self.iteration_min \
            and iteration < self.iteration_max:

            # Write the hdf5 file if needed
            self.write_hdf5( iteration )

            


    def create_dir( self, dir_path) :
        """
        Check whether the directory exists, and if not create it.

        Parameter
        ---------
        dir_path : string
           Relative path from the directory where the diagnostics
           are written
        """
        # The following operations are done only by the first processor.
        if self.rank == 0 :

            # Get the full path
            full_path = os.path.join( self.write_dir, dir_path )

            # Check wether it exists, and create it if needed
            if os.path.exists(full_path) == False :
                try:
                    os.makedirs(full_path)
                except OSError :
                    pass

    def setup_openpmd_file( self, f, iteration, time, dt ) :
        """
        Sets the attributes of the hdf5 file, that comply with OpenPMD

        Parameter
        ---------
        f : an h5py.File object

        iteration: int
            The iteration number of this diagnostic

        time: float (seconds)
            The physical time at this iteration

        dt: float (seconds)
            The timestep of the simulation
        """
        # Set the attributes of the HDF5 file

        # General attributes
        f.attrs["openPMD"] = np.string_("1.0.0")
        f.attrs["openPMDextension"] = np.uint32(1)
        f.attrs["software"] = np.string_("fbpic " + fbpic_version)
        f.attrs["date"] = np.string_(
            datetime.datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %z'))
        f.attrs["meshesPath"] = np.string_("fields/")
        f.attrs["particlesPath"] = np.string_("particles/")
        f.attrs["iterationEncoding"] = np.string_("fileBased")
        f.attrs["iterationFormat"] =  np.string_("data%T.h5")

        # Setup the basePath
        f.attrs["basePath"] = np.string_("/data/%T/")
        base_path = "/data/%d/" %iteration
        bp = f.require_group( base_path )
        bp.attrs["time"] = time
        bp.attrs["dt"] = dt
        bp.attrs["timeUnitSI"] = 1.

    def setup_openpmd_record( self, dset, quantity ) :
        """
        Sets the attributes of a record, that comply with OpenPMD

        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object

        quantity : string
           The name of the record considered
        """
        if quantity.startswith('rho'): # particle density such as rho_electrons
            quantity = 'rho'

        dset.attrs["unitDimension"] = unit_dimension_dict[quantity]
        # No time offset (approximation)
        dset.attrs["timeOffset"] = 0.

    def setup_openpmd_component( self, dset ) :
        """
        Sets the attributes of a component, that comply with OpenPMD

        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object
        """
        dset.attrs["unitSI"] = 1.


class TracerDiagnostic(TracksOpenPMDDiagnostic) :
    """
    Class that defines the particle diagnostics to be performed.
    """

    def __init__(self, period=None, species={}, comm=None,
        particle_data=["position", "momentum", "weighting"],
        select=None, write_dir=None, iteration_min=0, iteration_max=np.inf,
        subsampling_fraction=None, dt_period=None ) :
        """
        Initialize the particle diagnostics.

        Parameters
        ----------
        period : int, optional
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`). Specify either this or
            `dt_period`.

        dt_period : float (in seconds), optional
            The period of the diagnostics, in physical time of the simulation.
            Specify either this or `period`

        species : a dictionary of :any:`Particles` objects
            The object that is written (e.g. elec)
            is assigned to the particle name of this species.
            (e.g. {"electrons": elec })

        comm : an fbpic BoundaryCommunicator object or None
            If this is not None, the data is gathered by the communicator, and
            guard cells are removed.
            Otherwise, each rank writes its own data, including guard cells.
            (Make sure to use different write_dir in this case.)

        particle_data : a list of strings, optional
            Possible values are:
            ["position", "momentum", "weighting", "E" , "B", "gamma"]
            "E" and "B" writes the E and B fields at the particles' positions,
            respectively, but is turned off by default.
            "gamma" writes the particles' Lorentz factor.
            By default, if a particle is tracked, its id is always written.

        select : dict, optional
            Either None or a dictionary of rules
            to select the particles, of the form
            'x' : [-4., 10.]   (Particles having x between -4 and 10 microns)
            'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
            'uz' : [5., None]  (Particles with uz above 5 mc)

        write_dir : a list of strings, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        iteration_min, iteration_max: ints
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)

        subsampling_fraction : float, optional
            If this is not None, the particle data is subsampled with
            subsampling_fraction probability
        """
        # Check input
        if len(species) == 0:
            raise ValueError("You need to pass an non-empty `species_dict`.")
        # Build an ordered list of species. (This is needed since the order
        # of the keys is not well defined, so each MPI rank could go through
        # the species in a different order, if species_dict.keys() is used.)
        self.species_names_list = sorted( species.keys() )
        # Extract the timestep from the first species
        first_species = species[self.species_names_list[0]]
        self.dt = first_species.dt

        # General setup (uses the above timestep)
        TracksOpenPMDDiagnostic.__init__(self, period, comm, write_dir,
                        iteration_min, iteration_max,
                        dt_period=dt_period, dt_sim=self.dt )

        # Register the arguments
        self.species_dict = species
        self.select = select
        self.subsampling_fraction = subsampling_fraction

        self.p_data = None

        self.id_array = None
        self.track_time = None
        # For each species, get the particle arrays to be written
        self.array_quantities_dict = {}
        self.constant_quantities_dict = {}
        for species_name in self.species_names_list:
            species = self.species_dict[species_name]
            # Get the list of quantities that are written as arrays
            self.array_quantities_dict[species_name] = []
            self.array_quantities_dict[species_name] += ['time','id','p_data']
            self.array_quantities_dict[species_name] += ['x','y','z']
            self.array_quantities_dict[species_name] += ['ux','uy','uz']
            # Get the list of quantities that are constant
            self.constant_quantities_dict[species_name] = ["mass"]
            # For ionizable particles, the charge must be treated as an array
            if species.ionizer is not None:
                self.array_quantities_dict[species_name] += ["charge"]
            else:
                self.constant_quantities_dict[species_name] += ["charge"]


    def setup_openpmd_species_group( self, grp, species, constant_quantities ) :
        """
        Set the attributes that are specific to the particle group

        Parameter
        ---------
        grp : an h5py.Group object
            Contains all the species

        species : a fbpic Particle object

        constant_quantities: list of strings
            The scalar quantities to be written for this particle
        """
        # Generic attributes
        grp.attrs["particleShape"] = 1.
        grp.attrs["currentDeposition"] = np.string_("directMorseNielson")
        grp.attrs["particleSmoothing"] = np.string_("none")
        grp.attrs["particlePush"] = np.string_("Vay")
        grp.attrs["particleInterpolation"] = np.string_("uniform")

        # Setup constant datasets (e.g. charge, mass)
        for quantity in constant_quantities:
            grp.require_group( quantity )
            self.setup_openpmd_species_record( grp[quantity], quantity )
            self.setup_openpmd_species_component( grp[quantity], quantity )
            grp[quantity].attrs["shape"] = np.array([1], dtype=np.uint64)
        # Set the corresponding values
        grp["mass"].attrs["value"] = species.m
        if "charge" in constant_quantities:
            grp["charge"].attrs["value"] = species.q

        # Set the position records (required in openPMD)
        quantity = "positionOffset"
        grp.require_group(quantity)
        self.setup_openpmd_species_record( grp[quantity], quantity )
        for quantity in [ "positionOffset/x", "positionOffset/y",
                            "positionOffset/z"] :
            grp.require_group(quantity)
            self.setup_openpmd_species_component( grp[quantity], quantity )
            grp[quantity].attrs["shape"] = np.array([1], dtype=np.uint64)

        # Set the corresponding values
        grp["positionOffset/x"].attrs["value"] = 0.
        grp["positionOffset/y"].attrs["value"] = 0.
        grp["positionOffset/z"].attrs["value"] = 0.

    def setup_openpmd_species_record( self, grp, quantity ) :
        """
        Set the attributes that are specific to a species record

        Parameter
        ---------
        grp : an h5py.Group object or h5py.Dataset
            The group that correspond to `quantity`
            (in particular, its path must end with "/<quantity>")

        quantity : string
            The name of the record being setup
            e.g. "position", "momentum"
        """
        # Generic setup
        self.setup_openpmd_record( grp, quantity )

        # Weighting information
        grp.attrs["macroWeighted"] = macro_weighted_dict[quantity]
        grp.attrs["weightingPower"] = weighting_power_dict[quantity]

    def setup_openpmd_species_component( self, grp, quantity ) :
        """
        Set the attributes that are specific to a species component

        Parameter
        ---------
        grp : an h5py.Group object or h5py.Dataset

        quantity : string
            The name of the component
        """
        self.setup_openpmd_component( grp )

    def write_hdf5( self, iteration ) :
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        
        

        # Create the file and setup the openPMD structure (only first proc)
        if self.rank == 0:
            filename = "tracks%08d.h5" %iteration
            fullpath = os.path.join( self.write_dir, "tracks", filename )
            f = h5py.File( fullpath, mode="a" )

            # Setup its attributes
            self.setup_openpmd_file( f, iteration, iteration*self.dt, self.dt)

        # Loop over the different species and
        # particle quantities that should be written
        for species_name in self.species_names_list:

            # Check if the species exists
            species = self.species_dict[species_name]
            if species is None :
                # If not, immediately go to the next species_name
                continue

            # Setup the species group (only first proc)
            if self.rank==0:
                species_path = "/data/%d/tracks/%s" %(
                    iteration, species_name)
                # Create and setup the h5py.Group species_grp
                species_grp = f.require_group( species_path )
                self.setup_openpmd_species_group( species_grp, species,
                                self.constant_quantities_dict[species_name])
            else:
                species_grp = None


            # Write the datasets for each particle datatype

            self.write_tracks( species_grp, species)

        # Close the file
        if self.rank == 0:
            f.close()
            self.track_time = None
            self.id_array = None
            self.p_data = None



    def store_data(self,iteration):

        # Receive data from the GPU if needed
        for species_name in self.species_names_list:
            species = self.species_dict[species_name]
            if species.use_cuda :
                species.receive_particles_from_gpu()


        # Select the particles that will be written

        n = species.Ntot
        if self.comm is not None:
            # Multi-proc output
            if self.comm.size > 1:
                n_rank = self.comm.mpi_comm.allgather(n)
            else:
                n_rank = [n]
            Ntot = sum(n_rank)
        else:
            # Single-proc output
            n_rank = None
            Ntot = n
        

        id_array = self.get_dataset( species, "id",
                                        n_rank, Ntot )

        q_list = ["x","y","z","ux","uy","uz"]

        N_dims = len(q_list)
        p_data_t0 = np.zeros((Ntot,N_dims))

        if Ntot>0:
            # get quantities
            for nD, q_label in enumerate(q_list):
                p_data_t0[:,nD] = self.get_dataset( species, q_label,
                                        n_rank, Ntot )
           
            # merge with exisiting data
            if self.rank == 0:
                if self.id_array is None:
                    self.id_array = np.array(id_array*1).astype(np.uint64)
                    self.track_time = np.array(iteration*self.dt)
                    self.p_data = p_data_t0[np.newaxis,:,:] # first axis is along iterations, second is particle, third is axis
                    
                else:
                    for id in id_array:
                        if id not in self.id_array:
                            self.id_array = np.append(self.id_array,np.uint64(id))
                            N_t = np.shape(self.p_data)[0]
                            self.p_data = np.append(self.p_data,np.zeros((N_t,1,N_dims)),axis=1)

                    N_id = len(self.id_array)
                    self.p_data = np.append(self.p_data,np.zeros((1,N_id,N_dims)),axis=0)

                    for source_ind,id in enumerate(id_array):
                        p_ind = np.where(self.id_array==id)
                        self.p_data[-1,p_ind,:] = p_data_t0[np.newaxis,source_ind,:]

                    self.track_time = np.append(self.track_time,iteration*self.dt)

                                    
        # Send data to the GPU if needed
        for species_name in self.species_names_list:
            species = self.species_dict[species_name]
            if species.use_cuda :
                species.send_particles_to_gpu()
        

    def write_tracks( self, species_grp, species ) :
        """
        Write all the particle data sets for one given species

        species_grp : an h5py.Group
            The group where to write the species considered

        species : an fbpic.Particles object
        	The species object to get the particle data from

        n_rank : list of ints
            A list containing the number of particles to send on each proc

        Ntot : int
        	Contains the global number of particles

        """
        # Loop through the quantities and write them
        # for quantity in ['time','id','p_data'] :

        if self.rank == 0:
            quantity_path = "tracks/%s" %('time')
            self.write_dataset(species_grp, species, quantity_path, 'time',
                    self.track_time )
            quantity_path = "tracks/%s" %('id')
            self.write_dataset(species_grp, species, quantity_path, 'id',
                    self.id_array )
            quantity_path = "tracks/%s" %('p_data')
            self.write_dataset(species_grp, species, quantity_path, 'p_data',
                    self.p_data )
           
      
            # Setup the hdf5 groups for "position", "momentum", "E", "B"
            try:
                self.setup_openpmd_species_record(
                    species_grp["tracks"], "position" )
            except:
                pass



    def write_dataset( self, species_grp, species, path, quantity,
                     quantity_array ) :
        """
        Write a given dataset

        Parameters
        ----------
        species_grp : an h5py.Group
            The group where to write the species considered

        species : a warp Species object
        	The species object to get the particle data from

        path : string
            The relative path where to write the dataset,
            inside the species_grp

        quantity : string
            Describes which quantity is written
            x, y, z, ux, uy, uz, w, id, gamma

        n_rank : list of ints
            A list containing the number of particles to send on each proc

        Ntot : int
        	Contains the global number of particles


        """
        # Create the dataset and setup its attributes
        if self.rank==0:
            
            if quantity == "id":
                dtype = 'uint64'
            else:
                dtype = 'f8'
            # If the dataset already exists, remove it.
            # (This avoids errors with diags from previous simulations,
            # in case the number of particles is not exactly the same.)
            if path in species_grp:
                del species_grp[path]
            if quantity_array is not None:
                datashape = np.shape(quantity_array)
                if not datashape:
                    datashape = (1,)

                dset = species_grp.create_dataset(path, datashape, dtype=dtype )
                self.setup_openpmd_species_component( dset, quantity )

                if np.size(quantity_array)>0:
                    dset[:] = quantity_array

    def get_dataset( self, species, quantity, n_rank, Ntot ) :
        """
        Extract the array 

        species : a Particles object
        	The species object to get the particle data from

        quantity : string
            The quantity to be extracted (e.g. 'x', 'uz', 'w')

        n_rank: list of ints
        	A list containing the number of particles to send on each proc

        Ntot : int
            Length of the final array (selected + gathered from all proc)
        """
        # Extract the quantity
        if quantity == "id":
            quantity_one_proc = species.tracker.id
        elif quantity == "charge":
            quantity_one_proc = constants.e * species.ionizer.ionization_level
        elif quantity == "w":
            quantity_one_proc = species.w
        elif quantity == "gamma":
            quantity_one_proc = 1.0/getattr( species, "inv_gamma" )
        else:
            quantity_one_proc = getattr( species, quantity )

        # If this is the momentum, multiply by the proper factor
        # (only for species that have a mass)
        # if quantity in ['ux', 'uy', 'uz']:
        #     if species.m>0:
        #         scale_factor = species.m * constants.c
        #         quantity_one_proc = quantity_one_proc*scale_factor
        if self.comm is not None:
            quantity_all_proc = self.comm.gather_ptcl_array(
                quantity_one_proc, n_rank, Ntot )
        else:
            quantity_all_proc = quantity_one_proc

        # Return the results
        return( quantity_all_proc )


class CustomInjector( object ):
    """
    Class that stores a number of attributes that are needed for
    custom injection by a moving window.
    """

    def __init__(self,user_p_data,
            n,dens_func=None):
        """
        Initialize continuous injection

        Parameters
        ----------
        user_p_data is a 6 x N array containing (x,y,z,ux,uy,uz) values for N particles
        """
        # Register properties of the injected plasma
        self.user_p_data = user_p_data
        self.N_p = np.shape(user_p_data)[1]
        self.p_injected = np.zeros(self.N_p)
        self.n=n
        self.dens_func = dens_func

        self.p_trace_data = None


        # Register variables that define the positions
        # where the plasma is injected.
        p_gamma = np.sqrt(1 + np.sum(user_p_data[3:,:]**2,axis=0))

        # These variables are set by `initialize_injection_positions`
        self.z_inject = None
        self.z_end_plasma = None


    def initialize_injection_positions( self, comm, v_moving_window,
                                        species_z, dt ):
        """
        Initialize the positions that keep track of the injection of particles.
        This is automatically called at the beginning of `step`.

        Parameters
        ----------
        comm: a BoundaryCommunicator object
            Contains information about grid MPI decomposition
        v_moving_window: float (in m/s)
            The speed of the moving window
        species_z: 1darray of float (in m)
            (One element per macroparticle)
            Used in order to infer the position of the end of the plasma
        dt: float (in s)
            Timestep of the simulation
        """
        # The injection position is only initialized for the last proc
        if comm.rank != comm.size-1:
            return
        # Initialize the injection position only if it has not be initialized
        if self.z_inject is not None:
            return

        # Initialize plasma *ahead* of the right *physical*
        # boundary of the box in the damping region (including the
        # injection area) so that after `exchange_period` iterations
        # (without adding new plasma), there will still be plasma
        # inside the physical domain and the damping region (without the
        # injection area). This ensures that there are never particles in the
        # rightmost guard region and that there are always particles inside
        # the damped region, where the field can be non-zero. New particles,
        # which are injected in the Injection region, do not see any fields.
        _, zmax_global_domain_with_damp = comm.get_zmin_zmax( local=False,
                                    with_damp=True, with_guard=False )
        self.z_inject = zmax_global_domain_with_damp \
                + (3-comm.n_inject)*comm.dz \
                + comm.exchange_period*dt*(v_moving_window)
   
        _, zmax_global_physical_domain = comm.get_zmin_zmax( local=False,
                                    with_damp=False, with_guard=False )
        self.z_end_plasma = zmax_global_physical_domain
        self.p_vol = comm.dz**3


    def reset_injection_positions( self ):
        """
        Reset the variables that keep track of continuous injection to `None`
        This is typically called when restarting a simulation from a checkpoint
        """
        self.nz_inject = None
        self.z_inject = None
        self.z_end_plasma = None

    def increment_injection_positions( self, v_moving_window, duration ):
        """
        Update the positions between which the new particles will be generated,
        the next time when `generate_particles` is called.
        This function is automatically called when the moving window moves.

        Parameters
        ----------
        v_moving_window: float (in m/s)
            The speed of the moving window

        duration: float (in seconds)
            The duration since the last time that the moving window moved.
        """
        # Move the injection position
        self.z_inject += v_moving_window * duration
       


    def generate_particles( self, time ):
        """
        Generate new particles at the right end of the plasma
        (i.e. between z_end_plasma - nz_inject*dz and z_end_plasma)

        Parameters
        ----------
        time: float (in second)
            The current physical time of the simulation
        """
        # Create a temporary density function that takes into
        # account the fact that the plasma has moved
        if self.dens_func is not None:
            def dens_func( z, x,y):
                return( self.dens_func( z-self.v_end_plasma*time, x,y ) )
            dens_func = None

        # Create new particle cells
        # Determine the positions between which new particles will be created
        zmax = self.z_inject
        

        p_select = (self.user_p_data[2,:]<zmax)*(self.p_injected<1)
        N_select = np.sum(p_select)
        self.p_injected = self.p_injected + p_select
        p_data = self.user_p_data[:,p_select]
        # Create the particles
        if N_select>0:
    
            x = p_data[0,:].flatten()
            y = p_data[1,:].flatten()
            z = p_data[2,:].flatten()
            uz = p_data[0,:].flatten()
            ux = p_data[1,:].flatten()
            uy = p_data[2,:].flatten()
            inv_gamma = 1./np.sqrt( 1 + ux**2 + uy**2 + uz**2 )
            Ntot = len(x)
            w = self.n * self.p_vol*np.ones_like(x)
            # Modulate it by the density profile
            if dens_func is not None :
                w *= dens_func( z, x,y )
            
        else:
            Ntot = 0
            x = np.empty(0)
            y = np.empty(0)
            z = np.empty(0)
            ux = np.empty(0)
            uy = np.empty(0)
            uz = np.empty(0)
            inv_gamma = np.empty(0)
            w = np.empty(0)

        return( Ntot, x, y, z, ux, uy, uz, inv_gamma, w )





