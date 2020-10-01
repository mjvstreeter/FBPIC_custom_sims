import h5py
import matplotlib.pyplot as plt
from matplotlib import animation 
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.integrate import cumtrapz
from scipy.signal import hilbert
import matplotlib
import numpy as np
from scipy.constants import c, e, m_e, pi, m_p

def imagesc(I,ax = None,  x=None, y=None, **kwargs):
    """ display image with axes using pyplot
    recreates matlab imagesc functionality (roughly)
    argument I = 2D numpy array for plotting
    kwargs:
        ax = axes to plot on, if None will make a new one
        x = horizontal  axis - just uses first an last values to set extent
        y = vetical axis like x
        **kwargs anthing else which is passed to imshow except extent and aspec which are set
    """
    if ax is None:
        plt.figure()
        ax = plt.axes()
    if x is None:
        Nx = np.size(I, axis=1)
        x = np.arange(Nx)
    if y is None:
        Ny = np.size(I, axis=0)
        y = np.arange(Ny)
    ext = (x[0], x[-1], y[-1], y[0])
    return ax.imshow(I, extent=ext, aspect='auto', **kwargs)


def img_rgba_composite(img1,alpha1,img2,alpha2):
    """ Build a composite image from two color images
    img_rgba_composite(img1,alpha1,img2,alpha2)
    img1 is the base image with transparency given by alpha1 (can be all ones)
    img2 is the top image with transparency alpha2
    returns  imgComp (the composite image) ,imgAlpha (the composite image total transparency)
    """
    imgComp = img2*alpha2 + img1*alpha1*(1-alpha2)
    imgAlpha = alpha2 + alpha1*(1-alpha2)
    imgComp = imgComp/imgAlpha
    imgComp = np.clip(imgComp,0,1)
    return imgComp,imgAlpha

def get_a_zr(Ex,Ex_info):
    """ Function for calculating vector potential using integration of the E_field
    """
    N_r_zero = int(len(Ex_info.r)/2)
    a_zr = np.fliplr(cumtrapz(np.fliplr(Ex),x=Ex_info.z/c,axis=1,initial=0))*e/(m_e*c)

    return a_zr


class vid_maker():
    """ Class for making picures and videos from FBPIC simulations with track data
        Tracks should be created in the simulation using tracer_injector.py 
        Author Matthew Streeter 2020
    """
    def __init__(self,n_e0,ts,tracks_ts,
                 r_min=None,r_max=None,inj_z_um=None,
                 omega_0 = 2*pi*c/800e-9,a_ref=1):
        # things that are set at the start which should not be changed
        self.n_e0 = n_e0 # reference density for colormap
        self.ts = ts # fieldd data from fbpic
        self.tracks_ts = tracks_ts # track data using tracer_injector.py 
        self._N = len(self.ts.t) # number of file dumps in sim
        self.p_id = None # just to signify that tracks are not yet loaded

        # things that can be set with inputs but which can be changed
        self.r_min = r_min # min axis limit for figure, if None will be extent of data  
        self.r_max = r_max # max axis limit
        self.inj_z_um = inj_z_um # plots dashed line at this plane
        self.omega_0 = omega_0 # laser frequency
        self.a_ref = a_ref # max norm vector potential for color map

        # below here are set by default but can be changed
        # density stuff
        self.n_e0_lim = 2 # max density for colormap
        self.n_e_cmap = matplotlib.cm.get_cmap('gray_r') 
        # track display settings
        self.track_cmap = matplotlib.cm.get_cmap('cool')
        self.t_line = 50e-15 # length of trail
        self.track_linewidth = 2 # width of trail
        self.ball_size = [20]
        self.ball_linewidth = 1
        self.ball_edgecolors = [[0.0,0,0]]
        # laser colormap
        self.a_cmap = matplotlib.cm.get_cmap('inferno')
        # video settings
        self.color_particles = color_particles
        self.vid_filepath ='particle_vid.mp4'
        self.vid_dpi = 300
        self.vid_fps = 24
        self.vid_bitrate = 1800
        
    def make_video(self):

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.vid_fps,bitrate=self.vid_bitrate)
    
        self.initialise_figure()
        # look for track data that is already loaded
        if self.p_id is None:
            self.get_tracks() # other wise load it (can take some time)
        # update the coloring of the particles
        self.c,self.p_sel = self.color_particles(self.p_t,self.p_x,self.p_y,self.p_z,
                    self.p_px,self.p_py,self.p_pz,self.p_id,self.track_cmap)
        print('N = ' + str(self._N))
    
        print('about to start animation')
        ani = animation.FuncAnimation(self.fig,self._frameUpdate,
                                      frames = self._N,blit=False,  repeat=False)

        
        print('Saving file')
        ani.save(self.vid_filepath, writer=writer,dpi=self.vid_dpi )
        print('Done')
        
    def _frameUpdate(self,n):
        """ Function for updating the animation frame with the given iteraction n
        """
        print(f'Animating frame {n}')
        t0 = self.ts.t[n]
        z_um, r_um, img_comp  = self._get_data(t0)
        self._remove_tracks()
        self.ax.ih.set_data(img_comp)
        self.ax.ih.set_extent((np.min(z_um),np.max(z_um),
                               np.min(r_um),np.max(r_um)))
        self._plot_tracks(t0)
        self.ax.set_xlim((np.min(z_um),np.max(z_um)))
        self.ax.set_ylim((self.r_min,self.r_max))
        
        
    def _plot_tracks(self,t0):
        self.tracks = []
        self.balls = []
        ax = self.ax
        p_t = self.p_t
        for p_ind in range(self.N_p):
            if self.p_sel[p_ind]:
                x = self.p_x[:,p_ind]*1e6
                z = self.p_z[:,p_ind]*1e6
                # currently only tracks particles that have positive p_z
                # it may plot uninitialised particles otherwise
                iSel = np.where((z>0)*(p_t<=t0)*(p_t>(t0-self.t_line)))
                if iSel[0].shape[0]>0:
                    z_plot = z[iSel]
                    x_plot = x[iSel]
                    t_plot = p_t[iSel]

                    # trail behind particles fades relative to time
                    t_norm = plt.Normalize(t_plot.max()-self.t_line, t_plot.max())
                    line_colors = [list(self.c[p_ind,0:3])+[aV] for aV in t_norm(t_plot)]
                    line_cmap = ListedColormap(line_colors)

                    points = np.array([z_plot,x_plot]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc =LineCollection(segments, cmap=line_cmap, norm=t_norm)
                    # Set the values used for colormapping
                    lc.set_array(p_t[iSel])
                    lc.set_linewidth(self.track_linewidth)
                    self.tracks.append(ax.add_collection(lc))

                    # plot current positions
                    self.balls.append(ax.scatter(z_plot[-1],x_plot[-1],
                                                 c=self.c[p_ind,:].reshape(1,-1),
                                                s = self.ball_size,
                                                 edgecolors=self.ball_edgecolors,linewidths=self.ball_linewidth)
                                      )
                else:

                    self.tracks.append(None)
                    self.balls.append(None)
            else:

                self.tracks.append(None)
                self.balls.append(None)

        
    def _remove_tracks(self):
        # this removes all previous tracks from the figure, required before adding new tracks
        if self.tracks is not None:
            for th in self.tracks:
                if th is not None:
                    th.remove()
            for bh in self.balls:
                if bh is not None:
                    bh.remove()
        self.tracks = None
        self.balls = None
            
        
    def initialise_figure(self):
        """ function to setup a figure window for displaying the animiation frames
        """
        z_um, r_um, img_comp =self._get_data(0)
        if self.r_min is None:
            self.r_min = np.min(r_um)
        if self.r_max is None:
            self.r_max = np.max(r_um)
            
        self.fig,self.ax = plt.subplots(1,1,figsize=(18/2.54,18/2.54/1.62),dpi=150)
        self.ax.ih = imagesc(img_comp,x=z_um,y=r_um,ax=self.ax)
        self.ax.ph = self.ax.plot(r_um*0,r_um,'k--',lw=2)
        if self.inj_z_um is not None:
            self.ax.ph2 = self.ax.plot(r_um*0+self.inj_z_um,r_um,'k--',lw=2)
        self.ax.set_xlim((np.min(z_um),np.max(z_um)))
        self.ax.set_ylim((self.r_min,self.r_max))
        self.ax.set_xlabel(r'$z$ [$\mu$m]')
        self.ax.set_ylabel(r'$x$ [$\mu$m]')
        self.fig.tight_layout()
        self.tracks = None
        self.balls = None
        
                           
       
       
    def _get_data(self,t0):
        rho,rho_info = self.ts.get_field('rho',t=t0,plot=False)
        n_e = -rho/e/self.n_e0
        Ex,Ex_info = self.ts.get_field('E',coord='x',m=1,t=t0,plot=False)
        
        z_um = rho_info.z*1e6
        r_um = rho_info.r*1e6
        
        a_zx = e*Ex/(m_e*c*self.omega_0)
        a_env = np.abs(hilbert(a_zx, N=None, axis=1))

        a_color = self.a_cmap(np.clip((a_env/self.a_ref)/2+0.5,0,1))
        a_alpha = np.clip(np.abs(a_env/self.a_ref),0,1)[:,:,np.newaxis]
        n_e_color = self.n_e_cmap(np.clip(n_e/self.n_e0_lim,0,1),alpha=1)
        n_e_alpha = np.ones_like(n_e)[:,:,np.newaxis]

        img_comp,img_alpha = img_rgba_composite(n_e_color,n_e_alpha,a_color,a_alpha)

        
        return z_um, r_um, img_comp 
        
    def get_tracks(self):
        """ this function reads the track files and collates the particle info
        can take a long time especially if there are a lot of particles or file dumps
        """
        tracks_ts = self.tracks_ts
        p_t = []
        p_id = None
        for file_name in tracks_ts.h5_files:

            file_handle = h5py.File(file_name, 'r')
            dset = file_handle['data']
            iteration = list(dset.keys())[0]
            grp = dset[iteration]
            if 'tracks' in list(grp['tracks']['tracer'].keys()):
                trk = grp['tracks']['tracer']['tracks']
                if 'id' in list(trk.keys()):
                    id_array = trk['id']
                    if np.size(id_array.shape)>0:
                        if p_id is None:
                            p_id = id_array
                            p_t = trk['time']
                        else:
                            p_id = np.union1d(p_id, id_array)
                            p_t = np.append(p_t,trk['time'])

        N_t = len(p_t)
        N_id = len(p_id)
        p_x = np.zeros((N_t,N_id))
        p_y = np.zeros((N_t,N_id))
        p_z = np.zeros((N_t,N_id))
        p_px = np.zeros((N_t,N_id))
        p_py = np.zeros((N_t,N_id))
        p_pz = np.zeros((N_t,N_id))

        t_ind = 0
        for file_name in tracks_ts.h5_files:

            file_handle = h5py.File(file_name, 'r')
            list(file_handle.keys())
            dset = file_handle['data']
            iteration = list(dset.keys())[0]
            grp = dset[iteration]
            if 'tracks' in list(grp['tracks']['tracer'].keys()):
                trk = grp['tracks']['tracer']['tracks']
                if 'id' in list(trk.keys()):
                    id_array = trk['id']
                    if np.size(id_array.shape)>0:
                        p_data = trk['p_data']

                        t_ind2 = t_ind+np.shape(p_data)[0]
                        if np.size(id_array)>0:
                            for n,id_ in enumerate(id_array):
                                ind = np.where(id_==p_id)[0][0]
                                p_x[t_ind:t_ind2,ind] = np.reshape(p_data[:,n,0],(1,1,-1))
                                p_y[t_ind:t_ind2,ind] = np.reshape(p_data[:,n,1],(1,1,-1))
                                p_z[t_ind:t_ind2,ind] = np.reshape(p_data[:,n,2],(1,1,-1))
                                p_px[t_ind:t_ind2,ind] = np.reshape(p_data[:,n,3],(1,1,-1))
                                p_py[t_ind:t_ind2,ind] = np.reshape(p_data[:,n,4],(1,1,-1))
                                p_pz[t_ind:t_ind2,ind] = np.reshape(p_data[:,n,5],(1,1,-1))

                        t_ind = t_ind2
        self.p_t = p_t
        self.p_x = p_x
        self.p_y =p_y
        self.p_z = p_z
        self.p_px = p_px
        self.p_py =p_py
        self.p_pz = p_pz
        self.p_id = p_id
        self.N_p = len(p_id)
        # update the coloring of the particles
        self.c,self.p_sel = self.color_particles(p_t,p_x,p_y,p_z,p_px,p_py,p_pz,p_id,self.track_cmap)
        

def color_particles(p_t,p_x,p_y,p_z,p_px,p_py,p_pz,p_id,track_cmap):
    """ This is an example coloring scheme. 
    You can copy and past this into your script/notebook and then edit to your preferences
    Then set as the object color_particles method i.e. 
    VM = vid_maker(arg1,arg2,etc)
    VM.color_particles = color_particles
    """
    c_ind = p_x[0,:]
    
    c_sel = np.ones_like(p_id)>0
    for n in range(0,len(p_id)):
        iSel = np.where(p_z[:,n]>0)
        c_ind[n] = p_x[iSel[0][0],n]
    
    c_ind_sel = c_ind[np.where(c_sel)]
    y_norm = plt.Normalize(np.min(c_ind[c_sel]),np.max(c_ind[c_sel]))
    c = track_cmap(y_norm(c_ind))

    #self.c = np.where(c_ind>5,[0,0,0,1],[1,0,0,1])
    p_sel = c_sel

    return c, p_sel

