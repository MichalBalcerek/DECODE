from os import close
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.signal
import torch
import tifffile
#from tifffile import imsave

import decode.generic.emitter

# np.random.seed(1337)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def Reflect_Boundary_square(radius,x_old,y_old,jumpx,jumpy):
    """ 
    Input : position and jump
    Output : Corrected position if necessary : reflection on borders of a circle of radius radius
    """
    nJx=x_old+jumpx;
    nJy=y_old+jumpy;
    outx_sup=nJx>radius;
    outy_sup=nJy>radius;
    outx_inf=x_old+jumpx<-radius;
    outy_inf=y_old+jumpy<-radius;


    inx = np.bitwise_and(x_old+jumpx<radius, x_old+jumpx>-radius)
    iny = np.bitwise_and(y_old+jumpy<radius, y_old+jumpy>-radius)

    x=copy.deepcopy(x_old);
    y=copy.deepcopy(y_old);

    x[inx]=x[inx]+jumpx[inx];
    y[iny]=y[iny]+jumpy[iny];


    x[outx_sup]=(2*radius-nJx[outx_sup]);
    y[outy_sup]=(2*radius-nJy[outy_sup])
    x[outx_inf]=(-2*radius-nJx[outx_inf]);
    y[outy_inf]=(-2*radius-nJy[outy_inf]);
    return x, y


def Reflect_Boundary_circle(radius,x_old,y_old,jumpx,jumpy):
    """ 
    Input : position and jump
    Output : Corrected position if necessary : reflection on borders of a circle of radius radius
    """
    nJx=x_old+jumpx;
    nJy=y_old+jumpy;
    outx_sup=nJx>radius;
    outy_sup=nJy>radius;
    outx_inf=x_old+jumpx<-radius;
    outy_inf=y_old+jumpy<-radius;


    inx = np.bitwise_and(x_old+jumpx<radius, x_old+jumpx>-radius)
    iny = np.bitwise_and(y_old+jumpy<radius, y_old+jumpy>-radius)

    x=copy.deepcopy(x_old);
    y=copy.deepcopy(y_old);

    x[inx]=x[inx]+jumpx[inx];
    y[iny]=y[iny]+jumpy[iny];


    x[outx_sup]=(2*radius-nJx[outx_sup]);
    y[outy_sup]=(2*radius-nJy[outy_sup])
    x[outx_inf]=(-2*radius-nJx[outx_inf]);
    y[outy_inf]=(-2*radius-nJy[outy_inf]);
    return x, y

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def mat2gray(mat):
    """ convert matrix to gray image (0-255) """
#     mat = mat.astype(np.uint16)
#     depthGray = bytescale(mat)
    minItem = mat.min()
    maxItem = mat.max()
    
    grayImg = mat.copy()
    if (maxItem - minItem) != 0:
        tmpMat = (mat - minItem) / float(maxItem - minItem) # scale to [0-1]
        grayImg = tmpMat * 255
        grayImg = grayImg.astype(np.float32) # MB: previously uint8
    
    return grayImg


def simulate_movie(X, Y, pixel_number, boundary_type, saved, direct_show, file_name, movie_param, save_path, downscale_factor = 1, frame_duration = 30, dt = 5):
    """ 
    Simulate a movie from input trajectory coordinates
    X,Y = (N * (frame_duration//dt) x M) arrays
    M - number of trajectories  
    N - trajectory duration
    pixel_number = number of pixel along each dimension (image is a square)
    boundary_type=  molecules can diffuse
                    -'square': everywhere
                    -'circle': within the disk of diameter pixelnumber
    saved = 1 if movies is saved
    direct_show = 1 if should be shown as bit is being made
    file_name = name of the movies to be saved
    """

    #radius = pixel_number/2
    t, M = np.shape(X)
    d_index = frame_duration//dt
    t = t//(frame_duration//dt) # since we will have t frames, whereas N * (frame_duration//dt) was number of subframes
    
    stack = np.zeros(shape = (t, pixel_number//downscale_factor, pixel_number//downscale_factor), dtype = np.float32) # frame, pixel, pixel ## previously uint8 
    # print("shape(stack) = ", np.shape(stack))
    matrix = np.zeros(shape = (pixel_number, pixel_number)) # these are not downscaled; stack is
    matrix_inter = np.zeros(shape = (pixel_number, pixel_number))
    # print("shape(matrix) = ", np.shape(matrix))
    
    ## movie parameters
    background = movie_param.background;
    sigma_dye = movie_param.sigma_dye//movie_param.dx; # sigma of PSF in pixel
    sigma_noise = movie_param.sigma_noise; #150; % level of gaussian noise on each image
    PSF_intensity = movie_param.PSF_intensity; # Maximum of hte PSF
    dx = movie_param.dx; # pixel size

    hsize=20; # size of 2d Gaussian filter
    filterG = matlab_style_gauss2D(shape = (hsize, hsize), sigma = sigma_dye);
    filterG = PSF_intensity*filterG/np.max(filterG);

    # xcoord = [np.arange(start = 1, stop = pixel_number+dx, step = dx)] # not needed as wer use rint for the closest
    # ycoord = [np.arange(start = 1, stop = pixel_number+dx, step = dx)]
    X_discrete = np.zeros(shape = (t, M))
    Y_discrete = np.zeros(shape = (t, M))
    cp_t = 0
    for T in range(t):
        
        matrix[:,:] = 0 

        for _ in np.arange(start = 0, stop = frame_duration, step = dt): # inteframe loop and adding psf and noise
            matrix_inter[:,:] = 0
            x = X[cp_t, :]
            y = Y[cp_t, :]
            cp_t = cp_t + 1
            ## simulate PSFs
            
            # for inter_dt in range(frame_duration//dt):
            #print(np.argmin(np.abs( np.transpose(xcoord) - x), axis = 0))

            # x/dx normalization: check
            closestIndexX = np.rint(x/dx).astype(np.int32) # np.argmin(np.abs( np.transpose(xcoord) - x), axis = 0) # we could round
            closestIndexY = np.rint(y/dx).astype(np.int32) # np.argmin(np.abs( np.transpose(ycoord) - y), axis = 0) 
        
            #print(closestIndexX)
            matrix_inter[closestIndexX, closestIndexY] = 1
            matrix_inter = scipy.signal.convolve2d(matrix_inter, filterG, 'same')
            
            matrix = matrix + matrix_inter

        matrix = matrix/(frame_duration//dt) # normalization over number of subframes
        matrix = matrix + background * np.ones(shape = (pixel_number, pixel_number))
            
        ## add noise
        matrix = matrix + sigma_noise * np.random.normal(size = matrix.shape) # normal noise
        # matrix = matrix + sigma_noise * np.random.exponential(sigma_noise, size = matrix.shape)
        matrix = mat2gray(matrix)
        
        ## display
        if direct_show == 1 & T == t-1:
            plt.imshow(matrix, cmap='gray', vmin=0, vmax=255)
        

        # saving data
        X_discrete[T, :] = closestIndexX/downscale_factor
        Y_discrete[T, :] = closestIndexY/downscale_factor
        #if saved == 1:
        stack[T, :, :] = (matrix[0::downscale_factor, 0::downscale_factor] - np.min(matrix))/(np.max(matrix) - np.min(matrix)) * 500;#.astype(int);
    # stack = stack.astype(int)
    X_f = X[(d_index-1)::d_index, :]
    Y_f = Y[(d_index-1)::d_index, :]
    
    # print(stack.dtype)
    if saved==1: # MB: should work/works in the notebook
        tifffile.imwrite(file_name+'.tif', stack, imagej=True, metadata={'axes': 'TYX', 'fps': 10.0}) #check if TYX or TXY

    stack = torch.tensor(stack)

    return stack, X_f, Y_f, X_discrete, Y_discrete

def simulate_BM(N, M, sigma=1.0, frame_duration = 30, dt = 5):
    """
    N - number of frames (trajectories' length)
    M - number of molecules (number of trajectories)
    frame_duration - duration of the frames in the movies
    dt - one step duration
    """
    X = sigma * np.cumsum(np.random.normal(0, 1, size = (N * (frame_duration//dt), M)), axis = 0) # location, scale, size
    Y = sigma * np.cumsum(np.random.normal(0, 1, size = (N * (frame_duration//dt), M)), axis = 0)

    return X, Y

def simulate_trajectories(X, Y, pixel_number, boundary_type):
    """ 
    Simulate a movie from input trajectory coordinates
    X, Y = (N * (frame_duration//dt) x M) arrays  
    M - number of trajectories  
    N * (frame_duration//dt) - duration of trajectories for the movie
    pixel_number - number of pixel along each dimension 
    boundary_type -  molecules can diffuse
                    -'square': everywhere
                    -'circle': within the disk of diameter pixelnumber
    """

    radius = pixel_number/2
    incX = np.diff(np.vstack((X[0,:], X)), axis = 0)
    incY = np.diff(np.vstack((Y[0,:], Y)), axis = 0)
    t, M = np.shape(X)
    

    # stack = np.zeros(shape = (t, pixel_number//downscale_factor, pixel_number//downscale_factor), dtype = np.float32) # frame, pixel, pixel
    # #print("shape(stack) = ", np.shape(stack)) # 1.10.2021
    # matrix = np.zeros(shape = (pixel_number, pixel_number))
    # matrix_inter = np.zeros(shape = (pixel_number, pixel_number))
    #print("shape(matrix) = ", np.shape(matrix)) # 1.10.2021
    
    # dx = 1.0
    # xcoord = [np.arange(start = -radius+1, stop = radius+dx, step = dx)]
    # ycoord = [np.arange(start = -radius+1, stop = radius+dx, step = dx)]
    
    X_f = np.zeros(shape = (t, M))
    Y_f = np.zeros(shape = (t, M))
    # X_discreet = np.zeros(shape = (t, M))
    # Y_discreet = np.zeros(shape = (t, M))

    for T in range(t):
        if T == 0: # initial time
            if boundary_type == "square":
                x = (2*np.random.rand(M) - 1) * radius
                y = (2*np.random.rand(M) - 1) * radius
            elif boundary_type == "circle":
                ang_init = 2 * np.pi * np.random.rand(M)
                r = radius * np.sqrt(np.random.rand(M))
                x = r * np.cos(ang_init)
                y = r * np.sin(ang_init)
        else:
            jumpx = incX[T, :]
            jumpy = incY[T, :]
#             print("size jumpx", jumpy.shape)
            if boundary_type == "square":
                [x, y] = Reflect_Boundary_square(radius, x, y, jumpx, jumpy)
            elif boundary_type == "circle":
                [x, y] = Reflect_Boundary_circle(radius, x, y, jumpx, jumpy)
            
        X_f[T, :] = x
        Y_f[T, :] = y
        
    X_f = X_f + radius - 0.5 # 0.5 is the correction 
    Y_f = Y_f + radius - 0.5
    
    return X_f, Y_f # X_discrete, Y_discrete

def movie_to_emitter(X, Y, split = True):
    """
    Converting numpy array to DECODE EmitterSet class 
    """
    frames, N = np.shape(X) # number of frames, number of emitters
    # x_torch = torch.from_numpy(X) # MB not used
    # y_torch = torch.from_numpy(Y)
    frame_indices = torch.repeat_interleave(torch.arange(0, frames), N)

    xyz = torch.empty(N*frames, 3)

    for j in range(frames):
        for k in range(N):
            xyz[k+j*N] = torch.tensor([X[j, k], Y[j, k], -1])    # MB decode.EmitterSet() -> -1 because of the weird results during the training

    phot_ = 3000*torch.ones(N*frames)# torch.rand(N*frames) # MB how to obtain photons in simulations for each emitter? # is it PSF intensity

    if split: 
        em = [decode.generic.emitter.EmitterSet(xyz = xyz[frame_indices==k, :], frame_ix = frame_indices[frame_indices==k],
            phot = phot_[frame_indices==k], xy_unit='px', px_size=(100., 100.)) for k in range(frames)]       # MB split version
    else:
        em = decode.generic.emitter.EmitterSet(xyz = xyz, frame_ix = frame_indices,
            phot = phot_, xy_unit='px', px_size=(100., 100.))
    return em # check

class Movie_Parameters:
    def __init__(self, background, sigma_dye, sigma_noise, PSF_intensity, dx):
        self.background = background;
        self.sigma_dye = sigma_dye # % sigma of PSF in pixel
        self.sigma_noise = sigma_noise #%150; % level of gaussian noise on each image
        self.PSF_intensity= PSF_intensity #% Maximum of hte PSF
        self.dx = dx # pixel size