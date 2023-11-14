import numpy as np
from numpy.ma.core import MaskedArray
from .piv_motion import particle_image_velocimetry
from .vet_motion import _cost_function
from scipy.ndimage.interpolation import zoom
from scipy.optimize import minimize

from rcit.util.general_methods import get_padding
from rcit.util.post_process.post_process import error_vector_interp_2d
from rcit.util.post_process.post_process import interpolate_sparse_motion
from rcit.util.post_process.post_process import motion_interp_option
from rcit.util.post_process.post_process import smoothing
from rcit.util.pre_process.pre_process import isolate_echo_remove


def vet_cost_function_gradient(*args, **kwargs):
    kwargs["gradient"] = True
    return vet_cost_function(*args, **kwargs)


def vet_cost_function(sector_displacement_1d,
                      input_images,
                      blocks_shape,
                      mask,
                      smooth_gain,
                      debug=False,
                      gradient=False):
    """
    Variational Echo Tracking Cost Function.

    .. _`scipy.optimize.minimize` :\
    https://docs.scipy.org/doc/scipy-0.18.1/reference/\
    generated/scipy.optimize.minimize.html

    This function is designed to be used with the `scipy.optimize.minimize`_

    The function first argument is the variable to be used in the
    minimization procedure.

    The sector displacement must be a flat array compatible with the
    dimensions of the input image and sectors shape (see parameters section
    below for more details).

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/np.ndarray.html


    Parameters
    sector_displacement_1d : ndarray_
        Array of displacements to apply to each sector. The dimensions are:
        sector_displacement_2d
        [ x (0) or y (1) displacement, i index of sector, j index of sector ].
        The shape of the sector displacements must be compatible with the
        input image and the block shape.
        The shape should be (2, mx, my) where mx and my are the numbers of
        sectors in the x and the y dimension.

    template_image : ndarray_  (ndim=2)
        Target image array (nx by ny pixels) where the sector displacement
        is applied.

    input_image : ndarray_ (ndim=2)
        Image array to be used as reference (nx by ny pixels).

    blocks_shape : ndarray_ (ndim=2)
        Number of sectors in each dimension (x and y).
        blocks_shape.shape = (mx,my)

    mask : ndarray_ (ndim=2)
        Data mask. If is True, the data is marked as not valid and is not
        used in the computations.

    smooth_gain : float
        Smoothness constrain gain

    debug : bool, optional
        If True, print debugging information.

    gradient : bool, optional
        If True, the gradient of the morphing function is returned.

    Returns
    penalty or  gradient values.

    penalty : float
        Value of the cost function

    gradient_values : ndarray (float64 ,ndim = 3), optional
        If gradient keyword is True, the gradient of the function is also
        returned.

    """
    
    sector_displacement_2d = sector_displacement_1d.reshape(*((2,) + tuple(blocks_shape)))
    
    if input_images.shape[0] == 3:
        three_times = True
        previous_image = input_images[0]
        center_image = input_images[1]
        next_image = input_images[2]
    
    else:
        previous_image = None
        center_image = input_images[0]
        next_image = input_images[1]
        three_times = False
    
    if gradient:
        gradient_values = _cost_function(sector_displacement_2d,
                                         center_image,
                                         next_image,
                                         mask,
                                         smooth_gain,
                                         gradient=True)
        if three_times:
            gradient_values += _cost_function(sector_displacement_2d,
                                              previous_image,
                                              center_image,
                                              mask,
                                              smooth_gain,
                                              gradient=True)
        
        return gradient_values.ravel()
    
    else:
        residuals, smoothness_penalty = _cost_function(sector_displacement_2d,
                                                       center_image,
                                                       next_image,
                                                       mask,
                                                       smooth_gain,
                                                       gradient=False)
        
        if three_times:
            _residuals, _smoothness = _cost_function(sector_displacement_2d,
                                                     previous_image,
                                                     center_image,
                                                     mask,
                                                     smooth_gain,
                                                     gradient=False)
            
            residuals += _residuals
            smoothness_penalty += _smoothness
        
        if debug:
            print("\nresiduals", residuals)
            print("smoothness_penalty", smoothness_penalty)
        
        return residuals + smoothness_penalty


def global_motion_generation_vet(
        radar_image_first,
        radar_image_second,
        sectors=((32, 16, 4, 2), (32, 16, 4, 2)),
        smooth_gain=1e6,
        first_guess=None,
        intermediate_steps=False,
        verbose=True,
        indexing='ij',
        options=None):
    """
    Variational Echo Tracking Algorithm presented in
    `Laroche and Zawadzki (1995)`_  and used in the McGill Algorithm for
    Prediction by Lagrangian Extrapolation (MAPLE) described in
    `Germann and Zawadzki (2002)`_.

    .. _`Laroche and Zawadzki (1995)`:\
        http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

    .. _`Germann and Zawadzki (2002)`:\
        http://dx.doi.org/10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2

    This algorithm computes the displacement field between two images
    (the input_image with respect to the template image).
    The displacement is sought by minimizing sum of the residuals of the
    squared differences of the images pixels and the contribution of a
    smoothness constrain.

    In order to find the minimum a scaling guess procedure is applied,
    from larger scales
    to a finer scale. This reduces the changes that the minimization procedure
    converges to a local minimum. The scaling guess is defined by the scaling
    sectors (see **sectors** keyword).

    The smoothness of the returned displacement field is controlled by the
    smoothness constrain gain (**smooth_gain** keyword).

    If a first guess is not given, zero displacements are used as the first guess.

    To minimize the cost function, the `scipy minimization`_ function is used
    with the 'CG' method. This method proved to give the best results under
    any different conditions and is the most similar one to the original VET
    implementation in `Laroche and Zawadzki (1995)`_.

    The method CG uses a nonlinear conjugate gradient algorithm by Polak and
    Ribiere, a variant of the Fletcher-Reeves method described in
    Nocedal and Wright (2006), pp. 120-122.


    .. _MaskedArray: https://docs.scipy.org/doc/numpy/reference/\
        maskedarray.baseclass.html#np.ma.MaskedArray

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/np.ndarray.html

    Parameters
    input_images : ndarray_ or MaskedArray
        Input images, sequence of 2D arrays, or 3D arrays.
        The first dimension represents the images time dimension.

        The template_image (first element in first dimensions) denotes the
        reference image used to obtain the displacement (2D array).
        The second is the target image.

        The expected dimensions are (2,nx,ny).
        Be aware the 2D images dimensions correspond to (lon,lat) or (x,y).

    sectors : list or array, optional
        The number of sectors for each dimension used in the scaling procedure.
        If dimension is 1, the same sectors will be used both image dimensions
        (x and y). If is 2D, each row determines the sectors of each dimension.

    smooth_gain : float, optional, Smooth gain factor

    first_guess : ndarray_, optional_
        The shape of the first guess should have the same shape as the initial
        sectors shapes used in the scaling procedure.
        If first_guess is not present zeros are used as first guess.

        E.g.:
            If the first sector shape in the scaling procedure is (ni,nj), then
            the first_guess should have (2, ni, nj ) shape.

    intermediate_steps : bool, optional
        If True, also return a list with the first guesses obtained during the
        scaling procedure. False, by default.

    verbose : bool, optional
        Verbosity enabled if True (default).

    indexing : str, optional
        Input indexing order.'ij' and 'xy' indicates that the
        dimensions of the input are (time, longitude, latitude), while
        'yx' indicates (time, latitude, longitude).

    options : dict, optional
        A dictionary of solver options.
        See `scipy minimization`_ function for more details.

    . _`scipy minimization` : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Returns
    displacement_field : ndarray_
        Displacement Field (2D array representing the transformation) that
        warps the template image into the input image.

        The dimensions of the displacement field is (2,lon,lat).
        The first dimension indicates the displacement along x (0) or y (1).

    intermediate_steps : list of ndarray_
        List with the first guesses obtained during the scaling procedure.

    References
    Laroche, S., and I. Zawadzki, 1995:
    Retrievals of horizontal winds from single-Doppler clear-air data by
    methods of cross-correlation and variational analysis.
    J. Atmos. Oceanic Technol., 12, 721–738.
    doi: http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

    Germann, U. and I. Zawadzki, 2002:
    Scale-Dependence of the Predictability of Precipitation from Continental
    Radar Images.  Part I: Description of the Methodology.
    Mon. Wea. Rev., 130, 2859–2873,
    doi: 10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2.

    Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.

    """
    
    input_images = np.asarray((radar_image_first, radar_image_second,))
    if verbose:
        def debug_print(*args, **kwargs):
            print(*args, **kwargs)
    else:
        def debug_print(*args, **kwargs):
            del args
            del kwargs
    
    if options is None:
        options = dict()
    else:
        options = dict(options)
    
    method = 'CG'
    
    options.setdefault('eps', 0.1)
    options.setdefault('gtol', 0.1)
    options.setdefault('maxiter', 100)
    options.setdefault('disp', True)
    
    # Set to None to suppress pylint warning.
    pad_i = None
    pad_j = None
    sectors_in_i = None
    sectors_in_j = None
    
    debug_print("Running VET algorithm")
    
    if indexing == 'yx':
        input_images = np.swapaxes(input_images, 1, 2)
        if first_guess is not None:
            first_guess = np.swapaxes(first_guess, 1, 2)
    
    if (input_images.ndim != 3) or (1 < input_images.shape[0] > 3):
        raise ValueError("input_images dimension mismatch.\n" +
                         "input_images.shape: " + str(input_images.shape) +
                         "\n(2, x, y ) dimensions expected.")
    
    valid_indexing = ['yx', 'xy', 'ij']
    
    if indexing not in valid_indexing:
        raise ValueError("Invalid indexing valus: {0}\n".format(indexing)
                         + "Supported values: {0}".format(str(valid_indexing)))
    
    # Get mask
    if isinstance(input_images, MaskedArray):
        mask = np.ma.getmaskarray(input_images)
    else:
        # Mask invalid data
        input_images = np.ma.masked_invalid(input_images)
        mask = np.ma.getmaskarray(input_images)
    
    input_images[mask] = 0  # Remove any Nan from the raw data
    
    # Create a 2D mask with the right data type for _vet
    mask = np.asarray(np.any(mask, axis=0), dtype='int8')
    
    input_images = np.asarray(input_images.data, dtype='float64')
    
    # Check that the sectors divide the domain
    sectors = np.asarray(sectors, dtype="int")
    
    if sectors.ndim == 1:
        
        new_sectors = (np.zeros((2,) + sectors.shape, dtype='int')
                       + sectors.reshape((1, sectors.shape[0]))
                       )
        sectors = new_sectors
    elif sectors.ndim > 2 or sectors.ndim < 1:
        raise ValueError("Incorrect sectors dimensions.\n"
                         + "Only 1D or 2D arrays are supported to define"
                         + "the number of sectors used in"
                         + "the scaling procedure")
    
    # Sort sectors in descending order
    sectors[0, :].sort()
    sectors[1, :].sort()
    
    # Prepare first guest
    first_guess_shape = (2, int(sectors[0, 0]), int(sectors[1, 0]))
    
    if first_guess is None:
        first_guess = np.zeros(first_guess_shape, order='C')
    else:
        if first_guess.shape != first_guess_shape:
            raise ValueError(
                "The shape of the initial guess do not match the number of "
                + "sectors of the first scaling guess\n"
                + "first_guess.shape={}\n".format(str(first_guess.shape))
                + "Expected shape={}".format(str(first_guess_shape)))
        else:
            first_guess = np.asarray(first_guess, order='C', dtype='float64')
    
    scaling_guesses = list()
    
    previous_sectors_in_i = sectors[0, 0]
    previous_sectors_in_j = sectors[1, 0]
    
    for n, (sectors_in_i, sectors_in_j) in enumerate(zip(sectors[0, :],
                                                         sectors[1, :])):
        
        # Minimize for each sector size
        pad_i = get_padding(input_images.shape[1], sectors_in_i)
        pad_j = get_padding(input_images.shape[2], sectors_in_j)
        
        if (pad_i != (0, 0)) or (pad_j != (0, 0)):
            
            _input_images = np.pad(input_images, ((0, 0), pad_i, pad_j), 'edge')
            
            _mask = np.pad(mask, (pad_i, pad_j), 'constant', constant_values=1)
            
            if first_guess is None:
                first_guess = np.pad(first_guess, ((0, 0), pad_i, pad_j), 'edge')
        else:
            _input_images = input_images
            _mask = mask
        
        sector_shape = (_input_images.shape[1] // sectors_in_i,
                        _input_images.shape[2] // sectors_in_j)
        
        debug_print("original image shape: " + str(_input_images.shape))
        debug_print("padded image shape: " + str(_input_images.shape))
        debug_print("padded template_image image shape: "
                    + str(_input_images.shape))
        
        debug_print("\nNumber of sectors: {0:d},{1:d}".format(sectors_in_i, sectors_in_j))
        
        debug_print("Sector Shape:", sector_shape)
        
        if n > 0:
            first_guess = zoom(first_guess, (1, sectors_in_i/previous_sectors_in_i, sectors_in_j/previous_sectors_in_j),
                               order=1, mode='nearest')
        
        debug_print("Minimizing")
        
        result = minimize(vet_cost_function,
                          first_guess.flatten(),
                          jac=vet_cost_function_gradient,
                          args=(_input_images,
                                (sectors_in_i, sectors_in_j),
                                _mask,
                                smooth_gain),
                          method=method,
                          options=options)
        
        first_guess = result.x.reshape(*first_guess.shape)
        
        if verbose:
            vet_cost_function(result.x,
                              _input_images,
                              (sectors_in_i, sectors_in_j),
                              _mask,
                              smooth_gain,
                              debug=True)
        
        scaling_guesses.append(first_guess)
        
        previous_sectors_in_i = sectors_in_i
        previous_sectors_in_j = sectors_in_j
    
    first_guess = zoom(first_guess, (1, _input_images.shape[1]/sectors_in_i, _input_images.shape[2]/sectors_in_j), order=1, mode='nearest')
    
    # Remove the extra padding if any
    
    ni = _input_images.shape[1]
    nj = _input_images.shape[2]
    
    first_guess = first_guess[:, pad_i[0]:ni - pad_i[1], pad_j[0]:nj - pad_j[1]]
    
    if intermediate_steps:
        return first_guess, scaling_guesses
    
    return first_guess


def global_motion_generation_piv(radar_image_first,
                                 radar_image_second,
                                 nx_pixel,
                                 ny_pixel,
                                 overlap_x,
                                 overlap_y,
                                 iu_max,
                                 iv_max,
                                 grid_size,
                                 filt_eps,
                                 filt_thres):
    """
    global motion estimation with Particle Image Velocimetry Method
    which is cited from He Ting et al, 2019,
    "New Algorithm For Rain Cell Identification and Tracking In Rainfall Events Analysis",
    https://doi.org/10.3390/atmos10090532.

    Particle Image Velocimetry method is kind of a global optical flow method
    which can capture the instantaneous velocity of an object between two successive time step.
    The PIV method is more robust with motion estimation methods based on cross-correlation.

    :param:time_list: int, time steps for global motion estimation,
                           length of time list is consist with first dim of input_rainfall.

           input_rainfall: ndarray([time_list, m, n]), input spatial rainfall.

           nx_pixel、nx_pixel: int, window size in horizontal and vertical direction.
                                    Value of nx_pixel and ny_pixel must be divided with no remainder.

           overlap_x、overlap_y：float, overlap rate in horizontal and vertical direction.
                                 Value selection must between 0 and 0.9.

           iu_max、iv_max：float, maximum value of motion velocity at horizontal and vertical direction.

           filt_thres：float, threshold of motion post filtering. Value selection is recommended between 0 and 3

    :return: loc: ndarray([m/nx_pixel * n/ny_pixel，2]), center position of global motion vector (x and y direction).

             veclocity: ndarray([time_list-1, m/nx_pixel*n/ny_pixel, m/nx_pixel*n/ny_pixel]), estimated global motion vectors,
                        the first dimension of result is time steps, the second and the third dimension are the motion vector at
                        horizontal and vertical direction.
    """
    ref_img_t1 = isolate_echo_remove(radar_image_first, n=3, thr=np.min(radar_image_first))
    ref_img_t2 = isolate_echo_remove(radar_image_second, n=3, thr=np.min(radar_image_second))
    
    loc_x, loc_y, v_x, v_y = particle_image_velocimetry(ref_img_t1,
                                                        ref_img_t2,
                                                        nx_pixel,
                                                        ny_pixel,
                                                        overlap_x,
                                                        overlap_y,
                                                        iu_max,
                                                        iv_max)
    
    # v_filt_x, v_filt_y, filt_err = error_vector_interp_2d(v_x, v_y, filt_eps, filt_thres)
    
    v_interp_x = motion_interp_option(v_x, 3)
    v_interp_y = motion_interp_option(v_y, 3)
    
    loc_grid_x, loc_grid_y = np.meshgrid(loc_x, loc_y)
    
    grid_x_init, grid_y_init, v_final = interpolate_sparse_motion(loc_grid_x, loc_grid_y, v_interp_x, v_interp_y,
                                                                  (grid_size[0], grid_size[1]),
                                                                  function="multiquadric", epsilon=None, smooth=0.5, nchunks=10)
    
    v_final_x_1, v_final_y_1, filter_err = error_vector_interp_2d(v_final[0, :, :], v_final[1, :, :], filt_eps, filt_thres)
    
    v_final_x_1 = smoothing(v_final_x_1, mode='gauss', ksize=21)
    v_final_y_1 = smoothing(v_final_y_1, mode='gauss', ksize=21)
    
    velocity = np.zeros((2, v_final_x_1.shape[0], v_final_x_1.shape[1]), dtype=float)
    
    velocity[0, :, :] = v_final_x_1
    velocity[1, :, :] = v_final_y_1
    
    loc = [grid_x_init, grid_y_init]
    
    return loc, velocity

