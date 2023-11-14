import numpy as np
import cv2
import scipy.ndimage as ndi
import skimage.exposure as exp

import matplotlib as mpl
import csv

from numpy.ma.core import MaskedArray
from skimage.util.shape import view_as_blocks
from ..motion.vet_motion import _warp


def round_int(scalar):
    """
    Round number to nearest integer. Returns and integer value.
    """
    return int(np.round(scalar))


def ceil_int(scalar):
    """
    Round number to nearest integer. Returns and integer value.
    """
    return int(np.ceil(scalar))


def get_padding(dimension_size, sectors):
    """
    Get the padding at each side of the obe dimensions of the image,
    the new image dimensions are divided evenly in the number of *sectors* specified
    :param dimension_size: int Actual dimension size
    :param sectors: int number od sectors over which the image will be divided
    :return: pad_before, pad_after int, int
             Padding at each side of the image for the corresponding dimension.
    """

    reminder = dimension_size % sectors
    
    if reminder != 0:
        pad = sectors - reminder
        pad_before = pad // 2
        if pad % 2 == 0:
            pad_after = pad_before
        else:
            pad_after = pad_before + 1
        
        return pad_before, pad_after
    
    return 0, 0


def morph(image, displacement, gradient=False):
    """
    Morph image by applying a displacement field (Warping).

    The new image is created by selecting for each position the values of the
    input image at the positions given by the x and y displacements.
    The routine works in a backward sense.
    The displacement vectors have to refer to their destination.

    For more information in Morphing functions see Section 3 in
    `Beezley and Mandel (2008)`_.

    Beezley, J. D., & Mandel, J. (2008).
    Morphing ensemble Kalman filters. Tellus A, 60(1), 131-140.

    .. _`Beezley and Mandel (2008)`: http://dx.doi.org/10.1111/\
    j.1600-0870.2007.00275.x


    The displacement field in x and y directions and the image must have the
    same dimensions.

    The morphing is executed in parallel over x axis.

    The value of displaced pixels that fall outside the limits takes the
    value of the nearest edge. Those pixels are indicated by values greater
    than 1 in the output mask.

    Parameters
    ----------

    image : ndarray (ndim = 2)
        Image to morph

    displacement : ndarray (ndim = 3)
        Displacement field to be applied (Warping). The first dimension
        corresponds to the coordinate to displace.

        The dimensions are:
        displacement [ i/x (0) or j/y (1) ,
                      i index of pixel, j index of pixel ]


    gradient : bool, optional
        If True, the gradient of the morphing function is returned.


    Returns
    -------

    image : ndarray (float64 ,ndim = 2)
        Morphed image.

    mask : ndarray (int8 ,ndim = 2)
        Invalid values mask. Points outside the boundaries are masked.
        Values greater than 1, indicate masked values.

    gradient_values : ndarray (float64 ,ndim = 3), optional
        If gradient keyword is True, the gradient of the function is also
        returned.

    """
    
    if not isinstance(image, MaskedArray):
        _mask = np.zeros_like(image, dtype='int8')
    else:
        _mask = np.asarray(np.ma.getmaskarray(image),
                           dtype='int8')
    
    _image = np.asarray(image, dtype='float64', order='C')
    _displacement = np.asarray(displacement, dtype='float64', order='C')
    
    # return _warp(_image, _displacement, gradient=gradient)
    return _warp(_image, _mask, _displacement, gradient=gradient)


def downsize(input_array, x_factor, y_factor=None):
    """
    Reduce resolution of an array by neighbourhood averaging (2D averaging)
    of x_factor by y_factor elements.

    Parameters

    input_array: ndarray
        Array to downsize by neighbourhood averaging

    x_factor : int
        factor of downsizing in the x dimension

    y_factor : int
        factor of downsizing in the y dimension

    Returns

    downsized_array : MaskedArray
        Downsized array with the invalid entries masked.

    """
    
    x_factor = int(x_factor)
    
    if y_factor is None:
        y_factor = x_factor
    else:
        y_factor = int(y_factor)
    
    input_block_view = view_as_blocks(input_array, (x_factor, y_factor))
    
    data = input_block_view.mean(-1).mean(-1)
    
    return np.ma.masked_invalid(data)

def reflectivity_colormap():
    reflectivity_colors = ["#FFFFFF",  # 0 --- 0.02 --- 0.002
                           "#00FFFF",  # 4 --- 0.04 --- 0.003
                           "#43C6DB",  # 8 --- 0.07 --- 0.006
                           "#0000A0",  # 12 --- 0.14 --- 0.012
                           "#00FF00",  # 16 --- 0.27 --- 0.022
                           "#52D017",  # 20 --- 0.52 --- 0.043
                           "#347235",  # 24 --- 0.99 --- 0.082
                           "#FFFF00",  # 28 --- 1.89 --- 0.157
                           "#EAC117",  # 32 --- 3.61 --- 0.301
                           "#F88017",  # 36 --- 6.91 --- 0.576
                           "#FF0000",  # 40 --- 13.21 --- 1.101
                           "#E41B17",  # 44 --- 25.27 --- 2.106
                           "#C11B17",  # 48 --- 48.34 --- 4.029
                           "#F660AB",  # 52 --- 92.48 --- 7.706
                           "#8E35EF",  # 56 --- 176.90 --- 14.741
                           "#000000",  # 60 --- 338.38 --- 28.199
                           ]
    cmap = mpl.colors.ListedColormap(reflectivity_colors)
    return cmap

def write_csv(csv_path, M):
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(M)

def grad_cal(img0, img2):
    # img0 = img0 / 255
    # img2 = img2 / 255

    # kernels
    kernel_x = np.array([[-1, 1], [-1, 1]]) / 4
    kernel_y = np.array([[-1, -1], [1, 1]]) / 4
    kernel_t = np.array([[1, 1], [1, 1]]) / 4

    # calculating grdients by convolving kernels
    fx = ndi.convolve(input=img0, weights=kernel_x) + ndi.convolve(input=img2, weights=kernel_x)
    fy = ndi.convolve(input=img0, weights=kernel_y) + ndi.convolve(input=img2, weights=kernel_y)
    ft = ndi.convolve(input=img2, weights=kernel_t) + ndi.convolve(input=img0, weights=-1 * kernel_t)


def pyramid_down(img, levels):
    """

    :param img: first radar image
    :param levels: No. of labels in the pyramid
    :return: pyramid
    """

    # Initializing pyramid
    pyr = np.zeros((img.shape[0], img.shape[1], levels))

    # updating first real size radar image at first position
    pyr[:, :, 0] = img
    shapes = [[img.shape[0], img.shape[1]]]

    # updating other values
    for i in range(1, levels):
        temp = cv2.pyrDown(img)
        pyr[0:temp.shape[0], 0:temp.shape[1], i] = temp
        shapes.append([temp.shape[0], temp.shape[1]])
        img = temp

    return pyr, shapes

def global_motion_static(global_motion):
    # generating the prevailing wind direction for each time step according to the generated motion vector
    direction = np.arctan2(global_motion[0], global_motion[1])*180/np.pi
    wind_direction = direction + 180 * np.ones(direction.shape, dtype=float)
    wind_direction_hist, wind_direction_bins = exp.histogram(wind_direction[np.isfinite(wind_direction)], nbins=1)

    # generating the mean speed for each time step according to the generated motion vector
    speeds = np.sqrt(np.power(global_motion[0], 2) + np.power(global_motion[1], 2))
    speed_1 = np.reshape(speeds, (speeds.shape[0] * speeds.shape[1], 1))
    wind_speed = speed_1[~np.isnan(speed_1).any(axis=1), :]
    mean_speed = np.nanmean(wind_speed)

    return wind_direction_bins, mean_speed