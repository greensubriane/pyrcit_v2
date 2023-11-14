import cv2
import numpy as np
# import cupy as cp
import scipy.signal as signal

from cmath import log10, sqrt
from tkinter import image_names
import time
import json

from rcit.util.pre_process.curvature_filter.cf_interface import cf_filter
from rcit.util.pre_process.input_neis import neighbors_eight


def input_filter(returnMat, filter_type):
    # for all nan values in the returnMat, they are all converted to 0
    origin_img = outs = np.nan_to_num(returnMat)

    # choosing filtering methods if the filter type is 'mf', median filter method is applied
    if filter_type == 'mf':
        outs = signal.medfilt2d(outs, (3, 3))

    # if the filter type is 'cf', Gaussian Curvature method is applied
    if filter_type == 'cf':
        outs = cf_filter(outs, 'gc')

    # getting the neighbor pixels for each pixel by method 'neighbors_eight',
    # the number of neighbors is 8
    nei_pixels = neighbors_eight(outs)
    # nei_pixels = pixel_neighbor_identification(outs, 8)
    row_nei, col_nei = nei_pixels.shape
    row_outs, col_outs = outs.shape

    length_with_null_values = np.zeros([row_nei, 1], dtype=int)
    nei_mat_pixel = np.zeros([row_nei, 2], dtype=float)

    index_x = list(range(0, row_nei))
    for i in index_x:
        nan_list = np.argwhere(np.isnan(nei_pixels[i, :]))
        length_with_null_values[i] = len(nan_list)

    nei_mat_pixel[:, 0] = nei_pixels[:, 0]
    nei_mat_pixel[:, 1] = length_with_null_values[:, 0]

    nei_mat_pixel[:, 0][nei_mat_pixel[:, 0] <= 0] = np.nan

    pixels = np.transpose(nei_mat_pixel[:, 0])
    filter_pixel = np.reshape(pixels, (row_outs, col_outs))
    return origin_img, filter_pixel


def isolate_echo_remove(R, n=3, thr=0):
    """Apply a binary morphological opening to filter small isolated echoes.

    Parameters
    ----------
    R : array-like
        Array of shape (m,n) containing the input precipitation field.
    n : int
        The structuring element size [px].
    thr : float
        The rain/no-rain threshold to convert the image into a binary image.

    Returns
    -------
    R : array
        Array of shape (m,n) containing the cleaned precipitation field.
    """

    # convert to binary image (rain/no rain)
    field_bin = np.ndarray.astype(R > thr, "uint8")

    # build a structuring element of size (nx)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

    # apply morphological opening (i.e. erosion then dilation)
    field_bin_out = cv2.morphologyEx(field_bin, cv2.MORPH_OPEN, kernel)

    # build mask to be applied on the original image
    mask = (field_bin - field_bin_out) > 0

    # filter out small isolated echoes based on mask
    R[mask] = np.nanmin(R)

    return R


def outside_in_fill(image):
    """
    Outside in fill mentioned in paper
    :param image: Image matrix to be filled
    :return: output
    """

    rows, cols = image.shape[:2]

    col_start = 0
    col_end = cols
    row_start = 0
    row_end = rows
    lastValid = np.full([2], np.nan)
    while col_start < col_end or row_start < row_end:
        for c in range(col_start, col_end):
            if not np.isnan(image[row_start, c, 0]):
                lastValid = image[row_start, c, :]
            else:
                image[row_start, c, :] = lastValid

        for r in range(row_start, row_end):
            if not np.isnan(image[r, col_end - 1, 0]):
                lastValid = image[r, col_end - 1, :]
            else:
                image[r, col_end - 1, :] = lastValid

        for c in reversed(range(col_start, col_end)):
            if not np.isnan(image[row_end - 1, c, 0]):
                lastValid = image[row_end - 1, c, :]
            else:
                image[row_end - 1, c, :] = lastValid

        for r in reversed(range(row_start, row_end)):
            if not np.isnan(image[r, col_start, 0]):
                lastValid = image[r, col_start, :]
            else:
                image[r, col_start, :] = lastValid

        if col_start < col_end:
            col_start = col_start + 1
            col_end = col_end - 1

        if row_start < row_end:
            row_start = row_start + 1
            row_end = row_end - 1

    output = image

    return output

def get_video_frames(path):
    """
    Given the path of the video capture, returns the list of frames.
    Frames are converted in grayscale.

    Argss:
        path (str): path to the video capture

    Returns:
        frames (list):  list of grayscale frames of the specified video
    """
    cap = cv2.VideoCapture(path)
    flag = True
    frames = list()
    while flag:
        if cap.grab():
            flag, frame = cap.retrieve()
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        else:
            flag = False
    return frames


def get_pyramids(original_image, levels=3):
    """
    Rturns a list of downsampled images, obtained with the Gaussian pyramid method. The length of the list corresponds to the number of levels selected.

    Args:
        original_image (np.ndarray): the image to build the pyramid with
        levels (int): the number of levels (downsampling steps), default to 3

    Returns:
        pyramid (list): the listwith the various levels of the gaussian pyramid of the image.
    """
    pyramid = [original_image]
    curr = original_image
    for i in range(1, levels):
        scaled = cv2.pyrDown(curr)
        curr = scaled
        pyramid.insert(0, scaled)
    return pyramid


def draw_motion_field(frame, motion_field):
    height, width = frame.shape
    frame_dummy = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    mf_height, mf_width, _ = motion_field.shape
    bs = height // mf_height

    for y in range(0, mf_height):
        for x in range(0, mf_width):
            idx_x = x * bs + bs // 2
            idx_y = y * bs + bs // 2
            mv_x, mv_y = motion_field[y][x]

            cv2.arrowedLine(
                frame_dummy,
                (idx_x, idx_y),
                (int(idx_x + mv_x), int(idx_y + mv_y)),
                # (120, 120, 120),
                (0, 0, 255),
                1,
                line_type=cv2.LINE_AA,
            )
    return frame_dummy


def timer(func):
    """
    Decorator that prints the time of execution of a certain function.

    Args:
        func (Callable[[Callable], Callable]): the function that has to be decorated (timed)

    Returns:
        wrapper (Callable[[any], any]): the decorated function
    """

    def wrapper(*args, **kwargs):
        start = int(time.time())
        ret = func(*args, **kwargs)
        end = int(time.time())
        print(f"Execution of '{func.__name__}' in {end-start}s")
        return ret

    return wrapper


def PSNR(original, noisy):
    """
    Computes the peak sognal to noise ratio.

    Args:
        original (np.ndarray): original image
        noisy (np.ndarray): noisy image

    Returns:
        float: the measure of PSNR
    """
    mse = np.mean((original.astype("int") - noisy.astype("int")) ** 2)
    if mse == 0:  # there is no noise
        return -1
    max_value = 255.0
    psnr = 20 * log10(max_value / sqrt(mse))
    return psnr


def create_video_from_frames(frame_path, num_frames, video_name, fps=30):
    import os

    img_array = []
    img_names = []
    for i in range(3, num_frames):
        s = str(i - 3) + "-" + str(i) + ".png"
        img_names.append(s)
    for img in img_names:
        image = cv2.imread(frame_path + img)
        img_array.append(image)
    height, width, layers = img_array[0].shape
    video = cv2.VideoWriter(video_name, 0, fps, (width, height))

    for image in img_array:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def some_data(psnr_path: str) -> None:
    psnrs = {}
    with open(psnr_path, "r") as f:
        psnrs = json.load(f)

    psnrs_np = np.zeros(shape=[len(psnrs), 1])
    count = 0
    for frames in psnrs:
        cut = psnrs[frames].index("+")
        num = psnrs[frames][1:cut]
        psnrs_np[count] = num
        count += 1

    avg = psnrs_np.sum() / len(psnrs)
    diff = np.zeros(shape=[len(psnrs), 1])

    for value in psnrs_np:
        idx = psnrs_np.tolist().index(value)
        diff[idx] = (value - avg) ** 2

    var = diff.sum() / len(psnrs)

    print("Average: {:.3f}".format(avg))
    print("Variance: {:.3f}".format(var))
    print("Standard deviation: {:.3f}".format(var ** (1 / 2)))
    print("Highest: {:.3f}".format(psnrs_np.max()))
    print("Lowest: {:.3f}".format(psnrs_np.min()))
