# -*- coding: utf-8 -*-

#
# Licensed under the BSD-3-Clause license
# Copyright (c) 2021, Ting He
#

"""
Cython module for rain cell segmentation used in Rain Cell Identification and TRacking (RCIT) Algorithm
"""

import cv2
import cython
import math as mt
import numpy as np
cimport numpy as np
import sys

from scipy import ndimage
from skimage import measure
from skimage import morphology
from skimage.feature import peak_local_max as plm

from rcit.util.pre_process.input_neis import neighbors_eight as get_neighbor
from rcit.util.pre_process.local_peak_search import find

ctypedef np.float64_t float64

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def segment_empirical(np.ndarray[float64, ndim=2] filtered_image,
                      float64 thr_intensity,
                      float64 thr_area):
    """
    segmentation method based on RCIT algorithm
    """
    filtered_image[filtered_image <= thr_intensity] = np.nan
    filtered_image = np.nan_to_num(filtered_image)
    fimage = filtered_image.astype('int')
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbors_count = ndimage.convolve(fimage, k, mode='constant', cval=1)
    neighbors_count[~fimage.astype('bool')] = 0
    spur_image = neighbors_count > 1
    for _ in range(1):
        mimage = spur_image(filtered_image)
    binary_image = mimage.astype('int')

    # 8 connective region labeling
    labels = measure.label(binary_image, background=0, connectivity=2)
    label_cells = morphology.remove_small_objects(labels, min_size=thr_area, connectivity=1)
    return label_cells

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def segment_watershed(np.ndarray[float64, ndim=2] filtered_image,
                      float64 thr_intensity,
                      float64 thr_area):
    """
    segmentation method based on WaterShed Segment method
    """
    filtered_image[filtered_image <= thr_intensity] = np.nan
    filtered_image = np.nan_to_num(filtered_image)

    # siz = filtered_image.shape
    cdef int row = filtered_image.shape[0]
    cdef int col = filtered_image.shape[1]
    filtered_image_channel = np.ndarray((row, col, 3), dtype=np.uint8)
    filtered_image_channel[:, :, 0] = filtered_image_channel[:, :, 1] = filtered_image_channel[:, :, 2] = filtered_image

    # binary matrix process based on threshold
    ret0, thresh_mat = cv2.threshold(filtered_image_channel[:, :, 0], thr_intensity, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # ret0, thresh_mat = cv2.threshold(filtered_image_channel[:, :, 0], thr_intensity, 1, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)

    # identifying background zone
    data_bg = cv2.dilate(thresh_mat, kernel, iterations=3)

    # identifying forehead zone by Euclidean distance
    dist = cv2.distanceTransform(thresh_mat, cv2.DIST_L2, 3)  # acquiring distance
    ret1, data_fg = cv2.threshold(dist, dist.max() * 0.1, 1, cv2.THRESH_BINARY)  # acquiring foreground

    # identifying unknown zone
    data_fg = np.uint8(data_fg)
    unknown = cv2.subtract(data_bg, data_fg)

    # marking labels
    ret2, markers = cv2.connectedComponents(data_fg)
    markers1 = markers + 1
    markers1[unknown == 1] = 0

    # applying watershed segmenting method
    segment = cv2.watershed(filtered_image_channel, markers=markers1)
    segment[segment == -1] = 0
    segment[segment == 1] = 0

    # using morphology method to remove small sized objects, the size threshold is thr_area
    label_cells = morphology.remove_small_objects(segment, min_size=thr_area, connectivity=1)
    return label_cells

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def conv_rain_cell_segmentation(np.ndarray[float64, ndim=2] seg_ref_ini,
                                segment_type,
                                float64 diff_ref = 10,
                                float64 thr_conv_area = 4):
    """
    convective rain cell segment, this part is based on the algorithm of CELLTRACK and TRACE3D:
    Hana Kyznarová and Petr Novák: CELLTRACK — Convective cell tracking algorithm
    and its use for deriving life cycle characteristics
    Jan Handwerker: Cell tracking with TRACE3D — a new algorithm
    """
    cdef float64 max_ref_seg_ini = seg_ref_ini.max()
    cdef float64 con_thr = max_ref_seg_ini - diff_ref

    # defining the segment method,
    # this segment rule is used for segmenting convective cells based RCIT and Watershed method
    segment_rule = {'RCIT': segment_empirical, 'WaterShed': segment_watershed}
    seg_method = segment_rule.get(segment_type)
    label_conv_cells = seg_method(seg_ref_ini, con_thr, thr_conv_area)
    return label_conv_cells

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline neighbors_process(np.ndarray[float64, ndim=2] cells,
                              # np.ndarray[np.int64_t, ndim=2] cell_peak,
                              cell_peak,
                              float64 thr_intensity,
                              label_index):
    index = np.where(cell_peak == label_index)

    for i in range(len(index[0])):
        nei_0 = cells[index[0][i], index[1][i]]

        if 0 <= index[0][i] + 1 <= cells.shape[0] - 1 and 0 <= index[1][i] + 1 <= cells.shape[1] - 1:
            nei_1 = cells[index[0][i], index[1][i] + 1]
            nei_2 = cells[index[0][i] + 1, index[1][i] + 1]
            nei_3 = cells[index[0][i] + 1, index[1][i]]
            nei_4 = cells[index[0][i] + 1, index[1][i] - 1]
            nei_5 = cells[index[0][i], index[1][i] - 1]
            nei_6 = cells[index[0][i] - 1, index[1][i] - 1]
            nei_7 = cells[index[0][i] - 1, index[1][i]]
            nei_8 = cells[index[0][i] - 1, index[1][i] + 1]

            if nei_0 >= nei_1 > thr_intensity and cell_peak[index[0][i], index[1][i] + 1] == 0:
                # if cell_peak[index[0][i], index[1][i] + 1] == 0:
                cell_peak[index[0][i], index[1][i] + 1] = label_index

            if nei_0 >= nei_2 > thr_intensity and cell_peak[index[0][i] + 1, index[1][i] + 1] == 0:
                # if cell_peak[index[0][i] + 1, index[1][i] + 1] == 0:
                cell_peak[index[0][i] + 1, index[1][i] + 1] = label_index

            if nei_0 >= nei_3 > thr_intensity and cell_peak[index[0][i] + 1, index[1][i]] == 0:
                # if cell_peak[index[0][i] + 1, index[1][i]] == 0:
                cell_peak[index[0][i] + 1, index[1][i]] = label_index

            if nei_0 >= nei_4 > thr_intensity and cell_peak[index[0][i] + 1, index[1][i] - 1] == 0:
                # if cell_peak[index[0][i] + 1, index[1][i] - 1] == 0:
                cell_peak[index[0][i] + 1, index[1][i] - 1] = label_index

            if nei_0 >= nei_5 > thr_intensity and cell_peak[index[0][i], index[1][i] - 1] == 0:
                # if cell_peak[index[0][i], index[1][i] - 1] == 0:
                cell_peak[index[0][i], index[1][i] - 1] = label_index

            if nei_0 >= nei_6 > thr_intensity and cell_peak[index[0][i] - 1, index[1][i] - 1] == 0:
                # if cell_peak[index[0][i] - 1, index[1][i] - 1] == 0:
                cell_peak[index[0][i] - 1, index[1][i] - 1] = label_index

            if nei_0 >= nei_7 > thr_intensity and cell_peak[index[0][i] - 1, index[1][i]] == 0:
                # if cell_peak[index[0][i] - 1, index[1][i]] == 0:
                cell_peak[index[0][i] - 1, index[1][i]] = label_index

            if nei_0 >= nei_8 > thr_intensity and cell_peak[index[0][i] - 1, index[1][i] + 1] == 0:
                # if cell_peak[index[0][i] - 1, index[1][i] + 1] == 0:
                cell_peak[index[0][i] - 1, index[1][i] + 1] = label_index

    return cell_peak

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def rain_cell_modelling(np.ndarray[float64, ndim=2] filtered_image,
                        float64 thr_intensity,
                        float64 thr_area,
                        float64 thr_intensity_peak,
                        ):
    """
    Rainy pixel segment, there are two segment approach,
    one is based on a heuristic segmenting approach from RCIT algorithm(segment_RCIT);
    another way is based on watershed segment method (segment_RCIT).

    for both segmenting methods, inputs are:
    filtered_image - radar reflectivity ref_img after filtering;
    thr_intensity - thresholds for generating binary ref_img, the default value is 19 dBZ;
    thr_area - area thresholds for eliminating small segment area after segmenting process, the default value is 4km2

    outputs are segment - segmented area with labeled number

    cite: rain cell modelling code which is inspired from the reference:
    'Spatial patterns in thunderstorm rainfall events and their coupling with watershed hydrological response,
    Efrat Morin etal 2005'

    :param filtered_image: radar intensity or reflectivity maps with filtering procedure
    :param thr_intensity: intensity or ref threshold for segment identification
    :param thr_area: area threshold for eliminating or merging 'small' segment
    :param thr_intensity_peak: peak intensity or ref threshold for eliminating or merging 'small' or 'not compatible'
    segments
    :return: label_cells 'labels of final segmented rain cells'
    """

    cdef int rows = filtered_image.shape[0]
    cdef int cols = filtered_image.shape[1]
    # cdef np.ndarray[np.int64_t, ndim=2] cell_label = np.zeros((rows, cols), dtype='int')
    # cdef np.ndarray[np.int64_t, ndim=2] cell_labels = np.zeros((rows, cols), dtype='int')
    # cdef np.ndarray[np.int64_t, ndim=2] label_cells = np.zeros((rows, cols), dtype='int')

    cell_label = np.zeros((rows, cols), dtype='int')
    cell_labels = np.zeros((rows, cols), dtype='int')
    label_cells = np.zeros((rows, cols), dtype='int')

    # the first step is search the local peak
    # identification is based on pixel neighbors, 4 or 8 neighbor is for option
    # the output for this step is 'local_peaks'
    local_peaks = plm(filtered_image, min_distance=1, threshold_abs=thr_intensity)

    #local_peaks = find(filtered_image, search_radius=1, threshold=thr_intensity)
    # print(local_peaks.shape[0])

    if local_peaks.shape[0] != 0:
        for index_peak in range(local_peaks.shape[0]):
            """
            the second step is to identify all candidate segments for all loca peaks,
            all segments are gradually expanded to neighboring pixels with rainfall values
            lower than or equal to the ones already existing in the segment
            but higher than a rainfall threshold (thr_intensity, user defined parameter).
            The new segment is not in any existed segment
            the outputs of this step is：label_cells, segment_label_matrix
            """

            segment_label_matrix = np.zeros((rows, cols), dtype = 'int')
            segment_label_matrix[int(local_peaks[index_peak][0]), int(local_peaks[index_peak][1])] = index_peak + 1

            for expand_num in range(1, np.min([rows, cols]) - 1):
                segment_label_matrix = neighbors_process(filtered_image,
                                                         segment_label_matrix,
                                                         thr_intensity,
                                                         index_peak + 1)
                if np.max(segment_label_matrix) == expand_num:
                    break

            temp_index = np.where((segment_label_matrix == index_peak + 1))
            cell_label[segment_label_matrix == index_peak + 1] = index_peak + 1

    # the third step is to eliminate or merge 'small' or 'not compatible' segments,
    # if the segment is 'isolated', is the area of segment is lower than thr_area,
    # or peak intensity is lower than thr_intensity_peak, then it is eliminated,
    # otherwise it is merged with its neighbor segment.

    neighbour_cell_labels = get_neighbor(cell_label, 8)
    all_cell_label_column = neighbour_cell_labels[:, 0]
    cells_props = measure.regionprops(label_image=cell_label, intensity_image=filtered_image)

    for p in cells_props:
        nei_cell_label = neighbour_cell_labels[all_cell_label_column == p.label]
        intensity = filtered_image[cell_label == p.label]

        nei_matrix = nei_cell_label[:, 1:8]
        bordering_cell_labels = nei_matrix[(nei_matrix != p.label) & (nei_matrix != np.nan) & (nei_matrix > 0)]
        unique_bordering_cell_label = np.unique(bordering_cell_labels)

        inten = np.zeros((rows, cols), dtype=float)
        inten[np.where(cell_label == p.label)] = filtered_image[np.where(cell_label == p.label)]

        index = np.argwhere(inten == np.nanmax(inten))

        if len(np.where(cell_label == p.label)[0]) < thr_area or np.nanmax(intensity) < thr_intensity_peak or \
                (intensity > np.nanmax(intensity)).sum() > 0:
            # judge the cell is isolated or neighboured
            if unique_bordering_cell_label.size == 0:
                # isolated situation - label of isolated segments are set to 0
                all_cell_label_column[all_cell_label_column == p.label] = 0

            if unique_bordering_cell_label.size > 0:
                dis_peak_int = np.zeros((unique_bordering_cell_label.size, 4), dtype='float')

                # neighbored situation
                for j in range(unique_bordering_cell_label.size):
                    nei_label = np.zeros((rows, cols), dtype='int')
                    index_nei_label = np.where(cell_label == int(unique_bordering_cell_label[j]))
                    nei_label[index_nei_label] = cell_label[index_nei_label]
                    neicell_prop = measure.regionprops(label_image=nei_label, intensity_image=filtered_image)

                    for p_nei in neicell_prop:
                        # use coords of peak intensity pixel insted weighted centroid
                        intensity_nei = filtered_image[nei_label == p_nei.label]
                        inten_nei = np.zeros((rows, cols), dtype=float)
                        inten_nei[np.where(cell_label == p_nei.label)] = filtered_image[np.where(cell_label == p_nei.label)]
                        index_nei = np.argwhere(inten_nei == np.nanmax(intensity_nei))

                        dis = mt.sqrt(mt.pow(index[0][0] - index_nei[0][0], 2) +
                                      mt.pow(index[0][1] - index_nei[0][1], 2))

                        peak_int = p_nei.intensity_max
                        area = p_nei.area

                    dis_peak_int[j, 0] = int(unique_bordering_cell_label[j])
                    dis_peak_int[j, 1] = dis
                    dis_peak_int[j, 2] = peak_int
                    dis_peak_int[j, 3] = area

                f_label_nei = dis_peak_int[np.argwhere(dis_peak_int[:, 1] == np.nanmin(dis_peak_int[:, 1])), 0]

                for i in range(len(f_label_nei)):
                    all_cell_label_column[all_cell_label_column == int(f_label_nei[i])] = p.label

    labels_cell = np.reshape(all_cell_label_column, (rows, cols), order='C')
    unique_labels = np.unique(labels_cell)

    for i in range(unique_labels.size):
        temp_index = np.where(labels_cell == unique_labels[i])
        cell_labels[temp_index] = unique_labels[i]

    # final processing step： cells with intensity with area or intensity less than a given threshold
    neighbours = get_neighbor(cell_labels, 8)
    cell_labels_column = neighbours[:, 0]

    prop_cell = measure.regionprops(label_image=cell_labels, intensity_image=filtered_image)

    for p_cell in prop_cell:
        f_cell_label = neighbours[cell_labels_column == p_cell.label]
        intensity_cell = filtered_image[cell_labels == p_cell.label]

        if len(np.where(cell_labels == p_cell.label)[0]) < thr_area or np.nanmax(intensity_cell) < thr_intensity_peak:
            cell_labels_column[cell_labels_column == p_cell.label] = 0

    f_label = np.reshape(cell_labels_column, (rows, cols), order='C')
    unique_label_f = np.unique(f_label)
    label_index = 0

    for i in range(unique_label_f.size):
        temp_index = np.where(f_label == unique_label_f[i])
        label_cells[temp_index] = label_index
        label_index = label_index + 1

    return label_cells
