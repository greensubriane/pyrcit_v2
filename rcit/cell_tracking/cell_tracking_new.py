import numpy as np
import sys
import pandas as pd
import csv
import wradlib as wrl
import copy
import warnings

from rcit.motion.global_motion_estimation import global_motion_generation_piv
from rcit.radar_data_io.read_dx_data import read_dwd_dx_radar_data

from rcit.util.pre_process.pre_process import input_filter
from rcit.cell_segment.cell_props import get_props_modelled_cell
from rcit.cell_segment.cell_props import excell_model_func, dexcell_model_func, hycell_model_func
from rcit.cell_segment.cell_segment_methods import rain_cell_modelling

from rcit.util.general_methods import global_motion_static

import math as mt
from ismember import ismember

# set forecast dataset, totally we choose 27 radar images, the first three is used for global motion calculation,
# and the next 24 is used for 2-hr nowcasting and correspondly verification
# 2007-05-26, 0-27(00:00); 244-271(20:20); 252-279(21:00).
# 2008-07-19 192-219(16:00);
# 2008-07-26 0-27(00:00); 192-219(16:00); 217-244(18:05).

f = open("/Users/heting/Documents/IDEAproject/pyrcit_v2/070526/ein_bild.dat")
lines = f.readlines()
lines1 = lines[244:271]
radar_images = np.full((27, 256, 256), np.nan)
radar_images_filtered = np.full((27, 256, 256), np.nan)
radar_images_intensities = np.full((27, 256, 256), np.nan)
labeled_rain_cells = np.zeros((27, 256, 256), dtype='int')

# property list of identified rain cell
cell_label = []
cell_area = []
cell_center = []
cell_coord = []
cell_bbox = []
cell_bbox_boun = []

# excuation of segmentation pocedure for the selected radar images
# and the EXCELL rainfall model is applied to model the segmented rain cells

t = 0
r2 = 5

for lines2 in lines1:
    line = lines2.split('\n')
    line1 = lines2.split('.ess\n')
    print(line1[0][2:])
    print(line[0])

    reflectivity, intensity = read_dwd_dx_radar_data("/Users/heting/Documents/IDEAproject/pyrcit_v2/070526/" +
                                                     line[0], 6.964, 51.4064,
                                                     152, 128000)
    origin_ref, filter_ref = input_filter(reflectivity, 'cf')
    origin_intensity, filter_intensity = input_filter(intensity, 'cf')
    filter_intensity[np.isnan(filter_intensity)] = 0
    intensity[np.isnan(intensity)] = 0
    filter_intensity_1 = copy.deepcopy(filter_intensity)
    filter_intensity_2 = copy.deepcopy(filter_intensity)

    label_modelled_cell = rain_cell_modelling(filter_intensity_2, r2, 9, 35)

    radar_images[t, :, :] = reflectivity
    radar_images_filtered[t, :, :] = filter_intensity_1
    radar_images_intensities[t, :, :] = intensity
    labeled_rain_cells[t, :, :] = label_modelled_cell

    cell_props, hycell_props, df_cells = get_props_modelled_cell(filter_intensity_1, label_modelled_cell)
    print(df_cells)

    label = [temp[0] for temp in cell_props]
    area = [temp[1] for temp in cell_props]
    peak_r = [temp[2] for temp in hycell_props]
    center = [temp[2] for temp in cell_props]
    coords = [temp[3] for temp in cell_props]
    bbox = [temp[4] for temp in cell_props]
    bbox_search = [temp[5] for temp in cell_props]
    ellip = [temp[3] for temp in hycell_props]
    r_mean = [temp[4] for temp in hycell_props]
    rms_r = [temp[5] for temp in hycell_props]
    grad_mean = [temp[6] for temp in hycell_props]
    rms_grad = [temp[7] for temp in hycell_props]
    long_axis = [temp[8] for temp in hycell_props]

    '''
    print(label)
    print(area)
    print(center)
    print(coords)
    print(bbox)
    print(bbox_search)
    '''

    params_excell, model_params = excell_model_func(r2, area, peak_r, rms_r, ellip, grad_mean, rms_grad)

    print(model_params)

    cell_label.append(label)
    cell_area.append(area)
    cell_center.append(center)
    cell_coord.append(coords)
    cell_bbox.append(bbox)
    cell_bbox_boun.append(bbox_search)

    radar_images_intensities[t, :, :][labeled_rain_cells[t, :, :]==0]=0

    t += 1



time_list = 27
'''
# Global motion estimation with pyRCIT algorithm

global_vectors_RCIT = [[] for i in range(0, time_list)]
vectors_RCIT = np.zeros((time_list, 2), dtype=float)

# global_vectors_VET = [[] for i in range(0, time_list)]
# vectors_VET = np.zeros((time_list, 2), dtype=float)
# factors = [2, 4, 8, 16, 32]

time_lists = np.arange(0, time_list-1)

for i in time_lists:
    print(i)
    i = int(i)

    location_motion_RCIT, motion_RCIT = global_motion_generation_piv(radar_images_filtered[i], radar_images_filtered[i+1], 16, 16, 0.5, 0.5, 5, 5, [256, 256], 0.1, 5)
    wind_direction_bins_rcit, mean_speed_rcit = global_motion_static(motion_RCIT)

    print(motion_RCIT.shape)
    global_vectors_RCIT[i] = motion_RCIT
    vectors_RCIT[i, 0] = wind_direction_bins_rcit
    vectors_RCIT[i, 1] = mean_speed_rcit

    print('prevailing wind direction with rcit is: ', vectors_RCIT[i, 0], 'mean speed with rcit is: ', vectors_RCIT[i, 1])
'''

def get_child_rain_cell(time_step,
                        cell_label,
                        cell_coords,
                        cell_boundary_box,
                        cell_boundary_box_search_box,
                        cell_center,
                        cell_area
):
    candidate_child_cells_center = [[] for x in range(0, time_step)]
    candidate_child_cells_area = [[] for x in range(0, time_step)]
    candidate_child_cells_label = [[] for x in range(0, time_step)]
    candidate_child_cells_coords = [[] for x in range(0, time_step)]

    cells_boundary_box_next_time = [[] for x in range(0, time_step + 1)]
    cells_center_next_time = [[] for x in range(0, time_step + 1)]
    cells_area_next_time = [[] for x in range(0, time_step + 1)]
    cells_label_next_time = [[] for x in range(0, time_step + 1)]
    cells_coords_next_time = [[] for x in range(0, time_step + 1)]

    cells_boundary_box_next_time[0:time_step] = cell_boundary_box
    cells_center_next_time[0:time_step] = cell_center
    cells_area_next_time[0:time_step] = cell_area
    cells_label_next_time[0:time_step] = cell_label
    cells_coords_next_time[0:time_step] = cell_coords

    for i1 in range(time_step):
        print(cell_label[i1])
        print(cell_area[i1])

        boundary_box_next_time = cells_boundary_box_next_time[i1 + 1]
        center_next_time = cells_center_next_time[i1 + 1]
        area_next_time = cells_area_next_time[i1 + 1]
        label_next_time = cells_label_next_time[i1 + 1]
        coords_next_time = cells_coords_next_time[i1 + 1]

        boundary_box_search_box = cell_boundary_box_search_box[i1]
        rows = len(boundary_box_search_box)

        actual_cells_center = [[] for x in range(0, rows)]
        actual_cells_area = [[] for x in range(0, rows)]
        actual_cells_label = [[] for x in range(0, rows)]
        actual_cells_coords = [[] for x in range(0, rows)]

        for i2 in range(rows):

            boundary_box_search_box_x_min = boundary_box_search_box[i2][0]
            boundary_box_search_box_x_max = boundary_box_search_box[i2][2]

            boundary_box_search_box_y_min = boundary_box_search_box[i2][1]
            boundary_box_search_box_y_max = boundary_box_search_box[i2][3]

            if boundary_box_next_time is not None:
                rows_1 = len(boundary_box_next_time)
                candidate_child_cell_center = [[] for x in range(0, rows_1)]
                candidate_child_cell_area = [[] for x in range(0, rows_1)]
                candidate_child_cell_label = [[] for x in range(0, rows_1)]
                candidate_child_cell_coords = [[] for x in range(0, rows_1)]

                for i3 in range(rows_1):

                    boundary_box_next_time_x_min = boundary_box_next_time[i3][0]
                    boundary_box_next_time_x_max = boundary_box_next_time[i3][2]

                    boundary_box_next_time_y_min = boundary_box_next_time[i3][1]
                    boundary_box_next_time_y_max = boundary_box_next_time[i3][3]

                    if (
                            boundary_box_search_box_x_max <= boundary_box_next_time_x_min or
                            boundary_box_next_time_x_max <= boundary_box_search_box_x_min) and \
                            (boundary_box_search_box_y_max <= boundary_box_next_time_y_min or
                             boundary_box_next_time_y_max <= boundary_box_search_box_y_min):
                        candidate_child_cell_center[i3] = (0, 0)
                        candidate_child_cell_area[i3] = 0
                        candidate_child_cell_label[i3] = 0
                        candidate_child_cell_coords[i3] = []

                    else:
                        candidate_child_cell_center[i3] = center_next_time[i3]
                        candidate_child_cell_area[i3] = area_next_time[i3]
                        candidate_child_cell_label[i3] = label_next_time[i3]
                        candidate_child_cell_coords[i3] = coords_next_time[i3]
            else:
                candidate_child_cell_center = []
                candidate_child_cell_area = []
                candidate_child_cell_label = []
                candidate_child_cell_coords = []

            candidate_child_cell_label_1 = [i for i in candidate_child_cell_label if i != 0]
            candidate_child_cell_area_1 = [i for i in candidate_child_cell_area if i != 0]
            candidate_child_cell_center_1 = [i for i in candidate_child_cell_center if i != (0, 0)]
            candidate_child_cell_coords_1 = [i for i in candidate_child_cell_coords if i != []]

            actual_cells_center[i2] = candidate_child_cell_center_1
            actual_cells_area[i2] = candidate_child_cell_area_1
            actual_cells_label[i2] = candidate_child_cell_label_1
            actual_cells_coords[i2] = candidate_child_cell_coords_1

        candidate_child_cells_center[i1] = actual_cells_center
        candidate_child_cells_area[i1] = actual_cells_area
        candidate_child_cells_label[i1] = actual_cells_label
        candidate_child_cells_coords[i1] = actual_cells_coords

    return candidate_child_cells_label, candidate_child_cells_center, \
        candidate_child_cells_area, candidate_child_cells_coords


'''
def get_most_likely_child_rain_cell(
        time_step, cell_center, cell_label, cell_coords, cell_area, candidate_child_cells_center,
        candidate_child_cells_label, candidate_child_cells_coords, candidate_child_cells_area, global_vectors,
        angle_1, angle_2, distance_coefficient
):
    most_likely_child_cells = [[] for x in range(0, time_step)]

    for i in range(time_step - 1):
        center = cell_center[i]
        area = cell_area[i]
        label = cell_label[i]
        coords = cell_coords[i]
        child_cell_labels = candidate_child_cells_label[i]
        child_cell_center = candidate_child_cells_center[i]
        child_cell_coords = candidate_child_cells_coords[i]
        child_cell_area = candidate_child_cells_area[i]
        most_likely_child_cell = [[] for x in range(0, len(label))]

        for i1 in range(len(label)):
            number_child_cells = len(child_cell_labels[i1])

            # determination of most likely child rain cells from the distance and
            # angle difference (for the case of single child rain cells)
            if number_child_cells == 1:
                a = np.asarray(child_cell_coords[i1][number_child_cells - 1])
                b = np.asarray(coords[i1])
                Iloc, idx = ismember(a, b, 'rows')
                a[Iloc] == b[idx]
                overlap = len(a[Iloc]) / (a.shape[0] + b.shape[0] - len(a[Iloc]))
                dis_horizontal = center[i1][0] - child_cell_center[i1][number_child_cells - 1][0]
                dis_vertical = center[i1][1] - child_cell_center[i1][number_child_cells - 1][1]
                direction = mt.atan2(dis_vertical, dis_horizontal) * 180 / np.pi + 180
                speed = mt.sqrt(mt.pow(dis_horizontal, 2) + mt.pow(dis_vertical, 2))
                if overlap > 0 or (overlap == 0 and
                                   (mt.fabs(direction - global_vectors[i, 0])) <= angle_1 and
                                   speed <= distance_coefficient * global_vectors[i, 1]):
                    most_likely_child_cell[i1] = child_cell_labels[i1][number_child_cells - 1]
                    # most_likely_child_cell[i1] = [child_cell_center[i1][number_child_cells - 1],
                    #                               child_cell_labels[i1][number_child_cells - 1]]
                else:
                    most_likely_child_cell[i1] = [0]
                    # most_likely_child_cell[i1] = [0, 0, 0]

            # determination of most likely child rain cells from the distance,
            # angle and area difference (for the case of multi child rain cells)
            elif number_child_cells > 1:
                area_difference = []
                temp_likely_child_cell = []
                temp_likely_child_cell_1 = []

                for i2 in range(number_child_cells):
                    a_1 = np.asarray(child_cell_coords[i1][i2])
                    b_1 = np.asarray(coords[i1])
                    Iloc_1, idx_1 = ismember(a_1, b_1, 'rows')
                    a_1[Iloc_1] == b_1[idx_1]
                    overlap_1 = len(a_1[Iloc_1]) / (a_1.shape[0] + b_1.shape[0] - len(a_1[Iloc_1]))
                    dis_horizontal_1 = center[i1][0] - child_cell_center[i1][i2][0]
                    dis_vertical_1 = center[i1][1] - child_cell_center[i1][i2][1]
                    direction_1 = mt.atan2(dis_vertical_1, dis_horizontal_1) * 180 / np.pi + 180
                    speed_1 = mt.sqrt(mt.pow(dis_horizontal_1, 2) + mt.pow(dis_vertical_1, 2))
                    if overlap_1 > 0 or (overlap_1 == 0 and
                                         (mt.fabs(direction_1 - global_vectors[i, 0]) <= angle_2
                                          and speed_1 <= distance_coefficient * global_vectors[i, 1])):
                        most_likely_child_cell[i1] = child_cell_labels[i1][i2]

                        # most_likely_child_cell[i1] = [child_cell_center[i1][i2][0],
                        #                               child_cell_center[i1][i2][1],
                        #                               child_cell_labels[i1][i2]]

                    if overlap_1 == 0 and (mt.fabs(direction_1 - global_vectors[i, 0]) > angle_2 or
                                           speed_1 > distance_coefficient * global_vectors[i, 1]):
                        temp_likely_child_cell.append(child_cell_labels[i1][i2])
                        # temp_likely_child_cell.append([child_cell_center[i1][i2][0],
                        #                                child_cell_center[i1][i2][1],
                        #                                child_cell_labels[i1][i2]])
                        area_difference.append(mt.fabs(area[i1] - child_cell_area[i1][i2]))

                if area_difference:
                    for i3 in range(len(area_difference)):
                        if area_difference[i3] == min(area_difference):
                            temp_likely_child_cell_1.append(temp_likely_child_cell[i3])
                        else:
                            most_likely_child_cell[i1] = 0
                            # most_likely_child_cell[i1] = [0, 0, 0]

                    most_likely_child_cell[i1] = temp_likely_child_cell_1

        most_likely_child_cells[i] = most_likely_child_cell

    return most_likely_child_cells
'''

# get the trajectory of rain cell by the VET algorithm
all_child_cell_label, all_child_cell_center, all_child_cell_area, all_child_cell_coord = get_child_rain_cell(time_list, cell_label, cell_coord, cell_bbox, cell_bbox_boun, cell_center, cell_area)

# print(all_child_cell_label)
# print(all_child_cell_area)
# print(all_child_cell_center)
# print(all_child_cell_coord)

# child_cells = get_most_likely_child_rain_cell(time_list, cell_center, cell_label, cell_coord, cell_area, all_child_cell_center, all_child_cell_label, all_child_cell_coord, all_child_cell_area, vectors_RCIT, 40, 20, 4)

'''
for i in range(time_list):
    line1 = lines1[i].split('.ess\n')
    print(len(child_cells[i]))
    for j in range(len(child_cells[i])):
        print('Child cell Id for cell ' + str(j + 1) + ' at time ' + line1[0][2:] + ' is: ', child_cells[i][j])
'''
