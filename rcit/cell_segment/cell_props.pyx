# -*- coding: utf-8 -*-

#
# Licensed under the BSD-3-Clause license
# Copyright (c) 2020, Ting He
#

"""
Cython module for rain cell properties extraction used in Rain Cell Identification and TRacking (RCIT) Algorithm
"""

import pandas as pd
import numpy as np
cimport numpy as np
from skimage import measure
import math as mt

import cython
ctypedef np.float64_t float64

from sympy import symbols, log, erf, nsolve, sqrt, elliptic_e, pi

from .cell_segment_methods import segment_empirical
from .cell_segment_methods import segment_watershed
from .cell_segment_methods import conv_rain_cell_segmentation


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def excell_model_func(float64 r2,
                      area,
                      peak_r,
                      rms_r,
                      ellip,
                      grad_mean,
                      rms_grad
                      ):
    """
    EXCELL rain field model, cited from C. Capsoni et al, 1987,
    "Data and theory for a new model of the horizontal structure of rain cells for propagation applications"
    :param r2:
    :param area:
    :param peak_r:
    :param rms_r:
    :param ellip:
    :param grad_mean:
    :param rms_grad:
    :return:
    """

    re = symbols("re")
    params_excell = []

    if len(area) == len(peak_r) == len(rms_r) == len(ellip) == len(grad_mean) == len(rms_grad):
       for i in range(len(area)):
           eq1 = 2/(log(re/r2)**2)*(re-r2*(1+log(re/r2)))-area[i]

           eq2 = 2/(log(re/r2)**2)*(re**2-r2**2*(1+2*log(re/r2)))-rms_r[i]**2

           eq3 = (sqrt((2*ellip[i])/(pi*area[i]*log(re/r2)**2))*
                  elliptic_e(sqrt(1-ellip[i]**2))*(re-r2*(1+log(re/r2)))-grad_mean[i])

           eq4 = pi/(4*area[i])*(ellip[i]+1/ellip[i])*(re**2-r2**2*(1+2*log(re/r2)))-rms_grad[i]**2

           eqs = [eq1, eq2, eq3, eq4]
           result = nsolve(eqs, [re], [peak_r[i]], verify=False)

           r_max = result[0]
           be = sqrt(area[i]/(pi*ellip[i]*log(result[0]/r2)**2))
           ae = be*ellip[i]

           entry_params = [np.round(float(r_max), 2), np.round(float(ae), 2), np.round(float(be), 2)]
           params_excell.append(entry_params)

    model_params = pd.DataFrame(params_excell, columns=['peak rate', 'short axis', 'long axis'])

    return params_excell, model_params


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dexcell_model_func(float64 r2,
                       float64 area,
                       float64 peak_r,
                       float64 r_mean,
                       float64 rms_r,
                       float64 ellip,
                       float64 grad_mean,
                       float64 rms_grad,
                       float64 Rms_ini,
                       float64 Rth_ini
                       ):

    Rms, Rth = symbols("Rms Rth", real=True)
    Rm = peak_r

    eq1 = (log(Rms/Rth)**2/log(Rm/Rth)**2*(Rm-Rth*(1+log(Rm/Rth)))/((1/2)*log(Rms/r2)**2)+
           (Rth*(1+log(Rms/Rth))-r2*(1+log(Rms/r2)))/((1/2)*log(Rms/r2)**2)-r_mean)

    eq2 = (log(Rms/Rth)**2/log(Rm/Rth)**2*((Rm**2-Rth**2*(1+2*log(Rm/Rth)))/(2*log(Rms/r2)**2))+
           (Rth**2*(1+2*log(Rms/Rth))-r2**2*(1+2*log(Rms/r2)))/(2*log(Rms/r2)**2)-rms_r**2)

    eq3 = (4*elliptic_e(sqrt(1-ellip**2))/(log(Rms/r2)*sqrt(pi*ellip*area))*
           (log(Rms/Rth)/log(Rm/Rth)*(Rm-Rth)-r2*(1+log(Rms/r2)))-grad_mean)

    eq4 = pi*(ellip+1/ellip)/(4*area)*(Rm**2-Rth*(1+2*log(Rm/Rth))+
                                       Rth**2*(1+2*log(Rms/Rth))-r2**2*(1+2*log(Rms/r2)))-rms_grad**2

    eqs = [eq1, eq2, eq3, eq4]

    result = nsolve(eqs, [Rms, Rth], [Rms_ini, Rth_ini], verify=False)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def hycell_model_func(float64 r2,
                      float64 area,
                      float64 peak_r,
                      float64 r_mean,
                      float64 rms_r,
                      float64 ellip,
                      float64 grad_mean,
                      float64 rms_grad,
                      float64 re_ini,
                      float64 r1_ini):

    re, r1 = symbols("re r1", real=True)
    rg = peak_r

    eq1 = log(re/r2)**(-2)*(log(re/r1)**2*log(rg/r1)**(-1)*(rg-r1)+2*r1*(1+log(re/r1))-2*r2*(1+log(re/r2)))-r_mean

    eq2 = log(re/r2)**(-2)*(log(rg/r1)**(-1)*log(re/r1)**2*(rg**2-r1**2)+
                            r1**2*(1+2*log(re/r1))-r2**2*(1+2*log(re/r2)))-2*rms_r**2

    eq3 = (4*elliptic_e(sqrt(1-ellip**2))/(sqrt(area*pi*ellip)*log(re/r2))*
           (rg*sqrt(pi)/2*erf(sqrt(log(rg/r1)))*log(re/r1)*log(rg/r1)**(-1/2)+r1-r2*(1+log(re/r2)))-grad_mean)

    eq4 = pi/2*(ellip+1/ellip)*(rg**2-r1**2*(1+2*log(rg/r1))+r1**2/2*(1+2*log(re/r1))-
                                r2**2/2*(1+2*log(re/r2)))-area*rms_grad**2

    eqs = [eq1, eq2, eq3, eq4]

    result = nsolve(eqs, [re, r1], [re_ini, r1_ini], verify=False)

    return result


@cython.cdivision(True)
cdef inline ref_from_labelled_cell(label_cell,
                                   int label_num,
                                   np.ndarray[float64, ndim=2] filtered_image):

    cdef int row = filtered_image.shape[0]
    cdef int col = filtered_image.shape[1]
    cdef np.ndarray[float64, ndim=2] ref_cell = np.zeros((row, col), dtype=np.float)
    ref_cell[label_cell == label_num] = filtered_image[label_cell == label_num]

    return ref_cell


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# compute region properties
def get_props_modelled_cell(np.ndarray[float64, ndim=2] intensity_image,
                            label_cell):

    # np.ndarray[int, ndim=2] label_cell

    cell_props = []
    hycell_props = []
    append_cell_props = cell_props.append
    append_hycell_props = hycell_props.append
    cells_props = measure.regionprops(label_image=label_cell, intensity_image=intensity_image)


    for p in cells_props:
        rain_rate = p.image_intensity.reshape(p.image_intensity.shape[0]*p.image_intensity.shape[1], 1)

        dif_h = np.asarray(np.gradient(p.image_intensity)[1])
        dif_v = np.asarray(np.gradient(p.image_intensity)[0])

        square_g = dif_h**2 + dif_v**2
        squareroot_g =  np.sqrt(square_g)

        entry_rainy = [p.label,
                       p.area,
                       p.centroid_weighted,
                       p.coords,
                       p.bbox,
                       [p.bbox[0]-10,p.bbox[1]-10,p.bbox[2]+10,p.bbox[3]+10]]

        entry_hycell = [p.label,
                        p.area,
                        np.round(p.intensity_max, 2),
                        np.round(p.axis_minor_length/p.axis_major_length, 2),
                        np.round(p.intensity_mean, 2),
                        np.round(mt.sqrt(sum([x ** 2 for x in rain_rate])/len(rain_rate)), 2),
                        np.round(np.mean(squareroot_g[:]), 2),
                        np.round(mt.sqrt(np.sum(square_g[:])/(square_g.shape[0]*square_g.shape[1])), 2),
                        np.round(p.axis_major_length)]

        append_cell_props(entry_rainy)
        append_hycell_props(entry_hycell)

        # get the sizes for each of the remaining objects and store in dataframe
    df_cells = pd.DataFrame(hycell_props, columns=['label',
                                                   'area',
                                                   'peak_r',
                                                   'e',
                                                   'a_r',
                                                   'rms_r',
                                                   'a_grad',
                                                   'rms_grad',
                                                   'long_a'])

    return cell_props, hycell_props, df_cells

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# compute region properties of onvective cells , which is only valied for RCIT and Watershed algorithm
def get_props_conv_cell(np.ndarray[float64, ndim=2] intensity_image,
                        label_cell,
                        seg_type,
                        float64 thr_conv_intensity,
                        float64 thr_area,
                        float64 intensity_diff = 10
                        ):

    cell_props_conv = []
    append_cell_props_conv = cell_props_conv.append

    # acquiring the segment method. segment type: 'RCIT': RCIT algorithm, 'WaterShed': watershed algorithm
    # label_cell = rain_cell_modelling(intensity_image, thr_intensity, thr_area)
    cells_props = measure.regionprops(label_image=label_cell, intensity_image=intensity_image)

    for p in cells_props:
        # rain_rate = p.image_intensity
        # rain_intensity = rain_rate.reshape((rain_rate.shape[0]*rain_rate.shape[1], 1))

        entry_rainy = [p.label,
                       p.area,
                       p.intensity_max,
                       p.centroid_weighted,
                       p.coords,
                       p.bbox,
                       [p.bbox[0]-10,p.bbox[1]-10,p.bbox[2]+10,p.bbox[3]+10]]

        # get the intensity image with only segmented rain cells,
        ref_cell = ref_from_labelled_cell(label_cell, p.label, intensity_image)

        # for each cell from RCIT or Watershed, get the convective cells and their properties
        segment_rule = {'RCIT': segment_empirical, 'WaterShed': segment_watershed}
        method = segment_rule.get(seg_type)
        label_cell_conv_ini = method(ref_cell, thr_conv_intensity, thr_area)
        prop_label_cell_conv_ini = measure.regionprops(label_image=label_cell_conv_ini)

        label = 1
        for p1 in prop_label_cell_conv_ini:
            ref_cell_ini = ref_from_labelled_cell(label_cell_conv_ini, p1.label, intensity_image)

            # get convective rain cells by the method similar to the 'Trace3D' algorithm
            label_cell_conv_second = conv_rain_cell_segmentation(ref_cell_ini, seg_type)

            prop_label_cell_conv_second =  measure.regionprops(label_image=label_cell_conv_second,
                                                               intensity=intensity_image)

            # indexing the properties of convective cells identified in the rainy cells
            for p2 in prop_label_cell_conv_second:
                if p2.label > 0 and p2.max_intensity - intensity_diff >= thr_conv_intensity:
                    # rain_rate = p2.image_intensity
                    # rain_intensity = rain_rate.reshape((rain_rate.shape[0] * rain_rate.shape[1], 1))

                    entry_conv_second = [p2.label,
                                         p2.area,
                                         p2.intensity_max,
                                         p2.centroid_weighted,
                                         p2.coords,
                                         p2.bbox,
                                         [p2.bbox[0]-10,p2.bbox[1]-10,p2.bbox[2]+10,p2.bbox[3]+10]]
                    label = label + 1
                append_cell_props_conv(entry_conv_second)

    # same with rainy cells, but for the convective rain cells
    df_conv_cells = pd.DataFrame(cell_props_conv,columns=['label',
                                                          'area',
                                                          'peak_r',
                                                          'e',
                                                          'a_r',
                                                          'rms_r',
                                                          'a_grad',
                                                          'rms_grad',
                                                          'long_a'])


    return cell_props_conv