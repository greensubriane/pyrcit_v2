# read DWD base format radar data by wradlib
import numpy as np
import wradlib as wrl
from IPython.core.display_functions import display
from wradlib.ipol import xr


def read_dwd_dx_radar_data(filename, radar_loc_x, radar_loc_y, radar_elev, radar_pol_range):
    radar_location = (radar_loc_x, radar_loc_y, radar_elev)
    # radar_location = (6.96454, 51.40643, 152), this location is for essen radr

    ranges = np.arange(0, radar_pol_range, 1000.0)

    # reading original radar data
    data_dBZ, metadata = wrl.io.read_dx(filename)
    # print(data_dBZ.shape)
    # print(metadata.keys())
    da = wrl.georef.create_xarray_dataarray(data_dBZ, r=ranges, phi=metadata["azim"], theta=metadata["elev"],
                                            site=radar_location, sweep_mode="azimuth_surveillance")

    da = da.wrl.georef.georeference()
    # display(da)

    # projection to UTM Zone32
    utm = wrl.georef.epsg_to_osr(32632)
    da_utm = da.wrl.georef.reproject(trg_crs=utm)

    # Clutter removing
    clutter = da.wrl.classify.filter_gabella(tr1=12, n_p=6, tr2=1.1)
    data_no_clutter = da.wrl.ipol.interpolate_polar(clutter)
    # Attenuation correction
    pia = data_no_clutter.wrl.atten.correct_attenuation_constrained(a_max=1.67e-4, a_min=2.33e-5,
                                                                    n_a=100, b_max=0.7, b_min=0.65, n_b=6,
                                                                    gate_length=1.,
                                                                    constraints=[wrl.atten.constraint_dbz,
                                                                                 wrl.atten.constraint_pia],
                                                                    constraint_args=[[59.0], [20.0]])
    data_attcorr = data_no_clutter + pia

    # transfer decibel into rainfall intensity
    data_Z = data_attcorr.wrl.trafo.idecibel()
    intensity = data_Z.wrl.zr.z_to_r(a=256., b=1.42)

    # gridding operation
    xgrid = np.linspace(intensity.x.min(), intensity.x.max(), 256)
    ygrid = np.linspace(intensity.y.min(), intensity.y.max(), 256)

    cart = xr.Dataset(coords={"x": (["x"], xgrid), "y": (["y"], ygrid)})


    gridded_ref = data_attcorr.wrl.comp.togrid(cart, radius=radar_pol_range,
                                               center=(intensity.x.mean(), intensity.y.mean()),
                                               interpol=wrl.ipol.Idw)
    gridded_intensity = intensity.wrl.comp.togrid(cart, radius=radar_pol_range,
                                                  center=(intensity.x.mean(), intensity.y.mean()),
                                                  interpol=wrl.ipol.Idw)

    gridded_refs = np.ma.masked_invalid(gridded_ref).reshape((len(xgrid), len(ygrid)))
    where_are_inf_ref = np.isinf(gridded_refs)
    gridded_refs[where_are_inf_ref] = 0

    gridded_intensities = np.ma.masked_invalid(gridded_intensity).reshape((len(xgrid), len(ygrid)))
    where_are_inf_intensity = np.isinf(gridded_intensities)
    gridded_intensities[where_are_inf_intensity] = 0

    return gridded_refs, gridded_intensities
