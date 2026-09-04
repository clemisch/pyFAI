import h5py

from .models import ImageIndices
from .utils import (
    get_axes_index,
    get_dataset_name,
    get_radial_dataset,
    get_signal_dataset,
)


class Point:

    def __init__(self,
                 indices: ImageIndices,
                 url_nxdata_path: str):
        self.indices = indices
        row = indices.row
        col = indices.col
        file_name, nxdata_path = url_nxdata_path.split("?")

        with h5py.File(file_name, "r") as h5file:
            nxdata = h5file[nxdata_path]
            intensity_dset = get_signal_dataset(h5file, nxdata_path, default="intensity")
            axes_index = get_axes_index(intensity_dset)
            selection = [slice(None)] * intensity_dset.ndim
            selection[axes_index.slow] = row
            selection[axes_index.fast] = col
            self._intensity_curve = intensity_dset[tuple(selection)]
            self._y_name = intensity_dset.attrs.get("long_name", "Intensity")

            uncertainty_name = intensity_dset.attrs.get("uncertainties")
            if isinstance(uncertainty_name, bytes):
                uncertainty_name = uncertainty_name.decode()
            if uncertainty_name is not None:
                if uncertainty_name.startswith("/"):
                    uncertainty_dset = h5file[uncertainty_name]
                else:
                    uncertainty_dset = nxdata[uncertainty_name]
            elif "errors" in nxdata:
                uncertainty_dset = nxdata["errors"]
            else:
                uncertainty_dset = None

            if uncertainty_dset is None:
                self._uncertainty_curve = None
            else:
                axes_index = get_axes_index(uncertainty_dset)
                selection = [slice(None)] * uncertainty_dset.ndim
                selection[axes_index.slow] = row
                selection[axes_index.fast] = col
                self._uncertainty_curve = uncertainty_dset[tuple(selection)]
                if self._uncertainty_curve.shape != self._intensity_curve.shape:
                    raise RuntimeError(
                        "Intensity and uncertainty curves have different shapes: "
                        f"{self._intensity_curve.shape} and "
                        f"{self._uncertainty_curve.shape}"
                    )

            radial_dset = get_radial_dataset(h5file,
                                             nxdata_path=nxdata_path,
                                             size=self._intensity_curve.size)
            self._radial_curve = radial_dset[()]
            self._x_name = get_dataset_name(radial_dset)
            self._x_unit = radial_dset.attrs.get(
                "unit", radial_dset.attrs.get("units")
            )

    def __repr__(self) -> str:
        return str(self.indices)

    def get_curve(self):
        return self._intensity_curve

    def get_radial_curve(self):
        return self._radial_curve

    def get_uncertainty_curve(self):
        return self._uncertainty_curve

    def get_x_name(self):
        return self._x_name

    def get_x_unit(self):
        return self._x_unit

    def get_y_name(self):
        return self._y_name
