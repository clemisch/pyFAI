#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023-2026 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Loïc Huder (loic.huder@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""Tool to visualize diffraction maps."""
from __future__ import annotations

__authors__ = ["Loïc Huder", "E. Gutierrez-Fernandez", "Jérôme Kieffer"]
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/01/2026"
__status__ = "development"

import json
import logging
import os.path
import posixpath
from string import digits

import h5py
import numpy
from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.plot.items.image import ImageBase
from silx.image.marchingsquares import find_contours

from ...io.diffmap_config import DiffmapConfig
from ...io.integration_config import WorkerConfig
from ...utils.mathutil import binning
from .models import ImageIndices
from .point import Point
from .utils import (
    compute_radial_values,
    get_axes_dataset,
    get_axes_index,
    get_dataset,
    get_indices_from_values,
    get_mask_image,
    get_radial_dataset,
    get_signal_dataset,
)
from .widgets.DiffractionImagePlotWidget import DiffractionImagePlotWidget
from .widgets.IntegratedPatternPlotWidget import IntegratedPatternPlotWidget
from .widgets.MapPlotWidget import MapPlotWidget
from .widgets.RietveldRefinementWidget import (
    RietveldRefinementThread,
    RietveldRefinementWidget,
)
from .widgets.TitleWidget import TitleWidget

logger = logging.getLogger(__name__)


class MainWindow(qt.QMainWindow):
    sigFileChanged = qt.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._file_name: str | None = None

        self.setWindowTitle("PyFAI-diffmap viewer")

        self._image_plot_widget = DiffractionImagePlotWidget(self)
        self._image_plot_widget.setDefaultColormap(
            Colormap("gray", normalization="log")
        )
        self._image_plot_widget.setKeepDataAspectRatio(True)
        self._image_plot_widget.plotClicked.connect(self.onMouseClickOnImage)

        self._map_plot_widget = MapPlotWidget(self)
        self._map_plot_widget.clearPointsSignal.connect(self.clearPoints)
        self._map_plot_widget.setDefaultColormap(
            Colormap("viridis", normalization="log")
        )
        self._map_plot_widget.plotClicked.connect(self.selectMapPoint)
        self._map_plot_widget.pinContextEntrySelected.connect(self.fixMapPoint)
        self._map_plot_widget.setBackgroundClicked.connect(self.setNewBackgroundCurve)

        self.sigFileChanged.connect(self._map_plot_widget.onFileChange)

        self._integrated_plot_widget = IntegratedPatternPlotWidget(self)
        self._integrated_plot_widget.roi.sigRegionChanged.connect(self.onRoiEdition)
        self._integrated_plot_widget.roi.sigRegionChanged.connect(self.drawContoursOnImage)

        self._title_widget = TitleWidget(self)

        self._refinement_widget = RietveldRefinementWidget(self)
        refinement_dock = qt.QDockWidget("Rietveld refinement", self)
        refinement_dock.setWidget(self._refinement_widget)
        self.addDockWidget(qt.Qt.RightDockWidgetArea, refinement_dock)
        self._refinement_widget.refinementRequested.connect(
            self.runRietveldRefinement
        )

        self._central_widget = qt.QWidget()
        layout = qt.QGridLayout(self._central_widget)
        layout.setSpacing(0)
        layout.addWidget(self._title_widget, 0, 0, 1, 2)
        layout.addWidget(self._image_plot_widget, 1, 0, 2, 1)
        layout.addWidget(self._map_plot_widget, 1, 1)
        layout.addWidget(self._integrated_plot_widget, 2, 1)
        self._central_widget.setLayout(layout)
        self.setCentralWidget(self._central_widget)

        self._unfixed_indices = None
        self._fixed_indices = set()
        self._background_point = None
        self._refinement_thread = None
        self.worker_config = None

        # declaration of instance variables
        self._map_ptr = None # This is the map of the indices of input frame

    def initData(self,
                 file_name: str,
                 dataset_path: str="/entry_0000/measurement/images_0001",
                 nxprocess_path: str="/entry_0000/pyFAI",
                 ):

        self._file_name = os.path.abspath(file_name)
        self._dataset_paths = {}
        self._nxprocess_path = nxprocess_path

        self.sigFileChanged.emit(self._file_name)

        with h5py.File(self._file_name, "r") as h5file:
            nxprocess = h5file[self._nxprocess_path]
            nxdata = nxprocess["result"]
            map_dataset = get_signal_dataset(nxdata, default="intensity")
            axes_index = get_axes_index(map_dataset)
            map_data  = map_dataset[()].sum(axis=axes_index.radial)
            try:
                slow = get_axes_dataset(nxdata, dim=axes_index.slow, default="slow")
            except (KeyError, RuntimeError):
                slow_label = slow_values = None
            else:
                slow_label = slow.attrs.get("long_name", "Y")
                slow_values = slow[()]
            try:
                fast = get_axes_dataset(nxdata, dim=axes_index.fast, default="fast")
            except (KeyError, RuntimeError):
                fast_values = fast_label = None
            else:
                fast_label = fast.attrs.get("long_name", "X")
                fast_values = fast[()]

            pyFAI_config_as_str = get_dataset(
                parent=nxprocess,
                path="configuration/data")[()]
            pyFAI_config_as_dict = json.loads(pyFAI_config_as_str)
            if "diffmap_config_version" in pyFAI_config_as_dict:
                diffmap_config = DiffmapConfig.from_dict(pyFAI_config_as_dict, inplace=True)
                self.worker_config = diffmap_config.ai
            else:
                self.worker_config = WorkerConfig.from_dict(pyFAI_config_as_dict, inplace=True)

            radial_dset = get_radial_dataset(nxdata, size=self.worker_config.nbpt_rad)
            radial_values = radial_dset[()]
            delta_radial = (radial_values[-1] - radial_values[0]) / len(radial_values)

            if "offset" in nxprocess:
                self._offset = nxprocess["offset"][()]
            else:
                self._offset = 0

            try:
                self._map_ptr = get_dataset(nxdata, "map_ptr")[()]
            except (KeyError, RuntimeError):
                logger.warning("No `map_ptr` dataset in NXdata: guessing the frame indices !")
                self._map_ptr = numpy.arange(self._offset, self._offset + map_data.size)
                self._map_ptr.shape = map_data.shape

            _dataset_path = dataset_path.rstrip(digits)
            path, base = posixpath.split(_dataset_path)

            try:
                image_grp = h5file[path]
            except KeyError:
                self.warning(f"Cannot access diffraction images at {path}: no such path.")
            else:
                if isinstance(image_grp, h5py.Group):
                    lst = []
                    for key in image_grp:
                        try:
                            ds = image_grp[key]
                        except KeyError:
                            self.warning(f"Cannot access diffraction images at {path}/{key}: not a valid dataset.")
                        else:
                            if key.startswith(base) and isinstance(ds, h5py.Dataset):
                                lst.append(key)

                    lst.sort()
                    for key in lst:
                        self._dataset_paths[posixpath.join(path, key)] = len(image_grp[key])
                else:
                    self.warning(f"Cannot access diffraction images at {path}: not a group.")

        self._radial_matrix = compute_radial_values(self.worker_config)
        self._delta_radial_over_2 = delta_radial / 2

        wavelength = self.worker_config.poni.wavelength
        if wavelength is not None:
            self._refinement_widget.setWavelength(wavelength * 1e10)
        self._refinement_widget.setRadialRange(
            float(radial_values[0]), float(radial_values[-1])
        )
        cif_directory = os.path.dirname(self._file_name)
        cifs = sorted(
            os.path.join(cif_directory, filename)
            for filename in os.listdir(cif_directory)
            if filename.lower().endswith(".cif")
        )
        self._refinement_widget.setCifPaths(cifs)
        self._refinement_widget.setStatus("Ready")

        self._title_widget.setText(os.path.basename(file_name))
        self._map_plot_widget.setScatterData(map_data, fast_values, slow_values, fast_label, slow_label)
        # BUG: selectMapPoint(0, 0) does not work at first render cause the picking fails
        initial_indices = ImageIndices(0, 0)
        self._unfixed_indices = initial_indices
        self.displayPatternAtIndices(initial_indices, legend="INTEGRATE")
        self.displayImageAtIndices(initial_indices)
        self._map_plot_widget.addMarker(
            0,
            0,
            color=self.getCurveColor(legend="INTEGRATE"),
            symbol="o",
            legend="MAP_LOCATION",
        )

    def getRoiRadialRange(self) -> tuple[float | None, float | None]:
        return self._integrated_plot_widget.roi.getRange()

    def displayPatternAtIndices(self,
                                indices: ImageIndices,
                                legend: str,
                                color: str=None):
        if self._file_name is None:
            return
        point = Point(indices,
                      url_nxdata_path=f"{self._file_name}?{self._nxprocess_path}/result"
        )

        if self._background_point:
            curve = point.get_curve() - self._background_point.get_curve()
        else:
            curve = point.get_curve()

        self._integrated_plot_widget.addCurve(
            x=point.get_radial_curve(),
            y=curve,
            legend=legend,
            selectable=False,
            resetzoom=self._integrated_plot_widget.getGraphXLimits() == (0, 100),
        )
        self._integrated_plot_widget.setGraphXLabel(point.get_x_name())
        self._integrated_plot_widget.setGraphYLabel(point.get_y_name())

    def getMask(self, image, maskfile=None):
        """returns a 2D array of boolean with invalid pixels masked,
        combination of Detector mask, static & dynamic mask.
        Handles detector/mask image binning on the fly

        :param image: 2D array image with data, used for dynamic masking
        :param maskfile: filename or URL pointing to a static mask
        :return: 2D array
        """
        if maskfile:
            mask_image = get_mask_image(maskfile, image.shape)
        else:
            mask_image = None

        detector = self.worker_config.poni.detector
        if not detector:
            return mask_image

        detector_mask = detector.mask
        if detector.shape != image.shape:
            detector.guess_binning(image)
            detector_mask = binning(detector_mask, detector.binning)

        if mask_image is None:
            detector.mask = detector_mask
        elif detector_mask is None:
            detector.mask = mask_image
        else:
            detector.mask = numpy.logical_or(mask_image, detector_mask)

        return detector.dynamic_mask(image)

    def displayImageAtIndices(self, indices: ImageIndices):
        if self._file_name is None:
            return
        row = indices.row
        col = indices.col

        with h5py.File(self._file_name, "r") as h5file:
            nxprocess = h5file[self._nxprocess_path]
            map_dataset = get_signal_dataset(nxprocess, "result", default="intensity")
            axes_index = get_axes_index(map_dataset)
            map_shape = map_dataset.shape
            if self._map_ptr is None:
                logger.warning("No `map_ptr` defined: guessing the frame indices !")
                image_index = row * map_shape[axes_index.fast] + col + self._offset
            else:
                image_index = self._map_ptr[row, col]

            if self._dataset_paths:
                for dataset_path, size in self._dataset_paths.items():
                    if image_index < size:
                        break
                    else:
                        image_index -= size
            else:
                self.warning(f"No diffraction data images found in {self._file_name}")
                return
            try:
                image_dset = get_dataset(h5file, dataset_path)
            except KeyError:
                image_link = h5file.get(dataset_path, getlink=True)
                self.warning(f"Cannot access diffraction images at {image_link}")
                return

            if image_index >= len(image_dset):
                return

            image = image_dset[image_index]

            if "maskfile" in h5file[self._nxprocess_path]:
                maskfile = bytes.decode(h5file[self._nxprocess_path]["maskfile"][()])
            else:
                maskfile = None

        image_base = ImageBase(data=image, mask=self.getMask(image, maskfile))
        title = f"{image_dset.file.filename}\n::{image_dset.name}\n#{image_index}"
        self._image_plot_widget.setImageData(image_base.getValueData(), title)

    def selectMapPoint(self, x: float, y: float):
        indices = self._map_plot_widget.getImageIndices(x, y)
        if indices is None:
            return

        if indices == self._unfixed_indices:
            return
        else:
            self._unfixed_indices = indices

        self.clearRietveldCurves()
        self._refinement_widget.setStatus("Ready")
        self.displayPatternAtIndices(indices, legend="INTEGRATE")
        self.displayImageAtIndices(indices)
        pixel_center_coords = self._map_plot_widget.findCenterOfNearestPixel(x, y)
        self._map_plot_widget.addMarker(
            *pixel_center_coords,
            color=self.getCurveColor(legend="INTEGRATE"),
            symbol="o",
            legend="MAP_LOCATION",
        )

    def fixMapPoint(self, x: float, y: float):
        indices = self._map_plot_widget.getImageIndices(x, y)

        if indices is None:
            return
        # Remove curve and marker if the fixing point is the last clicked
        if indices == self._unfixed_indices:
            self._unfixed_point = None
            self._integrated_plot_widget.removeCurve(legend="INTEGRATE")
            self._map_plot_widget.removeMarker(legend="MAP_LOCATION")

        # Unfix is the fixing point is already fixed
        if indices in self._fixed_indices:
            self.removeMapPoint(indices=indices)

        # Fix the point
        else:
            self._fixed_indices.add(indices)

            legend = f"INTEGRATE_{indices.row}_{indices.col}"
            self.displayPatternAtIndices(
                indices,
                legend=legend,
            )
            used_color = self._integrated_plot_widget.getCurve(legend=legend).getColor()

            self.displayImageAtIndices(indices)
            pixel_center_coords = self._map_plot_widget.findCenterOfNearestPixel(x, y)
            self._map_plot_widget.addMarker(
                *pixel_center_coords,
                color=used_color,
                symbol="d",
                legend=f"MAP_LOCATION_{indices.row}_{indices.col}",
            )

    def removeMapPoint(self, indices: ImageIndices):
        self._fixed_indices.remove(indices)
        self._integrated_plot_widget.removeCurve(
            legend=f"INTEGRATE_{indices.row}_{indices.col}"
        )
        self._map_plot_widget.removeMarker(
            legend=f"MAP_LOCATION_{indices.row}_{indices.col}"
        )

    def onRoiEdition(self):
        v_min, v_max = self.getRoiRadialRange()
        if v_min is None or v_max is None:
            return

        self.displayAverageMap(v_min, v_max)

    def drawContoursOnImage(self):
        v_min, v_max = self.getRoiRadialRange()
        if v_min is None or v_max is None:
            return
        self._image_plot_widget.clearCurves()

        min_contours = find_contours(self._radial_matrix, v_min)
        for i, contour in enumerate(min_contours):
            self._image_plot_widget.addContour(contour, legend=f"min_contour_{i}")

        center_contours = find_contours(
            self._radial_matrix, v_min + (v_max - v_min) / 2
        )
        for i, contour in enumerate(center_contours):
            self._image_plot_widget.addContour(
                contour, legend=f"center_contour_{i}", linestyle=":"
            )

        max_contours = find_contours(self._radial_matrix, v_max)
        for i, contour in enumerate(max_contours):
            self._image_plot_widget.addContour(
                contour,
                legend=f"max_contour_{i}",
            )

    def displayAverageMap(self, v_min: float, v_max: float):
        if self._file_name is None:
            return

        with h5py.File(self._file_name, "r") as h5file:
            nxprocess = h5file.get(self._nxprocess_path)
            nxdata = nxprocess["result"]
            radial = get_radial_dataset(nxdata, size=self.worker_config.nbpt_rad)[()]
            i_min, i_max = get_indices_from_values(v_min, v_max, radial)
            full_map = get_signal_dataset(nxdata, default="intensity")
            axes_index = get_axes_index(full_map)
            if axes_index.radial == 2:
                map_data = full_map[:,:, i_min:i_max].mean(axis=2)
            else:
                map_data = full_map[i_min:i_max, :, : ].mean(axis=0)
            fast = get_axes_dataset(nxdata, dim=axes_index.fast, default="fast")
            slow = get_axes_dataset(nxdata, dim=axes_index.slow, default="slow")
            fast_name = fast.attrs.get("long_name", "X")
            fast_values = fast[()]
            slow_name = slow.attrs.get("long_name", "Y")
            slow_values = slow[()]
        self._map_plot_widget.setScatterData(map_data, fast_values, slow_values, fast_name, slow_name)

    def onMouseClickOnImage(self, x: float, y: float):
        indices = self._image_plot_widget.getImageIndices(x, y)
        if indices is None:
            return
        radial_value = self._radial_matrix[indices.row, indices.col]
        self._integrated_plot_widget.roi.setRange(
            radial_value - self._delta_radial_over_2,
            radial_value + self._delta_radial_over_2,
        )

    def getCurveColor(self, legend: str):
        curve = self._integrated_plot_widget.getCurve(legend=legend)
        if curve:
            return curve.getColor()
        else:
            return

    def getAvailableColor(self, legend: str):
        if self._integrated_plot_widget.getCurve(legend=legend):
            color = self._integrated_plot_widget.getCurve(legend=legend).getColor()
        else:
            color, _style = self._integrated_plot_widget._getColorAndStyle()
        return color

    def setNewBackgroundCurve(self, x: float, y: float):
        new_indices = self._map_plot_widget.getImageIndices(x, y)
        if new_indices is None or self._file_name is None:
            return

        new_background_point = Point(
            new_indices,
            url_nxdata_path=f"{self._file_name}?{self._nxprocess_path}/result"
        )

        # Unset the background if it's the same pixel and delete markers
        if (
            self._background_point
            and self._background_point.indices == new_background_point.indices
        ):
            self._map_plot_widget.removeMarker(legend="BG_LOCATION")
            self._background_point = None
        else:
            self._background_point = new_background_point
            pixel_center_coords = self._map_plot_widget.findCenterOfNearestPixel(x, y)
            self._map_plot_widget.addMarker(
                *pixel_center_coords, color="black", symbol="x", legend="BG_LOCATION"
            )

        # Refresh displayed curves
        if self._unfixed_indices:
            self.displayPatternAtIndices(self._unfixed_indices, legend="INTEGRATE")

        for indices in self._fixed_indices:
            self.displayPatternAtIndices(
                indices, legend=f"INTEGRATE_{indices.row}_{indices.col}"
            )

    def clearPoints(self):
        for indices in self._fixed_indices.copy():
            self.removeMapPoint(indices=indices)

    def clearRietveldCurves(self):
        for legend in list(self._integrated_plot_widget):
            if legend.startswith("Rietveld:"):
                self._integrated_plot_widget.removeCurve(legend=legend)
        self._integrated_plot_widget.setLegendsVisible(False)

    def runRietveldRefinement(self):
        if self._file_name is None or self._unfixed_indices is None:
            self.warning("No map point is selected for refinement")
            return
        if self._refinement_thread is not None:
            return

        point = Point(
            self._unfixed_indices,
            url_nxdata_path=f"{self._file_name}?{self._nxprocess_path}/result",
        )
        radial_unit = point.get_x_unit()
        if isinstance(radial_unit, bytes):
            radial_unit = radial_unit.decode()
        if radial_unit is not None and radial_unit.lower() not in {
            "deg",
            "degree",
            "degrees",
        }:
            self.warning(
                "Rietveld refinement currently requires a 2θ axis in degrees"
            )
            return

        cifs = self._refinement_widget.cifPaths()
        if not cifs:
            self.warning("Select at least one CIF before refinement")
            return

        ttheta_min, ttheta_max = self._refinement_widget.radialRange()
        if ttheta_min >= ttheta_max:
            self.warning("The refinement 2θ minimum must be below the maximum")
            return

        flags = self._refinement_widget.refinementFlags()
        schedule = []
        if flags["scale"] or flags["displacement"]:
            schedule.append(
                {
                    "n_iters": 20,
                    "scale": flags["scale"],
                    "displace_sample": flags["displacement"],
                    "rwp_tol": 0.01,
                }
            )
        if flags["unit_cell"]:
            schedule.append(
                {
                    "n_iters": 30,
                    "scale": flags["scale"],
                    "displace_sample": flags["displacement"],
                    "a": True,
                    "b": True,
                    "c": True,
                    "alpha": True,
                    "beta": True,
                    "gamma": True,
                    "rwp_tol": 0.01,
                }
            )
        if flags["peak_width"]:
            schedule.append(
                {
                    "n_iters": 100,
                    "scale": flags["scale"],
                    "displace_sample": flags["displacement"],
                    "a": flags["unit_cell"],
                    "b": flags["unit_cell"],
                    "c": flags["unit_cell"],
                    "alpha": flags["unit_cell"],
                    "beta": flags["unit_cell"],
                    "gamma": flags["unit_cell"],
                    "W": True,
                    "Eta0": True,
                    "rwp_tol": 0.01,
                }
            )
        if not schedule:
            self.warning("Select at least one refinement parameter")
            return

        inputs = {
            "ttheta_deg": point.get_radial_curve(),
            "intensity": point.get_curve(),
            "cifs": cifs,
            "instprm": {
                "lambda_A": self._refinement_widget.wavelength(),
                "W": 1e-6,
                "Eta0": 0.0,
            },
            "schedule": schedule,
            "ttheta_range_deg": (ttheta_min, ttheta_max),
        }
        intensity_error = point.get_uncertainty_curve()
        if intensity_error is not None:
            inputs["intensity_error"] = intensity_error

        self.clearRietveldCurves()
        self._refinement_widget.setRunning(True)
        self._refinement_widget.setStatus(
            f"Refining point [{point.indices.row}, {point.indices.col}]…"
        )
        self._refinement_thread = RietveldRefinementThread(
            inputs,
            point.indices,
            parent=self,
        )
        self._refinement_thread.finished.connect(
            self.onRietveldRefinementFinished
        )
        self._refinement_thread.start()

    def onRietveldRefinementFinished(self):
        thread = self._refinement_thread
        self._refinement_thread = None
        self._refinement_widget.setRunning(False)

        if thread.error is not None:
            logger.error("Rietveld refinement failed:\n%s", thread.error)
            message = thread.error.strip().splitlines()[-1]
            self._refinement_widget.setStatus(f"Refinement failed: {message}")
            self.warning(f"Rietveld refinement failed: {message}")
            thread.deleteLater()
            return

        if thread.indices != self._unfixed_indices:
            self._refinement_widget.setStatus(
                "Refinement finished for a point which is no longer selected"
            )
            thread.deleteLater()
            return

        result = thread.result
        x = result["ttheta_deg"]
        background = result["background"]
        self._integrated_plot_widget.addCurve(
            x,
            result["calculated"],
            legend="Rietveld: total",
            color="#d62728",
            linewidth=1.5,
            selectable=False,
            resetzoom=False,
        )
        self._integrated_plot_widget.addCurve(
            x,
            background,
            legend="Rietveld: background",
            color="#7f7f7f",
            linewidth=1.0,
            selectable=False,
            resetzoom=False,
        )
        phase_colors = (
            "#2ca02c",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#bcbd22",
            "#17becf",
            "#ff7f0e",
        )
        for index, (phase, phase_calculated) in enumerate(
            result["phase_patterns"].items()
        ):
            self._integrated_plot_widget.addCurve(
                x,
                background + phase_calculated,
                legend=f"Rietveld: {phase}",
                color=phase_colors[index % len(phase_colors)],
                linewidth=1.0,
                selectable=False,
                resetzoom=False,
            )

        self._integrated_plot_widget.setLegendsVisible(True)
        rwp = result["history"][-1]["Rw"]
        self._refinement_widget.setStatus(
            f"Point [{thread.indices.row}, {thread.indices.col}], Rwp {rwp:.2f}%"
        )
        thread.deleteLater()

    def closeEvent(self, event):
        if self._refinement_thread is not None:
            self._refinement_thread.wait()
        super().closeEvent(event)

    def warning(self, error_msg):
        """Log a warning both in the terminal and in the status bar if possible

        :param error_msg: string with the message
        """
        logger.warning(error_msg)
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(error_msg)
