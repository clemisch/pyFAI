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
import sys
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
from .widgets.BackgroundWidget import BackgroundDialog
from .widgets.ReflectionOverlayWidget import ReflectionOverlayDialog
from .widgets.RietveldRefinementWidget import (
    RietveldRefinementDialog,
    RietveldRefinementProcess,
    RietveldRefinementThread,
)

logger = logging.getLogger(__name__)


class MainWindow(qt.QMainWindow):
    sigFileChanged = qt.Signal(str)

    def __init__(self, rietveld_python=None) -> None:
        super().__init__()
        self._file_name: str | None = None
        self._unfixed_indices = None
        self._fixed_indices = set()
        self._background_point = None
        self._map_plot_widgets = []
        self._rietveld_python = rietveld_python or sys.executable
        self._mapped_refinement_result = None
        self._mapped_refinement_flags = None
        self._reflection_widget = None
        self._reflection_wavelength = None
        self._reflection_radial_range = None
        self._reflection_cif_paths = []
        self._background_widget = BackgroundDialog(self)
        self._background_widget.computeRequested.connect(self.computeBackground)
        self._background_widget.displayChanged.connect(self.refreshBackgroundDisplay)

        self.setWindowTitle("PyFAI-diffmap viewer")

        self._image_plot_widget = DiffractionImagePlotWidget(self)
        self._image_plot_widget.setDefaultColormap(
            Colormap("gray", normalization="log")
        )
        self._image_plot_widget.setKeepDataAspectRatio(True)
        self._image_plot_widget.plotClicked.connect(self.onMouseClickOnImage)

        self._map_tab_widget = qt.QTabWidget(self)
        self._map_tab_widget.setTabsClosable(True)
        self._map_tab_widget.tabCloseRequested.connect(self.removeMapTab)
        self._map_plot_widget = self.addMapTab("2θ ROI", closable=False)
        self._map_plot_widget.setDefaultColormap(
            Colormap("viridis", normalization="log")
        )

        self._integrated_plot_widget = IntegratedPatternPlotWidget(self)
        self._integrated_plot_widget.roi.sigRegionChanged.connect(self.onRoiEdition)
        self._integrated_plot_widget.roi.sigRegionChanged.connect(self.drawContoursOnImage)

        self._refinement_widget = RietveldRefinementDialog(self)
        self._rietveld_phase_paths = {}
        self._point_refinements = {}
        self._preview_generation = 0
        self._preview_timer = qt.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self.runRietveldRefinement)
        self._refinement_widget.phase_colors.changed.connect(self.refreshRietveldColors)
        self._refinement_widget.displayChanged.connect(self.refreshRietveldVisibility)
        self._refinement_widget.refinementRequested.connect(
            self.scheduleRietveldRefinement
        )
        self._refinement_widget.mapRefinementRequested.connect(
            self.runRietveldMapRefinement
        )
        self._refinement_widget.mapRequested.connect(self.showRietveldMap)
        self._integrated_plot_widget.refinementRequested.connect(
            self.showRietveldRefinement
        )
        self._integrated_plot_widget.reflectionOverlayRequested.connect(
            self.showReflectionOverlay
        )
        self._integrated_plot_widget.backgroundRequested.connect(self._background_widget.show)

        self._central_widget = qt.QWidget()
        right_splitter = qt.QSplitter(qt.Qt.Orientation.Vertical, self)
        right_splitter.addWidget(self._map_tab_widget)
        right_splitter.addWidget(self._integrated_plot_widget)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setHandleWidth(6)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)

        plot_splitter = qt.QSplitter(qt.Qt.Orientation.Horizontal, self)
        plot_splitter.addWidget(self._image_plot_widget)
        plot_splitter.addWidget(right_splitter)
        plot_splitter.setChildrenCollapsible(False)
        plot_splitter.setHandleWidth(6)
        plot_splitter.setStretchFactor(0, 1)
        plot_splitter.setStretchFactor(1, 1)

        layout = qt.QVBoxLayout(self._central_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(plot_splitter, 1)
        self._central_widget.setLayout(layout)
        self.setCentralWidget(self._central_widget)

        self._refinement_thread = None
        self._refinement_process = None
        self.worker_config = None

        # declaration of instance variables
        self._map_ptr = None # This is the map of the indices of input frame

    def addMapTab(
        self,
        title: str,
        image: numpy.ndarray | None = None,
        x: numpy.ndarray | None = None,
        y: numpy.ndarray | None = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        closable: bool = True,
    ) -> MapPlotWidget:
        map_plot_widget = MapPlotWidget(self._map_tab_widget)
        map_plot_widget.setDefaultColormap(Colormap("viridis"))
        map_plot_widget.clearPointsSignal.connect(self.clearPoints)
        map_plot_widget.plotClicked.connect(self.selectMapPoint)
        map_plot_widget.pinContextEntrySelected.connect(self.fixMapPoint)
        map_plot_widget.setBackgroundClicked.connect(self.setNewBackgroundCurve)
        self.sigFileChanged.connect(map_plot_widget.onFileChange)
        if self._file_name is not None:
            map_plot_widget.onFileChange(self._file_name)
        if image is not None:
            map_plot_widget.setScatterData(image, x, y, xlabel, ylabel)

        index = self._map_tab_widget.addTab(map_plot_widget, title)
        self._map_plot_widgets.append(map_plot_widget)
        self._map_tab_widget.setCurrentIndex(index)
        if not closable:
            tab_bar = self._map_tab_widget.tabBar()
            tab_bar.setTabButton(
                index, qt.QTabBar.ButtonPosition.LeftSide, None
            )
            tab_bar.setTabButton(
                index, qt.QTabBar.ButtonPosition.RightSide, None
            )

        if image is not None:
            if self._unfixed_indices is not None:
                self.setMapMarker(
                    self._unfixed_indices,
                    color=self.getCurveColor(legend="INTEGRATE"),
                    symbol="o",
                    legend="MAP_LOCATION",
                )
            for indices in self._fixed_indices:
                legend = f"INTEGRATE_{indices.row}_{indices.col}"
                self.setMapMarker(
                    indices,
                    color=self.getCurveColor(legend=legend),
                    symbol="d",
                    legend=f"MAP_LOCATION_{indices.row}_{indices.col}",
                )
            if self._background_point is not None:
                self.setMapMarker(
                    self._background_point.indices,
                    color="black",
                    symbol="x",
                    legend="BG_LOCATION",
                )

        return map_plot_widget

    def removeMapTab(self, index: int):
        if index == 0:
            return
        map_plot_widget = self._map_tab_widget.widget(index)
        self._map_tab_widget.removeTab(index)
        self._map_plot_widgets.remove(map_plot_widget)
        map_plot_widget.deleteLater()

    def setMapMarker(self, indices: ImageIndices, **kwargs):
        for map_plot_widget in self._map_plot_widgets:
            coordinates = map_plot_widget.getMapPointCoordinates(indices)
            if coordinates is not None:
                map_plot_widget.addMarker(*coordinates, **kwargs)

    def removeMapMarker(self, legend: str):
        for map_plot_widget in self._map_plot_widgets:
            map_plot_widget.removeMarker(legend=legend)

    def initData(self,
                 file_name: str,
                 dataset_path: str="/entry_0000/measurement/images_0001",
                 nxprocess_path: str="/entry_0000/pyFAI",
                 ):

        self._file_name = os.path.abspath(file_name)
        self._mapped_refinement_result = None
        self._mapped_refinement_flags = None
        while self._map_tab_widget.count() > 1:
            self.removeMapTab(1)
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
        wavelength_A = None if wavelength is None else wavelength * 1e10
        self._reflection_wavelength = wavelength_A
        self._reflection_radial_range = (
            float(radial_values[0]),
            float(radial_values[-1]),
            abs(float(delta_radial)),
        )
        self._background_widget.reset(radial_values)
        self._point_refinements.clear()
        self._refinement_widget.setWavelength(wavelength_A)
        self._refinement_widget.setRadialRange(
            float(radial_values[0]), float(radial_values[-1])
        )
        cif_directory = os.path.dirname(self._file_name)
        cifs = sorted(
            os.path.join(cif_directory, filename)
            for filename in os.listdir(cif_directory)
            if filename.lower().endswith(".cif")
        )
        self._reflection_cif_paths = cifs
        self._refinement_widget.setCifPaths(cifs)
        if self._reflection_widget is not None:
            self._reflection_widget.setWavelength(wavelength_A)
            self._reflection_widget.setRadialRange(*self._reflection_radial_range)
            self._reflection_widget.setCifPaths(cifs)

        self._map_plot_widget.setScatterData(map_data, fast_values, slow_values, fast_label, slow_label)
        # BUG: selectMapPoint(0, 0) does not work at first render cause the picking fails
        initial_indices = ImageIndices(0, 0)
        self._unfixed_indices = initial_indices
        self.displayPatternAtIndices(initial_indices, legend="INTEGRATE")
        self.displayImageAtIndices(initial_indices)
        self.setMapMarker(
            initial_indices,
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

        x = point.get_radial_curve()
        baseline_result = self._background_widget.resultForPoint(indices)
        if legend == "INTEGRATE":
            self._integrated_plot_widget.removeCurve("Estimated background")
        if baseline_result is not None:
            start, stop = baseline_result["slice"]
            baseline = baseline_result["background"]
            if self._background_point is not None:
                baseline = baseline - self._background_point.get_curve()[start:stop]
            if self._background_widget.subtract.isChecked():
                x = x[start:stop]
                curve = curve[start:stop] - baseline
            elif legend == "INTEGRATE":
                self._integrated_plot_widget.addDataCurve(
                    x[start:stop], baseline, legend="Estimated background",
                    color="#ff7f0e", selectable=False, resetzoom=False,
                )

        self._integrated_plot_widget.addDataCurve(
            x=x,
            y=curve,
            legend=legend,
            selectable=False,
            resetzoom=self._integrated_plot_widget.getGraphXLimits() == (0, 100),
        )
        self._integrated_plot_widget.setGraphXLabel(point.get_x_name())
        label = point.get_y_name()
        if baseline_result is not None and self._background_widget.subtract.isChecked():
            label += " − background"
        if legend == "INTEGRATE":
            self._integrated_plot_widget.setDataYLabel(label)
        if legend == "INTEGRATE" and self._background_widget.process is None:
            dialog = self._background_widget
            if dialog.map_result is not None:
                message = "Mapped backgrounds available. ROI map uses the last full-map computation."
                if indices in dialog.results:
                    message += " Histogram uses a newer single-point preview."
            elif baseline_result is not None:
                message = "Selected-point background available; ROI map is uncorrected."
            else:
                message = "No background for this point; histogram and map are uncorrected."
            dialog.status.setText(message)

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
        map_plot_widget = self.sender()
        if not isinstance(map_plot_widget, MapPlotWidget):
            map_plot_widget = self._map_plot_widget
        indices = map_plot_widget.getImageIndices(x, y)
        if indices is None:
            return

        if indices == self._unfixed_indices:
            return
        else:
            self._unfixed_indices = indices

        self.clearRietveldCurves()
        if self._mapped_refinement_result is None:
            self._refinement_widget.clearResult()
        else:
            self._refinement_widget.setResult(
                self._mapped_refinement_result,
                self._mapped_refinement_flags,
                indices=indices,
            )
        self.displayPatternAtIndices(indices, legend="INTEGRATE")
        self.displayAvailableRefinement()
        self.scheduleRietveldRefinement()
        self.displayImageAtIndices(indices)
        self.setMapMarker(
            indices,
            color=self.getCurveColor(legend="INTEGRATE"),
            symbol="o",
            legend="MAP_LOCATION",
        )

    def fixMapPoint(self, x: float, y: float):
        map_plot_widget = self.sender()
        if not isinstance(map_plot_widget, MapPlotWidget):
            map_plot_widget = self._map_plot_widget
        indices = map_plot_widget.getImageIndices(x, y)

        if indices is None:
            return
        # Remove curve and marker if the fixing point is the last clicked
        if indices == self._unfixed_indices:
            self._unfixed_point = None
            self._integrated_plot_widget.removeCurve(legend="INTEGRATE")
            self.removeMapMarker(legend="MAP_LOCATION")

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
            self.setMapMarker(
                indices,
                color=used_color,
                symbol="d",
                legend=f"MAP_LOCATION_{indices.row}_{indices.col}",
            )

    def removeMapPoint(self, indices: ImageIndices):
        self._fixed_indices.remove(indices)
        self._integrated_plot_widget.removeCurve(
            legend=f"INTEGRATE_{indices.row}_{indices.col}"
        )
        self.removeMapMarker(legend=f"MAP_LOCATION_{indices.row}_{indices.col}")

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
            i_min = max(0, i_min)
            i_max = min(len(radial), i_max)
            # An unset or sub-bin ROI can select no samples. Keep the current
            # map rather than replacing it with the mean of an empty slice.
            if i_min >= i_max:
                return
            full_map = get_signal_dataset(nxdata, default="intensity")
            axes_index = get_axes_index(full_map)
            if axes_index.radial == 2:
                map_data = full_map[:,:, i_min:i_max].mean(axis=2)
            else:
                map_data = full_map[i_min:i_max, :, : ].mean(axis=0)
            background = self._background_widget.map_result
            corrected = False
            if self._background_widget.subtract.isChecked() and background is not None:
                start, stop = background["slice"]
                if start <= i_min < i_max <= stop:
                    map_data = map_data - background["background"][
                        :, :, i_min - start:i_max - start
                    ].mean(axis=2)
                    corrected = True
                else:
                    self._background_widget.status.setText(
                        "ROI is outside the computed background range; map is uncorrected."
                    )
            fast = get_axes_dataset(nxdata, dim=axes_index.fast, default="fast")
            slow = get_axes_dataset(nxdata, dim=axes_index.slow, default="slow")
            fast_name = fast.attrs.get("long_name", "X")
            fast_values = fast[()]
            slow_name = slow.attrs.get("long_name", "Y")
            slow_values = slow[()]
        self._map_plot_widget.setScatterData(map_data, fast_values, slow_values, fast_name, slow_name)
        self._map_tab_widget.setTabText(0, "2θ ROI (no bkg)" if corrected else "2θ ROI")
        colormap = self._map_plot_widget.getScatter("MAP").getColormap()
        dialog = self._background_widget
        if corrected and dialog.map_normalization is None:
            dialog.map_normalization = colormap.getNormalization()
            colormap.setNormalization("linear")
        elif not corrected and dialog.map_normalization is not None:
            colormap.setNormalization(dialog.map_normalization)
            dialog.map_normalization = None

    def computeBackground(self, mapped):
        dialog = self._background_widget
        if self._file_name is None or self._unfixed_indices is None or dialog.process is not None:
            return
        point = Point(self._unfixed_indices, f"{self._file_name}?{self._nxprocess_path}/result")
        unit = point.get_x_unit()
        if isinstance(unit, bytes):
            unit = unit.decode()
        if unit is not None and unit.lower() not in {"2th_deg", "2theta_deg", "deg", "degree", "degrees", "°"}:
            dialog.status.setText("Background controls currently require a 2θ grid in degrees.")
            self.warning(dialog.status.text())
            return
        if mapped:
            with h5py.File(self._file_name, "r") as handle:
                nxdata = handle[self._nxprocess_path + "/result"]
                signal = get_signal_dataset(nxdata, default="intensity")
                radial = get_radial_dataset(nxdata, size=self.worker_config.nbpt_rad)
                axes = get_axes_index(signal)
                inputs = {
                    "filename": self._file_name, "intensity_path": signal.name,
                    "ttheta_path": radial.name,
                    "dimensions": (axes.slow, axes.fast, axes.radial),
                }
        else:
            inputs = {"ttheta_deg": point.get_radial_curve(), "observed": point.get_curve()}
        dialog.start(self._rietveld_python, inputs, mapped, self._unfixed_indices, self._file_name)
        dialog.process.completed.connect(self.backgroundFinished)
        dialog.process.startRefinement()

    def backgroundFinished(self):
        dialog = self._background_widget
        process = dialog.process
        dialog.process = None
        dialog.single.setEnabled(True)
        dialog.mapped.setEnabled(True)
        if process.error_text:
            dialog.status.setText(process.error_text.strip().splitlines()[-1])
            dialog.status.setToolTip(process.error_text)
            self.warning(dialog.status.text())
        elif (process.background_filename == self._file_name
              and process.background_generation == dialog.data_generation):
            if process.background_mapped:
                dialog.map_result = process.result
                dialog.results.clear()
            else:
                # A single preview supersedes only this point, not the map.
                dialog.results[process.background_indices] = process.result
            self.refreshBackgroundDisplay()
        process.deleteLater()

    def refreshBackgroundDisplay(self):
        if self._file_name is None or self._unfixed_indices is None:
            return
        self.clearRietveldCurves()
        self.displayPatternAtIndices(self._unfixed_indices, legend="INTEGRATE")
        for indices in self._fixed_indices:
            self.displayPatternAtIndices(indices, legend=f"INTEGRATE_{indices.row}_{indices.col}")
        if self._background_widget.map_result is not None:
            self.onRoiEdition()
        self.displayAvailableRefinement()

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
        map_plot_widget = self.sender()
        if not isinstance(map_plot_widget, MapPlotWidget):
            map_plot_widget = self._map_plot_widget
        new_indices = map_plot_widget.getImageIndices(x, y)
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
            self.removeMapMarker(legend="BG_LOCATION")
            self._background_point = None
        else:
            self._background_point = new_background_point
            self.setMapMarker(
                new_indices,
                color="black",
                symbol="x",
                legend="BG_LOCATION",
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

    def refreshRietveldColors(self):
        for phase, path in self._rietveld_phase_paths.items():
            curve = self._integrated_plot_widget.getCurve(f"Rietveld: {phase}")
            if curve is not None:
                curve.setColor(self._refinement_widget.phase_colors.get(path))
        self._integrated_plot_widget.scheduleLegendUpdate()

    def refreshRietveldVisibility(self):
        dialog = self._refinement_widget
        shown = dialog.shownCifPaths()
        visibility = {
            "Rietveld: total": dialog.show_total.isChecked(),
            "Rietveld: background": dialog.show_background.isChecked(),
        }
        for phase, path in self._rietveld_phase_paths.items():
            visibility[f"Rietveld: {phase}"] = os.path.realpath(path) in shown
        for legend, visible in visibility.items():
            curve = self._integrated_plot_widget.getCurve(legend)
            if curve is not None:
                curve.setVisible(visible)
        self._integrated_plot_widget.scheduleLegendUpdate()

    def showRietveldRefinement(self):
        self._refinement_widget.show()
        self._refinement_widget.raise_()
        self._refinement_widget.activateWindow()

    def showReflectionOverlay(self):
        if self._reflection_widget is None:
            try:
                self._reflection_widget = ReflectionOverlayDialog(
                    self, phase_colors=self._refinement_widget.phase_colors
                )
            except ImportError:
                self.warning(
                    "Expected reflection positions require pymatgen. "
                    "Install it with `pip install pymatgen`."
                )
                return
            self._reflection_widget.overlayChanged.connect(
                self._integrated_plot_widget.setReflectionOverlays
            )
            self._reflection_widget.setWavelength(self._reflection_wavelength)
            if self._reflection_radial_range is not None:
                self._reflection_widget.setRadialRange(
                    *self._reflection_radial_range
                )
            self._reflection_widget.setCifPaths(self._reflection_cif_paths)
        self._reflection_widget.show()
        self._reflection_widget.raise_()
        self._reflection_widget.activateWindow()

    def _getRietveldInputs(self, mapped=False):
        if self._file_name is None or self._unfixed_indices is None:
            self.warning("No map point is selected for refinement")
            return None

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
            return None

        cifs = self._refinement_widget.enabledCifPaths()
        if not cifs:
            self.warning("Select at least one CIF before refinement")
            return None

        ttheta_min, ttheta_max = self._refinement_widget.radialRange()
        if ttheta_min >= ttheta_max:
            self.warning("The refinement 2θ minimum must be below the maximum")
            return None

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
            return None

        inputs = {
            "cifs": cifs,
            "instprm": {
                "lambda_A": self._refinement_widget.wavelength(),
                "W": 1e-6,
                "Eta0": 0.0,
            },
            "schedule": schedule,
            "ttheta_range_deg": (ttheta_min, ttheta_max),
        }
        if mapped:
            inputs["nxdata_url"] = (
                f"{self._file_name}::{self._nxprocess_path}/result"
            )
        else:
            inputs["ttheta_deg"] = point.get_radial_curve()
            inputs["intensity"] = point.get_curve()
            intensity_error = point.get_uncertainty_curve()
            if intensity_error is not None:
                inputs["intensity_error"] = intensity_error

        return inputs, flags, point

    def scheduleRietveldRefinement(self):
        self._preview_generation += 1
        self._preview_timer.stop()
        self._point_refinements.clear()
        if self._refinement_widget._run_button.isChecked():
            self.clearRietveldCurves()
            self._preview_timer.start()

    def runRietveldRefinement(self):
        if not self._refinement_widget._run_button.isChecked() or self._file_name is None:
            return
        if self._refinement_thread is not None or self._refinement_process is not None:
            return
        setup = self._getRietveldInputs()
        if setup is None:
            return
        inputs, flags, point = setup

        self.clearRietveldCurves()
        self._refinement_widget.setRunning(True)
        self._refinement_thread = RietveldRefinementThread(
            inputs,
            point.indices,
            flags,
            parent=self,
        )
        self._rietveld_phase_paths = {
            os.path.splitext(os.path.basename(path))[0]: path for path in inputs["cifs"]
        }
        self._refinement_thread.phase_paths = dict(self._rietveld_phase_paths)
        self._refinement_thread.preview_generation = self._preview_generation
        self._refinement_thread.finished.connect(
            self.onRietveldRefinementFinished
        )
        self._refinement_thread.start()

    def runRietveldMapRefinement(self):
        if self._refinement_thread is not None or self._refinement_process is not None:
            return
        setup = self._getRietveldInputs(mapped=True)
        if setup is None:
            return
        inputs, flags, _point = setup

        graph = {
            "graph": {"id": "rietveld_refine_map", "schema_version": "1.2"},
            "nodes": [
                {
                    "id": "refinement",
                    "task_type": "class",
                    "task_identifier": (
                        "ewoksxrpd.tasks.rietveld.RietveldRefineMap"
                    ),
                    "default_inputs": [
                        {"name": name, "value": value}
                        for name, value in inputs.items()
                    ],
                }
            ],
            "links": [],
        }
        self.clearRietveldCurves()
        self._refinement_widget.setRunning(True, mapped=True)
        self._refinement_process = RietveldRefinementProcess(
            self._rietveld_python,
            graph,
            flags,
            parent=self,
        )
        self._refinement_process.completed.connect(
            self.onRietveldMapRefinementFinished
        )
        self._refinement_process.startRefinement()

    def onRietveldRefinementFinished(self):
        thread = self._refinement_thread
        self._refinement_thread = None
        self._refinement_widget.setRunning(False)

        if thread.preview_generation != self._preview_generation:
            thread.deleteLater()
            if self._refinement_widget._run_button.isChecked():
                self._preview_timer.start()
            return

        if thread.error is not None:
            logger.error("Rietveld refinement failed:\n%s", thread.error)
            message = thread.error.strip().splitlines()[-1]
            self.warning(f"Rietveld refinement failed: {message}")
            thread.deleteLater()
            return

        result = thread.result
        paths = thread.phase_paths
        self._point_refinements[thread.indices] = (result, thread.refinement_flags, paths)
        thread.deleteLater()
        self.displayAvailableRefinement()

    def displayAvailableRefinement(self):
        self.clearRietveldCurves()
        cached = self._point_refinements.get(self._unfixed_indices)
        if cached is not None:
            result, flags, self._rietveld_phase_paths = cached
            self._refinement_widget.setResult(result, flags)
        else:
            return
        x = result["ttheta_deg"]
        selected = numpy.ones(len(x), dtype=bool)
        baseline = 0
        preview = self._background_widget.resultForPoint(self._unfixed_indices)
        if self._background_widget.subtract.isChecked() and preview is not None:
            grid = preview["ttheta_deg"]
            selected = (x >= grid[0]) & (x <= grid[-1])
            baseline = numpy.interp(x[selected], grid, preview["background"])
        x = x[selected]
        background = result["background"][selected] - baseline
        self._integrated_plot_widget.addDataCurve(
            x,
            result["calculated"][selected] - baseline,
            legend="Rietveld: total",
            color="#d62728",
            linewidth=1.5,
            selectable=False,
            resetzoom=False,
        )
        self._integrated_plot_widget.addDataCurve(
            x,
            background,
            legend="Rietveld: background",
            color="#7f7f7f",
            linewidth=1.0,
            selectable=False,
            resetzoom=False,
        )
        for phase, phase_calculated in result["phase_patterns"].items():
            self._integrated_plot_widget.addDataCurve(
                x,
                background + phase_calculated[selected],
                legend=f"Rietveld: {phase}",
                color=self._refinement_widget.phase_colors.get(self._rietveld_phase_paths[phase]),
                linewidth=1.0,
                selectable=False,
                resetzoom=False,
            )


        self.refreshRietveldVisibility()

    def onRietveldMapRefinementFinished(self):
        process = self._refinement_process
        if process is None:
            return
        self._refinement_process = None
        self._refinement_widget.setRunning(False)

        if process.error_text is not None:
            logger.error("Mapped Rietveld refinement failed:\n%s", process.error_text)
            message = process.error_text.strip().splitlines()[-1]
            self.warning(f"Mapped Rietveld refinement failed: {message}")
            process.deleteLater()
            if self._refinement_widget._run_button.isChecked():
                self._preview_timer.start()
            return

        for index in range(1, self._map_tab_widget.count()):
            title = self._map_tab_widget.tabText(index)
            if not title.endswith("*"):
                self._map_tab_widget.setTabText(index, title + "*")
        self._mapped_refinement_result = process.result
        self._mapped_refinement_flags = process.refinement_flags
        self._refinement_widget.setResult(
            process.result,
            process.refinement_flags,
            indices=self._unfixed_indices,
        )
        self.showRietveldMap("Rwp", process.result["stages"][-1]["Rw"])
        self.displayAvailableRefinement()
        self.scheduleRietveldRefinement()
        process.deleteLater()

    def showRietveldMap(self, title, map_data):
        for index in range(1, self._map_tab_widget.count()):
            if self._map_tab_widget.tabText(index) == title:
                self._map_tab_widget.setCurrentIndex(index)
                return

        axes = self._mapped_refinement_result["axes"]
        self.addMapTab(
            title,
            numpy.asarray(map_data),
            x=numpy.asarray(axes[1]["values"]),
            y=numpy.asarray(axes[0]["values"]),
            xlabel=axes[1]["label"],
            ylabel=axes[0]["label"],
        )

    def closeEvent(self, event):
        self._preview_timer.stop()
        self._preview_generation += 1
        self._refinement_widget._run_button.blockSignals(True)
        self._refinement_widget._run_button.setChecked(False)
        if self._background_widget.process is not None:
            self._background_widget.process.kill()
            self._background_widget.process.waitForFinished()
        if self._refinement_thread is not None:
            self._refinement_thread.wait()
        if self._refinement_process is not None:
            process = self._refinement_process
            process.kill()
            process.waitForFinished()
        super().closeEvent(event)

    def warning(self, error_msg):
        """Log a warning both in the terminal and in the status bar if possible

        :param error_msg: string with the message
        """
        logger.warning(error_msg)
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(error_msg)
