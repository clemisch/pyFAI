#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023-2024 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Loïc Huder"
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/03/2024"
__status__ = "development"

import numpy
from matplotlib import scale as mscale
from matplotlib import ticker
from matplotlib import transforms
from silx.gui import icons, qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.LegendSelector import LegendsDockWidget
from silx.gui.plot.actions.control import ResetZoomAction
from silx.gui.plot.actions.io import SaveAction
from silx.gui.plot.actions.mode import PanModeAction, ZoomModeAction
from silx.gui.plot.items import Curve
from silx.gui.plot.tools import PositionInfo
from silx.gui.plot.tools.roi import RegionOfInterestManager

from ..HorizontalRangeROI import HorizontalRangeROI
from ..models import ROI_COLOR
from .RoiModeAction import RoiModeAction
from .RoiRangeWidget import RoiRangeWidget


class SignedSquareRootScale(mscale.ScaleBase):
    """Square-root axis scale which preserves negative values."""

    name = "signed_sqrt"

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    class Transform(transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, values):
            values = numpy.asarray(values)
            return numpy.sign(values) * numpy.sqrt(numpy.abs(values))

        def inverted(self):
            return SignedSquareRootScale.InvertedTransform()

    class InvertedTransform(transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, values):
            values = numpy.asarray(values)
            return numpy.sign(values) * numpy.square(values)

        def inverted(self):
            return SignedSquareRootScale.Transform()

    def get_transform(self):
        return self.Transform()


mscale.register_scale(SignedSquareRootScale)


class IntegratedPatternPlotWidget(PlotWidget):
    refinementRequested = qt.Signal()

    def __init__(self, parent=None, backend=None):
        super().__init__(parent, backend)
        self.sigPlotSignal.connect(self.onRectDraw)

        self._roi_manager = RegionOfInterestManager(parent=self)
        self.roi = self._initRoi()
        self._roi_manager.addRoi(self.roi)

        self._roi_range = RoiRangeWidget(self)
        # Interconnect the ROI and the ROI range widget
        self._roi_range.updated.connect(self.roi.setRange)
        self.roi.sigRegionChanged.connect(self.updateRoiRangeWidget)

        self._toolbar = self._initToolbar()
        self.addToolBar(self._toolbar)

        self._legends = LegendsDockWidget(parent=self, plot=self)
        self.addDockWidget(qt.Qt.RightDockWidgetArea, self._legends)
        self._legends.hide()

        self._statusBar = self._initStatusBar()
        centralWidget = self._initCentralWidget(self._statusBar)
        self.setCentralWidget(centralWidget)

    def __iter__(self):
        yield from self.getAllCurves(just_legend=True)

    def setLegendsVisible(self, visible):
        self._legends.setVisible(visible)

    def _initRoi(self):
        roi = HorizontalRangeROI()
        roi.setColor(ROI_COLOR)
        roi.setEditable(True)

        return roi

    def _initToolbar(self):
        toolbar = qt.QToolBar()
        toolbar.addAction(ResetZoomAction(self, toolbar))
        toolbar.addSeparator()
        toolbar.addAction(PanModeAction(self, toolbar))
        toolbar.addAction(ZoomModeAction(self, toolbar))
        self._y_scale_button = qt.QToolButton(toolbar)
        self._y_scale_button.setPopupMode(qt.QToolButton.ToolButtonPopupMode.InstantPopup)
        y_scale_menu = qt.QMenu(self._y_scale_button)
        self._y_scale_actions = {}
        y_scale_group = qt.QActionGroup(self._y_scale_button)
        y_scale_group.setExclusive(True)
        for scale, text, icon in (
            ("linear", "Linear Y-axis", "yscale-linear"),
            ("log", "Logarithmic Y-axis", "yscale-log"),
            ("asinh", "Arcsinh Y-axis", "yscale-asinh"),
            ("signed_sqrt", "Square-root Y-axis", "math-amplitude"),
        ):
            action = qt.QAction(icons.getQIcon(icon), text, y_scale_group)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, selected_scale=scale: self.setYAxisScale(
                    selected_scale
                )
            )
            y_scale_group.addAction(action)
            y_scale_menu.addAction(action)
            self._y_scale_actions[scale] = action
        self._y_scale_button.setMenu(y_scale_menu)
        self._y_scale_actions["linear"].setChecked(True)
        self._y_scale_button.setIcon(icons.getQIcon("yscale-linear"))
        self._y_scale_button.setToolTip("Y-axis scale is linear")
        toolbar.addWidget(self._y_scale_button)
        roiAction = RoiModeAction(self, toolbar)
        toolbar.addAction(roiAction)
        # Start in ROI mode
        roiAction._actionTriggered()

        toolbar.addSeparator()
        toolbar.addAction(SaveAction(self, toolbar))
        toolbar.addSeparator()
        refinementAction = qt.QAction(
            icons.getQIcon("math-fit"), "Rietveld refinement", toolbar
        )
        refinementAction.setToolTip("Open Rietveld refinement")
        refinementAction.triggered.connect(self.refinementRequested)
        toolbar.addAction(refinementAction)
        return toolbar

    def setYAxisScale(self, scale):
        axis = self.getYAxis()
        backend = self.getBackend()
        if scale == "signed_sqrt":
            if not hasattr(backend, "ax"):
                raise RuntimeError(
                    "Square-root Y-axis scaling requires the Matplotlib backend"
                )
            if axis.getScale() != "linear":
                axis.setScale("linear")
            backend.setYAxisScale("signed_sqrt")
            icon = "math-amplitude"
            tooltip = "Y-axis scale is signed square root"
        else:
            if axis.getScale() == scale:
                backend.setYAxisScale(scale)
            else:
                axis.setScale(scale)
            icon = f"yscale-{scale}"
            tooltip = f"Y-axis scale is {scale}"

        self._y_scale_actions[scale].setChecked(True)
        self._y_scale_button.setIcon(icons.getQIcon(icon))
        self._y_scale_button.setToolTip(tooltip)
        self.resetZoom()

    def _initStatusBar(self):
        converters = (
            ("X", lambda x, y: x),
            ("Data", self._dataConverter),
        )
        return PositionInfo(plot=self, converters=converters)

    def _initCentralWidget(self, status_bar: qt.QWidget):
        gridLayout = qt.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.addWidget(self.getWidgetHandle(), 0, 0)
        gridLayout.addWidget(status_bar, 1, 0, 1, -1)
        gridLayout.addWidget(self._roi_range, 2, 0)

        gridLayout.setRowStretch(0, 1)
        centralWidget = qt.QWidget(self)
        centralWidget.setLayout(gridLayout)
        return centralWidget

    def _dataConverter(self, x_data, y_data):
        curves = self.getAllCurves()
        if not curves:
            return
        curve_item = curves[0]
        if not isinstance(curve_item, Curve):
            raise RuntimeError("`curve` is not a `silx.gui.plot.items.curve.Curve` instance")
        tmp = self.dataToPixel(x_data, y_data)
        if tmp:
            pixel_x, pixel_y = tmp
            picking_result = curve_item.pick(pixel_x, pixel_y)
        else:
            picking_result = None
        if picking_result is None:
            return
        indices_x = picking_result.getIndices(copy=False)
        curve_data = curve_item.getYData(copy=False)
        return curve_data[indices_x[0]]

    def onRectDraw(self, signal_data):
        if signal_data["event"] != "drawingFinished":
            return

        v_min, v_max = signal_data["xdata"]
        if v_max < v_min:
            v_min, v_max = v_max, v_min
        self.roi.setRange(v_min, v_max)

    def updateRoiRangeWidget(self):
        v_min, v_max = self.roi.getRange()
        if v_min is None or v_max is None:
            return

        self._roi_range.setRange(v_min, v_max)
