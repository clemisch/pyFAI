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
from silx.gui import icons, qt
from silx.gui.plot import PlotWidget
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
from .PlotColors import PLOT_COLORS


class IntegratedPatternPlotWidget(PlotWidget):
    refinementRequested = qt.Signal()
    reflectionOverlayRequested = qt.Signal()
    backgroundRequested = qt.Signal()

    def __init__(self, parent=None, backend=None):
        self._sqrt_mode = False
        self._curve_y_data = {}
        self._data_y_label = ""
        super().__init__(parent, backend)
        self.setDefaultColors(PLOT_COLORS)
        self.setDataMargins(0.02, 0.02, 0.02, 0.02)
        self.sigPlotSignal.connect(self.onRectDraw)

        self._roi_manager = RegionOfInterestManager(parent=self)
        self.roi = self._initRoi()
        self._roi_manager.addRoi(self.roi)

        self._roi_range = RoiRangeWidget(self)
        # Interconnect the ROI and the ROI range widget
        self._roi_range.updated.connect(self.roi.setRange)
        self.roi.sigRegionChanged.connect(self.updateRoiRangeWidget)
        self.fit_roi = HorizontalRangeROI()
        self.fit_roi.setEditable(True)
        self._roi_manager.addRoi(self.fit_roi)
        # RegionOfInterestManager assigns its default style when adding an ROI.
        self.fit_roi.setColor("#8000ff")
        # silx exposes no public option for hiding only the centre handle.
        self.fit_roi._markerCen.setVisible(False)
        self._fit_range = RoiRangeWidget(self, title="Fit bounds")
        self._fit_range.updated.connect(self.fit_roi.setRange)
        self.fit_roi.sigRegionChanged.connect(self.updateFitRangeWidget)

        self._toolbar = self._initToolbar()
        self.addToolBar(self._toolbar)

        self._legend_timer = qt.QTimer(self)
        self._legend_timer.setSingleShot(True)
        self._legend_timer.timeout.connect(self.updatePlotLegend)
        self.sigContentChanged.connect(self.scheduleLegendUpdate)

        self._statusBar = self._initStatusBar()
        centralWidget = self._initCentralWidget(self._statusBar)
        self.setCentralWidget(centralWidget)

        self._reflection_overlays = []
        self._reflection_items = []
        self._updating_reflection_items = False
        self.getYAxis().sigLimitsChanged.connect(self._updateReflectionItems)

    def __iter__(self):
        yield from self.getAllCurves(just_legend=True)

    def scheduleLegendUpdate(self, *args):
        self._legend_timer.start(0)

    def updatePlotLegend(self):
        from matplotlib.lines import Line2D

        backend = self.getBackend()
        if not hasattr(backend, "ax"):
            return
        handles = []
        for curve in self.getAllCurves():
            if not curve.isVisible():
                continue
            label = curve.getName()
            if label == "INTEGRATE":
                label = "Observed"
            elif label.startswith("Rietveld: "):
                label = label.removeprefix("Rietveld: ")
            handles.append(Line2D(
                [], [], color=curve.getColor(), linestyle=curve.getLineStyle(),
                label=label,
            ))
        for overlay in self._reflection_overlays:
            if not (overlay["reflections"] or overlay.get("sticks", [])) or not (
                overlay["show_ticks"] or overlay["show_lines"]
                or overlay.get("show_sticks", False)
            ):
                continue
            handles.append(Line2D(
                [], [], color=overlay["color"],
                linestyle="-" if overlay["show_lines"] else "none",
                marker=(
                    "|"
                    if overlay["show_ticks"] or overlay.get("show_sticks", False)
                    else None
                ),
                markersize=8,
                label=f"Reflections: {overlay['name']}",
            ))
        previous = backend.ax.get_legend()
        if previous is not None:
            previous.remove()
        if handles:
            legend = backend.ax.legend(handles=handles, loc="upper right", fontsize="small")
            legend.set_in_layout(False)
        backend.fig.canvas.draw_idle()

    def addDataCurve(self, x, y, legend, **kwargs):
        y = numpy.asarray(y)
        self._curve_y_data[legend] = numpy.array(y, copy=True)
        if self._sqrt_mode:
            y = numpy.sign(y) * numpy.sqrt(numpy.abs(y))
        return self.addCurve(x, y, legend=legend, **kwargs)

    def setDataYLabel(self, label):
        self._data_y_label = label
        if self._sqrt_mode:
            label = f"asqrt({label})"
        self.setGraphYLabel(label)

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
        roiAction = RoiModeAction(self, self.roi, toolbar)
        toolbar.addAction(roiAction)
        # Start in ROI mode
        roiAction.trigger()

        toolbar.addSeparator()
        toolbar.addAction(SaveAction(self, toolbar))
        toolbar.addSeparator()
        backgroundAction = qt.QAction(
            icons.getQIcon("math-substract"), "Background", toolbar
        )
        backgroundAction.setToolTip("Estimate or subtract histogram background")
        backgroundAction.triggered.connect(self.backgroundRequested)
        toolbar.addAction(backgroundAction)
        reflectionAction = qt.QAction(
            icons.getQIcon("math-peak"), "Expected reflections", toolbar
        )
        reflectionAction.setToolTip("Display expected reflection positions")
        reflectionAction.triggered.connect(self.reflectionOverlayRequested)
        toolbar.addAction(reflectionAction)
        refinementAction = qt.QAction(
            icons.getQIcon("math-fit"), "Rietveld refinement", toolbar
        )
        refinementAction.setToolTip("Open Rietveld refinement")
        refinementAction.triggered.connect(self.refinementRequested)
        toolbar.addAction(refinementAction)
        return toolbar

    def setReflectionOverlays(self, overlays):
        self._reflection_overlays = overlays
        self._updateReflectionItems()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_reflection_overlays") and self._reflection_overlays:
            qt.QTimer.singleShot(0, self._updateReflectionItems)

    def _updateReflectionItems(self, *args):
        if self._updating_reflection_items:
            return
        self._updating_reflection_items = True
        try:
            for legend, kind in self._reflection_items:
                self.remove(legend, kind=kind)
            self._reflection_items.clear()
            if not self._reflection_overlays:
                return

            left, top, width, height = self.getPlotBoundsInPixels()
            if width <= 0 or height <= 0:
                return
            bottom = top + height
            font = qt.QFont()
            font.setPointSize(7)

            for row, overlay in enumerate(self._reflection_overlays):
                reflections = overlay["reflections"]
                sticks = overlay.get("sticks", [])
                if not reflections and not sticks:
                    continue
                baseline_pixel = bottom - 7 - 19 * row
                top_pixel = baseline_pixel - 8
                baseline = self.pixelToData(left, baseline_pixel)[1]
                tick_top = self.pixelToData(left, top_pixel)[1]
                positions = numpy.asarray(
                    [reflection["position"] for reflection in reflections]
                )
                legend = f"Reflections: {overlay['name']}"
                if overlay["show_ticks"]:
                    tick_y = 0.5 * (baseline + tick_top)
                    for index, position in enumerate(positions):
                        marker_legend = f"{legend}: tick {index}"
                        marker = self.addMarker(
                            position,
                            tick_y,
                            legend=marker_legend,
                            color=overlay["color"],
                            symbol="|",
                            selectable=False,
                        )
                        marker.setSymbolSize(8)
                        self._reflection_items.append((marker_legend, "marker"))

                if overlay["show_lines"]:
                    color = qt.QColor(overlay["color"])
                    color.setAlphaF(0.6)
                    for index, position in enumerate(positions):
                        marker_legend = f"{legend}: line {index}"
                        marker = self.addXMarker(
                            position,
                            legend=marker_legend,
                            color=color,
                            selectable=False,
                        )
                        marker.setLineWidth(1.0)
                        self._reflection_items.append((marker_legend, "marker"))

                if overlay.get("show_sticks", False):
                    stick_baseline_pixel = bottom - 7
                    for index, stick in enumerate(sticks):
                        stick_height = (
                            height
                            * overlay.get("stick_height", 35.0)
                            / 100.0
                            * stick["intensity"]
                            / 100.0
                        )
                        center_pixel = stick_baseline_pixel - 0.5 * stick_height
                        center = self.pixelToData(left, center_pixel)[1]
                        symbol_size = (
                            stick_height * 72.0 / self.getBackend().fig.dpi
                        )
                        marker_legend = f"{legend}: stick {index}"
                        marker = self.addMarker(
                            stick["position"],
                            center,
                            legend=marker_legend,
                            color=overlay["color"],
                            symbol="|",
                            selectable=False,
                        )
                        marker.setSymbolSize(symbol_size)
                        self._reflection_items.append((marker_legend, "marker"))

                if overlay["show_labels"]:
                    for index, reflection in enumerate(reflections):
                        hkls = reflection["hkls"]
                        if not hkls:
                            continue
                        hkl = ",".join(str(value) for value in hkls[0])
                        suffix = f"+{len(hkls) - 1}" if len(hkls) > 1 else ""
                        marker_legend = f"{legend}: {index}"
                        marker = self.addMarker(
                            reflection["position"],
                            tick_top,
                            legend=marker_legend,
                            text=f"({hkl}){suffix}",
                            color=overlay["color"],
                            symbol=",",
                            selectable=False,
                        )
                        marker.setSymbolSize(1)
                        marker.setFont(font)
                        self._reflection_items.append((marker_legend, "marker"))
        finally:
            self._updating_reflection_items = False

    def setYAxisScale(self, scale):
        axis = self.getYAxis()
        backend = self.getBackend()
        if scale == "signed_sqrt":
            if axis.getScale() != "linear":
                axis.setScale("linear")
            self._sqrt_mode = True
            icon = "math-amplitude"
            tooltip = "Y data is transformed with signed square root"
        else:
            self._sqrt_mode = False
            if axis.getScale() == scale:
                backend.setYAxisScale(scale)
            else:
                axis.setScale(scale)
            icon = f"yscale-{scale}"
            tooltip = f"Y-axis scale is {scale}"

        for curve in self.getAllCurves():
            y = self._curve_y_data.get(curve.getName())
            if y is None:
                continue
            if self._sqrt_mode:
                y = numpy.sign(y) * numpy.sqrt(numpy.abs(y))
            curve.setData(
                curve.getXData(copy=False),
                y,
                xerror=curve.getXErrorData(copy=False),
                yerror=curve.getYErrorData(copy=False),
                baseline=curve.getBaseline(copy=False),
                copy=False,
            )

        label = self._data_y_label
        if self._sqrt_mode:
            label = f"asqrt({label})"
        self.setGraphYLabel(label)

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
        gridLayout.addWidget(self.getWidgetHandle(), 0, 0, 1, 2)
        gridLayout.addWidget(status_bar, 1, 0, 1, 2)
        gridLayout.addWidget(self._roi_range, 2, 0)
        gridLayout.addWidget(self._fit_range, 2, 1)

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

    def updateFitRangeWidget(self):
        self._fit_range.setRange(*self.fit_roi.getRange())

    def updateRoiRangeWidget(self):
        v_min, v_max = self.roi.getRange()
        if v_min is None or v_max is None:
            return

        self._roi_range.setRange(v_min, v_max)
