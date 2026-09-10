#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2026 European Synchrotron Radiation Facility, Grenoble, France
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

"""Controls and asynchronous execution for Rietveld refinement."""

from __future__ import annotations

import pickle
import traceback
from math import degrees
from pathlib import Path

import numpy
from silx.gui import qt

from .ModifierDoubleSpinBox import ModifierDoubleSpinBox
from .PlotColors import PLOT_COLORS


class PhaseColors(qt.QObject):
    changed = qt.Signal()
    palette = PLOT_COLORS[1:]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors = {}

    def get(self, path):
        key = str(Path(path).resolve())
        if key not in self.colors:
            self.colors[key] = self.palette[len(self.colors) % len(self.palette)]
        return self.colors[key]

    def set(self, path, color):
        self.colors[str(Path(path).resolve())] = color
        self.changed.emit()


class PhaseEosSettings(qt.QObject):
    """Equation-of-state settings shared by CIF path across GUI tools."""

    changed = qt.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = {}

    def get(self, path, reference=None):
        key = str(Path(path).resolve())
        if key not in self._settings:
            self._settings[key] = {
                "eos": {
                    "model": None,
                    "pressure": 0.0,
                    "k0": 160.0,
                    "k0p": 4.0,
                    "p0": 0.0,
                },
                "reference": reference,
                "revision": 0,
            }
        elif self._settings[key]["reference"] is None and reference is not None:
            self._settings[key]["reference"] = reference
        return self._settings[key]

    def notifyChanged(self, path):
        key = str(Path(path).resolve())
        settings = self._settings.get(key)
        if settings is None:
            return
        settings["revision"] += 1
        self.changed.emit(key)


class RietveldRefinementThread(qt.QThread):

    def __init__(self, inputs, indices, refinement_flags, parent=None):
        super().__init__(parent)
        self.inputs = inputs
        self.indices = indices
        self.refinement_flags = refinement_flags
        self.result = None
        self.error = None

    def run(self):
        try:
            from ewokscore import execute_graph

            graph = {
                "graph": {"id": "rietveld_refine_single", "schema_version": "1.2"},
                "nodes": [
                    {
                        "id": "refinement",
                        "task_type": "class",
                        "task_identifier": (
                            "ewoksxrpd.tasks.rietveld.RietveldRefineSingle"
                        ),
                        "default_inputs": [
                            {"name": name, "value": value}
                            for name, value in self.inputs.items()
                        ],
                    }
                ],
                "links": [],
            }
            outputs = execute_graph(
                graph,
                outputs=[{"id": "refinement", "name": "result"}],
            )
            self.result = outputs["result"]
        except Exception:
            self.error = traceback.format_exc()


class RietveldRefinementProcess(qt.QProcess):
    completed = qt.Signal()

    def __init__(self, python_executable, graph, refinement_flags, parent=None):
        super().__init__(parent)
        self.python_executable = python_executable
        self.refinement_flags = refinement_flags
        self.result = None
        self.error_text = None
        self._request = pickle.dumps(
            {
                "graph": graph,
                "outputs": [{"id": "refinement", "name": "result"}],
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        self.started.connect(self._sendRequest)
        self.finished.connect(self._readResult)
        self.errorOccurred.connect(self._processError)

    def startRefinement(self):
        self.start(
            self.python_executable,
            ["-m", "ewoksxrpd.tasks.execute_subprocess"],
        )

    def _sendRequest(self):
        self.write(self._request)
        self.closeWriteChannel()
        self._request = None

    def _readResult(self, exit_code, exit_status):
        if self.error_text is None:
            stderr = bytes(self.readAllStandardError()).decode(errors="replace")
            if exit_status != qt.QProcess.ExitStatus.NormalExit or exit_code != 0:
                self.error_text = stderr.strip() or (
                    f"refinement worker exited with code {exit_code}"
                )
            else:
                try:
                    outputs = pickle.loads(bytes(self.readAllStandardOutput()))
                    self.result = outputs["result"]
                except Exception:
                    self.error_text = traceback.format_exc()
                    if stderr:
                        self.error_text += "\nWorker output:\n" + stderr
        self.completed.emit()

    def _processError(self, error):
        if error == qt.QProcess.ProcessError.FailedToStart:
            self.error_text = self.errorString()
            self.completed.emit()


class CifListWidget(qt.QTreeWidget):
    filesDropped = qt.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(qt.QAbstractItemView.DragDropMode.DropOnly)

    @staticmethod
    def _cifPaths(mime_data):
        return [
            url.toLocalFile()
            for url in mime_data.urls()
            if url.isLocalFile() and url.toLocalFile().lower().endswith(".cif")
        ]

    def dragEnterEvent(self, event):
        if self._cifPaths(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._cifPaths(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = self._cifPaths(event.mimeData())
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class RietveldRefinementDialog(qt.QDialog):
    displayChanged = qt.Signal()
    refinementRequested = qt.Signal()
    mapRefinementRequested = qt.Signal()
    mapRequested = qt.Signal(str, object)
    mapUpdated = qt.Signal(str, object)

    def __init__(self, parent=None, phase_eos=None):
        super().__init__(parent)
        self.setWindowTitle("Rietveld refinement")
        self.setModal(False)
        self.resize(520, 720)

        self._wavelength = ModifierDoubleSpinBox(self)
        self._wavelength.setDecimals(6)
        self._wavelength.setRange(0.000001, 100.0)
        self._wavelength.setSingleStep(0.1)
        self._wavelength_lock = qt.QPushButton("Lock", self)
        self._wavelength_lock.setCheckable(True)
        self._wavelength_lock.setChecked(True)
        self._wavelength_lock.setToolTip("Lock wavelength editing")
        self._wavelength_lock.toggled.connect(self._wavelength.setDisabled)
        self._wavelength.setDisabled(True)
        self._file_wavelength = None
        self._load_wavelength = qt.QPushButton("From file", self)
        self._load_wavelength.setEnabled(False)
        self._load_wavelength.setToolTip(
            "Restore the wavelength from the diffmap integration configuration"
        )
        self._load_wavelength.clicked.connect(self._restoreFileWavelength)
        wavelength_layout = qt.QHBoxLayout()
        wavelength_layout.setContentsMargins(0, 0, 0, 0)
        wavelength_layout.addWidget(self._wavelength)
        wavelength_layout.addWidget(self._wavelength_lock)
        wavelength_layout.addWidget(self._load_wavelength)
        wavelength_widget = qt.QWidget(self)
        wavelength_widget.setLayout(wavelength_layout)

        self._ttheta_range = (0.0, 180.0)
        self.phase_eos = (
            phase_eos if phase_eos is not None else PhaseEosSettings(self)
        )
        self.phase_eos.changed.connect(self._eosChanged)
        self._phase_paths = {}
        self._result = None
        self._result_flags = None
        self._result_indices = None
        self._derived_maps = {}
        self._mapped_result = None
        self._mapped_flags = None
        self._mapped_indices = None
        self._mapped_phase_paths = {}
        self._pressure_cache = {}

        form = qt.QFormLayout()
        form.addRow("Wavelength [Å]", wavelength_widget)

        self._cifs = CifListWidget(self)
        self._cifs.setHeaderLabels(("Fit", "Show", "Name"))
        self._cifs.setRootIsDecorated(False)
        for column in (0, 1):
            self._cifs.header().setSectionResizeMode(column, qt.QHeaderView.ResizeMode.ResizeToContents)
        self._cifs.itemChanged.connect(self._phaseDisplayChanged)
        self.phase_colors = PhaseColors(self)
        self.phase_colors.changed.connect(self.refreshPhaseColors)
        self._cifs.setAlternatingRowColors(True)
        self._cifs.setContextMenuPolicy(qt.Qt.ContextMenuPolicy.CustomContextMenu)
        self._cifs.customContextMenuRequested.connect(self.choosePhaseColor)
        self._cifs.setSelectionMode(
            qt.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._cifs.setMinimumHeight(100)
        self._cifs.filesDropped.connect(self.addCifPaths)
        add_cifs = qt.QPushButton("Add CIF", self)
        add_cifs.clicked.connect(self._addCifs)
        remove_cifs = qt.QPushButton("Remove selected", self)
        remove_cifs.clicked.connect(self._removeSelectedCifs)
        cif_buttons = qt.QHBoxLayout()
        cif_buttons.addWidget(add_cifs)
        cif_buttons.addWidget(remove_cifs)
        self.show_total = qt.QCheckBox("Show total", self)
        self.show_background = qt.QCheckBox("Show bkg", self)
        for checkbox in (self.show_total, self.show_background):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self.displayChanged)
            cif_buttons.addWidget(checkbox)

        self._refine_scale = qt.QCheckBox("Phase scales", self)
        self._refine_scale.setChecked(True)
        self._refine_displacement = qt.QCheckBox("Sample displacement", self)
        self._refine_displacement.setChecked(True)
        self._refine_unit_cell = qt.QCheckBox("Unit cells", self)
        self._refine_unit_cell.setChecked(True)
        self._refine_peak_width = qt.QCheckBox("FWHM W & Shape Eta0", self)
        self._refine_peak_width.setChecked(True)

        parameters = qt.QGroupBox("Refine", self)
        parameters_layout = qt.QVBoxLayout(parameters)
        parameters_layout.addWidget(self._refine_scale)
        parameters_layout.addWidget(self._refine_displacement)
        parameters_layout.addWidget(self._refine_unit_cell)
        parameters_layout.addWidget(self._refine_peak_width)

        self._run_button = qt.QPushButton("Refine selected point", self)
        self._run_button.setCheckable(True)
        self._run_button.setToolTip("While enabled, fit the selected point using the current settings")
        self._run_button.toggled.connect(self.refinementRequested)
        for edit in (self._wavelength,):
            edit.valueChanged.connect(self.refinementRequested)
        for checkbox in (self._refine_scale, self._refine_displacement,
                         self._refine_unit_cell, self._refine_peak_width):
            checkbox.toggled.connect(self.refinementRequested)
        self._run_map_button = qt.QPushButton("Refine all points", self)
        self._run_map_button.clicked.connect(self.mapRefinementRequested)
        run_buttons = qt.QHBoxLayout()
        run_buttons.addWidget(self._run_button)
        run_buttons.addWidget(self._run_map_button)

        self._parameters = qt.QTreeWidget(self)
        self._parameters.setColumnCount(3)
        self._parameters.setHeaderLabels(("Parameter", "Value", "σ"))
        self._parameters.setAlternatingRowColors(True)
        self._parameters.setMinimumHeight(180)
        self._parameters.itemDoubleClicked.connect(self._requestMap)


        layout = qt.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._cifs)
        layout.addLayout(cif_buttons)
        layout.addWidget(parameters)
        layout.addLayout(run_buttons)
        layout.addWidget(self._parameters)

    def refreshPhaseColors(self):
        for index in range(self._cifs.topLevelItemCount()):
            item = self._cifs.topLevelItem(index)
            path = item.data(2, qt.Qt.ItemDataRole.UserRole)
            item.setForeground(2, qt.QColor(self.phase_colors.get(path)))

    def _phaseDisplayChanged(self, item, column):
        if column == 1:
            self.displayChanged.emit()
        elif column == 0:
            self.refinementRequested.emit()

    def shownCifPaths(self):
        paths = set()
        for index in range(self._cifs.topLevelItemCount()):
            item = self._cifs.topLevelItem(index)
            if item.checkState(1) == qt.Qt.CheckState.Checked:
                paths.add(str(Path(item.data(2, qt.Qt.ItemDataRole.UserRole)).resolve()))
        return paths

    def choosePhaseColor(self, position):
        item = self._cifs.itemAt(position)
        if item is None:
            return
        menu = qt.QMenu(self)
        action = menu.addAction("Choose color…")
        if menu.exec(self._cifs.viewport().mapToGlobal(position)) == action:
            path = item.data(2, qt.Qt.ItemDataRole.UserRole)
            color = qt.QColorDialog.getColor(qt.QColor(self.phase_colors.get(path)), self)
            if color.isValid():
                self.phase_colors.set(path, color.name())

    def _addCifs(self):
        filenames, _ = qt.QFileDialog.getOpenFileNames(
            self,
            "Select phase CIFs",
            "",
            "Crystallographic information files (*.cif);;All files (*)",
        )
        self.addCifPaths(filenames)

    def _removeSelectedCifs(self):
        for item in self._cifs.selectedItems():
            self._cifs.takeTopLevelItem(self._cifs.indexOfTopLevelItem(item))
        self.displayChanged.emit()
        self.refinementRequested.emit()

    def setWavelength(self, wavelength_A):
        self._file_wavelength = wavelength_A
        if wavelength_A is None:
            self._load_wavelength.setEnabled(False)
            self._load_wavelength.setToolTip(
                "The diffmap integration configuration has no wavelength"
            )
            return
        self._wavelength.setValue(wavelength_A)
        self._load_wavelength.setEnabled(True)
        self._load_wavelength.setToolTip(
            f"Restore the file wavelength ({wavelength_A:.6g} Å)"
        )

    def _restoreFileWavelength(self):
        if self._file_wavelength is not None:
            self._wavelength.setValue(self._file_wavelength)

    def wavelength(self):
        return self._wavelength.value()

    def setRadialRange(self, minimum, maximum):
        self._ttheta_range = (minimum, maximum)
        self.refinementRequested.emit()

    def radialRange(self):
        return self._ttheta_range

    def setCifPaths(self, paths):
        self._cifs.clear()
        self.addCifPaths(paths)

    def addCifPaths(self, paths):
        existing = set(self.cifPaths())
        for path in paths:
            path = str(path)
            if path not in existing:
                item = qt.QTreeWidgetItem(["", "", Path(path).stem])
                item.setFlags(item.flags() | qt.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, qt.Qt.CheckState.Checked)
                item.setCheckState(1, qt.Qt.CheckState.Checked)
                item.setData(2, qt.Qt.ItemDataRole.UserRole, path)
                item.setToolTip(2, path)
                self._cifs.addTopLevelItem(item)
                item.setForeground(2, qt.QColor(self.phase_colors.get(path)))
                existing.add(path)
        self.refinementRequested.emit()

    def cifPaths(self):
        return [
            self._cifs.topLevelItem(index).data(2, qt.Qt.ItemDataRole.UserRole)
            for index in range(self._cifs.topLevelItemCount())
        ]

    def enabledCifPaths(self):
        return [
            self._cifs.topLevelItem(index).data(2, qt.Qt.ItemDataRole.UserRole)
            for index in range(self._cifs.topLevelItemCount())
            if self._cifs.topLevelItem(index).checkState(0) == qt.Qt.CheckState.Checked
        ]

    def refinementFlags(self):
        return {
            "scale": self._refine_scale.isChecked(),
            "displacement": self._refine_displacement.isChecked(),
            "unit_cell": self._refine_unit_cell.isChecked(),
            "peak_width": self._refine_peak_width.isChecked(),
        }

    def setRunning(self, running, mapped=False):
        self._run_map_button.setEnabled(not running)
        self._run_map_button.setText(
            "Refining all points…"
            if running and mapped
            else "Refine all points"
        )

    def _requestMap(self, item):
        map_data = item.data(0, qt.Qt.ItemDataRole.UserRole)
        if map_data is not None:
            title = item.text(0)
            if title.endswith("]") and " [" in title:
                title = title.rsplit(" [", 1)[0]
            if item.parent() is not None and item.parent().text(0) != "Histogram":
                title = f"{item.parent().text(0)}: {title}"
            self.mapRequested.emit(title, map_data)

    def clearResult(self):
        self._result = None
        self._result_flags = None
        self._result_indices = None
        self._derived_maps.clear()
        self._mapped_result = None
        self._mapped_flags = None
        self._mapped_indices = None
        self._mapped_phase_paths = {}
        self._pressure_cache.clear()
        self._parameters.clear()

    def setPhasePaths(self, paths):
        self._phase_paths = {
            phase: str(Path(path).resolve()) for phase, path in paths.items()
        }

    def setResult(self, result, flags, indices=None):
        self._result = result
        self._result_flags = flags
        self._result_indices = indices
        if "stages" in result:
            self._mapped_result = result
            self._mapped_flags = flags
            self._mapped_indices = indices
            self._mapped_phase_paths = dict(self._phase_paths)
        self._derived_maps.clear()
        self._parameters.clear()
        if indices is None:
            history = result["history"][-1]
            map_index = None
        else:
            history = result["stages"][-1]
            map_index = (indices.row, indices.col)
        values = history["ref"]
        uncertainties = history["ref_std"]
        mapped_history = (
            None
            if self._mapped_result is None
            else self._mapped_result["stages"][-1]
        )
        mapped_values = (
            None if mapped_history is None else mapped_history["ref"]
        )

        histogram = qt.QTreeWidgetItem(self._parameters, ["Histogram"])
        rwp = history["Rw"] if map_index is None else history["Rw"][map_index]
        rwp_item = qt.QTreeWidgetItem(
            histogram, ["Rwp [%]", f"{rwp:.7g}", ""]
        )
        if map_index is not None or mapped_history is not None:
            map_data = (
                history["Rw"] if map_index is not None else mapped_history["Rw"]
            )
            rwp_item.setData(
                0, qt.Qt.ItemDataRole.UserRole, map_data
            )
        if "Rw_net" in history:
            rwp_net = (
                history["Rw_net"]
                if map_index is None
                else history["Rw_net"][map_index]
            )
            rwp_net_item = qt.QTreeWidgetItem(
                histogram,
                ["Rwp (no bkg) [%]", f"{rwp_net:.7g}", ""],
            )
            if map_index is not None or (
                mapped_history is not None and "Rw_net" in mapped_history
            ):
                map_data = (
                    history["Rw_net"]
                    if map_index is not None
                    else mapped_history["Rw_net"]
                )
                rwp_net_item.setData(
                    0, qt.Qt.ItemDataRole.UserRole, map_data
                )
        if flags["displacement"]:
            value = values["pp"]["2ThetaFlatDetDispRatio"]
            uncertainty = uncertainties["pp"]["2ThetaFlatDetDispRatio"]
            display_value = value if map_index is None else value[map_index]
            display_uncertainty = (
                uncertainty
                if map_index is None
                else uncertainty[map_index]
            )
            item = qt.QTreeWidgetItem(
                histogram,
                [
                    "Sample displacement ratio",
                    f"{display_value:.7g}",
                    f"{display_uncertainty:.3g}",
                ],
            )
            if map_index is not None or (
                mapped_values is not None
                and self._mapped_flags["displacement"]
            ):
                map_data = (
                    value
                    if map_index is not None
                    else mapped_values["pp"]["2ThetaFlatDetDispRatio"]
                )
                item.setData(0, qt.Qt.ItemDataRole.UserRole, map_data)

        for phase, phase_values in values["phases"].items():
            phase_item = qt.QTreeWidgetItem(self._parameters, [phase])
            phase_uncertainties = uncertainties["phases"][phase]
            mapped_phase_values = (
                None
                if mapped_values is None
                else mapped_values["phases"].get(phase)
            )
            if flags["scale"]:
                value = values["scales"][phase]
                uncertainty = uncertainties["scales"][phase]
                display_value = value if map_index is None else value[map_index]
                display_uncertainty = (
                    uncertainty
                    if map_index is None
                    else uncertainty[map_index]
                )
                item = qt.QTreeWidgetItem(
                    phase_item,
                    [
                        "Scale",
                        f"{display_value:.7g}",
                        f"{display_uncertainty:.3g}",
                    ],
                )
                if map_index is not None or (
                    mapped_phase_values is not None
                    and self._mapped_flags["scale"]
                ):
                    map_data = (
                        value
                        if map_index is not None
                        else mapped_values["scales"][phase]
                    )
                    item.setData(0, qt.Qt.ItemDataRole.UserRole, map_data)
            if flags["unit_cell"]:
                for parameter in ("a", "b", "c"):
                    value = phase_values[parameter]
                    uncertainty = phase_uncertainties[parameter]
                    display_value = value if map_index is None else value[map_index]
                    display_uncertainty = (
                        uncertainty
                        if map_index is None
                        else uncertainty[map_index]
                    )
                    item = qt.QTreeWidgetItem(
                        phase_item,
                        [
                            f"{parameter} [Å]",
                            f"{display_value:.7g}",
                            f"{display_uncertainty:.3g}",
                        ],
                    )
                    if map_index is not None or (
                        mapped_phase_values is not None
                        and self._mapped_flags["unit_cell"]
                    ):
                        map_data = (
                            value
                            if map_index is not None
                            else mapped_phase_values[parameter]
                        )
                        item.setData(0, qt.Qt.ItemDataRole.UserRole, map_data)
                for parameter, label in (
                    ("alpha", "α"),
                    ("beta", "β"),
                    ("gamma", "γ"),
                ):
                    value = phase_values[parameter]
                    uncertainty = phase_uncertainties[parameter]
                    display_value = value if map_index is None else value[map_index]
                    display_uncertainty = (
                        uncertainty
                        if map_index is None
                        else uncertainty[map_index]
                    )
                    item = qt.QTreeWidgetItem(
                        phase_item,
                        [
                            f"{label} [°]",
                            f"{degrees(display_value):.7g}",
                            f"{degrees(display_uncertainty):.3g}",
                        ],
                    )
                    if map_index is not None or (
                        mapped_phase_values is not None
                        and self._mapped_flags["unit_cell"]
                    ):
                        map_data = (
                            value
                            if map_index is not None
                            else mapped_phase_values[parameter]
                        )
                        item.setData(
                            0,
                            qt.Qt.ItemDataRole.UserRole,
                            map_data * degrees(1.0),
                        )

                volume = phase_values.get("cell_vol_A3")
                if volume is not None:
                    display_volume = volume if map_index is None else volume[map_index]
                    item = qt.QTreeWidgetItem(
                        phase_item,
                        ["Volume [Å³]", f"{display_volume:.7g}", ""],
                    )
                    if map_index is not None or (
                        mapped_phase_values is not None
                        and self._mapped_flags["unit_cell"]
                    ):
                        map_data = (
                            volume
                            if map_index is not None
                            else mapped_phase_values["cell_vol_A3"]
                        )
                        item.setData(0, qt.Qt.ItemDataRole.UserRole, map_data)
                    if map_index is not None:
                        self._derived_maps[f"{phase}: Volume"] = volume

                    path = self._phase_paths.get(phase)
                    eos_settings = (
                        None if path is None else self.phase_eos.get(path)
                    )
                    if (
                        eos_settings is not None
                        and eos_settings["eos"]["model"] is not None
                        and eos_settings["reference"] is not None
                    ):
                        from ....crystallography.cell import Cell
                        from ....crystallography.eos import EquationOfState

                        reference = eos_settings["reference"]
                        reference_volume = Cell(**reference).volume
                        settings = eos_settings["eos"]
                        eos = EquationOfState.factory(
                            settings["model"],
                            k0=settings["k0"],
                            k0p=settings["k0p"],
                            p0=settings["p0"],
                            v0=reference_volume,
                        )
                        cached = self._pressure_cache.get(phase)
                        if (
                            map_index is not None
                            and cached is not None
                            and cached[0] is result
                            and cached[1] == eos_settings["revision"]
                        ):
                            pressure = cached[2]
                        else:
                            volume_array = numpy.asarray(volume)
                            pressure = numpy.full(volume_array.shape, numpy.nan)
                            valid = numpy.isfinite(volume_array) & (volume_array > 0.0)
                            for index in numpy.flatnonzero(valid):
                                try:
                                    pressure.flat[index] = eos.pressure(
                                        volume=volume_array.flat[index]
                                    )
                                except (OverflowError, ValueError, ZeroDivisionError):
                                    pass
                            if volume_array.ndim == 0:
                                pressure = pressure.item()
                            if map_index is not None:
                                self._pressure_cache[phase] = (
                                    result,
                                    eos_settings["revision"],
                                    pressure,
                                )
                        display_pressure = (
                            pressure if map_index is None else pressure[map_index]
                        )
                        item = qt.QTreeWidgetItem(
                            phase_item,
                            ["Pressure [GPa]", f"{display_pressure:.7g}", ""],
                        )
                        if map_index is not None:
                            item.setData(0, qt.Qt.ItemDataRole.UserRole, pressure)
                            self._derived_maps[f"{phase}: Pressure"] = pressure
                        elif (
                            mapped_phase_values is not None
                            and self._mapped_flags["unit_cell"]
                            and cached is not None
                            and cached[0] is self._mapped_result
                            and cached[1] == eos_settings["revision"]
                        ):
                            item.setData(
                                0, qt.Qt.ItemDataRole.UserRole, cached[2]
                            )
            if flags["peak_width"]:
                for parameter, label in (("W", "W [rad²]"), ("Eta0", "Eta0")):
                    value = phase_values[parameter]
                    uncertainty = phase_uncertainties[parameter]
                    display_value = value if map_index is None else value[map_index]
                    display_uncertainty = (
                        uncertainty
                        if map_index is None
                        else uncertainty[map_index]
                    )
                    item = qt.QTreeWidgetItem(
                        phase_item,
                        [
                            label,
                            f"{display_value:.7g}",
                            f"{display_uncertainty:.3g}",
                        ],
                    )
                    if map_index is not None or (
                        mapped_phase_values is not None
                        and self._mapped_flags["peak_width"]
                    ):
                        map_data = (
                            value
                            if map_index is not None
                            else mapped_phase_values[parameter]
                        )
                        item.setData(0, qt.Qt.ItemDataRole.UserRole, map_data)

        self._parameters.expandAll()
        self._parameters.resizeColumnToContents(0)

    def _eosChanged(self, path):
        phase_paths = (
            self._mapped_phase_paths
            if self._mapped_result is not None
            else self._phase_paths
        )
        if self._result is None or path not in phase_paths.values():
            return
        current_result = self._result
        current_flags = self._result_flags
        current_indices = self._result_indices
        current_phase_paths = self._phase_paths
        affected_phases = [
            phase for phase, phase_path in phase_paths.items()
            if phase_path == path
        ]
        if self._mapped_result is not None:
            result = self._mapped_result
            flags = self._mapped_flags
            indices = self._mapped_indices
            self._phase_paths = self._mapped_phase_paths
        else:
            result = current_result
            flags = current_flags
            indices = current_indices
        self.setResult(
            result,
            flags,
            indices=indices,
        )
        for phase in affected_phases:
            title = f"{phase}: Pressure"
            self.mapUpdated.emit(title, self._derived_maps.get(title))
        if current_result is not result:
            self._phase_paths = current_phase_paths
            self.setResult(current_result, current_flags, indices=current_indices)
