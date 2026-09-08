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

from silx.gui import qt

from .ModifierDoubleSpinBox import ModifierDoubleSpinBox


class PhaseColors(qt.QObject):
    changed = qt.Signal()
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf")

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rietveld refinement")
        self.setModal(False)
        self.resize(520, 720)

        self._wavelength = ModifierDoubleSpinBox(self)
        self._wavelength.setDecimals(6)
        self._wavelength.setRange(0.000001, 100.0)
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
        add_cifs = qt.QPushButton("Add CIFs…", self)
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
        self._refine_peak_width = qt.QCheckBox("Peak width W and Eta0", self)
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
        self._parameters.clear()

    def setResult(self, result, flags, indices=None):
        self._parameters.clear()
        if indices is None:
            history = result["history"][-1]
            map_index = None
        else:
            history = result["stages"][-1]
            map_index = (indices.row, indices.col)
        values = history["ref"]
        uncertainties = history["ref_std"]

        histogram = qt.QTreeWidgetItem(self._parameters, ["Histogram"])
        rwp = history["Rw"] if map_index is None else history["Rw"][map_index]
        rwp_item = qt.QTreeWidgetItem(
            histogram, ["Rwp [%]", f"{rwp:.7g}", ""]
        )
        if map_index is not None:
            rwp_item.setData(
                0, qt.Qt.ItemDataRole.UserRole, history["Rw"]
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
            if map_index is not None:
                rwp_net_item.setData(
                    0, qt.Qt.ItemDataRole.UserRole, history["Rw_net"]
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
            if map_index is not None:
                item.setData(0, qt.Qt.ItemDataRole.UserRole, value)

        for phase, phase_values in values["phases"].items():
            phase_item = qt.QTreeWidgetItem(self._parameters, [phase])
            phase_uncertainties = uncertainties["phases"][phase]
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
                if map_index is not None:
                    item.setData(0, qt.Qt.ItemDataRole.UserRole, value)
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
                    if map_index is not None:
                        item.setData(0, qt.Qt.ItemDataRole.UserRole, value)
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
                    if map_index is not None:
                        item.setData(
                            0,
                            qt.Qt.ItemDataRole.UserRole,
                            value * degrees(1.0),
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
                    if map_index is not None:
                        item.setData(0, qt.Qt.ItemDataRole.UserRole, value)

        self._parameters.expandAll()
        self._parameters.resizeColumnToContents(0)
