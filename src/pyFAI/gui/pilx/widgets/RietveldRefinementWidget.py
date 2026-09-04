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

"""Controls and asynchronous execution for single-pattern refinement."""

from __future__ import annotations

import traceback
from pathlib import Path

from silx.gui import qt


class RietveldRefinementThread(qt.QThread):

    def __init__(self, inputs, indices, parent=None):
        super().__init__(parent)
        self.inputs = inputs
        self.indices = indices
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


class RietveldRefinementWidget(qt.QWidget):
    refinementRequested = qt.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._wavelength = qt.QDoubleSpinBox(self)
        self._wavelength.setDecimals(6)
        self._wavelength.setRange(0.000001, 100.0)
        self._wavelength.setSuffix(" Å")

        self._ttheta_min = qt.QDoubleSpinBox(self)
        self._ttheta_min.setDecimals(4)
        self._ttheta_min.setSuffix("°")
        self._ttheta_max = qt.QDoubleSpinBox(self)
        self._ttheta_max.setDecimals(4)
        self._ttheta_max.setSuffix("°")

        form = qt.QFormLayout()
        form.addRow("Wavelength λ", self._wavelength)
        form.addRow("Minimum 2θ", self._ttheta_min)
        form.addRow("Maximum 2θ", self._ttheta_max)

        self._cifs = qt.QListWidget(self)
        self._cifs.setSelectionMode(
            qt.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._cifs.setMinimumHeight(100)
        add_cifs = qt.QPushButton("Add CIFs…", self)
        add_cifs.clicked.connect(self._addCifs)
        remove_cifs = qt.QPushButton("Remove selected", self)
        remove_cifs.clicked.connect(self._removeSelectedCifs)
        cif_buttons = qt.QHBoxLayout()
        cif_buttons.addWidget(add_cifs)
        cif_buttons.addWidget(remove_cifs)

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
        self._run_button.clicked.connect(self.refinementRequested)
        self._status = qt.QLabel("Ready", self)
        self._status.setWordWrap(True)

        layout = qt.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(qt.QLabel("Phases", self))
        layout.addWidget(self._cifs)
        layout.addLayout(cif_buttons)
        layout.addWidget(parameters)
        layout.addWidget(self._run_button)
        layout.addWidget(self._status)
        layout.addStretch(1)

    def _addCifs(self):
        filenames, _ = qt.QFileDialog.getOpenFileNames(
            self,
            "Select phase CIFs",
            "",
            "Crystallographic information files (*.cif);;All files (*)",
        )
        existing = set(self.cifPaths())
        for filename in filenames:
            if filename not in existing:
                item = qt.QListWidgetItem(Path(filename).name)
                item.setData(qt.Qt.ItemDataRole.UserRole, filename)
                item.setToolTip(filename)
                self._cifs.addItem(item)
                existing.add(filename)

    def _removeSelectedCifs(self):
        for item in self._cifs.selectedItems():
            self._cifs.takeItem(self._cifs.row(item))

    def setWavelength(self, wavelength_A):
        self._wavelength.setValue(wavelength_A)

    def wavelength(self):
        return self._wavelength.value()

    def setRadialRange(self, minimum, maximum):
        self._ttheta_min.setRange(minimum, maximum)
        self._ttheta_max.setRange(minimum, maximum)
        refinement_minimum = max(minimum, 3.0)
        refinement_maximum = min(maximum, 40.0)
        if refinement_minimum >= refinement_maximum:
            refinement_minimum = minimum
            refinement_maximum = maximum
        self._ttheta_min.setValue(refinement_minimum)
        self._ttheta_max.setValue(refinement_maximum)

    def radialRange(self):
        return self._ttheta_min.value(), self._ttheta_max.value()

    def setCifPaths(self, paths):
        self._cifs.clear()
        for path in paths:
            path = str(path)
            item = qt.QListWidgetItem(Path(path).name)
            item.setData(qt.Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self._cifs.addItem(item)

    def cifPaths(self):
        return [
            self._cifs.item(index).data(qt.Qt.ItemDataRole.UserRole)
            for index in range(self._cifs.count())
        ]

    def refinementFlags(self):
        return {
            "scale": self._refine_scale.isChecked(),
            "displacement": self._refine_displacement.isChecked(),
            "unit_cell": self._refine_unit_cell.isChecked(),
            "peak_width": self._refine_peak_width.isChecked(),
        }

    def setRunning(self, running):
        self._run_button.setEnabled(not running)
        self._run_button.setText(
            "Refinement running…" if running else "Refine selected point"
        )

    def setStatus(self, text):
        self._status.setText(text)
