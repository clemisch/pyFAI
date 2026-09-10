#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2026 European Synchrotron Radiation Facility
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

"""Controls for displaying expected powder-reflection positions."""

from __future__ import annotations

import logging
from pathlib import Path
import warnings

import numpy
from silx.gui import qt

from .ModifierDoubleSpinBox import ModifierDoubleSpinBox
from .PlotColors import PLOT_COLORS


_logger = logging.getLogger(__name__)


class ReflectionPhaseList(qt.QTreeWidget):
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


class ReflectionOverlayDialog(qt.QDialog):
    overlayChanged = qt.Signal(object)

    _COLORS = PLOT_COLORS[1:]

    def __init__(self, parent=None, phase_colors=None, phase_eos=None):
        super().__init__(parent)
        self.phase_colors = phase_colors
        self.phase_eos = phase_eos
        if phase_colors is not None:
            phase_colors.changed.connect(self.refreshPhaseColors)
        self.setWindowTitle("Expected reflections")
        self.setModal(False)
        self.resize(600, 800)

        from pymatgen.core import Lattice
        from pymatgen.core import Structure
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        from pymatgen.io.cif import CifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number

        from ....crystallography.eos import BirchMurnaghan, Murnaghan, Vinet

        self._Lattice = Lattice
        self._Structure = Structure
        self._XRDCalculator = XRDCalculator
        self._CifParser = CifParser
        self._SpaceGroup = SpaceGroup
        self._SpacegroupAnalyzer = SpacegroupAnalyzer
        self._eos_classes = {
            "Birch-Murnaghan": BirchMurnaghan,
            "Vinet": Vinet,
            "Murnaghan": Murnaghan,
        }
        self._phases = []
        self._file_wavelength = None
        self._ttheta_range = (0.0, 90.0)
        self._updating_controls = False

        self._update_timer = qt.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(120)
        self._update_timer.timeout.connect(self._updateReflections)

        self._wavelength = ModifierDoubleSpinBox(self)
        self._wavelength.setDecimals(6)
        self._wavelength.setRange(0.000001, 100.0)
        self._wavelength.setSingleStep(0.1)
        self._wavelength.valueChanged.connect(self._invalidateAllPhases)
        self._wavelength_lock = qt.QPushButton("Lock", self)
        self._wavelength_lock.setCheckable(True)
        self._wavelength_lock.setChecked(True)
        self._wavelength_lock.setToolTip("Lock wavelength editing")
        self._wavelength_lock.toggled.connect(self._wavelength.setDisabled)
        self._wavelength.setDisabled(True)
        self._load_wavelength = qt.QPushButton("From file", self)
        self._load_wavelength.setEnabled(False)
        self._load_wavelength.clicked.connect(self._restoreFileWavelength)
        wavelength_layout = qt.QHBoxLayout()
        wavelength_layout.setContentsMargins(0, 0, 0, 0)
        wavelength_layout.addWidget(self._wavelength)
        wavelength_layout.addWidget(self._wavelength_lock)
        wavelength_layout.addWidget(self._load_wavelength)
        wavelength_widget = qt.QWidget(self)
        wavelength_widget.setLayout(wavelength_layout)

        general_form = qt.QFormLayout()
        general_form.addRow("Wavelength [Å]", wavelength_widget)

        self._phase_list = ReflectionPhaseList(self)
        self._phase_list.setColumnCount(3)
        self._phase_list.setHeaderLabels(("Phase", "Source", "Space group"))
        self._phase_list.setRootIsDecorated(False)
        self._phase_list.setAlternatingRowColors(True)
        self._phase_list.setContextMenuPolicy(qt.Qt.ContextMenuPolicy.CustomContextMenu)
        self._phase_list.customContextMenuRequested.connect(self.choosePhaseColor)
        self._phase_list.setSelectionMode(
            qt.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._phase_list.filesDropped.connect(self.addCifPaths)
        self._phase_list.itemChanged.connect(self._phaseItemChanged)
        self._phase_list.currentItemChanged.connect(self._currentPhaseChanged)

        add_cifs = qt.QPushButton("Add CIF", self)
        add_cifs.clicked.connect(self._addCifs)
        add_manual = qt.QPushButton("Add manual phase", self)
        add_manual.clicked.connect(self._addManualPhase)
        remove_phase = qt.QPushButton("Remove selected", self)
        remove_phase.clicked.connect(self._removeSelectedPhase)
        phase_buttons = qt.QHBoxLayout()
        phase_buttons.addWidget(add_cifs)
        phase_buttons.addWidget(add_manual)
        phase_buttons.addWidget(remove_phase)

        self._name = qt.QLineEdit(self)
        self._name.editingFinished.connect(self._phaseNameChanged)
        self._source = qt.QLineEdit(self)
        self._source.setEnabled(False)
        self._space_group = qt.QComboBox(self)
        for number in range(1, 231):
            symbol = sg_symbol_from_int_number(number)
            self._space_group.addItem(f"{number}: {symbol}", number)
        self._space_group.currentIndexChanged.connect(self._spaceGroupChanged)
        self._space_group_lock = qt.QPushButton("Lock", self)
        self._space_group_lock.setCheckable(True)
        self._space_group_lock.setChecked(True)
        self._space_group_lock.setToolTip(
            "Changing a CIF space group converts it to a manual phase"
        )
        self._space_group_lock.toggled.connect(self._spaceGroupLockChanged)
        self._space_group.setDisabled(True)
        space_group_layout = qt.QHBoxLayout()
        space_group_layout.setContentsMargins(0, 0, 0, 0)
        space_group_layout.addWidget(self._space_group)
        space_group_layout.addWidget(self._space_group_lock)

        self._cell_edits = {}
        for parameter in ("a", "b", "c", "alpha", "beta", "gamma"):
            edit = ModifierDoubleSpinBox(self)
            edit.setDecimals(6)
            if parameter in ("a", "b", "c"):
                edit.setRange(0.000001, 10000.0)
                edit.setSingleStep(0.1)
            else:
                edit.setRange(0.000001, 179.999999)
            edit.valueChanged.connect(self._cellChanged)
            self._cell_edits[parameter] = edit

        phase_form = qt.QFormLayout()
        phase_form.addRow("Name", self._name)
        phase_form.addRow("Source", self._source)
        phase_form.addRow("Space group", space_group_layout)
        for parameter, label in (
            ("a", "a [Å]"),
            ("b", "b [Å]"),
            ("c", "c [Å]"),
            ("alpha", "α [deg]"),
            ("beta", "β [deg]"),
            ("gamma", "γ [deg]"),
        ):
            phase_form.addRow(label, self._cell_edits[parameter])
        phase_details = qt.QWidget(self)
        phase_details.setLayout(phase_form)

        self._eos_model = qt.QComboBox(self)
        self._eos_model.addItem("None", None)
        self._eos_model.addItem("Birch–Murnaghan", "Birch-Murnaghan")
        self._eos_model.addItem("Vinet", "Vinet")
        self._eos_model.addItem("Murnaghan", "Murnaghan")
        self._eos_model.currentIndexChanged.connect(self._eosSettingsChanged)
        self._eos_pressure = ModifierDoubleSpinBox(self)
        self._eos_pressure.setDecimals(4)
        self._eos_pressure.setRange(-10000.0, 10000.0)
        self._eos_pressure.setSingleStep(1.0)
        self._eos_pressure.valueChanged.connect(self._eosSettingsChanged)
        self._eos_k0 = ModifierDoubleSpinBox(self)
        self._eos_k0.setDecimals(4)
        self._eos_k0.setRange(0.0001, 1000000.0)
        self._eos_k0.setSingleStep(10.0)
        self._eos_k0.setValue(160.0)
        self._eos_k0.valueChanged.connect(self._eosSettingsChanged)
        self._eos_k0p = ModifierDoubleSpinBox(self)
        self._eos_k0p.setDecimals(4)
        self._eos_k0p.setRange(0.0001, 100.0)
        self._eos_k0p.setSingleStep(0.1)
        self._eos_k0p.setValue(4.0)
        self._eos_k0p.valueChanged.connect(self._eosSettingsChanged)
        self._eos_p0 = ModifierDoubleSpinBox(self)
        self._eos_p0.setDecimals(4)
        self._eos_p0.setRange(-10000.0, 10000.0)
        self._eos_p0.setSingleStep(1.0)
        self._eos_p0.valueChanged.connect(self._eosSettingsChanged)
        self._load_jcpds = qt.QPushButton("Load JCPDS", self)
        self._load_jcpds.setToolTip(
            "Load the reference cell and equation of state from a JCPDS file"
        )
        self._load_jcpds.clicked.connect(self._loadJcpds)
        self._apply_eos = qt.QPushButton("Apply EoS", self)
        self._apply_eos.setToolTip(
            "Apply isotropic EoS scaling relative to the phase reference cell"
        )
        self._apply_eos.clicked.connect(self._applyEquationOfState)
        eos_form = qt.QFormLayout()
        eos_form.addRow("Model", self._eos_model)
        eos_form.addRow("Pressure [GPa]", self._eos_pressure)
        eos_form.addRow("K₀ [GPa]", self._eos_k0)
        eos_form.addRow("K₀′", self._eos_k0p)
        eos_form.addRow("Reference pressure P₀ [GPa]", self._eos_p0)
        eos_buttons = qt.QHBoxLayout()
        eos_buttons.addWidget(self._load_jcpds, 1)
        eos_buttons.addWidget(self._apply_eos, 1)
        eos_form.addRow(eos_buttons)
        eos_group = qt.QGroupBox("Equation of state", self)
        eos_group.setLayout(eos_form)

        self._show_labels = qt.QCheckBox("Tick labels", self)
        self._show_labels.toggled.connect(self._labelsChanged)
        self._show_ticks = qt.QCheckBox("Ticks", self)
        self._show_ticks.setChecked(True)
        self._show_ticks.toggled.connect(self._visualizationChanged)
        self._show_lines = qt.QCheckBox("Lines", self)
        self._show_lines.toggled.connect(self._visualizationChanged)
        self._show_sticks = qt.QCheckBox("Intensity sticks", self)
        self._show_sticks.toggled.connect(self._visualizationChanged)
        self._stick_cutoff = ModifierDoubleSpinBox(self)
        self._stick_cutoff.setDecimals(2)
        self._stick_cutoff.setRange(0.0, 100.0)
        self._stick_cutoff.setSingleStep(0.5)
        self._stick_cutoff.setValue(1.0)
        self._stick_cutoff.setEnabled(False)
        self._stick_cutoff.valueChanged.connect(self._emitOverlay)
        self._stick_height = ModifierDoubleSpinBox(self)
        self._stick_height.setDecimals(0)
        self._stick_height.setRange(1.0, 100.0)
        self._stick_height.setSingleStep(5.0)
        self._stick_height.setValue(35.0)
        self._stick_height.setEnabled(False)
        self._stick_height.valueChanged.connect(self._emitOverlay)
        self._condense = qt.QCheckBox("Group reflections", self)
        self._condense.setChecked(True)
        self._condense.toggled.connect(self._visualizationChanged)
        self._merge_tolerance = ModifierDoubleSpinBox(self)
        self._merge_tolerance.setDecimals(5)
        self._merge_tolerance.setRange(0.0, 10.0)
        self._merge_tolerance.setValue(0.01)
        self._merge_tolerance.valueChanged.connect(self._emitOverlay)
        visualization_form = qt.QFormLayout()
        visibility = qt.QHBoxLayout()
        visibility.addWidget(self._show_ticks)
        visibility.addWidget(self._show_lines)
        visibility.addWidget(self._show_labels)
        visibility.addStretch()
        visualization_form.addRow(visibility)
        sticks = qt.QHBoxLayout()
        sticks.addWidget(self._show_sticks)
        sticks.addStretch()
        sticks.addWidget(qt.QLabel("cutoff [%]", self))
        sticks.addWidget(self._stick_cutoff)
        sticks.addWidget(qt.QLabel("Rel. height [%]", self))
        sticks.addWidget(self._stick_height)
        sticks.setSpacing(4)
        visualization_form.addRow(sticks)
        grouping = qt.QHBoxLayout()
        grouping.addWidget(self._condense)
        grouping.addStretch()
        grouping.addWidget(qt.QLabel("tol [deg]", self))
        grouping.setSpacing(4)
        grouping.addWidget(self._merge_tolerance)
        visualization_form.addRow(grouping)
        visualization = qt.QGroupBox("Display", self)
        visualization.setLayout(visualization_form)

        self._message = qt.QLabel(self)
        self._message.setWordWrap(True)
        self._message.setStyleSheet("color: #b04040")

        layout = qt.QVBoxLayout(self)
        layout.addLayout(general_form)
        layout.addWidget(self._phase_list, 1)
        layout.addLayout(phase_buttons)
        layout.addWidget(phase_details)
        layout.addWidget(eos_group)
        layout.addWidget(visualization)
        layout.addWidget(self._message)
        self._setPhaseControlsEnabled(False)

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

    def setRadialRange(self, minimum, maximum, step=None):
        self._ttheta_range = (float(minimum), float(maximum))
        if step is not None and numpy.isfinite(step) and step > 0:
            self._merge_tolerance.setValue(float(step))
        self._invalidateAllPhases()

    def refreshPhaseColors(self):
        for index, phase in enumerate(self._phases):
            if phase["path"] is not None and self.phase_colors is not None:
                phase["color"] = self.phase_colors.get(phase["path"])
            self._phase_list.topLevelItem(index).setForeground(0, qt.QColor(phase["color"]))
        self._emitOverlay()

    def choosePhaseColor(self, position):
        item = self._phase_list.itemAt(position)
        if item is None:
            return
        phase = self._phases[self._phase_list.indexOfTopLevelItem(item)]
        menu = qt.QMenu(self)
        color_action = menu.addAction("Choose color…")
        reset_action = menu.addAction("Reset") if phase["path"] is not None else None
        selected = menu.exec(self._phase_list.viewport().mapToGlobal(position))
        if selected == color_action:
            color = qt.QColorDialog.getColor(qt.QColor(phase["color"]), self)
            if color.isValid():
                phase["color"] = color.name()
                if phase["path"] is not None and self.phase_colors is not None:
                    self.phase_colors.set(phase["path"], color.name())
                else:
                    self.refreshPhaseColors()
        elif reset_action is not None and selected == reset_action:
            self._resetPhase(phase, item)

    def _resetPhase(self, phase, item):
        phase.update(phase["cif_state"])
        phase["space_group"] = None
        phase["dirty"] = True
        self._currentPhaseChanged(item, item)
        self._updateModifiedState(phase, item)
        item.setText(1, phase["source"])
        item.setText(2, self._spaceGroupText(phase))
        if phase["visible"]:
            self._scheduleUpdate()
        else:
            self._emitOverlay()

    def setCifPaths(self, paths):
        self._phases.clear()
        self._phase_list.clear()
        self.addCifPaths(paths)

    def addCifPaths(self, paths):
        existing = {
            phase["path"] for phase in self._phases if phase["path"] is not None
        }
        errors = []
        for path in paths:
            path = str(Path(path).resolve())
            if path in existing:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    parser = self._CifParser(
                        path, occupancy_tolerance=float("inf")
                    )
                    structure = parser.parse_structures(primitive=False)[0]
                analyzer = self._SpacegroupAnalyzer(structure)
                space_group_number = analyzer.get_space_group_number()
                space_group_symbol = analyzer.get_space_group_symbol()
                lattice = structure.lattice
            except Exception as error:
                errors.append(f"{Path(path).name}: {error}")
                continue

            phase_warnings = []
            for message in parser.warnings:
                if message not in phase_warnings:
                    phase_warnings.append(message)
                    _logger.warning("%s: %s", Path(path).name, message)

            eos_reference = {
                "a": lattice.a,
                "b": lattice.b,
                "c": lattice.c,
                "alpha": lattice.alpha,
                "beta": lattice.beta,
                "gamma": lattice.gamma,
            }
            shared_eos = (
                None
                if self.phase_eos is None
                else self.phase_eos.get(path, reference=eos_reference)
            )
            phase = {
                "name": Path(path).stem,
                "path": path,
                "source": "CIF",
                "space_group_number": space_group_number,
                "space_group_symbol": space_group_symbol,
                "space_group": None,
                "crystal_system": analyzer.get_crystal_system(),
                "cell_mode": (
                    "rhombohedral"
                    if analyzer.get_crystal_system() == "trigonal"
                    and numpy.allclose(lattice.abc, lattice.a)
                    and numpy.allclose(lattice.angles, lattice.alpha)
                    else "conventional"
                ),
                "structure": structure,
                "a": lattice.a,
                "b": lattice.b,
                "c": lattice.c,
                "alpha": lattice.alpha,
                "beta": lattice.beta,
                "gamma": lattice.gamma,
                "visible": False,
                "color": self._COLORS[len(self._phases) % len(self._COLORS)],
                "reflections": [],
                "sticks": [],
                "labels_computed": False,
                "dirty": True,
                "warnings": phase_warnings,
                "eos": (
                    shared_eos["eos"]
                    if shared_eos is not None
                    else {
                        "model": None,
                        "pressure": 0.0,
                        "k0": 160.0,
                        "k0p": 4.0,
                        "p0": 0.0,
                    }
                ),
                "eos_reference": (
                    shared_eos["reference"]
                    if shared_eos is not None
                    else eos_reference
                ),
            }
            phase["cif_state"] = {
                key: phase[key]
                for key in (
                    "name", "source", "space_group_number", "space_group_symbol",
                    "crystal_system", "cell_mode", "a", "b", "c",
                    "alpha", "beta", "gamma",
                )
            }
            self._appendPhase(phase)
            existing.add(path)

        errors.extend(
            f"{phase['name']}: {message}"
            for phase in self._phases
            for message in phase["warnings"]
        )

        self._message.setText("\n".join(errors))
        self._emitOverlay()

    def _addCifs(self):
        filenames, _ = qt.QFileDialog.getOpenFileNames(
            self,
            "Select phases",
            "",
            "Crystallographic information files (*.cif);;All files (*)",
        )
        self.addCifPaths(filenames)

    def _addManualPhase(self):
        number = 1
        group = self._SpaceGroup.from_int_number(number)
        phase = {
            "name": f"Phase {len(self._phases) + 1}",
            "path": None,
            "source": "Manual",
            "space_group_number": number,
            "space_group_symbol": group.symbol,
            "space_group": group,
            "crystal_system": group.crystal_system,
            "cell_mode": "conventional",
            "a": 4.0,
            "b": 4.0,
            "c": 4.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
            "visible": False,
            "color": self._COLORS[len(self._phases) % len(self._COLORS)],
            "reflections": [],
            "sticks": [],
            "labels_computed": False,
            "dirty": True,
            "warnings": [],
            "eos": {
                "model": None,
                "pressure": 0.0,
                "k0": 160.0,
                "k0p": 4.0,
                "p0": 0.0,
            },
            "eos_reference": {
                "a": 4.0,
                "b": 4.0,
                "c": 4.0,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            },
        }
        self._appendPhase(phase)
        item = self._phase_list.topLevelItem(
            self._phase_list.topLevelItemCount() - 1
        )
        self._phase_list.setCurrentItem(item)
        self._emitOverlay()

    def _appendPhase(self, phase):
        if phase["path"] is not None and self.phase_colors is not None:
            phase["color"] = self.phase_colors.get(phase["path"])
        self._phases.append(phase)
        item = qt.QTreeWidgetItem(
            [phase["name"], phase["source"], self._spaceGroupText(phase)]
        )
        item.setFlags(item.flags() | qt.Qt.ItemFlag.ItemIsUserCheckable)
        check_state = (
            qt.Qt.CheckState.Checked
            if phase["visible"]
            else qt.Qt.CheckState.Unchecked
        )
        item.setCheckState(0, check_state)
        item.setForeground(0, qt.QColor(phase["color"]))
        if phase["path"] is not None:
            item.setToolTip(0, phase["path"])
        self._phase_list.addTopLevelItem(item)
        if self._phase_list.currentItem() is None:
            self._phase_list.setCurrentItem(item)

    def _spaceGroupText(self, phase):
        number = phase["space_group_number"]
        symbol = phase["space_group_symbol"]
        return f"{number}: {symbol}"

    def _removeSelectedPhase(self):
        item = self._phase_list.currentItem()
        if item is None:
            return
        index = self._phase_list.indexOfTopLevelItem(item)
        self._phases.pop(index)
        self._phase_list.takeTopLevelItem(index)
        self._emitOverlay()

    def _phaseItemChanged(self, item, column):
        if self._updating_controls or column != 0:
            return
        index = self._phase_list.indexOfTopLevelItem(item)
        phase = self._phases[index]
        phase["visible"] = item.checkState(0) == qt.Qt.CheckState.Checked
        if phase["visible"] and phase["dirty"]:
            self._scheduleUpdate()
        else:
            self._emitOverlay()

    def _currentPhaseChanged(self, current, previous):
        self._updating_controls = True
        try:
            if current is None:
                self._setPhaseControlsEnabled(False)
                return
            index = self._phase_list.indexOfTopLevelItem(current)
            phase = self._phases[index]
            self._setPhaseControlsEnabled(True)
            self._name.setText(phase["name"])
            self._source.setText(
                phase["path"] if phase["source"] == "CIF" else "Manual"
            )
            index = self._space_group.findData(phase["space_group_number"])
            self._space_group.setCurrentIndex(index)
            for parameter, edit in self._cell_edits.items():
                edit.setValue(phase[parameter])
            eos = phase["eos"]
            self._eos_model.setCurrentIndex(
                self._eos_model.findData(eos["model"])
            )
            self._eos_pressure.setValue(eos["pressure"])
            self._eos_k0.setValue(eos["k0"])
            self._eos_k0p.setValue(eos["k0p"])
            self._eos_p0.setValue(eos["p0"])
            self._applyCellConstraints(phase)
            self._updateEosControlsEnabled()
        finally:
            self._updating_controls = False

    def _setPhaseControlsEnabled(self, enabled):
        self._name.setEnabled(enabled)
        self._space_group_lock.setEnabled(enabled)
        self._space_group.setEnabled(
            enabled and not self._space_group_lock.isChecked()
        )
        for edit in self._cell_edits.values():
            edit.setEnabled(enabled)
        self._eos_model.setEnabled(enabled)
        self._load_jcpds.setEnabled(enabled)
        self._updateEosControlsEnabled()

    def _updateEosControlsEnabled(self):
        enabled = self._phase_list.currentItem() is not None
        model_enabled = enabled and self._eos_model.currentData() is not None
        self._eos_pressure.setEnabled(model_enabled)
        self._eos_k0.setEnabled(model_enabled)
        self._eos_k0p.setEnabled(model_enabled)
        self._eos_p0.setEnabled(model_enabled)
        self._apply_eos.setEnabled(model_enabled)

    def _eosSettingsChanged(self):
        self._updateEosControlsEnabled()
        if self._updating_controls:
            return
        item = self._phase_list.currentItem()
        if item is None:
            return
        phase = self._phases[self._phase_list.indexOfTopLevelItem(item)]
        model = self._eos_model.currentData()
        if (
            phase["source"] == "Manual"
            and phase["eos"]["model"] is None
            and model is not None
        ):
            phase["eos_reference"] = {
                parameter: phase[parameter]
                for parameter in ("a", "b", "c", "alpha", "beta", "gamma")
            }
        previous_inverse_settings = (
            phase["eos"]["model"],
            phase["eos"]["k0"],
            phase["eos"]["k0p"],
            phase["eos"]["p0"],
        )
        phase["eos"].update(
            model=model,
            pressure=self._eos_pressure.value(),
            k0=self._eos_k0.value(),
            k0p=self._eos_k0p.value(),
            p0=self._eos_p0.value(),
        )
        inverse_settings = (
            phase["eos"]["model"],
            phase["eos"]["k0"],
            phase["eos"]["k0p"],
            phase["eos"]["p0"],
        )
        if (
            phase["path"] is not None
            and self.phase_eos is not None
            and inverse_settings != previous_inverse_settings
        ):
            self.phase_eos.notifyChanged(phase["path"])

    def _applyEquationOfState(self):
        item = self._phase_list.currentItem()
        if item is None:
            return
        phase = self._phases[self._phase_list.indexOfTopLevelItem(item)]
        settings = phase["eos"]
        model = settings["model"]
        if model is None:
            return
        try:
            eos = self._eos_classes[model](
                k0=settings["k0"],
                k0p=settings["k0p"],
                p0=settings["p0"],
            )
            ratio = eos.linear_ratio(pressure=settings["pressure"])
            if not numpy.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(f"invalid linear cell ratio {ratio}")
        except Exception as error:
            self._message.setText(f"{phase['name']}: {error}")
            return

        reference = phase["eos_reference"]
        values = {
            "a": reference["a"] * ratio,
            "b": reference["b"] * ratio,
            "c": reference["c"] * ratio,
            "alpha": reference["alpha"],
            "beta": reference["beta"],
            "gamma": reference["gamma"],
        }
        phase.update(values)
        phase["dirty"] = True
        self._updating_controls = True
        for parameter, edit in self._cell_edits.items():
            edit.setValue(values[parameter])
        self._updating_controls = False
        self._updateModifiedState(phase, item)
        self._scheduleUpdate()

    def _loadJcpds(self):
        item = self._phase_list.currentItem()
        if item is None:
            return
        phase = self._phases[self._phase_list.indexOfTopLevelItem(item)]
        directory = str(Path(phase["path"]).parent) if phase["path"] else ""
        filename, _ = qt.QFileDialog.getOpenFileName(
            self,
            "Load equation of state",
            directory,
            "JCPDS files (*.jcpds);;All files (*)",
        )
        if not filename:
            return

        try:
            from ....io.calibrant_config import CalibrantConfig

            config = CalibrantConfig.from_JCPDS(filename)
            cell = config.to_cell()
            if cell is None:
                raise ValueError("JCPDS file does not contain a usable reference cell")
            eos = config.eos
            thermal_ignored = False
            if hasattr(eos, "isothermal"):
                eos = eos.isothermal
                thermal_ignored = True
            if eos is None or eos.name not in self._eos_classes:
                raise ValueError(
                    "JCPDS file does not contain a supported compression EoS"
                )
        except Exception as error:
            self._message.setText(f"{Path(filename).name}: {error}")
            return

        reference = {
            parameter: getattr(cell, parameter)
            for parameter in ("a", "b", "c", "alpha", "beta", "gamma")
        }
        if phase["path"] is not None and self.phase_eos is not None:
            shared_eos = self.phase_eos.get(phase["path"])
            shared_eos["reference"] = reference
            phase["eos_reference"] = shared_eos["reference"]
        else:
            phase["eos_reference"] = reference
        phase["eos"].update(
            model=eos.name,
            k0=eos.k0,
            k0p=eos.k0p,
            p0=eos.p0,
        )
        self._updating_controls = True
        self._eos_model.setCurrentIndex(self._eos_model.findData(eos.name))
        self._eos_k0.setValue(eos.k0)
        self._eos_k0p.setValue(eos.k0p)
        self._eos_p0.setValue(eos.p0)
        self._updating_controls = False
        self._updateEosControlsEnabled()
        if phase["path"] is not None and self.phase_eos is not None:
            self.phase_eos.notifyChanged(phase["path"])
        if thermal_ignored:
            message = (
                f"{Path(filename).name}: thermal EoS parameters are not yet "
                "supported and were not loaded"
            )
            _logger.warning(message)
            self._message.setText(message)
        else:
            self._message.clear()

    def _spaceGroupLockChanged(self, locked):
        self._space_group.setEnabled(
            self._phase_list.currentItem() is not None and not locked
        )

    def _phaseNameChanged(self):
        if self._updating_controls:
            return
        item = self._phase_list.currentItem()
        if item is None:
            return
        index = self._phase_list.indexOfTopLevelItem(item)
        phase = self._phases[index]
        name = self._name.text().strip()
        if name:
            phase["name"] = name
            self._updateModifiedState(phase, item)
            self._emitOverlay()
        else:
            self._name.setText(phase["name"])

    def _spaceGroupChanged(self):
        if self._updating_controls:
            return
        item = self._phase_list.currentItem()
        if item is None:
            return
        index = self._phase_list.indexOfTopLevelItem(item)
        phase = self._phases[index]
        number = self._space_group.currentData()
        group = self._SpaceGroup.from_int_number(number)
        phase["space_group_number"] = number
        phase["space_group_symbol"] = group.symbol
        phase["space_group"] = group
        phase["crystal_system"] = group.crystal_system
        phase["cell_mode"] = "conventional"
        phase["source"] = "Manual"
        phase["sticks"] = []
        phase["dirty"] = True
        self._updating_controls = True
        item.setText(1, "Manual")
        item.setText(2, self._spaceGroupText(phase))
        self._source.setText("Manual")
        self._applyCellConstraints(phase)
        self._updating_controls = False
        self._cellChanged()

    def _applyCellConstraints(self, phase):
        system = phase["crystal_system"]
        enabled = {
            "a": True,
            "b": system in ("triclinic", "monoclinic", "orthorhombic"),
            "c": system != "cubic" and phase["cell_mode"] != "rhombohedral",
            "alpha": system == "triclinic" or phase["cell_mode"] == "rhombohedral",
            "beta": system in ("triclinic", "monoclinic"),
            "gamma": system == "triclinic",
        }
        for parameter, edit in self._cell_edits.items():
            edit.setEnabled(enabled[parameter])

    def _cellChanged(self):
        if self._updating_controls:
            return
        item = self._phase_list.currentItem()
        if item is None:
            return
        index = self._phase_list.indexOfTopLevelItem(item)
        phase = self._phases[index]
        values = {
            parameter: edit.value() for parameter, edit in self._cell_edits.items()
        }
        system = phase["crystal_system"]
        if system == "cubic":
            values.update(b=values["a"], c=values["a"], alpha=90.0, beta=90.0, gamma=90.0)
        elif phase["cell_mode"] == "rhombohedral":
            values["b"] = values["a"]
            values["c"] = values["a"]
            values["beta"] = values["alpha"]
            values["gamma"] = values["alpha"]
        elif system in ("tetragonal", "hexagonal", "trigonal"):
            values["b"] = values["a"]
            values["alpha"] = 90.0
            values["beta"] = 90.0
            values["gamma"] = 120.0 if system in ("hexagonal", "trigonal") else 90.0
        elif system == "orthorhombic":
            values.update(alpha=90.0, beta=90.0, gamma=90.0)
        elif system == "monoclinic":
            values.update(alpha=90.0, gamma=90.0)
        phase.update(values)
        phase["dirty"] = True

        self._updating_controls = True
        for parameter, edit in self._cell_edits.items():
            edit.setValue(values[parameter])
        self._updating_controls = False
        self._updateModifiedState(phase, item)
        self._scheduleUpdate()

    def _updateModifiedState(self, phase, item):
        modified = False
        original = phase.get("cif_state")
        if original is not None:
            for key, value in original.items():
                current = phase[key]
                if isinstance(value, float):
                    different = not numpy.isclose(current, value, rtol=1e-7, atol=5e-7)
                else:
                    different = current != value
                if different:
                    modified = True
                    break
        self._updating_controls = True
        item.setText(0, phase["name"] + ("*" if modified else ""))
        self._updating_controls = False

    def _scheduleUpdate(self):
        self._update_timer.start()

    def _invalidateAllPhases(self):
        for phase in self._phases:
            phase["dirty"] = True
        self._scheduleUpdate()

    def _updateReflections(self):
        wavelength = self._wavelength.value()
        minimum, maximum = self._ttheta_range
        maximum = min(maximum, 179.999999)
        errors = [
            f"{phase['name']}: {message}"
            for phase in self._phases
            for message in phase["warnings"]
        ]

        for phase in self._phases:
            if not phase["visible"] or not phase["dirty"]:
                continue
            try:
                lattice = self._Lattice.from_parameters(
                    phase["a"],
                    phase["b"],
                    phase["c"],
                    phase["alpha"],
                    phase["beta"],
                    phase["gamma"],
                )
                if self._show_sticks.isChecked() and phase["source"] == "CIF":
                    original = phase["structure"]
                    structure = self._Structure(
                        lattice,
                        [site.species for site in original],
                        original.frac_coords,
                        site_properties=original.site_properties,
                    )
                    pattern = self._XRDCalculator(wavelength=wavelength).get_pattern(
                        structure,
                        scaled=True,
                        two_theta_range=(minimum, maximum),
                    )
                    phase["sticks"] = [
                        {"position": float(position), "intensity": float(intensity)}
                        for position, intensity in zip(pattern.x, pattern.y)
                    ]
                elif phase["source"] != "CIF":
                    phase["sticks"] = []
                group = phase["space_group"]
                if group is None:
                    group_symbol = phase["space_group_symbol"]
                    if (
                        phase["cell_mode"] == "rhombohedral"
                        and group_symbol.startswith("R")
                    ):
                        group_symbol += ":R"
                    group = self._SpaceGroup(group_symbol)
                    phase["space_group"] = group
                min_r = 2.0 * numpy.sin(numpy.deg2rad(max(minimum, 0.0) / 2.0)) / wavelength
                max_r = 2.0 * numpy.sin(numpy.deg2rad(maximum / 2.0)) / wavelength
                reciprocal = lattice.reciprocal_lattice_crystallographic
                points = reciprocal.get_points_in_sphere(
                    [[0.0, 0.0, 0.0]], [0.0, 0.0, 0.0], max_r
                )
                hkls = numpy.asarray(
                    [numpy.rint(point[0]).astype(int) for point in points if point[1] >= min_r and point[1] > 0.0]
                )
                lengths = numpy.asarray(
                    [point[1] for point in points if point[1] >= min_r and point[1] > 0.0]
                )
                if hkls.size == 0:
                    phase["reflections"] = []
                    phase["labels_computed"] = self._show_labels.isChecked()
                    phase["dirty"] = False
                    continue

                # Unit weights on a generic full orbit identify systematic
                # absences. They are not used as physical reflection intensities.
                general_position = numpy.asarray(
                    group.get_orbit([0.123456789, 0.234567891, 0.345678912])
                )
                allowed = numpy.empty(len(hkls), dtype=bool)
                chunk_size = 50000
                for start in range(0, len(hkls), chunk_size):
                    stop = min(start + chunk_size, len(hkls))
                    phases = 2j * numpy.pi * numpy.dot(
                        hkls[start:stop], general_position.T
                    )
                    amplitudes = numpy.abs(numpy.exp(phases).sum(axis=1))
                    allowed[start:stop] = amplitudes > 1e-7 * len(general_position)
                hkls = hkls[allowed]
                lengths = lengths[allowed]

                arguments = wavelength * lengths / 2.0
                physical = arguments <= 1.0
                hkls = hkls[physical]
                positions = numpy.rad2deg(2.0 * numpy.arcsin(arguments[physical]))
                keys = numpy.round(positions, 7)
                order = numpy.argsort(keys)
                keys = keys[order]
                hkls = hkls[order]
                unique_positions, starts = numpy.unique(keys, return_index=True)
                stops = numpy.append(starts[1:], len(keys))
                reflections = []
                if self._show_labels.isChecked():
                    rotations = []
                    for operation in group.symmetry_ops:
                        rotation = numpy.rint(
                            numpy.linalg.inv(operation.rotation_matrix).T
                        ).astype(int)
                        if not any(
                            numpy.array_equal(rotation, known)
                            for known in rotations
                        ):
                            rotations.append(rotation)

                    use_hkil = phase["crystal_system"] in (
                        "hexagonal",
                        "trigonal",
                    )
                    for position, start, stop in zip(
                        unique_positions, starts, stops
                    ):
                        remaining = {tuple(hkl) for hkl in hkls[start:stop]}
                        families = set()
                        while remaining:
                            hkl = numpy.asarray(remaining.pop())
                            equivalent = set()
                            for rotation in rotations:
                                transformed = tuple((rotation @ hkl).tolist())
                                equivalent.add(transformed)
                                equivalent.add(tuple((-rotation @ hkl).tolist()))
                            families.add(max(equivalent))
                            remaining.difference_update(equivalent)
                        if use_hkil:
                            families = {
                                (h, k, -h - k, l) for h, k, l in families
                            }
                        reflections.append(
                            {
                                "position": position,
                                "hkls": sorted(families, reverse=True),
                            }
                        )
                else:
                    reflections = [
                        {"position": position, "hkls": []}
                        for position in unique_positions
                    ]
                phase["reflections"] = reflections
                phase["labels_computed"] = self._show_labels.isChecked()
                phase["dirty"] = False
            except Exception as error:
                phase["reflections"] = []
                phase["labels_computed"] = False
                phase["dirty"] = False
                errors.append(f"{phase['name']}: {error}")

        self._message.setText("\n".join(errors))
        self._emitOverlay()

    def _visualizationChanged(self):
        self._merge_tolerance.setEnabled(self._condense.isChecked())
        self._stick_cutoff.setEnabled(self._show_sticks.isChecked())
        self._stick_height.setEnabled(self._show_sticks.isChecked())
        if self.sender() is self._show_sticks and self._show_sticks.isChecked():
            needs_update = False
            for phase in self._phases:
                if phase["visible"] and phase["source"] == "CIF":
                    phase["dirty"] = True
                    needs_update = True
            if needs_update:
                self._scheduleUpdate()
                return
        self._emitOverlay()

    def _labelsChanged(self, visible):
        if not visible:
            self._emitOverlay()
            return
        needs_update = False
        for phase in self._phases:
            if phase["visible"] and not phase["labels_computed"]:
                phase["dirty"] = True
                needs_update = True
        if needs_update:
            self._scheduleUpdate()
        else:
            self._emitOverlay()

    def _emitOverlay(self):
        overlays = []
        tolerance = self._merge_tolerance.value()
        for phase in self._phases:
            if not phase["visible"]:
                continue
            minimum, maximum = self._ttheta_range
            reflections = [entry for entry in phase["reflections"]
                           if minimum <= entry["position"] <= maximum]
            sticks = [
                entry for entry in phase["sticks"]
                if minimum <= entry["position"] <= maximum
                and entry["intensity"] >= self._stick_cutoff.value()
            ]
            if self._condense.isChecked() and reflections:
                condensed = []
                current = [reflections[0]]
                for reflection in reflections[1:]:
                    if reflection["position"] - current[-1]["position"] <= tolerance:
                        current.append(reflection)
                    else:
                        hkls = sorted(
                            {hkl for entry in current for hkl in entry["hkls"]},
                            reverse=True,
                        )
                        condensed.append(
                            {
                                "position": numpy.mean(
                                    [entry["position"] for entry in current]
                                ),
                                "hkls": hkls,
                            }
                        )
                        current = [reflection]
                hkls = sorted(
                    {hkl for entry in current for hkl in entry["hkls"]},
                    reverse=True,
                )
                condensed.append(
                    {
                        "position": numpy.mean(
                            [entry["position"] for entry in current]
                        ),
                        "hkls": hkls,
                    }
                )
                reflections = condensed

            overlays.append(
                {
                    "name": phase["name"],
                    "color": phase["color"],
                    "reflections": reflections,
                    "sticks": sticks,
                    "show_labels": self._show_labels.isChecked(),
                    "show_ticks": self._show_ticks.isChecked(),
                    "show_lines": self._show_lines.isChecked(),
                    "show_sticks": self._show_sticks.isChecked(),
                    "stick_height": self._stick_height.value(),
                }
            )
        self.overlayChanged.emit(overlays)
