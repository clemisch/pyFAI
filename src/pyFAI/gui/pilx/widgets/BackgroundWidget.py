"""Optional background estimation controls, executed in the external Ewoks worker."""

import numpy
from silx.gui import qt

from .RietveldRefinementWidget import RietveldRefinementProcess


class BackgroundDialog(qt.QDialog):
    computeRequested = qt.Signal(bool)
    displayChanged = qt.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.radial_values = None
        self.setWindowTitle("Histogram background")
        self.minimum = qt.QDoubleSpinBox(self)
        self.maximum = qt.QDoubleSpinBox(self)
        for edit in (self.minimum, self.maximum):
            edit.setDecimals(6)
            edit.setRange(0, 180)
        self.automatic = qt.QPushButton("Auto", self)
        self.automatic.setCheckable(True)
        self.automatic.setChecked(True)
        self.smoothness = qt.QDoubleSpinBox(self)
        self.smoothness.setRange(0, 20)
        self.smoothness.setSingleStep(0.1)
        self.smoothness.setValue(6)
        self.smoothness.setEnabled(False)
        self.automatic.toggled.connect(self.smoothness.setDisabled)
        self.automatic.toggled.connect(self.updateAutomaticSmoothness)
        self.minimum.valueChanged.connect(self.updateAutomaticSmoothness)
        self.maximum.valueChanged.connect(self.updateAutomaticSmoothness)
        self.overlay = qt.QRadioButton("Overlay background", self)
        self.subtract = qt.QRadioButton("Substract background", self)
        self.overlay.setChecked(True)
        self.subtract.toggled.connect(self.displayChanged)
        self.single = qt.QPushButton("Compute histogram background", self)
        self.mapped = qt.QPushButton("Compute all backgrounds", self)
        self.single.clicked.connect(lambda: self.computeRequested.emit(False))
        self.mapped.clicked.connect(lambda: self.computeRequested.emit(True))
        self.status = qt.QLabel("No backgrounds computed", self)
        self.status.setWordWrap(True)
        self.status.hide()
        form = qt.QFormLayout(self)
        form.setVerticalSpacing(12)
        bounds = qt.QHBoxLayout()
        bounds.addWidget(self.minimum)
        bounds.addWidget(self.maximum)
        form.addRow("Min/Max 2θ [deg]", bounds)
        smoothness = qt.QHBoxLayout()
        smoothness.addWidget(self.automatic)
        smoothness.addWidget(self.smoothness)
        form.addRow("Smoothness", smoothness)
        for button in (self.single, self.mapped, self.overlay):
            separator = qt.QFrame(self)
            separator.setFrameShape(qt.QFrame.Shape.HLine)
            separator.setFrameShadow(qt.QFrame.Shadow.Sunken)
            form.addRow(separator)
            form.addRow(button)
        form.addRow(self.subtract)
        self.resize(self.sizeHint().width() + 30, self.sizeHint().height() + 30)
        self.process = None
        self.results = {}
        self.map_result = None
        self.map_normalization = None
        self.data_generation = 0

    def reset(self, radial_values):
        self.data_generation += 1
        self.radial_values = numpy.asarray(radial_values)
        minimum, maximum = float(radial_values[0]), float(radial_values[-1])
        self.radial_bounds = (minimum, maximum)
        self.minimum.setValue(minimum)
        self.maximum.setValue(maximum)
        self.results.clear()
        self.map_result = None
        self.subtract.blockSignals(True)
        self.overlay.setChecked(True)
        self.subtract.blockSignals(False)
        self.status.setText("No backgrounds computed")
        self.updateAutomaticSmoothness()

    def updateAutomaticSmoothness(self):
        if not self.automatic.isChecked() or self.radial_values is None:
            return
        lower, upper = self.selectedRange()
        start, stop = numpy.searchsorted(self.radial_values, (lower, upper))

        if stop - start >= 3:
            # Match xrdmap.background._auto_smoothness without importing the
            # refinement environment into the GUI process.
            value = min(10, (int(10 * numpy.log10(stop - start)**1.5) - 9.5) / 10.)
            self.smoothness.setValue(value)

    def resultForPoint(self, indices):
        if indices in self.results:
            return self.results[indices]
        if self.map_result is not None:
            result = dict(self.map_result)
            result["background"] = result["background"][indices.row, indices.col]
            return result
        return self.results.get(indices)

    def selectedRange(self):
        lower, upper = self.minimum.value(), self.maximum.value()
        # Preserve endpoint samples despite spin-box display rounding.
        if abs(lower - self.radial_bounds[0]) < 1e-6:
            lower = self.radial_bounds[0]
        if abs(upper - self.radial_bounds[1]) < 1e-6:
            upper = self.radial_bounds[1]
        return lower, upper

    def start(self, python, inputs, mapped, indices, filename):
        inputs.update(
            ttheta_range_deg=self.selectedRange(),
            smoothness=None if self.automatic.isChecked() else self.smoothness.value(),
        )
        task = "EstimateBackgroundMap" if mapped else "EstimateBackgroundSingle"
        graph = {
            "graph": {"id": "background", "schema_version": "1.2"},
            "nodes": [{
                "id": "refinement", "task_type": "class",
                "task_identifier": f"ewoksxrpd.tasks.estimate_background.{task}",
                "default_inputs": [{"name": key, "value": value} for key, value in inputs.items()],
            }],
            "links": [],
        }
        self.process = RietveldRefinementProcess(python, graph, {}, self)
        self.process.background_mapped = mapped
        self.process.background_indices = indices
        self.process.background_filename = filename
        self.process.background_generation = self.data_generation
        self.single.setEnabled(False)
        self.mapped.setEnabled(False)
        self.status.setText("Computing all backgrounds…" if mapped else "Computing background…")
