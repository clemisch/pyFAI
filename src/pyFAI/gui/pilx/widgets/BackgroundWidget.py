"""Optional background estimation controls, executed in the external Ewoks worker."""

from silx.gui import qt

from .RietveldRefinementWidget import RietveldRefinementProcess


class BackgroundDialog(qt.QDialog):
    computeRequested = qt.Signal(bool)
    displayChanged = qt.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histogram background")
        self.minimum = qt.QDoubleSpinBox(self)
        self.maximum = qt.QDoubleSpinBox(self)
        for edit in (self.minimum, self.maximum):
            edit.setDecimals(6)
            edit.setRange(0, 180)
        self.automatic = qt.QCheckBox("Automatic smoothness", self)
        self.automatic.setChecked(True)
        self.smoothness = qt.QDoubleSpinBox(self)
        self.smoothness.setRange(0, 20)
        self.smoothness.setSingleStep(0.1)
        self.smoothness.setValue(6)
        self.smoothness.setEnabled(False)
        self.automatic.toggled.connect(self.smoothness.setDisabled)
        self.subtract = qt.QCheckBox("Subtract background", self)
        self.subtract.toggled.connect(self.displayChanged)
        self.single = qt.QPushButton("Compute background", self)
        self.mapped = qt.QPushButton("Compute all backgrounds", self)
        self.single.clicked.connect(lambda: self.computeRequested.emit(False))
        self.mapped.clicked.connect(lambda: self.computeRequested.emit(True))
        self.status = qt.QLabel("No backgrounds computed", self)
        self.status.setWordWrap(True)
        form = qt.QFormLayout(self)
        form.addRow("Minimum 2θ [deg]", self.minimum)
        form.addRow("Maximum 2θ [deg]", self.maximum)
        form.addRow(self.automatic)
        form.addRow("Smoothness [log10 λ]", self.smoothness)
        form.addRow(self.single, self.mapped)
        form.addRow(self.subtract)
        form.addRow(self.status)
        self.process = None
        self.results = {}
        self.map_result = None
        self.map_normalization = None
        self.data_generation = 0

    def reset(self, minimum, maximum):
        self.data_generation += 1
        self.radial_bounds = (minimum, maximum)
        self.minimum.setValue(minimum)
        self.maximum.setValue(maximum)
        self.results.clear()
        self.map_result = None
        self.subtract.blockSignals(True)
        self.subtract.setChecked(False)
        self.subtract.blockSignals(False)
        self.status.setText("No backgrounds computed")

    def resultForPoint(self, indices):
        if indices in self.results:
            return self.results[indices]
        if self.map_result is not None:
            result = dict(self.map_result)
            result["background"] = result["background"][indices.row, indices.col]
            return result
        return self.results.get(indices)

    def start(self, python, inputs, mapped, indices, filename):
        lower, upper = self.minimum.value(), self.maximum.value()
        # Preserve endpoint samples despite spin-box display rounding.
        if abs(lower - self.radial_bounds[0]) < 1e-6:
            lower = self.radial_bounds[0]
        if abs(upper - self.radial_bounds[1]) < 1e-6:
            upper = self.radial_bounds[1]
        inputs.update(
            ttheta_range_deg=(lower, upper),
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
