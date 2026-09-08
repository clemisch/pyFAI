"""Double spin box with modifier-sensitive stepping."""

from silx.gui import qt


class ModifierDoubleSpinBox(qt.QDoubleSpinBox):
    """Use Shift for fine steps while retaining Qt's Ctrl acceleration."""

    def stepBy(self, steps):
        modifiers = qt.QApplication.keyboardModifiers()
        fine_step = (
            modifiers & qt.Qt.KeyboardModifier.ShiftModifier
            and not modifiers & qt.Qt.KeyboardModifier.ControlModifier
        )
        if fine_step:
            single_step = self.singleStep()
            self.setSingleStep(single_step / 10)
            try:
                super().stepBy(steps)
            finally:
                self.setSingleStep(single_step)
        else:
            super().stepBy(steps)
