import panel as pn

__all__ = ["ToolViewer"]


class ToolViewer(pn.Column):
    def __init__(self, name, viewable, **params):
        """
        Custom widget that shows a checkbox and another viewable below it.

        Args:
            viewable: The Panel viewable to be displayed below the checkbox.
            **params: Additional parameters to pass to pn.Column.
        """
        super().__init__(**params)
        self.checkbox = pn.widgets.Checkbox(name=name)
        self.viewable = viewable
        self.append(self.checkbox)
        self.append(self.viewable)
        self.checkbox.param.watch(self._update_visibility, "value")
        self._update_visibility(None)  # Initial visibility

    def _update_visibility(self, event):
        """Updates the visibility of the viewable based on the checkbox."""
        self.viewable.visible = self.checkbox.value
