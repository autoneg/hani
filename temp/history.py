import panel as pn
import pandas as pd
import param

pn.extension()


class History(pn.pane.Pane):
    """Custom Pane to display DataFrame history."""

    df = param.DataFrame(default=pd.DataFrame())

    def __init__(self, **params):
        super().__init__(**params)
        self._last_df_hash = hash(self.df.to_json(orient="records"))  # Add hash
        self._df_widget = pn.widgets.DataFrame(self.df)
        self.object = self._df_widget

    def _update_pane(self, *events) -> None:
        current_df_hash = hash(self.df.to_json(orient="records"))  # Add hash
        if current_df_hash != self._last_df_hash:  # Add hash
            self._df_widget.value = self.df
            self._last_df_hash = current_df_hash  # Add hash

    @param.depends("df", watch=True)
    def _update_on_df_change(self):
        self._update_pane()

    def _get_model(self, doc, root=None, parent=None, comm=None):
        return pn.Row(self.object)._get_model(doc, root, parent, comm)


# Initial DataFrame
df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

# History Pane
history_pane = History(df=df)

# Button to add a row
add_row_button = pn.widgets.Button(name="Add Row")


# Function to add a row to the DataFrame
def add_row(event):
    global df  # Use global df to modify it
    new_row = {"col1": len(df) + 1, "col2": len(df) + 3}  # Example new row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    history_pane.df = df  # Update the History pane's DataFrame


add_row_button.on_click(add_row)

# Layout
layout = pn.Column(add_row_button, history_pane)

layout.servable()
