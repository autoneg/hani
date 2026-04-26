import pandas as pd
import param

import panel as pn

from panel.widgets import IntSlider, Tabulator
from param.parameterized import Event

pn.extension("tabulator")


class DataExplorer(pn.viewable.Viewer):
    data = param.DataFrame(doc="Stores a DataFrame to explore")
    page_size = param.Integer(
        default=10, doc="Number of rows per page.", bounds=(1, None)
    )
    theme = param.Selector(
        default="simple",
        objects=["simple", "default", "site", "midnight"],
    )
    show_index = param.Boolean(
        default=True, doc="Whether or not to display the index of the data"
    )

    @pn.depends("data")
    def data_len(self):
        return str(len(self.data))

    def __panel__(self):
        print("presenting panel")
        table = Tabulator.from_param(
            self.param.data,
            page_size=self.param.page_size,
            sizing_mode="stretch_width",
            theme=self.param.theme,
            show_index=self.param.show_index,
        )
        return pn.Column(
            IntSlider.from_param(self.param.page_size, start=5, end=25, step=5),
            self.param.theme,
            self.param.show_index,
            table,
            pn.pane.Markdown(
                f"{type(table.value)=} , {type(self.data)=}, {type(self.param.data)=}\n\n"
            ),
            self.data_len,
        )


data_url = "https://assets.holoviz.org/panel/tutorials/turbines.csv.gz"
df1 = pn.cache(pd.read_csv)(data_url)
df = df1.head(5)

explorer = DataExplorer(data=df)


def add_records(event: Event = None, explorer=explorer):
    if event.type == "set":
        return
    print(f"Adding records: {event}")
    explorer.data = pd.concat((explorer.data, df), ignore_index=True)


btn = pn.widgets.Button(name="Add", on_click=add_records)
pn.Column(btn, explorer).servable()
