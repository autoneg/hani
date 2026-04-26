import pandas as pd
import panel as pn
import param
from panel.viewable import Viewer

pn.extension("tabulator")

data_url = "https://assets.holoviz.org/panel/tutorials/turbines.csv.gz"

turbines = pn.cache(pd.read_csv)(data_url)


class DataExplorer(Viewer):
    data = param.DataFrame(doc="Stores a DataFrame to explore")

    columns = param.ListSelector(
        default=["p_name", "t_state", "t_county", "p_year", "t_manu", "p_cap"]
    )

    year = param.Range(default=(1981, 2022), bounds=(1981, 2022))

    capacity = param.Range(default=(0, 1100), bounds=(0, 1100))

    def __init__(self, **params):
        super().__init__(**params)
        self.param.columns.objects = self.data.columns.to_list()

    @param.depends("data", "columns", "year", "capacity")
    def filtered_data(self):
        df = self.data
        return df[df.p_year.between(*self.year) & df.p_cap.between(*self.capacity)][
            self.columns
        ]

    @param.depends("filtered_data")
    def number_of_rows(self):
        return f"Rows: {len(self.filtered_data())}"

    def __panel__(self):
        return pn.Column(
            pn.Row(
                pn.widgets.MultiChoice.from_param(self.param.columns, width=400),
                pn.Column(self.param.year, self.param.capacity),
            ),
            self.number_of_rows,
            pn.widgets.Tabulator(self.filtered_data, page_size=10, pagination="remote"),
        )


DataExplorer(data=turbines).servable()
