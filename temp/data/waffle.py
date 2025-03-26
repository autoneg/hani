import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Open the HTML file using the default web browser (platform-specific)
import os
import platform
import webbrowser


# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Replace the encoding '..' with 0 and convert to integer
    df["Valor"] = df["Valor"].str.replace("..", "0").astype(int)
    return df  # Return the raw DataFrame


# Map codes to levels
def code2level(x):
    return str(x)


def calculate_percentage_by_education(df):
    """
    Groups the data by education level and calculates the percentage for each level.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.

    Returns:
    pd.Series: Series with the percentage for each education level.
    """
    # Group by 'NIV_EDUCA_esta' and sum the 'Valor' column
    value_by_education = df.groupby("NIV_EDUCA_esta")["Valor"].sum()

    # Calculate the percentage for each education level
    percentage_by_education = (value_by_education / value_by_education.sum()) * 100

    return percentage_by_education


def round_percentages_to_100(raw_percentages):
    """
    Rounds down the percentages and distributes the remainder to ensure the sum equals 100%.

    Parameters:
    raw_percentages (pd.Series): Series of raw percentages.

    Returns:
    pd.Series: A Series with rounded percentages summing to 100%.
    """
    # Round down the percentages and calculate the difference to 100%
    rounded_percentages = np.floor(raw_percentages)
    difference = int(100 - rounded_percentages.sum())

    # Distribute the difference by increasing the largest remainders
    remainders = raw_percentages - rounded_percentages
    indices = remainders.nlargest(difference).index
    rounded_percentages.loc[indices] += 1

    return rounded_percentages.astype(int)


# Function to rename and round percentages
def prepare_percentages(percentage_by_education):
    percentage_by_education = percentage_by_education[percentage_by_education > 0]
    return round_percentages_to_100(percentage_by_education)


def create_waffle_grid(rounded_percentages, grid_size_x, grid_size_y):
    """
    Creates a waffle grid from rounded percentages.

    Parameters:
    rounded_percentages (pd.Series): Series with rounded percentages.
    grid_size (int): Size of the grid.

    Returns:
    np.ndarray: The waffle grid with integer categories.
    """
    # Create the waffle grid directly using the integer indexes
    categories = np.repeat(rounded_percentages.index, rounded_percentages.values)

    # Convert to NumPy array and reshape into the desired grid size
    return np.array(categories).reshape(grid_size_y, grid_size_x)


def create_heatmap_trace(waffle_grid, custom_colors, code2level, rounded_percentages):
    """
    Creates a heatmap trace for the waffle chart.

    Parameters:
    waffle_grid (np.ndarray): The waffle grid.
    custom_colors (list): List of colors to use for the chart.
    level2code (dict): Mapping of categories to integer indices.
    rounded_percentages (pd.Series): Series with rounded percentages.

    Returns:
    plotly.graph_objects.Heatmap: The heatmap trace.
    """
    hover_text = np.vectorize(lambda x: f"{code2level(x)}: {rounded_percentages[x]}%")(
        waffle_grid
    )

    unique_indices = np.unique(waffle_grid)
    used_colors = [
        custom_colors[i] for i in range(len(custom_colors)) if i + 1 in unique_indices
    ]

    return go.Heatmap(
        z=waffle_grid,
        x=np.arange(waffle_grid.shape[1]),
        y=np.arange(waffle_grid.shape[0]),
        showscale=False,
        hoverinfo="text",
        text=hover_text,
        colorscale=used_colors,
    )


def create_border_trace(grid_size_x, grid_size_y):
    """
    Creates a border trace for the waffle chart.

    Parameters:
    grid_size (int): Size of the grid.

    Returns:
    plotly.graph_objects.Scatter: The border trace.
    """
    borders_x = []
    borders_y = []

    # Loop through each square in the grid to add borders
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            # Create the coordinates for each square's borders
            x0, y0 = j - 0.5, i - 0.5  # Bottom-left corner
            x1, y1 = j + 0.5, i - 0.5  # Bottom-right corner
            x2, y2 = j - 0.5, i + 0.5  # Top-left corner
            x3, y3 = j + 0.5, i + 0.5  # Top-right corner

            # Add four borders (left, bottom, top, right)
            borders_x.extend([x0, x1])  # Bottom border
            borders_y.extend([y0, y0])
            borders_x.extend([x1, x3])  # Right border
            borders_y.extend([y1, y3])

    # Create the border trace
    return go.Scatter(
        x=borders_x,
        y=borders_y,
        mode="lines",
        line=dict(color="white", width=3),  # Border color and width
        showlegend=False,
        hoverinfo="skip",
    )


def create_legend(rounded_percentages, custom_colors, code2level):
    """
    Creates the legend for the waffle chart.
    Parameters:
    rounded_percentages (pd.Series): Series with rounded percentages.
    custom_colors (list): List of colors to use for the chart.
    Returns:
    list: List of legend traces.
    """
    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=code2level(category),
            marker=dict(symbol="square", color=color, size=10),
            showlegend=True,
        )
        for category, color in zip(
            reversed(rounded_percentages.index),
            reversed(custom_colors[: len(rounded_percentages)]),
        )
    ]


def create_layout(title, subtitle, legend_name, footer):
    """
    Creates the layout for the waffle chart.

    Returns:
    plotly.graph_objects.Layout: The layout for the chart.
    """
    # Subtitle annotations
    subtitle_annotation = [
        dict(
            x=-0.1025,
            y=1.125,
            xref="paper",
            yref="paper",
            text=subtitle,
            showarrow=False,
            font=dict(size=14, color="grey"),
            xanchor="left",
        )
    ]

    # Footer annotations
    footer_annotation = [
        dict(
            x=-0.05,
            y=-0.10,
            xref="paper",
            yref="paper",
            text=footer,
            showarrow=False,
            font=dict(size=12, color="grey"),
            xanchor="left",
        )
    ]

    return go.Layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, tickvals=[]),
        yaxis=dict(showgrid=False, zeroline=False, tickvals=[]),
        height=550,
        width=900,
        plot_bgcolor="rgba(255,255,255,1)",
        paper_bgcolor="rgba(255,255,255,1)",
        legend=dict(
            title=legend_name,
            orientation="v",  # Legend displayed vertically
            x=2.25,  # Position the legend on the left side of the chart
            y=0.5,  # Vertical center position
            xanchor="right",  # Anchor the legend to the right of the `x` position
            yanchor="middle",  # Anchor the legend to the middle of the `y` position
        ),
        font=dict(family="Poppins"),
        annotations=footer_annotation + subtitle_annotation,
    )


def waffle():
    # Load and preprocess the data
    df = load_and_preprocess_data("education.csv")

    # Visualize the first rows of the dataset
    df.head()

    # Calculate percentages
    percentage_by_education = calculate_percentage_by_education(df)

    # Prepare rounded percentages
    percentage_by_education_rounded = prepare_percentages(percentage_by_education)

    # Visualize the resulting Series
    percentage_by_education_rounded
    # Define the grid size (10x10 = 100 squares)
    grid_size_x, grid_size_y = 10, 10

    # Create a 10x10 grid (total 100 squares)
    waffle_grid = create_waffle_grid(
        percentage_by_education_rounded, grid_size_x, grid_size_y
    )

    # Calculate percentages
    percentage_by_education = calculate_percentage_by_education(df)

    # Prepare rounded percentages
    percentage_by_education_rounded = prepare_percentages(percentage_by_education)

    custom_colors = ["#ECC199", "#CBDBA7", "#789342", "#76C1C3", "#548D95"]
    # Create the heatmap trace
    heatmap_trace = create_heatmap_trace(
        waffle_grid, custom_colors, code2level, percentage_by_education_rounded
    )

    # Create the figure
    go.Figure(data=[heatmap_trace])

    # Create the border trace
    border_trace = create_border_trace(grid_size_x, grid_size_y)

    # Create the legend
    legend_traces = create_legend(
        percentage_by_education_rounded, custom_colors, code2level
    )

    # Set chart details
    title = "Barcelona's Educational Landscape"
    subtitle = "Percentage of the Population Across Different Education Levels"
    legend_name = "Educational Levels"
    footer = 'Source: <a href="https://opendata-ajuntament.barcelona.cat/data/es/dataset/pad_mdbas_niv-educa-esta_sexe" target="_blank">Open Data Barcelona</a>. Retrieved October 3, 2024.'
    # Create layout with annotations
    layout = create_layout(title, subtitle, legend_name, footer)

    # Create the figure
    return go.Figure(data=[heatmap_trace, border_trace] + legend_traces, layout=layout)


def open_html_in_browser(filename):
    """Opens an HTML file in the default web browser."""
    filepath = os.path.abspath(filename)  # get the full path
    if platform.system() == "Windows":
        os.startfile(filepath)
    elif platform.system() == "Darwin":  # macOS
        webbrowser.open_new_tab(f"file://{filepath}")
    else:  # Linux/other Unix-like systems
        webbrowser.open_new_tab(f"file://{filepath}")


if __name__ == "__main__":
    fig = waffle()

    fig.write_html("my_plot.html")
    open_html_in_browser("my_plot.html")
