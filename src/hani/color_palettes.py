"""
Color-blind friendly color palettes for HANI visualizations.

Based on Paul Tol's color schemes which are designed to be distinguishable
for people with different types of color vision deficiency.
https://personal.sron.nl/~pault/
"""

# Color-blind friendly qualitative palette (for categorical data)
# Works well for protanopia, deuteranopia, and tritanopia
COLORBLIND_QUALITATIVE = [
    "#0077BB",  # Blue
    "#33BBEE",  # Cyan
    "#009988",  # Teal
    "#EE7733",  # Orange
    "#CC3311",  # Red
    "#EE3377",  # Magenta
    "#BBBBBB",  # Grey
    "#000000",  # Black
]

# Color-blind friendly sequential palette (for ordered/continuous data)
COLORBLIND_SEQUENTIAL = [
    "#FFFFE0",  # Lightest yellow
    "#FFEAA7",
    "#FDCB6E",
    "#E17055",
    "#D63031",  # Darkest red
]

# Color-blind friendly diverging palette (for data with meaningful middle point)
COLORBLIND_DIVERGING = [
    "#2166AC",  # Dark blue
    "#67A9CF",  # Light blue
    "#D1E5F0",  # Very light blue
    "#F7F7F7",  # Neutral grey
    "#FDDBC7",  # Very light red
    "#EF8A62",  # Light red
    "#B2182B",  # Dark red
]

# Default Plotly layout with color-blind friendly settings
DEFAULT_PLOT_LAYOUT = {
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "font": {"family": "Segoe UI, sans-serif", "color": "#282D3C"},
    "xaxis": {
        "gridcolor": "#E9ECEF",
        "linecolor": "#DEE2E6",
        "zerolinecolor": "#DEE2E6",
    },
    "yaxis": {
        "gridcolor": "#E9ECEF",
        "linecolor": "#DEE2E6",
        "zerolinecolor": "#DEE2E6",
    },
    "colorway": COLORBLIND_QUALITATIVE,
}


def get_colorblind_palette(
    n_colors: int = 8, palette_type: str = "qualitative"
) -> list[str]:
    """
    Get a color-blind friendly palette.

    Args:
        n_colors: Number of colors needed
        palette_type: Type of palette - "qualitative", "sequential", or "diverging"

    Returns:
        List of hex color codes
    """
    if palette_type == "qualitative":
        palette = COLORBLIND_QUALITATIVE
    elif palette_type == "sequential":
        palette = COLORBLIND_SEQUENTIAL
    elif palette_type == "diverging":
        palette = COLORBLIND_DIVERGING
    else:
        raise ValueError(f"Unknown palette type: {palette_type}")

    # If we need more colors than available, cycle through the palette
    if n_colors > len(palette):
        return (palette * ((n_colors // len(palette)) + 1))[:n_colors]

    return palette[:n_colors]


def apply_colorblind_theme(fig, palette_type: str = "qualitative"):
    """
    Apply color-blind friendly theme to a Plotly figure.

    Args:
        fig: Plotly figure object
        palette_type: Type of palette to use

    Returns:
        Modified figure with color-blind friendly colors
    """
    # Update layout with default settings
    fig.update_layout(**DEFAULT_PLOT_LAYOUT)

    # Update trace colors if needed
    n_traces = len(fig.data)
    if n_traces > 0:
        colors = get_colorblind_palette(n_traces, palette_type)
        for i, trace in enumerate(fig.data):
            if hasattr(trace, "marker"):
                trace.marker.color = colors[i % len(colors)]
            elif hasattr(trace, "line"):
                trace.line.color = colors[i % len(colors)]

    return fig
