"""
LFPS Spatial Mapping Module
==============================

Spatial interpolation and map generation for LFPS attributes.
Supports IDW, RBF, and linear/cubic interpolation methods.
Generates publication-quality maps with station overlay.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import griddata, RBFInterpolator
import logging

logger = logging.getLogger(__name__)


def create_interpolation_grid(lats, lons, resolution=100, padding_fraction=0.05):
    """
    Create a regular grid for interpolation.

    Parameters
    ----------
    lats : np.ndarray
        Station latitudes.
    lons : np.ndarray
        Station longitudes.
    resolution : int
        Number of grid points per axis.
    padding_fraction : float
        Fraction of range to add as padding.

    Returns
    -------
    grid_lon : np.ndarray
        2D grid of longitudes.
    grid_lat : np.ndarray
        2D grid of latitudes.
    """
    lat_range = lats.max() - lats.min()
    lon_range = lons.max() - lons.min()

    padding_lat = max(lat_range * padding_fraction, 0.001)
    padding_lon = max(lon_range * padding_fraction, 0.001)

    lat_lin = np.linspace(lats.min() - padding_lat, lats.max() + padding_lat, resolution)
    lon_lin = np.linspace(lons.min() - padding_lon, lons.max() + padding_lon, resolution)

    grid_lon, grid_lat = np.meshgrid(lon_lin, lat_lin)

    return grid_lon, grid_lat


def interpolate_idw(lons, lats, values, grid_lon, grid_lat, power=2.0,
                    min_distance=1e-10):
    """
    Inverse Distance Weighting (IDW) interpolation.

    Parameters
    ----------
    lons, lats : np.ndarray
        Station coordinates.
    values : np.ndarray
        Attribute values at stations.
    grid_lon, grid_lat : np.ndarray
        2D interpolation grid.
    power : float
        IDW power parameter (higher = more local influence).
    min_distance : float
        Minimum distance to avoid division by zero.

    Returns
    -------
    np.ndarray
        Interpolated grid values.
    """
    grid_shape = grid_lon.shape
    grid_values = np.zeros(grid_shape)

    flat_lon = grid_lon.ravel()
    flat_lat = grid_lat.ravel()

    for i in range(len(flat_lon)):
        distances = np.sqrt(
            (lons - flat_lon[i]) ** 2 + (lats - flat_lat[i]) ** 2
        )

        # Check if point coincides with a station
        min_dist = np.min(distances)
        if min_dist < min_distance:
            grid_values.ravel()[i] = values[np.argmin(distances)]
            continue

        weights = 1.0 / (distances ** power)
        grid_values.ravel()[i] = np.sum(weights * values) / np.sum(weights)

    return grid_values


def interpolate_rbf(lons, lats, values, grid_lon, grid_lat,
                    kernel="multiquadric", smoothing=0.0):
    """
    Radial Basis Function (RBF) interpolation.

    Parameters
    ----------
    lons, lats : np.ndarray
        Station coordinates.
    values : np.ndarray
        Attribute values.
    grid_lon, grid_lat : np.ndarray
        2D interpolation grid.
    kernel : str
        RBF kernel: 'multiquadric', 'inverse_multiquadric',
        'thin_plate_spline', 'gaussian', 'cubic', 'linear'.
    smoothing : float
        Smoothing parameter.

    Returns
    -------
    np.ndarray
        Interpolated grid values.
    """
    points = np.column_stack([lons, lats])
    grid_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])

    rbf = RBFInterpolator(points, values, kernel=kernel, smoothing=smoothing)
    grid_values = rbf(grid_points).reshape(grid_lon.shape)

    return grid_values


def interpolate_griddata(lons, lats, values, grid_lon, grid_lat, method="cubic"):
    """
    Scipy griddata interpolation.

    Parameters
    ----------
    method : str
        'linear', 'cubic', or 'nearest'
    """
    points = np.column_stack([lons, lats])
    grid_values = griddata(points, values, (grid_lon, grid_lat), method=method)

    # Fill NaN edges with nearest
    if np.any(np.isnan(grid_values)):
        nearest = griddata(points, values, (grid_lon, grid_lat), method="nearest")
        mask = np.isnan(grid_values)
        grid_values[mask] = nearest[mask]

    return grid_values


def interpolate_attribute(lons, lats, values, grid_lon, grid_lat, config):
    """
    Dispatch to the appropriate interpolation method.

    Parameters
    ----------
    lons, lats, values : np.ndarray
        Station data.
    grid_lon, grid_lat : np.ndarray
        Interpolation grid.
    config : dict
        Mapping configuration.

    Returns
    -------
    np.ndarray
        Interpolated grid values.
    """
    map_cfg = config.get("mapping", {})
    method = map_cfg.get("method", "idw")

    if method == "idw":
        power = map_cfg.get("idw_power", 2.0)
        return interpolate_idw(lons, lats, values, grid_lon, grid_lat, power=power)
    elif method == "rbf":
        kernel = map_cfg.get("rbf_function", "multiquadric")
        return interpolate_rbf(lons, lats, values, grid_lon, grid_lat, kernel=kernel)
    elif method in ["linear", "cubic", "nearest"]:
        return interpolate_griddata(lons, lats, values, grid_lon, grid_lat, method=method)
    else:
        logger.warning(f"Unknown method '{method}', falling back to IDW")
        return interpolate_idw(lons, lats, values, grid_lon, grid_lat)


def plot_attribute_map(grid_lon, grid_lat, grid_values, station_lons, station_lats,
                       station_values=None, title="LFPS Attribute Map",
                       colorbar_label="Value", cmap="RdYlGn_r",
                       station_labels=None, figsize=(10, 8),
                       show_stations=True, contour_levels=10,
                       vmin=None, vmax=None):
    """
    Generate a publication-quality attribute map.

    Parameters
    ----------
    grid_lon, grid_lat : np.ndarray
        2D interpolation grid.
    grid_values : np.ndarray
        Interpolated attribute values.
    station_lons, station_lats : np.ndarray
        Station coordinates.
    station_values : np.ndarray, optional
        Values at stations for color-coding.
    title : str
        Map title.
    colorbar_label : str
        Colorbar label.
    cmap : str
        Matplotlib colormap name.
    station_labels : list of str, optional
        Station ID labels.
    figsize : tuple
        Figure size.
    show_stations : bool
        Whether to plot station locations.
    contour_levels : int
        Number of contour levels.
    vmin, vmax : float, optional
        Colorbar limits.

    Returns
    -------
    matplotlib.figure.Figure
        Map figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")

    # Set color limits
    if vmin is None:
        vmin = np.nanpercentile(grid_values, 2)
    if vmax is None:
        vmax = np.nanpercentile(grid_values, 98)

    # Filled contour plot
    cf = ax.contourf(
        grid_lon, grid_lat, grid_values,
        levels=contour_levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend="both",
    )

    # Contour lines
    cs = ax.contour(
        grid_lon, grid_lat, grid_values,
        levels=contour_levels,
        colors="k",
        linewidths=0.3,
        alpha=0.4,
    )

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Station overlay
    if show_stations:
        if station_values is not None:
            scatter = ax.scatter(
                station_lons, station_lats,
                c=station_values,
                cmap=cmap,
                edgecolors="black",
                linewidths=1.0,
                s=60,
                zorder=5,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            ax.scatter(
                station_lons, station_lats,
                c="white",
                edgecolors="black",
                linewidths=1.0,
                s=40,
                zorder=5,
                marker="^",
            )

        # Station labels
        if station_labels is not None:
            for i, label in enumerate(station_labels):
                ax.annotate(
                    label,
                    (station_lons[i], station_lats[i]),
                    fontsize=6,
                    ha="left",
                    va="bottom",
                    xytext=(3, 3),
                    textcoords="offset points",
                    alpha=0.7,
                )

    # Formatting
    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(labelsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    return fig


def plot_composite_map(grid_lon, grid_lat, attribute_grids, station_lons, station_lats,
                       attribute_names, station_labels=None, figsize=(16, 12)):
    """
    Generate a composite figure with multiple attribute maps.

    Parameters
    ----------
    grid_lon, grid_lat : np.ndarray
        2D interpolation grid.
    attribute_grids : dict
        Dictionary of {attribute_name: grid_values}.
    station_lons, station_lats : np.ndarray
        Station coordinates.
    attribute_names : list
        List of attribute keys to plot.
    station_labels : list, optional
        Station ID labels.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Composite figure.
    """
    n_attrs = len(attribute_names)
    n_cols = min(3, n_attrs)
    n_rows = int(np.ceil(n_attrs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor="white")
    if n_attrs == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    # Colormap assignments for different attributes
    cmap_map = {
        "psd": "hot",
        "vhsr": "RdYlGn_r",
        "v/h": "RdYlGn_r",
        "rectilinearity": "YlOrRd",
        "dip": "coolwarm",
        "azimuth": "hsv",
        "planarity": "PuBuGn",
        "eigenvalue": "inferno",
        "energy": "hot",
        "peak_freq": "viridis",
        "peak_amp": "magma",
    }

    for i, attr_name in enumerate(attribute_names):
        ax = axes[i]
        grid = attribute_grids[attr_name]

        # Select colormap
        cmap = "RdYlGn_r"
        for key, cm in cmap_map.items():
            if key in attr_name.lower():
                cmap = cm
                break

        vmin = np.nanpercentile(grid, 2)
        vmax = np.nanpercentile(grid, 98)

        cf = ax.contourf(grid_lon, grid_lat, grid, levels=12, cmap=cmap,
                         vmin=vmin, vmax=vmax, extend="both")
        ax.contour(grid_lon, grid_lat, grid, levels=12, colors="k",
                   linewidths=0.2, alpha=0.3)

        cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

        ax.scatter(station_lons, station_lats, c="white", edgecolors="black",
                   linewidths=0.8, s=20, zorder=5, marker="^")

        ax.set_title(attr_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("LFPS Composite Attribute Maps", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_polarization_rose(azimuths, title="Azimuth Rose Diagram", figsize=(6, 6)):
    """
    Plot a rose diagram for azimuth distribution.

    Parameters
    ----------
    azimuths : np.ndarray or list
        Azimuth values in degrees.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, polar=True)

    # Convert to radians
    az_rad = np.radians(azimuths)

    # Create histogram
    n_bins = 36
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    counts, _ = np.histogram(az_rad, bins=bins)

    # Plot
    theta = bins[:-1] + np.diff(bins) / 2
    width = 2 * np.pi / n_bins

    bars = ax.bar(theta, counts, width=width, alpha=0.7,
                  color=plt.cm.RdYlBu(counts / counts.max() if counts.max() > 0 else counts),
                  edgecolor="black", linewidth=0.5)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


def normalize_attribute(values, method="minmax"):
    """
    Normalize attribute values for composite analysis.

    Parameters
    ----------
    values : np.ndarray
    method : str
        'minmax', 'zscore', or 'percentile'

    Returns
    -------
    np.ndarray
        Normalized values.
    """
    if method == "minmax":
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax - vmin > 0:
            return (values - vmin) / (vmax - vmin)
        return np.zeros_like(values)

    elif method == "zscore":
        mean, std = np.nanmean(values), np.nanstd(values)
        if std > 0:
            return (values - mean) / std
        return np.zeros_like(values)

    elif method == "percentile":
        from scipy.stats import rankdata
        ranks = rankdata(values, method="average")
        return ranks / len(values)

    return values
