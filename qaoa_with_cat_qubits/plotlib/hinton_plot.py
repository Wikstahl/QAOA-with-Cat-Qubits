import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm


def hinton(matrix, max_weight=None):
    """
    Draw Hinton diagram with RdBu colormap, white background, colorbar, black frame,
    custom x ticks on top, y ticks with labels, without a dashed grid and without minor ticks.
    """
    matrix = np.transpose(matrix)
    fig, ax = plt.subplots()
    ax = ax if ax is not None else plt.gca()
    # Using the RdBu colormap
    cmap = cm.viridis

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor("white")  # Setting background to white
    ax.set_aspect("equal", "box")

    # Adding a black frame around the plot
    for spine in ax.spines.values():
        spine.set_color("black")

    # Disable grid
    ax.grid(False)

    norm = plt.Normalize(0, 1)  # Normalizing for the range -1 to 1

    for (x, y), w in np.ndenumerate(matrix):
        color = cmap(norm(w))
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle(
            [x - size / 2, y - size / 2], size, size, facecolor=color, edgecolor=color
        )
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    # Create a mappable object for the colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    # Add colorbar
    plt.colorbar(mappable, ax=ax, orientation="vertical", ticks=[-1, -0.5, 0, 0.5, 1])

    # Set the custom tick labels
    tick_labels = ["I", "X", "Y", "Z"]
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, va="bottom")
    ax.set_yticklabels(tick_labels)

    # Remove minor ticks
    ax.tick_params(which="minor", size=0)

    # Move x ticks to the top
    ax.xaxis.tick_top()
    return fig, ax
