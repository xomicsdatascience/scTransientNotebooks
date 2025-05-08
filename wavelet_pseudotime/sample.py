def plot_df(df: pd.DataFrame, title: str = None, save=None, ax=None) -> None:
    """
    Plots each row of the DataFrame as a circle grouped by the 'source' column.
    The horizontal axis displays -log10(p_value) and the vertical positions
    are arranged based on the source group with added jitter.

    A legend is added for both the source groups and the circle size scale (intersection_size).

    Parameters:
        df (pd.DataFrame): A DataFrame containing the columns:
            - 'source': categorical column with 3 categories.
            - 'p_value': continuous values.
            - 'intersection_size': integers (will be used to scale circle sizes).
            - 'name': a descriptor for the row (unused in the plot).
    """
    # Compute the horizontal position: -log10(p_value)
    # (Make sure there are no p_value values equal to 0)
    df = df.copy()  # Avoid modifying the original DataFrame
    if ax is None:
        fig, ax = plt.subplots()

    if (df["p_value"] <= 0).any():
        raise ValueError("All p_value entries must be positive so that -log10 can be computed.")

    df["neg_log10"] = -np.log10(df["p_value"])

    # Create a mapping for each unique source to a base y-position.
    unique_sources = sorted(df["source"].unique())
    source_to_index = {source: idx for idx, source in enumerate(unique_sources, start=1)}

    # Map sources to base y positions.
    df["base_y"] = df["source"].map(source_to_index)

    # Add vertical jitter to separate the circles
    np.random.seed(0)  # For reproducibility
    jitter = np.random.uniform(-0.2, 0.2, size=len(df))
    df["y_pos"] = df["base_y"] + jitter

    # Create the plot
    # plt.figure(figsize=(4,4))

    # Plot each group with its own color and label.
    for source in unique_sources:
        subset = df[df["source"] == source]
        ax.scatter(
            subset["neg_log10"],
            subset["y_pos"],
            s=subset["intersection_size"] * 10,  # Scale circle sizes; adjust factor as needed.
            alpha=0.7,
            label=source,  # This will be used in the legend for sources.
            edgecolors="w"
        )

    ax.set_xlabel("-log10(p_value)")
    ax.set_yticks(list(source_to_index.values()), list(source_to_index.keys()))
    ax.set_ylim([0, 4])
    # plt.ylabel("Source Group")
    if title is None:
        ax.set_title("Function Enrichment Analysis")
    else:
        ax.set_title(title)

    # First, add the legend for the source groups.
    # source_legend = plt.legend(title="Source", loc="upper right")
    # plt.gca().add_artist(source_legend)

    # Now, create a legend for the circle sizes corresponding to 'intersection_size'.
    # Use three representative sizes: min, median, and max.
    size_min = df["intersection_size"].min()
    # size_median = int(df["intersection_size"].median())

    size_max = df["intersection_size"].max()
    size_median = int((size_min + size_max) / 2)  # int(df["intersection_size"].median())
    size_scale = 10  # This is the factor applied to intersection_size for the marker size

    sizes = [size_min, size_median, size_max]
    markers = [
        ax.scatter([], [], s=size * size_scale, color="gray", alpha=0.7, edgecolors="w")
        for size in sizes
    ]
    labels = [f"{size}" for size in sizes]

    ax.legend(markers, labels, title="Intersection Size", bbox_to_anchor=(1, 1),
              borderaxespad=0)  # , loc="lower right")
    ax.grid()
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    # plt.show()
