import cmasher as cmr
import numpy as np
import pandas as pd
import plotnine as p9
from box import Box
from pyprojroot import here


def make_performance_by_generation_faceted_plot(df):
    chosen_reweighting_temps = [1.2, 5.6, 10.0]
    df["reweighting_temp"] = df["reweighting_temp"].apply(lambda x: np.round(x, 1))
    df = df[df["reweighting_temp"].isin(chosen_reweighting_temps)]
    df_last = df[df["timestep"] == df["timestep"].max()]
    p = (
        p9.ggplot(
            df_last,
            mapping=p9.aes(
                x="generation",
                y="expected_true_probs_surprisal",
                color="rate",
                group="rate",
            ),
        )
        + p9.facet_wrap("~reweighting_temp")
        + p9.geom_line()
        + p9.labs(
            x="Generation",
            y="Score",
            color="Rate",
        )
        + p9.theme_tufte(base_size=18)
        + p9.theme(
            axis_title_x=p9.element_text(family="Charter"),
            axis_title_y=p9.element_text(family="Charter"),
            legend_title=p9.element_text(family="Charter"),
        )
    )
    return p


def make_eventual_performance_by_rate_facted_plot(df):
    chosen_reweighting_temps = [1.2, 5.6, 10.0]
    last_generation = df["generation"].max()
    df["reweighting_temp"] = df["reweighting_temp"].apply(lambda x: np.round(x, 1))
    df = df[df["reweighting_temp"].isin(chosen_reweighting_temps)]
    asocial_performance = df[
        (df["generation"] == 1) & (df["timestep"] == df["timestep"].max())
    ].iloc[0]["expected_true_probs_surprisal"]
    df_last = df[
        (df["generation"] == last_generation) & (df["timestep"] == df["timestep"].max())
    ]
    p = (
        p9.ggplot(
            df_last,
            mapping=p9.aes(x="rate", y="expected_true_probs_surprisal"),
        )
        + p9.facet_wrap("~reweighting_temp")
        + p9.geom_hline(
            yintercept=asocial_performance,
            color="darkgray",
            size=0.8,
        )
        + p9.geom_line(color="black")
        + p9.geom_point(color="black")
        + p9.theme_tufte()
        + p9.labs(x="Channel rate", y="Score")
        + p9.theme(
            axis_title_x=p9.element_text(family="Charter"),
        )
    )
    return p


def make_final_generation_heatmap(df):
    """Create a heatmap showing final generation performance across rate and reweighting_temp.

    Args:
        df: DataFrame with experimental results

    Returns:
        plotnine plot object
    """
    colors = cmr.ember(np.linspace(0, 1, 256))

    last_generation = df["generation"].max()
    df_last = df[
        (df["generation"] == last_generation) & (df["timestep"] == df["timestep"].max())
    ]

    # also remove rates > 10
    df_last = df_last[df_last["rate"] <= 10]

    df_last["rate"] = df_last["rate"].apply(lambda x: np.round(x, 2))
    df_last["reweighting_temp"] = df_last["reweighting_temp"].apply(
        lambda x: np.round(x, 2)
    )

    # Calculate tile dimensions based on actual spacing
    rate_unique = np.sort(df_last["rate"].unique())
    temp_unique = np.sort(df_last["reweighting_temp"].unique())

    # Function to calculate tile boundaries for each value
    def get_tile_boundaries(values, unique_vals):
        boundaries = []
        for val in values:
            idx = np.where(unique_vals == val)[0][0]
            if len(unique_vals) == 1:
                left = val - 0.5
                right = val + 0.5
            elif idx == 0:
                # First value: left edge to midpoint with next
                left = val - (unique_vals[1] - unique_vals[0]) / 4
                right = (unique_vals[0] + unique_vals[1]) / 2
            elif idx == len(unique_vals) - 1:
                # Last value: midpoint with previous to right edge
                left = (unique_vals[-2] + unique_vals[-1]) / 2
                right = val + (unique_vals[-1] - unique_vals[-2]) / 4
            else:
                # Middle values: midpoint to midpoint
                left = (unique_vals[idx - 1] + unique_vals[idx]) / 2
                right = (unique_vals[idx] + unique_vals[idx + 1]) / 2
            boundaries.append((left, right))
        return boundaries

    # Calculate tile boundaries
    rate_bounds = get_tile_boundaries(df_last["rate"], rate_unique)
    temp_bounds = get_tile_boundaries(df_last["reweighting_temp"], temp_unique)

    df_last["xmin"] = [b[0] for b in rate_bounds]
    df_last["xmax"] = [b[1] for b in rate_bounds]
    df_last["ymin"] = [b[0] for b in temp_bounds]
    df_last["ymax"] = [b[1] for b in temp_bounds]

    p = (
        p9.ggplot(
            df_last,
            mapping=p9.aes(
                xmin="xmin",
                xmax="xmax",
                ymin="ymin",
                ymax="ymax",
                fill="expected_true_probs_surprisal",
            ),
        )
        + p9.geom_rect()
        + p9.scale_fill_gradientn(colors=colors)
        + p9.scale_x_continuous(breaks=rate_unique)
        + p9.scale_y_continuous(breaks=temp_unique)
        + p9.labs(
            x="Channel rate",
            y="Reweighting temperature",
            fill="Gen. 20 score",
        )
        + p9.theme_tufte(base_size=18)
        + p9.theme(
            axis_title_x=p9.element_text(family="Charter"),
            axis_title_y=p9.element_text(family="Charter"),
            axis_text_x=p9.element_text(angle=90, size=10),
            legend_title=p9.element_text(family="Charter", size=10),
        )
    )

    return p


def main(args):
    df = pd.read_csv(here(f"data/{args.data_filename}.csv"))
    df["generation"] = df["generation"] + 1

    p_perf_by_gen = make_performance_by_generation_faceted_plot(df)
    p_perf_by_gen.save(
        here(f"figures/{args.data_filename}-performance_by_generation_ssl_faceted.pdf"),
        width=8,
        height=4,
    )

    p_perf_by_rate = make_eventual_performance_by_rate_facted_plot(df)
    p_perf_by_rate.save(
        here(
            f"figures/{args.data_filename}-eventual_performance_by_rate_ssl_faceted.pdf"
        ),
        width=8,
        height=4,
    )

    p_final_gen_heatmap = make_final_generation_heatmap(df)
    p_final_gen_heatmap.save(
        here(f"figures/{args.data_filename.replace('-processed', '')}-ssl_heatmap.pdf"),
        width=5,
        height=4,
    )


if __name__ == "__main__":
    config = Box(
        {
            "data_filename": "dirichlet_categorical_ssl-processed",
        }
    )
    main(config)
