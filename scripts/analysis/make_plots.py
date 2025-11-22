import pandas as pd
import plotnine as p9
from box import Box
from pyprojroot import here


def make_performance_by_generation_plot(df):
    df_last = df[df["timestep"] == df["timestep"].max()]
    if df["generation"].max() == 20:
        xticks = [0, 5, 10, 15, 20]
    elif df["generation"].max() == 10:
        xticks = [0, 2, 4, 6, 8, 10]
    else:
        raise ValueError(f"Weird number of generations: {df['generation'].max()}")
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
        + p9.geom_line()
        + p9.scale_x_continuous(breaks=xticks)
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


def make_eventual_performance_by_rate_plot(df):
    last_generation = df["generation"].max()
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
        + p9.geom_hline(
            yintercept=asocial_performance,
            color="darkgray",
            size=0.8,
        )
        + p9.geom_line(color="black")
        + p9.geom_point(color="black")
        + p9.theme_tufte(base_size=18)
        + p9.labs(x="Channel rate", y="Generation 20 Score")
        + p9.theme(
            axis_title_x=p9.element_text(family="Charter"),
            axis_title_y=p9.element_text(family="Charter"),
            legend_title=p9.element_text(family="Charter"),
        )
    )
    return p


def main(args):
    df = pd.read_csv(here(f"data/{args.data_filename}.csv"))
    condition_name = args.data_filename.split("-")[0]
    df["generation"] = df["generation"] + 1

    p_perf_by_gen = make_performance_by_generation_plot(df)
    if condition_name == "dirichlet_categorical":
        p_perf_by_gen.save(
            here(f"figures/{condition_name}-performance_by_generation.pdf"),
            width=6,
            height=5,
        )
    else:
        p_perf_by_gen.save(
            here(f"figures/{condition_name}-performance_by_generation.pdf"),
        )

    p_perf_by_rate = make_eventual_performance_by_rate_plot(df)
    if condition_name == "dirichlet_categorical":
        p_perf_by_rate.save(
            here(f"figures/{condition_name}-eventual_performance_by_rate.pdf"),
            width=6,
            height=5,
        )
    else:
        p_perf_by_rate.save(
            here(f"figures/{condition_name}-eventual_performance_by_rate.pdf"),
        )


if __name__ == "__main__":
    filenames = [
        "dirichlet_categorical-processed",
        "dirichlet_categorical_decreasing-processed",
        "dirichlet_categorical_farthertrueprobs-processed",
        "dirichlet_categorical_closertrueprobs-processed",
        "dirichlet_categorical_longlife-processed",
    ]
    for filename in filenames:
        config = Box(
            {
                "data_filename": filename,
            }
        )
        main(config)
