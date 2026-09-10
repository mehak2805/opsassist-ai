import pandas as pd


def calculate_scores():

    benchmark_file = "data/benchmark_results.csv"
    quality_file = "data/benchmark_quality_results.csv"

    benchmark_df = pd.read_csv(benchmark_file)
    quality_df = pd.read_csv(quality_file)

    df = benchmark_df.merge(
        quality_df,
        on="model"
    )

    # ---------------------------------------------
    # SPEED SCORE
    # Higher tokens/second = better
    # ---------------------------------------------

    max_speed = df["tokens_per_second"].max()

    df["speed_score"] = (
        df["tokens_per_second"] / max_speed
    ) * 100


    # ---------------------------------------------
    # MEMORY SCORE
    # Lower memory usage = better
    # ---------------------------------------------

    min_memory = df["peak_memory_mb"].min()

    df["memory_score"] = (
        min_memory / df["peak_memory_mb"]
    ) * 100


    # ---------------------------------------------
    # OVERALL SCORE
    #
    # Quality     = 50%
    # Speed       = 30%
    # Memory      = 20%
    # ---------------------------------------------

    df["overall_score"] = (
        df["quality_score"] * 0.50
        + df["speed_score"] * 0.30
        + df["memory_score"] * 0.20
    )

    df["overall_score"] = df["overall_score"].round(2)

    df["speed_score"] = df["speed_score"].round(2)
    df["memory_score"] = df["memory_score"].round(2)


    # ---------------------------------------------
    # RANK MODELS
    # ---------------------------------------------

    df["rank"] = (
        df["overall_score"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    df = df.sort_values(
        "overall_score",
        ascending=False
    )


    # ---------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------

    output_columns = [
        "model",
        "latency_seconds",
        "tokens_per_second",
        "peak_memory_mb",
        "quality_score",
        "speed_score",
        "memory_score",
        "overall_score",
        "rank"
    ]

    final_df = df[output_columns]

    final_df.to_csv(
        "data/model_scores.csv",
        index=False
    )


    # ---------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------

    print("\n" + "=" * 80)
    print("OPSASSIST AI - FINAL MODEL SCORE")
    print("=" * 80)

    print(
        final_df.to_string(
            index=False
        )
    )


    winner = final_df.iloc[0]

    print("\n" + "=" * 80)
    print("🏆 WINNING MODEL")
    print("=" * 80)

    print(
        f"{winner['model']} "
        f"with an overall score of "
        f"{winner['overall_score']}/100"
    )

    print(
        "\nResults saved to: "
        "data/model_scores.csv"
    )


if __name__ == "__main__":

    calculate_scores()