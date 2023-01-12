import numpy as np
import pandas as pd
import seaborn as sns

from rbo_weight import rbo_cumulative_weight, rbo_discrete_weight


def plot_cumulative_weight() -> None:
    ps = [0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
    ds = np.arange(1, 101)
    data = []
    for p in ps:
        for d in ds:
            weight = rbo_cumulative_weight(d, p)
            data.append([d, p, weight])
    df = pd.DataFrame(data, columns=["Rank", "RBO P", "Weight"])
    df2 = df.pivot(index="Rank", columns="RBO P", values="Weight")
    print(df2.to_markdown())
    df["RBO P"] = df["RBO P"].astype(str)
    g = sns.relplot(kind="line", data=df, x="Rank", y="Weight", hue="RBO P", aspect=2.0)
    g.ax.set_xticks([1] + list(range(5, 101, 5)))
    g.ax.set_yticks(np.arange(0, 1.1, 0.1))
    g.tight_layout()
    # g.savefig("rbo_cumulative_weight.pdf")
    g.savefig("cumulative.png")


def plot_rank_weight() -> None:
    ps = [0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
    ds = np.arange(1, 21)
    data = []
    for p in ps:
        for d in ds:
            weight = rbo_discrete_weight(d, p)
            data.append([d, p, weight])
    df = pd.DataFrame(data, columns=["Rank", "RBO P", "Weight"])
    df["RBO P"] = df["RBO P"].astype(str)
    g = sns.relplot(kind="line", data=df, x="Rank", y="Weight", hue="RBO P", aspect=2.0)
    g.ax.set_xticks(ds)
    g.tight_layout()
    g.savefig("discrete.png")


def main() -> None:
    sns.set_theme("paper", "darkgrid", font="Linux Biolinum O")
    plot_cumulative_weight()
    plot_rank_weight()


if __name__ == "__main__":
    main()
