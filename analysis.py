"""Exploratory analysis of Titanic passenger survival."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Titanic.csv"
FIGURES = ROOT / "figures"

AGE_BINS = [0, 12, 18, 35, 50, 80]
AGE_LABELS = ["0-12", "13-18", "19-35", "36-50", "51+"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Title"] = df["Name"].str.extract(r", ([^.]+)\.")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    df["FareQuartile"] = pd.qcut(
        df["Fare"], 4, labels=["Q1 cheapest", "Q2", "Q3", "Q4 dearest"]
    )
    return df


def survival_rate(series: pd.Series) -> float:
    return round(float(series.mean()) * 100, 1)


def print_overview(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("TITANIC — exploratory analysis")
    print("=" * 60)
    print(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"Overall survival: {survival_rate(df['Survived'])}% "
          f"({int(df['Survived'].sum())} of {len(df)})")
    print()
    print("Missing values")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(missing.to_string())
    print()
    print("Survival by sex")
    print(
        df.groupby("Sex")["Survived"]
        .agg(n="count", survivors="sum", rate="mean")
        .assign(rate=lambda x: (x["rate"] * 100).round(1))
        .to_string()
    )
    print()
    print("Survival by ticket class")
    print(
        df.groupby("Pclass")["Survived"]
        .agg(n="count", survivors="sum", rate="mean")
        .assign(rate=lambda x: (x["rate"] * 100).round(1))
        .to_string()
    )
    print()
    print("Survival by class and sex")
    print(
        df.groupby(["Pclass", "Sex"])["Survived"]
        .agg(n="count", rate="mean")
        .assign(rate=lambda x: (x["rate"] * 100).round(1))
        .to_string()
    )
    print()
    print("Survival by age group (known age only)")
    print(
        df.groupby("AgeGroup", observed=False)["Survived"]
        .agg(n="count", rate="mean")
        .assign(rate=lambda x: (x["rate"] * 100).round(1))
        .to_string()
    )
    print()
    print("Survival by family size")
    print(
        df.groupby("FamilySize")["Survived"]
        .agg(n="count", rate="mean")
        .assign(rate=lambda x: (x["rate"] * 100).round(1))
        .to_string()
    )
    print()
    print("Survival by embarkation port")
    print(
        df.groupby("Embarked", dropna=False)["Survived"]
        .agg(n="count", rate="mean")
        .assign(rate=lambda x: (x["rate"] * 100).round(1))
        .to_string()
    )
    print()
    print("Survival by title (top counts)")
    print(
        df.groupby("Title")["Survived"]
        .agg(n="count", rate="mean")
        .sort_values("n", ascending=False)
        .assign(rate=lambda x: (x["rate"] * 100).round(1))
        .head(8)
        .to_string()
    )


def save_figures(df: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    palette = {0: "#6b7280", 1: "#2563eb"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = df.groupby("Sex")["Survived"].mean() * 100
    rates.plot(kind="bar", ax=ax, color=["#db2777", "#2563eb"], width=0.6)
    ax.set_ylabel("Survival rate (%)")
    ax.set_xlabel("Sex")
    ax.set_title("Survival rate by sex")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "survival_by_sex.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = df.groupby("Pclass")["Survived"].mean() * 100
    rates.plot(kind="bar", ax=ax, color="#2563eb", width=0.6)
    ax.set_ylabel("Survival rate (%)")
    ax.set_xlabel("Ticket class (1 = highest)")
    ax.set_title("Survival rate by ticket class")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "survival_by_class.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    grouped = (
        df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack() * 100
    )
    grouped.plot(kind="bar", ax=ax, color=["#db2777", "#2563eb"], width=0.7)
    ax.set_ylabel("Survival rate (%)")
    ax.set_xlabel("Ticket class")
    ax.set_title("Survival rate by class and sex")
    ax.set_ylim(0, 100)
    ax.legend(title="Sex")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "survival_by_class_sex.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    known = df.dropna(subset=["Age"])
    sns.histplot(
        data=known,
        x="Age",
        hue="Survived",
        bins=30,
        palette=palette,
        ax=ax,
        element="step",
        stat="density",
        common_norm=False,
    )
    ax.set_title("Age distribution by survival")
    ax.set_xlabel("Age (years)")
    fig.tight_layout()
    fig.savefig(FIGURES / "age_by_survival.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    rates = df.groupby("AgeGroup", observed=False)["Survived"].mean() * 100
    rates.plot(kind="bar", ax=ax, color="#2563eb", width=0.7)
    ax.set_ylabel("Survival rate (%)")
    ax.set_xlabel("Age group")
    ax.set_title("Survival rate by age group")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "survival_by_age_group.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    rates = df.groupby("FamilySize")["Survived"].mean() * 100
    rates.plot(kind="bar", ax=ax, color="#2563eb", width=0.7)
    ax.set_ylabel("Survival rate (%)")
    ax.set_xlabel("Family size (self + SibSp + Parch)")
    ax.set_title("Survival rate by family size")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "survival_by_family_size.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    numeric = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]].corr()
    sns.heatmap(numeric, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation among numeric features")
    fig.tight_layout()
    fig.savefig(FIGURES / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = df.groupby("FareQuartile", observed=False)["Survived"].mean() * 100
    rates.plot(kind="bar", ax=ax, color="#2563eb", width=0.7)
    ax.set_ylabel("Survival rate (%)")
    ax.set_xlabel("Fare quartile")
    ax.set_title("Survival rate by fare quartile")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "survival_by_fare.png", dpi=150)
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    df = load_data()
    print_overview(df)
    save_figures(df)
    print()
    print(f"Figures saved to {FIGURES}")


if __name__ == "__main__":
    main()
