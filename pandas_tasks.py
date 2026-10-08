"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    # 1
    df = pd.read_csv(data.csv_path)

    # 2
    missing_by_column = {}
    for k, v in df.isnull().sum().items():
        missing_by_column[str(k)] = int(v)

    # 3
    adults_over_30_count = int((df["Age"] > 30).sum())

    #4
    mean_age_by_pclass = {}
    grouped = df.groupby("Pclass")["Age"].mean()
    for k, v in grouped.items():
        mean_age_by_pclass[int(k)] = float(v)

    #5
    survival_rate_by_pclass = {}
    for k, v in df.groupby("Pclass")["Survived"].mean().items():
        survival_rate_by_pclass[int(k)] = float(v)

    #6
    highest_fares = []
    for x in df["Fare"].nlargest(5):
        highest_fares.append(float(x))

    return TitanicSummary(
        row_count=len(df),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )