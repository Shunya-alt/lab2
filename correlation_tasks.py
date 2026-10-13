"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    # Загружаем таблицу 
    df = pd.read_csv(data.csv_path, sep="\t", na_values="NA")
    
    # Список признаков для анализа 
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    
    # Разделяем по полу
    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]
    
    # Корреляции для мужчин
    men_mri_correlation = {}
    for f in features:
        corr = men["MRI_Count"].corr(men[f])   
        men_mri_correlation[f] = float(corr)
    
    # Корреляции для женщин
    women_mri_correlation = {}
    for f in features:
        corr = women["MRI_Count"].corr(women[f])
        women_mri_correlation[f] = float(corr)
    
    # Ищем признак с максимальным модулем корреляции среди всех
    strongest_mri_feature = ""
    max_abs_corr = -1.0
    
    for group_corrs in (men_mri_correlation, women_mri_correlation):
        for feature, corr in group_corrs.items():
            if abs(corr) > max_abs_corr:
                max_abs_corr = abs(corr)
                strongest_mri_feature = feature
    
    # Результат
    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )
