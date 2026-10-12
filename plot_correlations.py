"""Графики для отчёта: scatter и heatmap по brainsize.txt."""
import matplotlib
matplotlib.use("TkAgg")   

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(path="brainsize.txt"):
    return pd.read_csv(path, sep="\t", na_values="NA")

def plot_scatter(df):
    """Scatter: MRI_Count vs каждый признак, отдельно для M и F."""
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]
    
    fig, axes = plt.subplots(len(features), 2, figsize=(10, 4 * len(features)))
    
    for i, feat in enumerate(features):
        # Мужчины - синий
        axes[i, 0].scatter(men[feat], men["MRI_Count"], color="steelblue")
        axes[i, 0].set_title(f"Мужчины: {feat} vs MRI_Count")
        axes[i, 0].set_xlabel(feat)
        axes[i, 0].set_ylabel("MRI_Count")
        
        # Женщины - красный
        axes[i, 1].scatter(women[feat], women["MRI_Count"], color="coral")
        axes[i, 1].set_title(f"Женщины: {feat} vs MRI_Count")
        axes[i, 1].set_xlabel(feat)
        axes[i, 1].set_ylabel("MRI_Count")
    
    plt.tight_layout()
    plt.savefig("scatter.png", dpi=100)
    plt.show()


def plot_heatmap(df):
    """Heatmap корреляций между всеми числовыми признаками."""
    # Только нужные колонки
    cols = ["FSIQ", "VIQ", "PIQ", "Weight", "Height", "MRI_Count"]
    
    men = df[df["Gender"] == "Male"][cols]
    women = df[df["Gender"] == "Female"][cols]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(men.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title("Корреляции: мужчины")
    
    sns.heatmap(women.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title("Корреляции: женщины")
    
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    df = load_data("brainsize.txt")
    plot_scatter(df)
    plot_heatmap(df)