from pathlib import Path
import os

import pandas as pd
from scipy.stats import rankdata, spearmanr

RESULTS_DIR = Path(os.environ.get("CSI_RESULTS_DIR", "results"))

DISTANCE_MAP = {
    "Manhattan (L1,1)": "Manhattan",
    "Manhattan (L1,2)": "Manhattan",
}

FAMILY_MAP = {
    "Manhattan": "Euclidean",
    "Euclidean": "Euclidean",
    "Normalized Euclidean": "Normalised",
    "Geodesic on S^(2D-1)": "Normalised",
    "Global phase, Chordal": "Global phase",
    "Global phase, Bures-Wasserstein": "Global phase",
    "Norm+global, Chordal": "Normalised+Global phase",
    "Norm+global, Geodesic on Grass.": "Normalised+Global phase",
    "Norm+global, Bures-Wasserstein": "Normalised+Global phase",
}

FAMILY_ORDER = [
    "Euclidean",
    "Normalised",
    "Global phase",
    "Normalised+Global phase",
]

METRIC_ORDER = [
    "Manhattan",
    "Euclidean",
    "Normalized Euclidean",
    "Geodesic on S^(2D-1)",
    "Global phase, Chordal",
    "Global phase, Bures-Wasserstein",
    "Norm+global, Chordal",
    "Norm+global, Bures-Wasserstein",
    "Norm+global, Geodesic on Grass.",
]

files = [
    RESULTS_DIR / "DICHASUS_Absolute_complex_Metrics_updated.xlsx",
    RESULTS_DIR / "CAEZ_Absolute_complex_Metrics.xlsx",
]

df_all = pd.concat(
    [pd.read_excel(file_path) for file_path in files],
    ignore_index=True,
)

results = []

for (dataset, setting), group in df_all.groupby(
    ["Dataset Name", "Setting"],
    sort=True,
):
    best_complex = round(
        group.loc[
            group["Data Representation"] == "Complex",
            "NPR",
        ].max(),
        3,
    )

    best_absolute = round(
        group.loc[
            group["Data Representation"] == "Absolute",
            "NPR",
        ].max(),
        3,
    )

    if best_complex > best_absolute:
        rank_complex, rank_absolute = 1, 2
    elif best_absolute > best_complex:
        rank_complex, rank_absolute = 2, 1
    else:
        rank_complex, rank_absolute = 1, 1

    results.append(
        {
            "Dataset": dataset,
            "Setting": setting,
            "Best NPR Complex": best_complex,
            "Rank Complex": rank_complex,
            "Best NPR Absolute": best_absolute,
            "Rank Absolute": rank_absolute,
        }
    )

    print(
        f"  {dataset} | {setting}: "
        f"Complex={best_complex:.3f} (rank {rank_complex})  "
        f"Absolute={best_absolute:.3f} (rank {rank_absolute})"
    )

results_df = pd.DataFrame(results)

avg_rank_complex = results_df["Rank Complex"].mean()
avg_rank_absolute = results_df["Rank Absolute"].mean()

rho, pval = spearmanr(
    results_df["Rank Complex"],
    results_df["Rank Absolute"],
)

print("\nComplex vs. magnitude: best NPR")
print(
    results_df[
        ["Dataset", "Setting", "Best NPR Complex", "Best NPR Absolute"]
    ].to_string(index=False)
)

print("\nComplex vs. magnitude: ranks")

rank_table = results_df[
    ["Dataset", "Setting", "Rank Complex", "Rank Absolute"]
].copy()

avg_row = pd.DataFrame(
    [
        {
            "Dataset": "Average Rank",
            "Setting": "",
            "Rank Complex": round(avg_rank_complex, 3),
            "Rank Absolute": round(avg_rank_absolute, 3),
        }
    ]
)

rank_table = pd.concat(
    [
        rank_table,
        pd.DataFrame([{column: "" for column in rank_table.columns}]),
        avg_row,
    ],
    ignore_index=True,
)

print(rank_table.to_string(index=False))

print("\nRank agreement")
print(f"Spearman rho = {rho:.4f}")
print(f"p-value      = {pval:.4f}")

files = {
    "DICHASUS": RESULTS_DIR / "DICHASUS_AVD_RASD_Metrics_updated.xlsx",
    "CAEZ-5G": RESULTS_DIR / "CAEZ_AVD_RASD_Metrics.xlsx",
}

results = []

for dataset, file_path in files.items():
    df = pd.read_excel(file_path)

    df = df[
        df["Data Representation"] != "Absolute"
    ].reset_index(drop=True)

    for setting, group in df.groupby("Setting"):
        best_avd = round(
            group.loc[
                group["Distance Avg Type"] == "AVD",
                "NPR",
            ].max(),
            3,
        )

        best_rasd = round(
            group.loc[
                group["Distance Avg Type"] == "sqAVD",
                "NPR",
            ].max(),
            3,
        )

        if best_avd > best_rasd:
            rank_avd, rank_rasd = 1, 2
        elif best_rasd > best_avd:
            rank_avd, rank_rasd = 2, 1
        else:
            rank_avd, rank_rasd = 1, 1

        results.append(
            {
                "Dataset": dataset,
                "Setting": setting,
                "Best NPR AVD": best_avd,
                "Rank AVD": rank_avd,
                "Best NPR RASD": best_rasd,
                "Rank RASD": rank_rasd,
            }
        )

results_df = pd.DataFrame(results)

avg_rank_avd = results_df["Rank AVD"].mean()
avg_rank_rasd = results_df["Rank RASD"].mean()

d = results_df["Rank AVD"] - results_df["Rank RASD"]
n = len(results_df)
rho = 1 - (6 * (d ** 2).sum()) / (n * (n ** 2 - 1))

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

print("\nAVD vs. RASD: best NPR")
print(results_df.to_string(index=False))

print("\nAVD vs. RASD: average rank")
print(
    pd.DataFrame(
        {
            "Method": ["AVD", "RASD"],
            "Average Rank": [
                round(avg_rank_avd, 3),
                round(avg_rank_rasd, 3),
            ],
        }
    ).to_string(index=False)
)

print("\nRank agreement")
print(f"Spearman rho = {rho:.4f}")

files = {
    "DICHASUS": (
        RESULTS_DIR / "DICHASUS_all_Metrics_updated.xlsx",
        "abs_com",
    ),
    "CAEZ-5G": (
        RESULTS_DIR / "CAEZ_all_Metrics.xlsx",
        "Metrics",
    ),
}

rows = []

for dataset, (file_path, sheet_name) in files.items():
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df = df[
        df["Data Representation"] != "Absolute"
    ].reset_index(drop=True)

    for setting, group in df.groupby("Setting"):
        feature_best = (
            group.groupby(
                "CSI Feature Vector",
                as_index=False,
            )["NPR"]
            .max()
            .rename(columns={"NPR": "Best_NPR"})
        )

        feature_best["Best_NPR"] = feature_best["Best_NPR"].round(3)

        feature_best["Rank"] = rankdata(
            -feature_best["Best_NPR"],
            method="min",
        ).astype(int)

        for _, row in feature_best.iterrows():
            rows.append(
                {
                    "Dataset": dataset,
                    "Setting": setting,
                    "CSI Feature": row["CSI Feature Vector"],
                    "Best_NPR": row["Best_NPR"],
                    "Rank": row["Rank"],
                }
            )

results_df = pd.DataFrame(rows)

summary = results_df.pivot_table(
    index=["Dataset", "Setting"],
    columns="CSI Feature",
    values=["Best_NPR", "Rank"],
)

summary.columns = [
    f"{group}_{feature}"
    for group, feature in summary.columns
]
summary = summary.reset_index()

avg_rank_df = (
    results_df.groupby(
        "CSI Feature",
        as_index=False,
    )["Rank"]
    .mean()
    .rename(columns={"Rank": "Average Rank"})
    .sort_values("Average Rank")
)

dichasus_avg = (
    results_df[results_df["Dataset"] == "DICHASUS"]
    .groupby("CSI Feature", as_index=False)["Rank"]
    .mean()
    .rename(columns={"Rank": "AvgRank_DICHASUS"})
)

caez_avg = (
    results_df[results_df["Dataset"] == "CAEZ-5G"]
    .groupby("CSI Feature", as_index=False)["Rank"]
    .mean()
    .rename(columns={"Rank": "AvgRank_CAEZ"})
)

corr_df = pd.merge(
    dichasus_avg,
    caez_avg,
    on="CSI Feature",
    how="inner",
)

rho, pval = spearmanr(
    corr_df["AvgRank_DICHASUS"],
    corr_df["AvgRank_CAEZ"],
)

npr_cols = [
    "Dataset",
    "Setting",
] + [
    column
    for column in summary.columns
    if column.startswith("Best_NPR_")
]

rank_cols = [
    "Dataset",
    "Setting",
] + [
    column
    for column in summary.columns
    if column.startswith("Rank_")
]

print("\nCSI features: best NPR")
print(summary[npr_cols].to_string(index=False))

rank_table = summary[rank_cols].copy()
rank_table.columns = [
    "Dataset",
    "Setting",
] + [
    column.replace("Rank_", "")
    for column in rank_cols[2:]
]

avg_row = {
    "Dataset": "Average Rank",
    "Setting": "",
}

for feature_column in rank_cols[2:]:
    feature = feature_column.replace("Rank_", "")
    avg_row[feature] = avg_rank_df.loc[
        avg_rank_df["CSI Feature"] == feature,
        "Average Rank",
    ].values[0]

empty_row = {column: "" for column in rank_table.columns}
rank_table = pd.concat(
    [
        rank_table,
        pd.DataFrame([empty_row]),
        pd.DataFrame([avg_row]),
    ],
    ignore_index=True,
)

print("\nCSI features: ranks")
print(rank_table.to_string(index=False))

print("\nCSI feature rank agreement: DICHASUS vs. CAEZ-5G")
print(f"Spearman rho = {rho:.4f}")
print(f"p-value      = {pval:.6f}")

rows = []

for dataset, (file_path, sheet_name) in files.items():
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df = df[
        df["Data Representation"] != "Absolute"
    ].copy()

    df = df[
        df["Distance Avg Type"] != "-"
    ].copy()

    df["Distance Metric"] = df["Distance Metric"].replace(DISTANCE_MAP)

    df = df[
        df["Distance Metric"].isin(METRIC_ORDER)
    ].copy()

    for setting, group in df.groupby("Setting"):
        best_metric = (
            group.groupby(
                "Distance Metric",
                as_index=False,
            )["NPR"]
            .max()
        )

        best_metric["NPR"] = best_metric["NPR"].round(3)

        best_metric["Rank"] = rankdata(
            -best_metric["NPR"],
            method="min",
        )

        row = {
            "Dataset": dataset,
            "Setting": setting,
        }

        for _, metric_row in best_metric.iterrows():
            metric = metric_row["Distance Metric"]
            row[f"Best_NPR_{metric}"] = metric_row["NPR"]
            row[f"Rank_{metric}"] = metric_row["Rank"]

        rows.append(row)

results_df = pd.DataFrame(rows)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

npr_cols = [
    "Dataset",
    "Setting",
] + [
    column
    for column in results_df.columns
    if column.startswith("Best_NPR_")
]

rank_cols_all = [
    "Dataset",
    "Setting",
] + [
    column
    for column in results_df.columns
    if column.startswith("Rank_")
]

print("\nDistance metrics: best NPR")

npr_table = (
    results_df[npr_cols]
    .set_index(["Dataset", "Setting"])
    .T
)

npr_table.index = npr_table.index.str.replace(
    "Best_NPR_",
    "",
    regex=False,
)
npr_table = npr_table.reindex(METRIC_ORDER)
npr_table.index.name = "Best NPR"
print(npr_table.to_string())

print("\nDistance metrics: ranks")

rank_table = (
    results_df[rank_cols_all]
    .set_index(["Dataset", "Setting"])
    .T
)

rank_table.index = rank_table.index.str.replace(
    "Rank_",
    "",
    regex=False,
)
rank_table["Average Rank"] = rank_table.mean(axis=1)
rank_table = rank_table.reindex(METRIC_ORDER)
rank_table.index.name = "Distance Metric"
print(rank_table.to_string())

rank_cols = [
    column
    for column in results_df.columns
    if column.startswith("Rank_")
]

dichasus_avg = results_df[
    results_df["Dataset"] == "DICHASUS"
][rank_cols].mean()

caez_avg = results_df[
    results_df["Dataset"] == "CAEZ-5G"
][rank_cols].mean()

rho, pval = spearmanr(
    dichasus_avg.values,
    caez_avg.values,
)

print("\nDistance-metric rank agreement: DICHASUS vs. CAEZ-5G")
print(f"Spearman rho = {rho:.4f}")
print(f"p-value      = {pval:.6f}")

rows = []

for dataset, (file_path, sheet_name) in files.items():
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df = df[
        df["Data Representation"] != "Absolute"
    ].copy()

    df = df[
        df["Distance Avg Type"] != "-"
    ].copy()

    df["Distance Metric"] = df["Distance Metric"].replace(DISTANCE_MAP)

    df = df[
        df["Distance Metric"].isin(FAMILY_MAP)
    ].copy()

    df["Distance Family"] = df["Distance Metric"].map(FAMILY_MAP)

    for setting, group in df.groupby("Setting"):
        best_family = (
            group.groupby(
                "Distance Family",
                as_index=False,
            )["NPR"]
            .max()
        )

        best_family["NPR"] = best_family["NPR"].round(3)

        best_family["Rank"] = rankdata(
            -best_family["NPR"],
            method="min",
        )

        row = {
            "Dataset": dataset,
            "Setting": setting,
        }

        for _, family_row in best_family.iterrows():
            family = family_row["Distance Family"]
            row[f"Best_NPR_{family}"] = family_row["NPR"]
            row[f"Rank_{family}"] = family_row["Rank"]

        rows.append(row)

results_df = pd.DataFrame(rows)

npr_cols = [
    "Dataset",
    "Setting",
] + [
    f"Best_NPR_{family}"
    for family in FAMILY_ORDER
]

print("\nDistance families: best NPR")

npr_table = (
    results_df[npr_cols]
    .set_index(["Dataset", "Setting"])
    .T
)

npr_table.index = npr_table.index.str.replace(
    "Best_NPR_",
    "",
    regex=False,
)
npr_table.index.name = "Best NPR"
print(npr_table.to_string())

rank_cols = [
    "Dataset",
    "Setting",
] + [
    f"Rank_{family}"
    for family in FAMILY_ORDER
]

rank_table = (
    results_df[rank_cols]
    .set_index(["Dataset", "Setting"])
    .T
)

rank_table.index = rank_table.index.str.replace(
    "Rank_",
    "",
    regex=False,
)
rank_table["Average Rank"] = rank_table.mean(axis=1)
rank_table = rank_table.reindex(FAMILY_ORDER)

print("\nDistance families: ranks")
print(rank_table.to_string())

family_rank_cols = [
    column
    for column in results_df.columns
    if column.startswith("Rank_")
]

dichasus_avg = results_df[
    results_df["Dataset"] == "DICHASUS"
][family_rank_cols].mean()

caez_avg = results_df[
    results_df["Dataset"] == "CAEZ-5G"
][family_rank_cols].mean()

rho, pval = spearmanr(
    dichasus_avg.values,
    caez_avg.values,
)

print("\nDistance-family rank agreement: DICHASUS vs. CAEZ-5G")
print(f"Spearman rho = {rho:.4f}")
print(f"p-value      = {pval:.6f}")
