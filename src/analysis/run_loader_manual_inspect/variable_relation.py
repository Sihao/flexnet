#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
some simple descriptive linear plots that assumes the data is already parsed and cleaned
one single df from multiple runs
"""


import numpy as np
import pandas, json
import seaborn as sns
from pathlib import Path
from pandas import DataFrame
import matplotlib.pyplot as plt
from src.analysis.run_loader_manual_inspect.imputations import (
    make_imputations_for_single_col,
)
from src.analysis.run_loader_manual_inspect.random_forest import (
    random_forest_select_features,
)
import scienceplots

FIGURE_SIZE = (10, 9)
FONTSIZE_TITLE = 32
FONTSIZE_LABEL = 27
FONTSIZE_TICK = 24
PAD_TITLE = 22
DPI_FIGURE = 300

plt.style.use(["science", "ieee"])
sns.set(style="white")


class DataframeParser:
    def __init__(self, csv_path: Path):
        self.dataframe = pandas.read_csv(csv_path)
        Path("__plots__").mkdir(parents=True, exist_ok=True)

    # ----[ plots ]----
    def plot_linear(self, prediction_col: str):
        """
        For each column in the dataframe that is not the prediction column,
        create a scatter plot with a linear regression line.
        Save each plot in the __plots__ folder.
        """
        dataframes_imputed: list[DataFrame] = make_imputations_for_single_col(
            self.dataframe, prediction_col
        )
        random_frame = dataframes_imputed[0]

        # Iterate over each column in the dataframe
        for col in random_frame.columns:
            if col != prediction_col:
                plt.figure(figsize=(8, 6))
                sns.regplot(
                    x=col,
                    y=prediction_col,
                    data=random_frame,
                    scatter_kws={"alpha": 0.5},
                )
                plt.title(f"Linear Regression: {col} vs {prediction_col}")
                plt.xlabel(col)
                plt.ylabel(prediction_col)
                plt.savefig(f"__plots__/{col}_vs_{prediction_col}.png")
                plt.close()

    def plot_heatmap(self):
        """
        Create a heatmap plot to visualize the density of each variable in the dataframe.
        Each column will have its own color coding to reflect its range of values.
        Boolean variables are treated as numeric (0 and 1).
        """
        dummy_frame = self.dataframe.copy()

        # Convert boolean columns to numeric (0 and 1)
        for col in dummy_frame.columns:
            if dummy_frame[col].dtype == bool:
                dummy_frame[col] = dummy_frame[col].astype(int)

        # Normalize the dataframe for heatmap
        normalized_df = (dummy_frame - dummy_frame.min()) / (
            dummy_frame.max() - dummy_frame.min()
        )

        plt.figure(figsize=(15, 10))
        sns.heatmap(normalized_df, cmap="viridis", cbar=True)
        plt.title("Heatmap of Variable Densities")
        plt.ylabel("Data Points")
        plt.xlabel("Variables")
        plt.tight_layout()
        plt.show()

    def plot_correlation(self):
        """
        Plot a heatmap using seaborn to visualize the density of each variable in the dataframe.
        """
        dummy_frame = self.dataframe.copy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(dummy_frame.corr(), annot=False, fmt=".2f", cmap="coolwarm")
        plt.tight_layout()
        plt.title("Heatmap of Variable Correlations")
        plt.show()

    # ----[ feature importances ]----
    def get_importances_ranking(
        self, colname: str, drop_cols: list = [], save_data_as=None
    ):
        """get the feature importances by other values"""
        dummy_frame = self.dataframe.copy()

        # [1] drop the columns that are not needed
        dummy_frame = dummy_frame.drop(
            columns=[
                col for col in dummy_frame.columns if any([i in col for i in drop_cols])
            ]
        )

        # [2] impute the data
        dataframes_imputed: list[DataFrame] = make_imputations_for_single_col(
            dummy_frame, colname
        )

        mean_importances, mean_r2_values = random_forest_select_features(
            dataframes_imputed, colname
        )

        # Sort the mean_importances dictionary by value in descending order
        sorted_importances = sorted(
            mean_importances.items(), key=lambda item: item[1], reverse=True
        )

        print(sorted_importances)
        print(f"mean r2: {mean_r2_values}")

        # Unpack the sorted items into lists
        feature_names, importance_scores = zip(*sorted_importances)

        # Plotting
        plt.figure(figsize=FIGURE_SIZE)
        plt.barh(feature_names, importance_scores, color="skyblue")
        plt.xlabel("Mean Importance Score Across DFs", fontsize=FONTSIZE_LABEL)
        plt.ylabel("Features", fontsize=FONTSIZE_LABEL)
        plt.title(f"R2: {mean_r2_values}", fontsize=FONTSIZE_TITLE)
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.tight_layout()
        plt.savefig(Path("__plots__", "_importances.png"), dpi=DPI_FIGURE)

        if save_data_as:
            save_content = sorted_importances.copy()
            save_content.append(("Mean R2 Score", mean_r2_values))
            save_content = {item[0]: item[1] for item in save_content}
            with open(save_data_as, "w") as writer:
                json.dump(save_content, writer, indent=4)

    def plot_group_relation(self, target_col: str):
        Path("__plots__").mkdir(parents=True, exist_ok=True)
        # find all categorical / non-float columns
        dataframes_imputed: list[DataFrame] = make_imputations_for_single_col(
            self.dataframe, target_col
        )
        dummy_frame = dataframes_imputed[0]

        group_columns = [
            col for col in dummy_frame.columns if dummy_frame[col].dtype != float
        ]
        group_columns = [col for col in group_columns if col != target_col]
        group_columns = [
            col for col in group_columns if len(dummy_frame[col].unique()) > 1
        ]

        for group_col in group_columns:
            plt.figure(figsize=FIGURE_SIZE)

            # Plot KDE for each unique category in group_col
            unique_groups = dummy_frame[group_col].unique()
            group_means = [
                (group, dummy_frame[dummy_frame[group_col] == group][target_col].mean())
                for group in unique_groups
            ]
            group_means = sorted(group_means, key=lambda x: x[1], reverse=True)

            for group, mean_value in group_means:
                subset = dummy_frame[dummy_frame[group_col] == group]
                mean_value = round(mean_value, 3)
                sns.kdeplot(
                    subset[target_col],
                    label=f"{group} {mean_value}",
                    fill=True,
                    bw_adjust=0.2,
                    warn_singular=False,
                )

            # Adding labels and title
            plt.title(
                f"KDE of {target_col} by {group_col.replace('_', ' ').title()}",
                fontsize=FONTSIZE_TITLE,
            )
            plt.xlabel(target_col, fontsize=FONTSIZE_LABEL)
            plt.ylabel("Density", fontsize=FONTSIZE_LABEL)
            plt.xticks(fontsize=FONTSIZE_TICK)
            plt.yticks(fontsize=FONTSIZE_TICK)

            # ax.axvline(mean_eigenvalue, color="r", linestyle="--", label="Mean")

            plt.legend(
                fontsize=FONTSIZE_TITLE,
                bbox_to_anchor=(1.5, 1),
            )

            try:
                plt.savefig(
                    Path("__plots__") / f"_{target_col}_{group_col}.png",
                    dpi=DPI_FIGURE,
                    bbox_inches="tight",
                )
            except OSError:
                plt.savefig(
                    Path("_plots__") / f"_{target_col[:10]}_{group_col[:10]}.png",
                    dpi=DPI_FIGURE,
                    bbox_inches="tight",
                )
            plt.close()

    # ----[helper]----
    def _inspect_missing_value_patterns(self, save_path: Path):
        frame = self.dataframe.copy()
        frame.columns = frame.columns.str.lower()

        # -------- figure bit --------
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            frame.isna().transpose(), cmap="YlGnBu", cbar_kws={"label": "Missing Data"}
        )

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()

        # -------- text bit --------
        missing_value_percentages = []
        for column in frame.columns:
            na_list = frame[column].isna().to_list()
            na_len = len([i for i in na_list if i])
            total_len = len(frame[column])
            missing_value_percentage = (na_len / total_len) * 100
            missing_value_percentages.append(missing_value_percentage)

        save_text = f"""
        among the total curated variables:
        max_missing_percentage: {max(missing_value_percentages)}%
        mean_missing_percentage: {np.mean(missing_value_percentages)}%
        median_missing_percentage: {np.median(missing_value_percentages)}
        """

        with open(save_path.parent / "_missing_values.MD", "w") as writer:
            writer.write(save_text)


if __name__ == "__main__":
    pass
