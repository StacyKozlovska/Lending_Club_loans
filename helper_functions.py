"""
Helper functions necessary for the data analysis of Podcast Reviews.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import re
import plotly.express as px
from varname import nameof
from pprint import pprint
import textwrap
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score

sns.set_style("darkgrid")

xticklabels_d = {
    "horizontalalignment": "right",
    "fontweight": "light",
    "fontsize": "x-large",
}


def plot_sns_barplot(
        data,
        x: str,
        y: str,
        x_label: str,
        y_label: str,
        title: str,
        hue=None,
        xtick_rot: int = 65,
        max_len_xtick_labels: int = 25,
        xticklabels: dict = xticklabels_d,
        my_figsize: (int, int) = (10, 7),
):
    """
    Function to automate seaborn
    barplot plotting.
    """
    # Figure Size
    fig = plt.figure(figsize=my_figsize)

    # Bar Plot
    ax = sns.barplot(x=data[x], y=data[y], hue=hue)
    f = lambda x: textwrap.fill(x.get_text(), max_len_xtick_labels)
    ax.set_xticklabels(map(f, ax.get_xticklabels()), rotation=xtick_rot, **xticklabels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.plot()


def plot_sns_countplot(
        data,
        x: str,
        plot_order,
        x_label: str,
        y_label: str,
        title: str,
        xtick_rot: int = 65,
        max_len_xtick_labels: int = 25,
        xticklabels: dict = xticklabels_d,
        my_figsize: (int, int) = (10, 7),
        percentages = False,
        palette = 'Set1',
):
    """
    Function to automate seaborn
    countplot plotting.
    """
    plt.figure(figsize=my_figsize)
    ax = sns.countplot(data=data, x=x, order=plot_order, hue=None, palette=palette)
    f = lambda x: textwrap.fill(x.get_text(), max_len_xtick_labels)
    ax.set_xticklabels(map(f, ax.get_xticklabels()), rotation=xtick_rot, **xticklabels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if percentages is True:
        total_height = len(data)
        for p in ax.patches:
            height = p.get_height()
            percentage = 100 * height / total_height
            ax.annotate(f'{percentage:.2f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom')

    ax.plot()
    return ax


def plot_sns_jointplot(
        data, x: str, y: str, title: str, xlim=(-20, 850), ylim=(3, 5.1), my_figsize=(8, 5)
):
    """
    Function to automate seaborn
    jointplot plotting.
    """
    g = sns.JointGrid(data, x=x, y=y)
    g.plot_joint(sns.scatterplot, s=100, alpha=0.5)
    g.ax_marg_x.set_xlim(*xlim)
    g.ax_marg_y.set_ylim(*ylim)
    g.plot_marginals(sns.histplot, kde=True)
    g.fig.set_size_inches(my_figsize)
    g.fig.suptitle(title)

    g.fig.show()


def visualize_violinplot(df, x: str, y: str, hue: str = None,
                         xtick_rot: int = 0):
    """
    Function to plot a violin plot with seaborn.
    """
    # Create the violin plot
    sns.violinplot(x=x, y=y, data=df, hue=hue)

    # Set the plot title and axes labels
    plt.title(f"{y.capitalize()} distribution by {x.capitalize()}")
    plt.xlabel(x.capitalize())
    plt.ylabel(y.capitalize())

    plt.xticks(rotation=xtick_rot)

    if hue is not None:
        plt.legend(title=hue, loc="upper right", bbox_to_anchor=(1.2, 1))
        plt.title(
            f"{y.capitalize()} distribution by {x.capitalize()} and {hue.capitalize()}"
        )
    # Show the plot
    plt.show()


def get_strong_corr_features(my_df, method: str, threshold: float):
    """
    Function that calculated correlation scores between features
    and select the top correlated ones based on a given
    threshold.
    The output is a Pandas Dataframe without duplicate pairs.
    """
    # create a dataframe with correlations
    corr_data_pa = my_df.corr(numeric_only=True, method=method)

    # Retain upper triangular values of correlation matrix and
    # make Lower triangular values Null
    upper_corr_data = corr_data_pa.where(np.triu(np.ones(corr_data_pa.shape), k=1).astype(bool))

    # Convert to 1-D series and drop Null values
    unique_corr_pairs = upper_corr_data.unstack().dropna()

    # Sort correlation pairs
    sorted_corr_data_pa = unique_corr_pairs.drop_duplicates()
    sorted_corr_data_pa.sort_index(inplace=True)

    strong_corr_pa = sorted_corr_data_pa[(sorted_corr_data_pa < -threshold) | (sorted_corr_data_pa > threshold)]
    strong_corr_pa.sort_values(ascending=False, inplace=True)

    heatmap_data_pa = pd.DataFrame(strong_corr_pa.reset_index())
    heatmap_data_pa.rename(
        columns={"level_0": "var_1", "level_1": "var_2", 0: "corr"}, inplace=True
    )

    return heatmap_data_pa


def plot_count_percent_barplots_by_category(
    my_df, cat_col: str, my_col: str, my_title: str, my_order=None
):
    """
    Function to visualize two plots side by side.
    The first plot shows the total count for each category.
    The second plot shows the shares for each category.
    """

    fig = plt.figure(figsize=(16, 8))
    grid = GridSpec(1, 2)

    ax1 = fig.add_subplot(grid[0, 0])

    # Set the color palette for the countplot
    palette = {0: "lightblue", 1: "steelblue"}

    sns.countplot(
        data=my_df.dropna(),
        x=cat_col,
        order=my_order,
        hue=my_col,
        palette=palette,
        ax=ax1,
        width=0.8,
    )
    ax1.set(xlabel=cat_col.capitalize(), ylabel="Count")
    ax2 = fig.add_subplot(grid[0, 1])

    share_home = my_df.groupby(cat_col)[my_col].mean()

    if my_order is not None:
        share_home = share_home.loc[my_order].sort_values(ascending=False)

    share_away = 1 - share_home

    # Plot the stacked bars
    ax2.bar(
        share_home.index,
        share_home,
        color="steelblue",
        label="Home",
    )
    ax2.bar(
        share_away.index,
        share_away,
        bottom=share_home,
        color="lightblue",
        label="Away",
    )

    ylabel_name = my_col.replace("_", " ")

    ax2.set(xlabel=cat_col.capitalize(), ylabel=f"Share of {ylabel_name.capitalize()}")
    ax2.set_ylim(0, 1)
    ax2.legend()

    # Rotate the x-axis labels by 45 degrees
    ax1.tick_params(axis="x", labelrotation=45)
    ax2.tick_params(axis="x", labelrotation=45)

    fig.suptitle(my_title, fontsize=16)

    plt.subplots_adjust(wspace=0.3)

    plt.show()


def plot_radial_graph_two_groups(features_list,
                                 category_1_vals,
                                 category_2_vals,
                                 cat_1_label: str,
                                 cat_2_label: str,
                                 plot_title: str):
    """
    Function to visualize the radial graph to
    compare multiple features between two groups of data.
    """
    N = len(features_list)

    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    plt.title(plot_title)

    ax.fill(angles, category_1_vals, 'b', alpha=0.1, label=cat_1_label)
    ax.set_xticks(angles)
    ax.set_xticklabels(features_list)

    ax.fill(angles, category_2_vals, 'r', alpha=0.1, label=cat_2_label)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()


def visualize_double_countplots(my_df,
                                feature: str,
                                hue_feature: str,
                                plot_1_title: str,
                                plot_2_title: str,
                                percentages = True,
                                x_axis_label_order = None,
                                left_plot_colors = None,
                                hue_feature_order=None,
                                hue_colors=None):
    """
    Function to create two countplots for a
    selected feature:
    left - usual countplot,
    right - countplot with additional feature (hue).
    """

    f, ax = plt.subplots(1, 2, figsize=(18, 8))

    sns.countplot(x=my_df[feature], order=x_axis_label_order, ax=ax[0], palette=left_plot_colors)
    ax[0].set_title(plot_1_title)

    if percentages:
        total = len(my_df)
        for p in ax[0].patches:
            height = p.get_height()
            ax[0].annotate(f'{100 * height / total:.2f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=12, color='black', weight='bold')

    sns.countplot(x=feature, hue=hue_feature, data=my_df, order=x_axis_label_order, ax=ax[1],
                  hue_order=hue_feature_order, palette=hue_colors)

    ax[1].set_title(plot_2_title)
    plt.show()


def visualize_double_barplot(df, my_title,
                             cat_col: str,
                             plot1_y_label: str,
                             plot1_x_col: str,
                             plot2_group1_col: str,
                             plot2_group2_col: str,
                             group1_label: str,
                             group2_label: str,
                             ylim_left=None):
    """
    Function to visualize two plots:
    left plot - bar plot,
    right plot - stacked 0-1 plot showing
    shares for each category.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.bar(df[cat_col], df[plot1_x_col], color='darkturquoise')

    ax1.set_xlabel(cat_col)
    ax1.set_ylabel(plot1_y_label)

    if ylim_left:
        ax1.set_ylim(ylim_left)

    # Right plot - share bar
    group1_share = df[plot2_group1_col] / (df[plot2_group1_col] + df[plot2_group2_col])
    group2_share = 1 - group1_share

    ax2.bar(df[cat_col], group1_share, color='steelblue', label=group1_label)
    ax2.bar(df[cat_col], group2_share, bottom=group1_share, color='lightblue', label=group2_label)

    ax2.set_xlabel(cat_col)
    ax2.set_ylabel(plot1_y_label)
    ax2.set_ylim(0, 1)

    ax1.tick_params(axis="x", labelrotation=45)
    ax2.tick_params(axis="x", labelrotation=45)

    ax2.legend()

    fig.suptitle(my_title, fontsize=16)

    plt.subplots_adjust(wspace=0.3)


def test_two_sample_ttest_assumptions(group_1, group_2, feature):
    """
    Function to check whether the assumptions for a two-sample
    t-test are met.
    """
    group0_data = group_1[feature]
    group1_data = group_2[feature]

    # Shapiro-Wilk test for normality
    shapiro_stat_group0, shapiro_pvalue_group0 = stats.shapiro(group0_data)
    shapiro_stat_group1, shapiro_pvalue_group1 = stats.shapiro(group1_data)

    # Levene's test for equal variances
    levene_stat, levene_pvalue = stats.levene(group0_data, group1_data)

    print(f"Results of the tests for assumptions for 2-sample t-tests for groups in {feature}:")
    print("-" * 70)

    print(
        "Shapiro-Wilk Test - Group 1: Statistic =",
        shapiro_stat_group0,
        "P-value =",
        shapiro_pvalue_group0,
    )
    print(
        "Shapiro-Wilk Test - Group 2: Statistic =",
        shapiro_stat_group1,
        "P-value =",
        shapiro_pvalue_group1,
    )
    print("Levene's Test - P-value =", levene_pvalue)

    if (
            shapiro_pvalue_group0 > 0.05
            and shapiro_pvalue_group1 > 0.05
            and levene_pvalue > 0.05
    ):
        print(
            "Both groups pass the normality and equal variance assumptions for the t-test."
        )
    else:
        print("One or both groups may not meet the assumptions for the t-test.")

    print("-" * 50)

    # Check normality using scipy.stats.normaltest
    normaltest_stat_group0, normaltest_pvalue_group0 = stats.normaltest(group0_data)
    normaltest_stat_group1, normaltest_pvalue_group1 = stats.normaltest(group1_data)

    print(
        "D'Agostino and Pearson's omnibus Test for Normality - Group 1: Statistic =",
        normaltest_stat_group0,
        "P-value =",
        normaltest_pvalue_group0,
    )

    print(
        "D'Agostino and Pearson's omnibus Test for Normality - Group 2: Statistic =",
        normaltest_stat_group1,
        "P-value =",
        normaltest_pvalue_group1,
    )

    alpha = 0.05
    if normaltest_pvalue_group0 > alpha and normaltest_pvalue_group1 > alpha:
        print("Both groups are approximately normally distributed.")
    else:
        print("One or both groups may not be normally distributed.")

    print("-" * 50)

    # Create Q-Q plots for both groups
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    stats.probplot(group0_data, plot=plt)
    plt.title(f"Q-Q Plot for {feature} - Group 1")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")

    plt.subplot(2, 1, 2)
    stats.probplot(group1_data, plot=plt)
    plt.title(f"Q-Q Plot {feature} - Group 2")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")

    plt.tight_layout()
    plt.show()


def chi_square_test(data, feature1, feature2, alpha=0.05):
    """
    Perform a Chi-Square Test of Independence for two categorical features.

    Parameters:
    - data: DataFrame containing the dataset.
    - feature1: Name of the first categorical feature.
    - feature2: Name of the second categorical feature.
    - alpha: Significance level (default is 0.05).

    Returns:
    - result: A dictionary containing the test results.
    """
    contingency_table = pd.crosstab(data[feature1], data[feature2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    result = {
        "Chi-Square Statistic": chi2,
        "P-value": p,
        "Degrees of Freedom": dof,
        "Expected Frequencies Table": expected,
        "Significance Level (alpha)": alpha
    }

    if p < alpha:
        result["Hypothesis Test Result"] = "Reject the null hypothesis: There is a significant association between the two features."
    else:
        result["Hypothesis Test Result"] = "Fail to reject the null hypothesis: There is no significant association between the two features."

    return result


def make_mi_scores(X, y, discrete_features):
    """
    Function to calculate mutual information scores.
    """
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    """
    Function to plot mutual information scores.
    """
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def avg_mi(df, feature_cols, target_cols):
    """
    Function to calculate average mutual info scores
    when there are multiple target columns.
    """
    mi_scores = []

    for col in feature_cols:
        mi_list = []
        for target_col in target_cols:
            mi = mutual_info_score(df[col], df[target_col])
            mi_list.append(mi)
        mi_avg = sum(mi_list) / len(mi_list)
        mi_scores.append(mi_avg)

    mi_avg_scores = pd.Series(mi_scores, index=feature_cols)
    mi_avg_scores.sort_values(ascending=False, inplace=True)

    return mi_avg_scores


def select_features(scores, k=10):
    top_k = list(scores.index[:k])
    print(f"Selected {len(top_k)} features: {top_k}")
    return top_k


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = range(len(scores))
    ticks = list(scores.index)

    plt.figure(figsize=(12, 8))
    plt.barh(width, scores)
    plt.yticks(width, ticks, ha='right')
    plt.title("Average Mutual Information Scores")

    threshold = scores.mean()
    plt.axvline(x=threshold, color='r')

    plt.xlabel("Mutual Information Score")
    plt.tight_layout()
    plt.show()


def remove_outliers(df):
    """
    Function to remove outliers
    using the IQR approach.
    """
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df_upd = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        return df_upd


def clean_string(series):
    """
    Function to clean string values
    of a Pandas Series of all
    non-alphanumeric characters.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a Pandas Series.")

    def clean_single_string(text):
        cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', str(text))
        return cleaned_text

    cleaned_series = series.apply(clean_single_string)

    return cleaned_series


def clean_columns(df):
    """
    Function to clean column names of a Pandas Dataframe.
    """
    df.columns = [c.strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]
    return df.columns


def clean_series(series):
    """
    Function to clean a Pandas Series
    of empty spaces, hyphens, and
    convert everything to lowercase.
    * Other version of the function:
    clean_columns(df)
    """
    series = series.str.replace(" ", "_")
    series = series.str.replace("-", "_")
    series = series.str.strip()
    series = series.str.lower()

    return series