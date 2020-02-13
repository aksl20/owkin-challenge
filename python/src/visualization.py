import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_diag_matrix_corr(df, figsize=(14, 12), show_labels=True):
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    f, ax = plt.subplots(figsize=figsize)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, xticklabels=show_labels,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, yticklabels=show_labels)
