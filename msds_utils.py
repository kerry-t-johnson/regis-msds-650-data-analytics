import math
import matplotlib
import matplotlib.cm
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

def summary_graphs(df: pd.DataFrame, measurements, boxplot=True, hist=True, probplot=True, title_prefix=''):
    ''' Creates a figure with multiple plots for each requested measurement.

        Parameters
        ----------
        df: pd.DataFrame
            The source dataframe
        measurements: str | list
            One or more columns (measurements) to graph
        title_prefix: str (optional)
            A prefix to apply to each plot title

    '''
    measurements = measurements if type(measurements) is list else [measurements]
    num_measurements = len(measurements)
    figures_per_measurement = (1 if boxplot else 0) + (1 if hist else 0) + (1 if probplot else 0)
    figure_cols = figures_per_measurement
    figure_rows = int(math.ceil((num_measurements * figures_per_measurement) / figure_cols))

    # A personal figure size preference, based on the number or rows and cols:
    matplotlib.pyplot.figure(figsize=(4.0 * figure_cols, 4.0 * figure_rows))

    for i in range(num_measurements):
        column_name = measurements[i]
        column_values = df[column_name]

        if boxplot:
            matplotlib.pyplot.subplot(figure_rows,
                                      figure_cols,
                                      (i * figures_per_measurement) + 1,
                                      title=f'{title_prefix}{column_name} Boxplot')
            matplotlib.pyplot.boxplot(column_values)

        if hist:
            ax = matplotlib.pyplot.subplot(figure_rows,
                                           figure_cols,
                                           (i * figures_per_measurement) + 2,
                                           title=f'{title_prefix}{column_name} Histogram')
            sns.histplot(column_values, kde=True, ax=ax)
            matplotlib.pyplot.axvline(column_values.mean(), color='red')
            matplotlib.pyplot.axvline(column_values.median(), color='black', linestyle='dashed')

        if probplot:
            ax = matplotlib.pyplot.subplot(figure_rows,
                                           figure_cols,
                                           (i * figures_per_measurement) + 3,
                                           title=f'{title_prefix}{column_name} Probability')
            scipy.stats.probplot(column_values, plot=ax)

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

def silhouette_graphs(silhouette_values, figure_cols=1, size=4.0, titles=None):
    figure_rows = math.ceil(len(silhouette_values) / figure_cols)

    # A personal figure size preference, based on the number or rows and cols:
    matplotlib.pyplot.figure(figsize=(size * figure_cols, size * figure_rows))

    for index, values in enumerate(silhouette_values):
        k = values['k']
        silhouette_mean = values['silhouette_mean']
        predictions = values['predictions']
        silhouette_values = values['silhouette_values']
        title = titles[index] if titles else f'Silhouette Plot k={k}'
        ax = matplotlib.pyplot.subplot(figure_rows,
                                       figure_cols,
                                       index + 1,
                                       title=title)

        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = silhouette_values[predictions == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = matplotlib.cm.nipy_spectral(float(i) / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0,
                             ith_cluster_silhouette_values,
                             facecolor=color,
                             edgecolor=color,
                             alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_mean, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.3, 0, 0.3])

    matplotlib.pyplot.show()
