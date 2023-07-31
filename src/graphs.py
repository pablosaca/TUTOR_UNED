"""
######## GRAPHS ########

:Author: Pablo Sánchez Cabrera
:email: psancabrera@gmail.com

"""

import pandas as pd
import os
from datetime import datetime
from beartype import beartype
from src.majorel._utils import StrOrNone, Error
from pandas.api.types import is_numeric_dtype
import plotly.express as px


@beartype
def histplot(data: pd.DataFrame,
             target: str,
             static_plot: bool = True,
             img_dir: str = "images"):
    """
    Function to graph a histplot of a variable from different dataframes.
    Each graph is saved in interactive format in the folder
    `img_dir` using as name the time it was executed.

    Parameters
    ----------
    data: DataFrame
        Table which contain the variable (or variables) to be plotted.
    target: str
        Name of the target variable for which we want to plot the distribution.
    static_plot: bool
         If `static_plot` True, a static graph will be returned.
         If `static_plot` False, an interactive graph
         will be returned.
    img_dir: str
        Image directory to save the images. If `img_dir` does not exist, it will be created.

    Returns
    -------
    figure:
        Histplot

    Raises
    ------
    ValueError
        If `target` does not exist in `data`.
    """

    if target not in data.columns:
        raise ValueError(f"{target} not available "
                         " in the dataframe")

    img_dir_full = os.path.join(os.getcwd(), img_dir)
    if not os.path.exists(img_dir_full):
        os.makedirs(img_dir_full)

    if not is_numeric_dtype(target):
        data[target] = data[target].astype("str")
        data_group = data.groupby(target).size().reset_index()
        data_group = data_group.rename(columns={0: "count"})

        fig = px.histogram(data_group, x="count", nbins=20)

        fig.update_layout(width=800,
                          height=500,
                          title_text=f'Histogram - count {target}',
                          xaxis_title=f"{target}"
                          )
        now = datetime.now()
        now_time = now.strftime("%Y%m%d%H%M%S")

        if static_plot:
            fig.write_html(f'{img_dir}/{now_time}.html')
            fig.show(renderer="png")
        else:
            fig.show()

    else:
        fig = px.histogram(data, x=target, nbins=20)

        fig.update_layout(width=800,
                          height=500,
                          title_text=f'Histogram - {target}',
                          xaxis_title="")
        now = datetime.now()
        now_time = now.strftime("%Y%m%d%H%M%S")

        if static_plot:
            fig.write_html(f'{img_dir}/{now_time}.html')
            fig.show(renderer="png")
        else:
            fig.show()


@beartype
def countplot(data: pd.DataFrame,
              target: str,
              var: StrOrNone = None,
              static_plot: bool = True,
              img_dir: str = "images"):
    """
    Function to graph the countplot of a variable from different dataframes.
    Each graph is saved in interactive format in the folder
    `img_dir` using as name the time it was executed.

    Parameters
    ----------
    data: DataFrame
        Table which contain the variable (or variables) to be plotted.
    target: str
        Name of the target variable for which we want to plot the distribution.
    var: str
        Name of the first variable for which we want to count the number of rows
    static_plot: bool
         If `static_plot` True, a static graph will be returned.
         If `static_plot` False, an interactive graph
         will be returned.
    img_dir: str
        Image directory to save the images. If `img_dir` does not exist, it will be created.

    Returns
    -------
    figure:
        Barplot

    Raises
    ------
    ValueError
        If `target` does not exist in `data`.
    ValueError
        If `var` does not exist in `data`.
    Error : Custom error
        All historical sample used to fit the model.
        Not available performance time series plot
    """

    if target not in data.columns:
        raise ValueError(f"{target} not available "
                         " in the dataframe")

    img_dir_full = os.path.join(os.getcwd(), img_dir)
    if not os.path.exists(img_dir_full):
        os.makedirs(img_dir_full)

    if var is not None:

        if var not in data.columns:
            raise ValueError(f"{var} not available "
                             " in the dataframe")

        X_features = data.copy()

        if is_numeric_dtype(X_features[var]) is True:

            X_copy = X_features.copy()

            levels = list(X_features[var].unique())
            d = {var: levels}  # key: column_name, value: number of factors/levels of variable

            list_two_levels = []
            for i, j in d.items():
                if len(j) == 2:  # two factors
                    list_two_levels.append(i)

            if list_two_levels:
                X_dummy = X_features[list_two_levels]
                X_dummy = pd.get_dummies(X_dummy, drop_first=True)
                X_copy = X_copy.drop(list_two_levels, axis=1)
                X_features = pd.concat([X_copy, X_dummy], axis=1)
            else:
                raise Error("Please check if this variable "
                            "is really a categorical variable. "
                            "In that case, convert it to the correct format")

        lista_var = [target, var]
        df = X_features[lista_var]
        df = df.sort_values(by=lista_var)

        if is_numeric_dtype(df[var]) is False:
            df[var] = data[var].astype("str")

        df = df.groupby(var).count().reset_index()

        fig = px.bar(df,
                     x=var,
                     y=target)

        fig.update_layout(width=800,
                          height=500,
                          title_text=f'Barplot of {target}, groupby {var}')

        now = datetime.now()
        now_time = now.strftime("%Y%m%d%H%M%S")

        if static_plot:
            fig.write_html(f'{img_dir}/{now_time}.html')
            fig.show(renderer="png")
        else:
            fig.show()

    else:

        df = data[target].value_counts().to_frame()
        df.index = df.index.astype("str")

        fig = px.bar(df,
                     x=var,
                     y=target)

        fig.update_layout(width=800,
                          height=500,
                          title_text=f'Barplot of {target}')

        now = datetime.now()
        now_time = now.strftime("%Y%m%d%H%M%S")

        if static_plot:
            fig.write_html(f'{img_dir}/{now_time}.html')
            fig.show(renderer="png")
        else:
            fig.show()
