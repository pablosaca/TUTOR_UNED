import pandas as pd
from beartype import beartype
from src._utils import IntOrNone


@beartype
def group_function(data1: pd.DataFrame,
                   data2: pd.DataFrame,
                   target1: str,
                   target2: str,
                   n_tweets: IntOrNone):
    """
    Function to calculate the results. Sentiment Analysis
    """

    data1 = pd.merge(data1,
                     data2,
                     how="left",
                     left_index=True,
                     right_index=True)

    df1 = data1.groupby(target1).size().reset_index()
    df1 = df1.rename(columns={0: f"total_{target1}"})

    df2 = data1.groupby([target1, target2]).size().reset_index()
    df2 = df2.rename(columns={0: f"total_{target2}"})

    df_fin = pd.merge(df2,
                      df1,
                      how="left",
                      on="label")

    df_fin = df_fin.assign(percentage_label_sent_label=df_fin[f"total_{target2}"]/df_fin[f"total_{target1}"] * 100)

    if n_tweets is not None:
        if n_tweets < 0:
            raise ValueError("Value must be greater than 0")
        else:
            df_fin = df_fin[df_fin[f"total_{target1}"] > n_tweets]

    print(f"Users analysed: {df_fin[target1].nunique()}")
    print(f"Total Users: {data1[target1].nunique()}")

    return df_fin
