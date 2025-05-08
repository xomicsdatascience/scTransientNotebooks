import pandas as pd


def get_distribution(df: pd.DataFrame,
                     key0: str,
                     value,
                     key1: str) -> pd.Series:
    """

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be examined
    key0
        Key
    value
    key1

    Returns
    -------

    """
    filtered_df = df[df[key0] == value]
    distribution = filtered_df[key1].value_counts()
    return distribution
