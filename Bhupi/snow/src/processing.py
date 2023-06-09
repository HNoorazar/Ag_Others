import numpy as np
import pandas as pd


def form_slopes(TS_df, window_size=4):
    """Returns slopes of connecting points that are window_size days apart.
        Here I assume TS_df has 365 days in it already where each day
        is a column day_1, day_2, ..., day_365 and each row is associated with
        an station in a given year.

    Hossein: April 26, 2023

    Arguments
    ---------
    TS_df : DataFrame
        pandas DataFrame. Each row is 365 days data and a given location and year.

    window_size : int
        Size of the window to consider.

    Returns
    -------
    Slopes connecting points that are window_size days apart.
    """
    columns_ = list(TS_df.columns)
    end = 365 - window_size + 1
    NA_columns = ["day_"] * window_size
    day_post = list(range(end, 366))
    assert len(NA_columns) == len(day_post)
    NA_columns = [m + str(n) for m, n in zip(NA_columns, day_post)]

    # for a_col in NA_columns:
    #     columns_.remove(a_col)
    left_columns_ = columns_[2 : end + 1]
    right_columns_ = columns_[2 + window_size :]

    for a in zip(left_columns_, right_columns_):
        LC = a[0]
        RC = a[1]
        TS_df.loc[:, LC] = TS_df.loc[:, RC] - TS_df.loc[:, LC]

    TS_df.drop(labels=NA_columns, axis="columns", inplace=True)
    TS_df[left_columns_] = TS_df[left_columns_] / window_size


def one_sided_smoothing(all_stations_years, window_size=5):
    """Returns a smoothed version of signal where signals are soomthed by the old neighbors.
               In other words, only past data contribute to adjusting the current value and
               future data play no role here.

    Side Note: perhaps it is better to break this into smaller functions.

    Hossein: March 29, 2023

    Arguments
    ---------
    all_stations_years : DataFrame
        pandas DataFrame. Each row is 365 days data for a given location and year.

    window_size : int
        Size of the window to consider.

    Returns
    -------
    A smoother version of data.
    """

    all_locs_years_smooth_1Sided = all_stations_years.copy()
    weights = np.arange(1, window_size + 1)
    stations_ = all_stations_years.station_name.unique()

    for a_loc in stations_:
        curr_loc = all_locs_years_smooth_1Sided[all_locs_years_smooth_1Sided.station_name == a_loc]
        years = curr_loc["year"].unique()  # year 2003 does not have all stations_!
        for a_year in years:
            a_signal = curr_loc.loc[curr_loc.year == a_year, "day_1":"day_365"]
            curr_idx = curr_loc.loc[curr_loc.year == a_year, "day_1":"day_365"].index[0]
            a_signal = pd.Series(a_signal.values[0])

            # moving average:
            # ma5=a_signal.rolling(window_size).mean().tolist()

            # weighted moving average. weights are not symmetric here.
            wma_5 = a_signal.rolling(window_size, center=False).apply(
                lambda a_signal: np.dot(a_signal, weights) / weights.sum(), raw=True
            )
            all_locs_years_smooth_1Sided.loc[curr_idx, "day_1":"day_365"] = wma_5.values

    """ 
        We lost some data at the beginning due to rolling window. 
        So, we replace them here
    """
    end = window_size - 1
    # NA_columns = list(all_locs_years_smooth_1Sided.columns[0:end])
    NA_columns = ["day_"] * end

    day_post = list(range(1, window_size))
    assert len(NA_columns) == len(day_post)

    NA_columns = [m + str(n) for m, n in zip(NA_columns, day_post)]

    for a_col in NA_columns:
        # all_locs_years_smooth_1Sided.loc[:, a_col] = all_locs_years_smooth_1Sided.iloc[:, end]
        all_locs_years_smooth_1Sided.loc[:, a_col] = all_locs_years_smooth_1Sided.loc[
            :, "day_" + str(end + 1)
        ]

    return all_locs_years_smooth_1Sided


def two_sided_smoothing(all_stations_years, window_size=5):
    """Returns a smoothed version of signal where signals are soomthed
               using both old and future data.

    Side Note: perhaps it is better to break this into smaller functions.

    Hossein: March 29, 2023

    Arguments
    ---------
    all_stations_years : DataFrame
        pandas DataFrame. Each row is 365 days data for a given location and year.


    window_size : int
        Size of the window to consider.

    Returns
    -------
    A smoother version of data.
    """
    all_locs_years_smooth_2Sided = all_stations_years.copy()
    stations_ = all_stations_years.station_name.unique()

    each_side_len = int((window_size - 1) / 2)
    weights = list(np.arange(1, each_side_len + 2))
    weights_rev = list(np.arange(2, each_side_len + 2))
    weights_rev.reverse()
    weights = weights_rev + weights
    weights = 1 / np.array(weights)

    for a_loc in stations_:
        curr_loc = all_locs_years_smooth_2Sided[all_locs_years_smooth_2Sided.station_name == a_loc]
        years = curr_loc["year"].unique()  # year 2003 does not have all stations_!
        for a_year in years:
            a_signal = curr_loc.loc[curr_loc.year == a_year, "day_1":"day_365"]
            curr_idx = curr_loc.loc[curr_loc.year == a_year, "day_1":"day_365"].index[0]
            a_signal = pd.Series(a_signal.values[0])

            # moving average:
            # ma5=a_signal.rolling(window_size_size).mean().tolist()

            # weighted moving average. weights are not symmetric here.
            wma = a_signal.rolling(window_size, center=True).apply(
                lambda a_signal: np.dot(a_signal, weights) / weights.sum(), raw=True
            )
            all_locs_years_smooth_2Sided.loc[curr_idx, "day_1":"day_365"] = wma.values

    del (a_loc, a_year, curr_loc)

    # **We lost some data at the beginning and end due to rolling window_size. So, we replace them here:**
    # first part of dataframe
    end_first = each_side_len

    # last part of dataframe
    begin = all_locs_years_smooth_2Sided.shape[1] - 2 - each_side_len
    end_last = all_locs_years_smooth_2Sided.shape[1] - 2

    NA_columns_first = list(all_locs_years_smooth_2Sided.columns[0:end_first])
    NA_columns_last = list(all_locs_years_smooth_2Sided.columns[begin:end_last])

    for a_col in NA_columns_first:
        all_locs_years_smooth_2Sided.loc[:, a_col] = all_locs_years_smooth_2Sided.iloc[:, end_first]

    for a_col in NA_columns_last:
        all_locs_years_smooth_2Sided.loc[:, a_col] = all_locs_years_smooth_2Sided.iloc[:, begin - 1]

    return all_locs_years_smooth_2Sided


def add_waterYear(DF_):
    """Returns a DataFrame with one added column; the water year.

    Arguments
    ---------
    DF_ : DataFrame
        pandas DataFrame. Each row is 365 days data for a given location and year.


    Returns
    -------
    A DataFrame with an additional column; water year.
    """
