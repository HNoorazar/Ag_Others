def diversify(prices_arr, portfolio_size, invest_fund=1, preferred=None):
    """Return (stocks indices, weights [norm of eita - eita bar]^-1) for a diversified portfolio
    Arguments
    ---------
    prices_arr : array like
        Stock prices: `prices_arr.shape = (Ndays, Nstocks)`
    portfolio_size : int
        Number of diverse stocks to choose.
    preferred : list of int
        List of preferred vectores. These columns should be in the
        diversified return

    Returns
    -------
    weights_dict : dict(ind:weight)
        Dictionary of recommended diversification. Keys are the indices,
        and the values are the associated weights.

    stocks : array
        Recommended list of `portfolio_size` stocks for a diversified portfolio
    weights : array_like
        Array of weights of investment for each stock

    Example
    -------
    >>> prices_arr = np.arange(1, 1251).reshape((250, 5)) # for plotting
    >>> diversify(prices_arr=prices_arr, portfolio_size=3, invest_fund=1000)
    OrderedDict([(3, 450.15...), (1, 312.327...), (0, 237.52...)])
    """



class StockData(abc.Mapping):
    """Stock data as a dictionary with tickers as keys.

    Data is stored on disk or in memory.

    Attributes
    ----------
    cache_type : "hdf5" | "pickle"
        Type of cache on disk.
    verbose : bool
        If True, then print diagnostic information.

    Examples
    --------
    Here is an example that does not use an on-disk cache:

    >>> data = StockData(cache_dir=None)
    >>> df = data['AA']    # Load data first time
    Downloading AA data...
    >>> df = data['AA']    # Now it is cached in memory.
    >>> list(data)
    ['AA']

    Now we use a cache:

    >>> import tempfile
    >>> cache_dir = tempfile.TemporaryDirectory()
    >>> data = StockData(cache_dir=cache_dir.name, start_date="2020-01-02", end_date="2020-12-30")
    >>> df = data["AA"]    # Load data first time
    Downloading AA data...
    >>> df = data["AA"]    # Now it is cached in memory.
    >>> data2 = StockData(cache_dir=cache_dir.name, start_date="2020-01-02", end_date="2020-12-30")
    >>> df2 = data2["AA"]   # This is from disk.
    >>> cache_dir.cleanup()
    """
