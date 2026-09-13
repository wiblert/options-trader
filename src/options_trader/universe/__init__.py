"""
universe — the investable-stock universe and the standardized cap-weighted
ticker sampler used to build reproducible test sets.

Two pieces:
    snapshot.py — fetches the S&P 500 constituents + market caps and FREEZES
        them to a dated CSV. The frozen file is the source of truth: it makes the
        sampled test set reproducible regardless of when the test is re-run.
    sampler.py — draws tickers from a frozen snapshot with probability
        proportional to market cap, seeded for determinism.
"""
