"""Theory-grounded embedding selection for chaos estimators (Plan V2, Phase 1).

The forecaster's val-MSE-optimal (dim, lag) is in general a poor Takens
embedding for the CHAOS ESTIMATORS (Rosenstein, expansion, FTLE): the MSE
search favors short lags that make coordinates nearly collinear (audit
A3/B1). This module implements the classical prescriptions instead:

- lag: first local minimum of the time-delayed mutual information
  I(x_t ; x_{t+lag}) (Fraser & Swinney 1986), with a fallback to the first
  lag where the autocorrelation drops below 1/e when no local minimum
  exists before ``max_lag``.
- dim: smallest dimension whose false-nearest-neighbor fraction is below
  1% (Kennel, Brown & Abarbanel 1992), with a Theiler window to exclude
  temporally correlated neighbors.

Numpy only. All functions are pure and deterministic given the seed.
"""

import numpy as np

from src.horizon_utils import embed_series

FNN_THRESHOLD = 0.01  # Kennel: accept the first dim with < 1% false neighbors


def _autocorrelation(series, max_lag):
    """Normalized autocorrelation for lags 0..max_lag (biased estimator)."""
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    acf = np.ones(max_lag + 1, dtype=np.float64)
    if denom <= 0.0:
        return acf
    for k in range(1, max_lag + 1):
        acf[k] = float(np.dot(x[:-k], x[k:])) / denom
    return acf


def _acf_first_below(series, threshold, max_lag):
    """First lag where the autocorrelation drops below ``threshold``.

    Returns ``max_lag`` when the autocorrelation never crosses the
    threshold within the search range.
    """
    acf = _autocorrelation(series, max_lag)
    below = np.nonzero(acf[1:] < threshold)[0]
    if below.size:
        return int(below[0] + 1)
    return int(max_lag)


def _auto_theiler(series):
    """Theiler window from the first zero crossing of the autocorrelation.

    Same recipe as ``estimate_lyapunov`` in horizon_utils: search up to
    min(1000, n//2) and clip the result to [10, max(10, n//10)].
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    max_lag = max(1, min(1000, n // 2))
    theiler = _acf_first_below(series, 0.0, max_lag)
    return int(np.clip(theiler, 10, max(10, n // 10)))


def mutual_information_lag(series, max_lag=100, bins=32):
    """Selects the embedding lag by the mutual-information criterion.

    Computes the histogram mutual information I(x_t ; x_{t+lag}) in nats
    for lags 0..max_lag (shared bin edges over the full series range) and
    returns ``(best_lag, mi_curve)`` where ``best_lag`` is the first local
    minimum of the curve (Fraser & Swinney 1986): the first lag l >= 1
    with mi[l] < mi[l-1] and mi[l] <= mi[l+1].

    Fallback when no local minimum exists before ``max_lag``: the first
    lag where the autocorrelation drops below 1/e (and ``max_lag`` itself
    if the autocorrelation never does).

    Returns:
        best_lag (int >= 1), mi_curve (ndarray of length max_lag + 1,
        index = lag; mi_curve[0] is the self-information of the binned
        series, the natural maximum of the curve).
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    max_lag = int(np.clip(max_lag, 1, max(1, n - 4)))
    mi_curve = np.zeros(max_lag + 1, dtype=np.float64)
    if n < 8 or series.std() <= 0.0:
        return 1, mi_curve

    edges = np.histogram_bin_edges(series, bins=bins)
    for lag in range(0, max_lag + 1):
        a = series if lag == 0 else series[:-lag]
        b = series if lag == 0 else series[lag:]
        joint, _, _ = np.histogram2d(a, b, bins=(edges, edges))
        total = joint.sum()
        if total <= 0:
            continue
        pxy = joint / total
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        outer = px[:, None] * py[None, :]
        nz = pxy > 0.0
        mi_curve[lag] = float(np.sum(pxy[nz] * np.log(pxy[nz] / outer[nz])))

    best_lag = None
    for lag in range(1, max_lag):
        if mi_curve[lag] < mi_curve[lag - 1] and mi_curve[lag] <= mi_curve[lag + 1]:
            best_lag = lag
            break
    if best_lag is None:
        best_lag = _acf_first_below(series, 1.0 / np.e, max_lag)
    return int(max(1, best_lag)), mi_curve


def false_nearest_neighbors(
    series,
    lag,
    max_dim=10,
    rtol=15.0,
    atol=2.0,
    theiler=None,
    max_points=2000,
    seed=0,
):
    """Selects the embedding dimension by false nearest neighbors (Kennel).

    For each dimension d in 1..max_dim, embeds the series with delay
    ``lag``, finds each point's nearest neighbor outside a Theiler window,
    and flags the pair as a false neighbor when either Kennel criterion
    holds (distances in the d- and (d+1)-dimensional embeddings):

      1. |x_{i+d*lag} - x_{j+d*lag}| / R_d(i, j) > rtol
      2. sqrt(R_d(i, j)^2 + |x_{i+d*lag} - x_{j+d*lag}|^2) / R_A > atol

    with R_A = std(series) (attractor size). ``theiler=None`` selects the
    window automatically from the first zero of the autocorrelation.
    At most ``max_points`` reference points are used (seeded subsample).

    Returns:
        best_dim (int): first d with FNN fraction < 1%; if no dimension
            reaches 1%, the evaluated d with the smallest fraction.
        fnn_fractions (ndarray): FNN fraction for d = 1..n_evaluated
            (n_evaluated <= max_dim; smaller when the series is too short
            for the larger dimensions).
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if max_dim < 1:
        raise ValueError("max_dim must be >= 1")
    scale = float(series.std())
    if n < 8 or scale <= 0.0:
        return 1, np.zeros(0, dtype=np.float64)
    if theiler is None:
        theiler = _auto_theiler(series)
    theiler = int(theiler)

    rng = np.random.default_rng(seed)
    fractions = []
    best_dim = None
    for d in range(1, max_dim + 1):
        m = n - d * lag  # points that also have the (d+1)-th coordinate
        if m <= theiler + 2:
            break  # series too short for this dimension
        emb = embed_series(series, d, lag)[:m]
        nxt = series[d * lag : d * lag + m]

        if max_points is not None and max_points < m:
            sample = rng.choice(m, size=max_points, replace=False)
        else:
            sample = np.arange(m)

        n_false = 0
        n_total = 0
        for i in sample:
            diff = emb - emb[i]
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            lo = max(0, i - theiler)
            hi = min(m, i + theiler + 1)
            dist[lo:hi] = np.inf
            j = int(np.argmin(dist))
            rd = dist[j]
            if not np.isfinite(rd):
                continue
            extra = abs(nxt[i] - nxt[j])
            n_total += 1
            if rd <= 1e-12:
                is_false = extra > 1e-12
            else:
                is_false = (extra / rd) > rtol or (
                    np.sqrt(rd * rd + extra * extra) / scale
                ) > atol
            if is_false:
                n_false += 1

        if n_total == 0:
            break
        frac = n_false / n_total
        fractions.append(frac)
        if best_dim is None and frac < FNN_THRESHOLD:
            best_dim = d

    fnn_fractions = np.asarray(fractions, dtype=np.float64)
    if best_dim is None:
        best_dim = int(np.argmin(fnn_fractions)) + 1 if fnn_fractions.size else 1
    return int(best_dim), fnn_fractions


def select_embedding(
    series,
    max_dim=10,
    max_lag=100,
    bins=32,
    rtol=15.0,
    atol=2.0,
    theiler=None,
    max_points=2000,
    seed=0,
):
    """Selects a Takens embedding (dim, lag) for chaos estimators.

    lag from the first local minimum of the mutual information, dim from
    the false-nearest-neighbor criterion at that lag. Intended as the
    default source of ``lyap_dim``/``lyap_lag`` when the user leaves them
    None (the forecaster keeps its own val-MSE embedding).

    Failure guard: when no dimension reaches the 1% FNN threshold at the
    MI-selected lag, the delay vectors at that lag do not unfold the data
    (noise-like embedding). This happens for strongly mixing maps, whose
    MI curve decays to the estimation-noise floor with no meaningful
    minimum; the standard prescription there is lag = 1 (Kantz &
    Schreiber 2004, section 3.3). We then retry FNN at lag = 1 and keep
    it if it reaches a lower FNN fraction.

    Returns:
        dict with keys ``dim`` (int), ``lag`` (int), ``mi_curve``
        (ndarray, index = lag) and ``fnn_fractions`` (ndarray, index
        = dim - 1, evaluated at the returned lag).
    """
    lag, mi_curve = mutual_information_lag(series, max_lag=max_lag, bins=bins)
    dim, fnn_fractions = false_nearest_neighbors(
        series,
        lag,
        max_dim=max_dim,
        rtol=rtol,
        atol=atol,
        theiler=theiler,
        max_points=max_points,
        seed=seed,
    )
    fnn_failed = fnn_fractions.size == 0 or fnn_fractions.min() >= FNN_THRESHOLD
    if fnn_failed and lag != 1:
        dim1, fnn1 = false_nearest_neighbors(
            series,
            1,
            max_dim=max_dim,
            rtol=rtol,
            atol=atol,
            theiler=theiler,
            max_points=max_points,
            seed=seed,
        )
        if fnn1.size and (
            fnn_fractions.size == 0 or fnn1.min() < fnn_fractions.min()
        ):
            lag, dim, fnn_fractions = 1, dim1, fnn1
    return {
        "dim": int(dim),
        "lag": int(lag),
        "mi_curve": mi_curve,
        "fnn_fractions": fnn_fractions,
    }
