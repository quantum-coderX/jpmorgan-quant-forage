import numpy as np

def optimal_binning_by_likelihood(fico_scores, defaults, K,
                                  smoothing=1.0,  # Laplace alpha
                                  min_bin_size=1):
    """
    Optimal contiguous partitioning of sorted FICO scores into K buckets
    maximizing sum of binomial log-likelihoods.
    - fico_scores: 1D array-like of numeric FICO values
    - defaults: 1D array-like of 0/1 default indicators (same length)
    - K: desired number of buckets (labels)
    - smoothing: Laplace smoothing (alpha). Use small positive to avoid log(0).
    - min_bin_size: minimum records per bucket (enforce if >1)
    Returns:
      boundaries: list of K-1 boundary FICO values (right-edge cutoffs)
      bucket_stats: list of dicts {n, k, p_hat} for each bucket (ordered low->high fico)
      rating_fn: function(fico) -> rating (1 best/high fico)
    """

    # Convert to numpy and sort by fico
    x = np.asarray(fico_scores)
    y = np.asarray(defaults).astype(int)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    n_total = len(x_sorted)
    if K <= 0 or K > n_total:
        raise ValueError("K must be between 1 and number of samples")

    # Option: compress identical FICO values to unique levels (keeping counts)
    # This reduces N for DP without losing contiguity.
    vals, idx_start, counts = np.unique(x_sorted, return_index=True, return_counts=True)
    # Build per-unique-value counts of defaults
    m = len(vals)
    defaults_per_val = np.zeros(m, dtype=int)
    for i, start in enumerate(idx_start):
        defaults_per_val[i] = y_sorted[start:start+counts[i]].sum()

    # Precompute cumulative sums for counts and defaults
    cum_n = np.concatenate(([0], np.cumsum(counts)))   # length m+1
    cum_k = np.concatenate(([0], np.cumsum(defaults_per_val)))

    # Helper: compute log-likelihood contribution for interval [i, j] inclusive
    # We'll return negative cost to maximize (so we maximize total ll)
    def interval_loglik(i, j):
        # i, j are 0-based indices into unique value arrays; interval includes i..j
        n = cum_n[j+1] - cum_n[i]
        k = cum_k[j+1] - cum_k[i]
        # enforce minimum bin size
        if n < min_bin_size:
            return -1e18  # extremely bad
        # Laplace: alpha smoothing
        alpha = smoothing
        p_hat = (k + alpha) / (n + 2*alpha)  # Beta( alpha, alpha ) prior
        # log-likelihood of bin (up to additive constant from binomial coeff)
        # ll = k*log(p_hat) + (n-k)*log(1-p_hat)
        # safe check: ensure within (0,1)
        if p_hat <= 0 or p_hat >= 1:
            return -1e18
        return k * np.log(p_hat) + (n - k) * np.log(1 - p_hat)

    # Precompute interval scores for all i<=j (m up to maybe thousands)
    score = np.full((m, m), -1e18)
    for i in range(m):
        for j in range(i, m):
            score[i, j] = interval_loglik(i, j)

    # DP: dp[t][j] best score using t buckets for prefix ending at j (inclusive index j)
    # We'll index t from 1..K, j from 0..m-1
    dp = np.full((K+1, m), -1e18)
    back = np.full((K+1, m), -1, dtype=int)

    # Base: t=1
    for j in range(m):
        dp[1, j] = score[0, j]
        back[1, j] = -1

    for t in range(2, K+1):
        for j in range(t-1, m):  # need at least t values to make t bins
            best_val = -1e18
            best_i = -1
            # try last cut between i..j where previous bucket ends at i-1
            # previous prefix end p = i-1 -> last bucket is i..j, i ranges from t-1-1?? min i = t-1-1? simpler:
            # i minimal = t-1 -1? easier: i from t-1-1? To be safe choose i from t-1-1? We'll just i from t-1..j
            # minimal i is (t-1)-th unique index because we need t-1 buckets for prefix i-1
            for i in range(t-1, j+1):
                prev = dp[t-1, i-1] if i-1 >= 0 else -1e18
                if prev <= -1e17:
                    continue
                val = prev + score[i, j]
                if val > best_val:
                    best_val = val
                    best_i = i
            dp[t, j] = best_val
            back[t, j] = best_i

    # Recover partition for dp[K, m-1]
    parts = []
    t = K
    j = m - 1
    if dp[K, j] <= -1e17:
        raise RuntimeError("No valid partition found — consider decreasing min_bin_size or K")
    while t >= 1:
        i = back[t, j]
        if i == -1:
            # should be base case for t==1
            i = 0
        parts.append((i, j))
        j = i - 1
        t -= 1
    parts.reverse()  # list of (i,j) intervals, low fico -> high fico

    # Build boundaries: right edge of each bucket except last
    boundaries = []
    bucket_stats = []
    for (i, j) in parts:
        n = cum_n[j+1] - cum_n[i]
        k = cum_k[j+1] - cum_k[i]
        p_hat = (k + smoothing) / (n + 2 * smoothing)
        bucket_stats.append({'n': int(n), 'k': int(k), 'p_hat': float(p_hat),
                             'fico_min': float(vals[i]), 'fico_max': float(vals[j])})
    # boundaries from bucket i: cutoff between bucket b and b+1 is max fico of bucket b
    for b in range(len(bucket_stats)-1):
        boundaries.append(bucket_stats[b]['fico_max'])

    # rating function: map fico to rating integer; rating 1 = best credit (highest fico)
    # currently buckets are ordered low->high fico (bucket 0 worst), so we reverse mapping
    def rating_fn(fico):
        # higher fico -> lower rating number (1 is best)
        # find bucket index (0-based low->high), then rating = K - idx
        idx = np.searchsorted(boundaries, fico, side='right')  # returns bucket index low->high
        rating = K - idx
        return int(rating)

    return boundaries, bucket_stats, rating_fn


# Example usage:
if __name__ == "__main__":
    # toy data
    np.random.seed(0)
    N = 2000
    # realistic-ish: FICO ~ mixture: good (700-850), mid (620-700), poor (300-619)
    fico = np.concatenate([
        np.random.normal(760, 30, size=900),
        np.random.normal(670, 25, size=700),
        np.random.normal(580, 40, size=400)
    ])
    fico = np.clip(fico, 300, 850)
    # simulate defaults: higher prob when fico lower
    p = 0.02 + 0.4 * (1 - (fico - 300) / (850 - 300))  # approx decreasing with fico
    defaults = (np.random.rand(len(fico)) < p).astype(int)

    K = 5
    boundaries, stats, rating_fn = optimal_binning_by_likelihood(fico, defaults, K, smoothing=1.0)

    print("Boundaries (right edge of buckets, low->high):", boundaries)
    for i, s in enumerate(stats):
        print(f"Bucket {i+1} (low->high) FICO {s['fico_min']:.1f}-{s['fico_max']:.1f}: n={s['n']}, k={s['k']}, p_hat={s['p_hat']:.3f}")
    # test mapping
    print("Rating for FICO 780:", rating_fn(780), "(1 best)")
    print("Rating for FICO 650:", rating_fn(650))
    print("Rating for FICO 540:", rating_fn(540)).
    