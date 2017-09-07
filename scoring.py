"""
    This module implements the SMART tf-idf weighting schemes described in (Salton, Gerard., 1971a)
    as well as the Okapi BM25 function described in (Singhal A., 2011).
"""

import logging
from math import log, sqrt
from numpy import mean

LOGGER = logging.getLogger(__name__)

LOG_BASE = 2.0
OKAPI_K1 = 1.2
OKAPI_K3 = 1000.0
OKAPI_B = 0.75

def bm25(tf, qtf, N, df, dl, avdl, k1=OKAPI_K1, k3=OKAPI_K3, b=OKAPI_B):
    """The Okapi BM25 function for a single term."""
    return log((N-df+0.5) / (df+0.5), LOG_BASE) * \
        ((k1+1) * tf) / (k1 * ((1-b) + b * dl / avdl) + tf) * \
        (k3+1)*qtf / (k3+qtf)

# The following methods implement tf-idf term frequency weighting.
def tf_n(tf, _):
    """Natural term frequency."""
    return tf

def tf_l(tf, _):
    """Logarithmic term frequency."""
    return 1.0 + log(tf, LOG_BASE)

def tf_d(tf, _):
    """Double-logarithmic term frequency."""
    return 1.0 + log(1.0 + log(tf, LOG_BASE), LOG_BASE)

def tf_a(tf, tfs):
    """Augmented term frequency."""
    return 0.5 + (0.5 * tf) / (max(tfs))

def tf_b(tf, _):
    """Boolean term frequency."""
    return 1.0 if tf > 0.0 else 0.0

def tf_L(tf, tfs):
    """Logarithmic averaged term frequency."""
    return (1.0 + log(tf)) / (1.0 + log(mean(tfs)))

# The following methods implement tf-idf document frequency weighting.
def df_n(*_):
    """No document frequency."""
    return 1.0

def df_f(df, N):
    """Inverse document frequency."""
    return log(1.0 * N / df, LOG_BASE)

def df_F(df, N):
    """Inverse document frequency taken to the power of 10."""
    return log(1.0 * N / df, LOG_BASE)**10

def df_t(df, N):
    """Inverse document frequency."""
    return log((N + 1.0) / df, LOG_BASE)

def df_p(df, N):
    """Probabilistic inverse document frequency."""
    return max(0.0, log(1.0 * (N - df) / df, LOG_BASE))

# The following methods implement tf-idf vector normalization techniques.
def norm_n(*_):
    """No normalization."""
    return 1.0

def norm_c(vector, *_):
    """Cosine normalization."""
    s2 = sum(weight**2 for term_id, weight in vector)
    return sqrt(s2)

def norm_u(_, pivot_stats, s):
    """Pivoted unique normalization."""
    return 1.0 - s + s * pivot_stats["u"] / pivot_stats["avgu"]

def norm_b(_, pivot_stats, s):
    """Pivoted byte normalization."""
    return 1.0 - s + s * pivot_stats["b"] / pivot_stats["avgb"]
