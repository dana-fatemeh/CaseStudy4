#!/usr/bin/env python3
"""
stat_analysis.py

Statistical analysis for CATS data (data.csv).

Assumes dataset is at: /mnt/data/data.csv

Outputs:
  Numeric results (CSVs, txt) -> /mnt/data/stat_results/
  Visualizations (PNGs)       -> /mnt/data/visualizations/

Visualizations created:
  - PCA cumulative explained variance (pca_explained.png)
  - PCA scatter of first 2 components (pca_scatter.png)
  - Top-20 correlation-change bar chart (corr_change_top_pairs.png)
  - Histograms (normal vs anomaly) for top 4 sensors by |Cohen's d|
      (hist_normal_anom_<sensor>.png)
  - Mutual information bar chart (mi_bar.png)
  - Segment duration histogram (segment_duration_hist.png)
  - Segment gap histogram (segment_gap_hist.png, if gaps exist)
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "data_final.csv"       # path to your CSV
OUT_DIR = "visualizations/stat_results"     # numeric outputs
VIS_DIR = "visualizations"   # figures

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

SENSORS = [
    'aimp','amud','arnd','asin1','asin2','adbr','adfl',
    'bed1','bed2','bfo1','bfo2','bso1','bso2','bso3',
    'ced1','cfo1','cso1'
]

# sampling / computational settings
COHEN_BOOTSTRAP = 200      # bootstrap reps for Cohen's d CI
MI_SAMPLE = 50000          # rows for MI computation
WINDOW = 60                # window size (rows)
STEP = 30                  # step between windows
MAX_WINDOW_SAMPLES = 20000 # max number of windows for PCA
PCA_N_COMPONENTS = 10
PERM_N = 500               # permutations for corr-change test

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

# -----------------------
# Utility functions
# -----------------------
def load_data(path):
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print("Loaded rows:", len(df))
    return df

def extract_segments(df):
    """Create contiguous anomaly segments (y==1)."""
    df = df.copy()
    df['is_anom'] = df['y'].astype(int)
    df['anom_group'] = (df['is_anom'].diff(1) != 0).cumsum() * df['is_anom']
    segments = []
    for g, group in df.groupby('anom_group'):
        if g == 0:
            continue
        start = group['timestamp'].iloc[0]
        end = group['timestamp'].iloc[-1]
        duration_s = (end - start).total_seconds() + 1
        cat = int(group['category'].mode()[0])
        segments.append({
            'group': int(g),
            'start': start,
            'end': end,
            'duration_s': duration_s,
            'category': cat,
            'n_rows': len(group)
        })
    segdf = pd.DataFrame(segments).sort_values('start').reset_index(drop=True)
    segdf.to_csv(os.path.join(OUT_DIR, "segments.csv"), index=False)
    print("Extracted segments:", len(segdf))
    return segdf

def cohens_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa = np.nanstd(a, ddof=1)
    sb = np.nanstd(b, ddof=1)
    pooled = np.sqrt(((na-1)*sa*sa + (nb-1)*sb*sb) / max((na+nb-2), 1))
    if pooled == 0:
        return 0.0
    return (np.nanmean(a) - np.nanmean(b)) / pooled

def bootstrap_cohen_ci(a, b, n_boot=200, seed=0):
    rng = np.random.RandomState(seed)
    combined = np.concatenate([a,b])
    na = len(a)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(combined, size=len(combined), replace=True)
        sa = samp[:na]; sb = samp[na:]
        boots.append(cohens_d(sa, sb))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return np.mean(boots), lo, hi

def permute_corr_change(x_norm, y_norm, x_anom, y_anom, n_perm=500, seed=0):
    """Permutation test on difference of correlations (anomaly - normal)."""
    base_norm = np.corrcoef(x_norm, y_norm)[0,1]
    base_anom = np.corrcoef(x_anom, y_anom)[0,1]
    base_diff = base_anom - base_norm

    combined = np.concatenate([
        np.vstack([x_norm, y_norm]).T,
        np.vstack([x_anom, y_anom]).T
    ], axis=0)
    n_norm = len(x_norm)
    rng = np.random.RandomState(seed)
    perm_diffs = []
    for _ in range(n_perm):
        p = rng.permutation(combined)
        n1 = p[:n_norm]
        n2 = p[n_norm:]
        try:
            c1 = np.corrcoef(n1[:,0], n1[:,1])[0,1]
            c2 = np.corrcoef(n2[:,0], n2[:,1])[0,1]
        except Exception:
            continue
        perm_diffs.append(c2 - c1)
    if len(perm_diffs) == 0:
        return base_diff, 1.0
    pval = np.mean(np.abs(perm_diffs) >= np.abs(base_diff))
    return base_diff, pval

def window_features(df, sensors, window=60, step=30, max_windows=20000):
    print("Computing windowed features ...")
    idxs = list(range(0, len(df) - window, step))
    if len(idxs) > max_windows:
        idxs = list(np.random.RandomState(RANDOM_SEED).choice(idxs, size=max_windows, replace=False))
    feats = []
    labels = []
    for start in idxs:
        sub = df.iloc[start:start+window]
        feat = []
        for s in sensors:
            arr = sub[s].values
            feat.extend([
                np.nanmean(arr),
                np.nanstd(arr),
                np.nanmin(arr),
                np.nanmax(arr),
                (arr[-1] - arr[0]) / (window + 1e-9)
            ])
        feats.append(feat)
        labels.append(int(sub['y'].any()))
    X = np.array(feats)
    y = np.array(labels)
    print("Window features shape:", X.shape)
    return X, y

# -----------------------
# Analysis steps
# -----------------------
def step_cohens_d(df, sensors, n_boot=200):
    print("Running Cohen's d (bootstrap) ...")
    a_df = df[df['y'] == 0]
    b_df = df[df['y'] == 1]
    rows = []
    for s in sensors:
        a = a_df[s].values
        b = b_df[s].values
        d = cohens_d(a, b)
        boot_mean, lo, hi = bootstrap_cohen_ci(a, b, n_boot=n_boot, seed=RANDOM_SEED)
        rows.append({
            'sensor': s,
            'cohens_d': d,
            'boot_mean': boot_mean,
            'ci_lo': lo,
            'ci_hi': hi
        })
    res = pd.DataFrame(rows).sort_values('cohens_d', key=lambda x: x.abs(), ascending=False)
    res.to_csv(os.path.join(OUT_DIR, "cohens_d.csv"), index=False)
    print("Saved Cohen's d results.")
    return res

def step_nonparam_tests(df, sensors):
    print("Running KS and Mann-Whitney tests (sanity only)...")
    from scipy.stats import ks_2samp, mannwhitneyu
    a_df = df[df['y'] == 0]
    b_df = df[df['y'] == 1]
    rows = []
    for s in sensors:
        a = a_df[s].values
        b = b_df[s].values
        try:
            ks = ks_2samp(a, b)
            mw = mannwhitneyu(a, b, alternative='two-sided')
            rows.append({
                'sensor': s,
                'ks_stat': ks.statistic,
                'ks_p': ks.pvalue,
                'mw_stat': mw.statistic,
                'mw_p': mw.pvalue
            })
        except Exception:
            rows.append({'sensor': s, 'ks_stat': np.nan, 'ks_p': np.nan, 'mw_stat': np.nan, 'mw_p': np.nan})
    statdf = pd.DataFrame(rows).sort_values('ks_stat', ascending=False)
    statdf.to_csv(os.path.join(OUT_DIR, "nonparam_tests.csv"), index=False)
    print("Saved nonparametric test results.")
    return statdf

def step_mutual_info(df, sensors, sample_size=50000):
    print("Computing mutual information (sampled) ...")
    N = len(df)
    sample_size = min(sample_size, N)
    idx = np.random.RandomState(RANDOM_SEED).choice(N, size=sample_size, replace=False)
    X_sample = df.iloc[idx][sensors].fillna(0).values
    y_sample = df.iloc[idx]['y'].astype(int).values
    mi = mutual_info_classif(X_sample, y_sample, random_state=RANDOM_SEED)
    midf = pd.DataFrame({'sensor': sensors, 'mi': mi}).sort_values('mi', ascending=False)
    midf.to_csv(os.path.join(OUT_DIR, "mi_scores.csv"), index=False)
    print("Saved MI scores.")
    # Visualization: MI bar chart
    plt.figure(figsize=(8, 5))
    plt.barh(midf['sensor'][::-1], midf['mi'][::-1])
    plt.xlabel("Mutual Information with Anomaly Label")
    plt.title("Sensor Mutual Information Ranking")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "mi_bar.png"))
    print("Saved MI bar chart.")
    return midf

def step_pca_window_features(df, sensors, window=60, step=30, max_windows=20000):
    X, y = window_features(df, sensors, window=window, step=step, max_windows=max_windows)
    print("Scaling features and running PCA ...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)
    proj = pca.fit_transform(Xs)

    # Explained variance plot
    evr = pca.explained_variance_ratio_.cumsum()
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(evr) + 1), evr, marker='o')
    plt.xlabel('PCA Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.title("PCA Explained Variance (Window Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "pca_explained.png"))

    # Scatter of first two components (sample)
    s0 = np.random.RandomState(RANDOM_SEED).choice(len(proj), size=min(4000, len(proj)), replace=False)
    plt.figure(figsize=(6, 5))
    plt.scatter(proj[s0][y[s0] == 0, 0], proj[s0][y[s0] == 0, 1], alpha=0.15, label="normal")
    plt.scatter(proj[s0][y[s0] == 1, 0], proj[s0][y[s0] == 1, 1], alpha=0.4, label="anomaly")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("PCA Scatter (Window Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "pca_scatter.png"))
    print("Saved PCA visualizations.")
    return pca, proj, X, y

def step_correlation_change(df, sensors, perm_n=500):
    print("Computing correlation matrices for normal and anomalous periods ...")
    corr_norm = df[df['y'] == 0][sensors].corr()
    corr_anom = df[df['y'] == 1][sensors].corr()
    diff = corr_anom - corr_norm

    pairs = []
    for i, s1 in enumerate(sensors):
        for j, s2 in enumerate(sensors):
            if j <= i:
                continue
            pairs.append((abs(diff.iloc[i, j]), s1, s2, diff.iloc[i, j]))
    pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)[:20]

    labels = [f"{a}-{b}" for _, a, b, _ in pairs_sorted]
    values = [v for v, _, _, _ in pairs_sorted]

    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], values[::-1], color="purple")
    plt.xlabel("Absolute change in correlation (anomaly - normal)")
    plt.title("Top 20 Sensor Pairs by Correlation Change")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "corr_change_top_pairs.png"))
    print("Saved correlation-change top pairs plot.")

    # Permutation tests on top pairs
    perm_results = []
    for absd, a, b, rawdiff in pairs_sorted:
        x_norm = df[df['y'] == 0][a].values
        y_norm = df[df['y'] == 0][b].values
        x_anom = df[df['y'] == 1][a].values
        y_anom = df[df['y'] == 1][b].values

        n_sub = 2000
        rng = np.random.RandomState(RANDOM_SEED)
        if len(x_norm) > n_sub:
            idxn = rng.choice(len(x_norm), n_sub, replace=False)
            x_norm_s, y_norm_s = x_norm[idxn], y_norm[idxn]
        else:
            x_norm_s, y_norm_s = x_norm, y_norm

        if len(x_anom) > n_sub:
            idxa = rng.choice(len(x_anom), n_sub, replace=False)
            x_anom_s, y_anom_s = x_anom[idxa], y_anom[idxa]
        else:
            x_anom_s, y_anom_s = x_anom, y_anom

        base_diff, pval = permute_corr_change(
            x_norm_s, y_norm_s, x_anom_s, y_anom_s,
            n_perm=min(perm_n, 1000), seed=RANDOM_SEED
        )
        perm_results.append({
            'sensor_a': a,
            'sensor_b': b,
            'corr_change': rawdiff,
            'abs_change': absd,
            'perm_pval': pval
        })
    permdf = pd.DataFrame(perm_results).sort_values('abs_change', ascending=False)
    permdf.to_csv(os.path.join(OUT_DIR, "corr_change_permutation.csv"), index=False)
    print("Saved permutation test results for correlation changes.")
    return diff, permdf

def step_duration_stats(segments):
    print("Computing duration and gap statistics ...")
    if len(segments) == 0:
        print("No segments found.")
        return None
    dur = segments['duration_s']
    summary = {
        'n_segments': int(len(segments)),
        'median_duration_s': float(dur.median()),
        'mean_duration_s': float(dur.mean()),
        'std_duration_s': float(dur.std()),
        'min_duration_s': float(dur.min()),
        'max_duration_s': float(dur.max())
    }
    # gaps between segments
    segs_sorted = segments.sort_values('start').reset_index(drop=True)
    if len(segs_sorted) > 1:
        gaps = (
            segs_sorted['start'].iloc[1:].reset_index(drop=True) -
            segs_sorted['end'].iloc[:-1].reset_index(drop=True)
        ).dt.total_seconds()
        summary['median_gap_s'] = float(gaps.median())
        summary['mean_gap_s'] = float(gaps.mean())

        # histogram of gaps
        plt.figure(figsize=(6, 4))
        plt.hist(gaps, bins=50)
        plt.xlabel("Gap between anomaly segments (s)")
        plt.ylabel("Count")
        plt.title("Distribution of Time Between Anomalous Segments")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "segment_gap_hist.png"))
        print("Saved segment gap histogram.")
    else:
        gaps = None

    # duration histogram
    plt.figure(figsize=(6, 4))
    plt.hist(dur, bins=50)
    plt.xlabel("Anomaly segment duration (s)")
    plt.ylabel("Count")
    plt.title("Distribution of Anomaly Segment Durations")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "segment_duration_hist.png"))
    print("Saved segment duration histogram.")

    fname = os.path.join(OUT_DIR, "segment_duration_summary.txt")
    with open(fname, "w") as f:
        f.write(str(summary))
    print("Saved duration summary:", fname)
    return summary

def step_hist_top_cohens(df, cohens_df, top_k=4):
    """Save histograms (normal vs anomaly) for top-K sensors by |Cohen's d|."""
    print(f"Saving histograms for top {top_k} sensors by |Cohen's d| ...")
    top = cohens_df.sort_values('cohens_d', key=lambda x: x.abs(), ascending=False).head(top_k)
    a_df = df[df['y'] == 0]
    b_df = df[df['y'] == 1]
    for s in top['sensor']:
        vals_norm = a_df[s].values
        vals_anom = b_df[s].values
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 2, 1)
        plt.hist(vals_norm, bins=100, alpha=0.8)
        plt.title(f"{s} — Normal")
        plt.subplot(1, 2, 2)
        plt.hist(vals_anom, bins=100, alpha=0.8, color="orange")
        plt.title(f"{s} — Anomaly")
        plt.suptitle(f"Distribution Comparison for {s}")
        plt.tight_layout()
        fname = os.path.join(VIS_DIR, f"hist_normal_anom_{s}.png")
        plt.savefig(fname)
        print("Saved", fname)

# -----------------------
# Main
# -----------------------
def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: data file not found at {DATA_PATH}")
        sys.exit(1)

    df = load_data(DATA_PATH)

    # basic check
    missing = [s for s in SENSORS if s not in df.columns]
    if missing:
        print("ERROR: missing sensors in CSV:", missing)
        sys.exit(1)

    # 1) Extract anomaly segments
    segments = extract_segments(df)

    # 2) Cohen's d + bootstrap CI
    cohens_df = step_cohens_d(df, SENSORS, n_boot=COHEN_BOOTSTRAP)

    # 3) Nonparametric tests
    _ = step_nonparam_tests(df, SENSORS)

    # 4) Mutual information + MI bar chart
    mi_df = step_mutual_info(df, SENSORS, sample_size=MI_SAMPLE)

    # 5) Windowed features + PCA visualizations
    _pca_model, _proj, _Xwnd, _ywnd = step_pca_window_features(
        df, SENSORS,
        window=WINDOW,
        step=STEP,
        max_windows=MAX_WINDOW_SAMPLES
    )

    # 6) Correlation change + bar chart + permutation p-values
    _diff_mat, _permdf = step_correlation_change(df, SENSORS, perm_n=PERM_N)

    # 7) Segment durations & gaps + histograms
    _dur_stats = step_duration_stats(segments)

    # 8) Histograms for top 4 sensors by |Cohen's d|
    step_hist_top_cohens(df, cohens_df, top_k=4)

    # Print quick summaries for console
    print("\nTop sensors by |Cohen's d|:\n", cohens_df[['sensor', 'cohens_d']].head(10))
    print("\nTop sensors by mutual information:\n", mi_df.head(10))

    print("\nAll numeric results saved to:", OUT_DIR)
    print("All visualizations saved to:", VIS_DIR)
    print("Script finished at", datetime.utcnow().isoformat(), "UTC")

if __name__ == "__main__":
    main()
