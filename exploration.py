# Chapter 1: Overview & Labels
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1) load file (adjust path if needed)
df = pd.read_csv("data.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

os.makedirs("visualizations", exist_ok=True)


# 2) basic counts
n_rows = len(df)
n_anom = int(df['y'].sum())
anomaly_pct = 100 * n_anom / n_rows
print(f"Rows: {n_rows:,}, Anomalous rows: {n_anom:,} ({anomaly_pct:.3f}%)")

# 3) category counts
cat_counts = df['category'].astype(int).value_counts().sort_index()
print("\nCategory counts:")
print(cat_counts)

fig, ax = plt.subplots(figsize=(8,3))
ax.bar(cat_counts.index.astype(str), cat_counts.values)
ax.set_yscale('log')
ax.set_title("Category Frequency (Log Scale)")
ax.set_xlabel("category")
ax.set_ylabel("log(count)")
plt.tight_layout()
# plt.show()
plt.savefig(r"visualizations\category_frequency_log.png")

anomaly_cats = cat_counts[cat_counts.index != 0]  # drop category 0
anom_total = anomaly_cats.sum()
pct = (anomaly_cats / anom_total) * 100

fig, ax = plt.subplots(figsize=(8,3))
ax.bar(pct.index.astype(str), pct.values, color='purple')
ax.set_title("Percentage Distribution of Anomaly Types")
ax.set_xlabel("category (1–13)")
ax.set_ylabel("% of all anomalies")
plt.tight_layout()
# plt.show()
plt.savefig(r"visualizations\anomaly_type_distribution.png")




# 5) plot: anomalies over time (rolling count)
df['is_anom'] = df['y'].astype(int)
# choose aggregation interval (e.g., 1 hour). for 1Hz data, 3600 rows ≈ 1 hour
agg_s = '1H'  # use pandas time resampling
try:
    ts = df.set_index('timestamp')['is_anom'].resample(agg_s).sum()
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(ts.index, ts.values)
    ax.set_ylabel("anomaly count per hour")
    ax.set_title("Anomalies over time (hourly)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(r"visualizations\anomalies_over_time.png")
except Exception as e:
    print("Resampling failed (check timestamp index). Error:", e)



#Chapter 2: Sensor Analysis
normal_start = df[df['y']==0].index[10000]
normal_range = range(normal_start, normal_start+200)
anom_start   = df[df['y']==1].index[0]
anom_range   = range(anom_start, anom_start+200)

def plot_groups(sensors, title):
    fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 3*len(sensors)), sharex=True)
    for i, s in enumerate(sensors):
        axes[i].plot(df.loc[normal_range, s].values, label='normal', alpha=0.8)
        axes[i].plot(df.loc[anom_range, s].values, label='anomaly', alpha=0.8)
        axes[i].set_ylabel(s)
        axes[i].legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("visualizations\\" + title.replace(" ", "_").lower() + ".png")
    # plt.show()

group1 = ['aimp','arnd','asin1']
group2 = ['bso1','ced1','cfo1']

plot_groups(group1, "Normal vs Anomalous (Group 1)")
plot_groups(group2, "Normal vs Anomalous (Group 2)")



## Chapter 3:Distributions: Normal vs Anomaly for Key Sensors
sensors = ['aimp', 'arnd', 'asin1', 'bso1', 'ced1', 'cfo1']
for s in sensors:
    normal_vals = df[df['y']==0][s].values
    anom_vals   = df[df['y']==1][s].values
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].hist(normal_vals, bins=100, alpha=0.7)
    axes[0].set_title(f"{s} — Normal")
    
    axes[1].hist(anom_vals, bins=100, alpha=0.7, color='orange')
    axes[1].set_title(f"{s} — Anomaly")
    
    plt.suptitle(f"Distribution Comparison for {s}")
    plt.tight_layout()
    plt.savefig("visualizations\\distribution_" + s + ".png")
    # plt.show()

##chpater 4: Correlation Analysis
sensors = ['aimp','amud','arnd','asin1','asin2','adbr','adfl',
           'bed1','bed2','bfo1','bfo2','bso1','bso2','bso3',
           'ced1','cfo1','cso1']

corr_norm = df[df['y']==0][sensors].corr()
corr_anom = df[df['y']==1][sensors].corr()


# Difference matrix
diff = corr_anom - corr_norm

# Collect top correlation changes
pairs = []
for i,s1 in enumerate(sensors):
    for j,s2 in enumerate(sensors):
        if i < j:
            pairs.append((abs(diff.iloc[i,j]), s1, s2))
pairs_sorted = sorted(pairs, reverse=True)[:20]
labels = [f"{a}-{b}" for _,a,b in pairs_sorted]
values = [v for v,_,_ in pairs_sorted]
plt.figure(figsize=(10,6))
plt.barh(labels, values, color="purple")
plt.title("Top 20 Sensor Pairs with Largest Correlation Change")
plt.xlabel("Absolute Change in Correlation")
plt.tight_layout()
plt.savefig("visualizations\\top_correlation_changes.png")
# plt.show()




from sklearn.preprocessing import StandardScaler

sensors = ['aimp', 'arnd', 'asin1', 'bso1', 'ced1', 'cfo1']

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[sensors] = scaler.fit_transform(df[sensors])

window = 200
step = 200

cat_stats = {cat: {s: [] for s in sensors}
             for cat in sorted(df['category'].unique()) if cat != 0}

for cat in cat_stats:
    cat_indices = df_scaled[df_scaled['category'] == cat].index.values
    for idx in cat_indices[::step]:
        segment = df_scaled.loc[idx:idx+window, sensors]
        if len(segment) == window+1:
            for s in sensors:
                cat_stats[cat][s].append(segment[s].mean())


from math import pi
import matplotlib.pyplot as plt
import numpy as np

def plot_category_signature(cat):
    values = [np.mean(cat_stats[cat][s]) for s in sensors]
    values += values[:1]  # close loop

    angles = [n / float(len(sensors)) * 2 * pi for n in range(len(sensors))]
    angles += angles[:1]

    plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sensors)
    plt.title(f"Category {cat} Sensor Signature")
    plt.tight_layout()
    plt.savefig(f"visualizations\\category_{cat}_signature.png")
    # plt.show()


plot_category_signature(1)
plot_category_signature(2)
plot_category_signature(3)
plot_category_signature(4)
plot_category_signature(5)
plot_category_signature(6)
plot_category_signature(7)
plot_category_signature(8)
plot_category_signature(9)
plot_category_signature(10)
plot_category_signature(11)
plot_category_signature(12)
plot_category_signature(13)


import matplotlib.pyplot as plt
import numpy as np
from math import pi

categories = sorted(cat_stats.keys())

angles = [n / float(len(sensors)) * 2 * pi for n in range(len(sensors))]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for cat in categories:
    values = [np.mean(cat_stats[cat][s]) for s in sensors]
    values += values[:1]
    ax.plot(angles, values, linewidth=1.2, label=f"Cat {cat}", alpha=0.7)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(sensors)
plt.title("Combined Sensor Signatures for All Anomaly Categories")
plt.legend(bbox_to_anchor=(1.1, 1.05))
# plt.show()
plt.savefig(r"visualizations\combined_category_signatures.png")
