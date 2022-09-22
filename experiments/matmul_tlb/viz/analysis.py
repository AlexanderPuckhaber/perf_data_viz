import pandas as pd
import os
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

csv_filepath = os.path.join('..', 'data', 'fix_ijk_vary_r-3.csv')

df = pd.read_csv(csv_filepath)

df_pivot = df.pivot_table(index=['i', 'j', 'k', 'bs', 'r'], columns='event', values='count', aggfunc=[np.mean])
print(df_pivot)

df_sel = df_pivot[('mean', )]

print(df_sel)
df_sel.reset_index(inplace=True)

# df_sel['matrix'] = str(df_sel['i']) + ' ' + str(df_sel['j'])

matrix_labels = []

for idx in range(0, df_sel.shape[0]):
  print(df_sel['i'][idx])
  matrix_labels.append('I: {} J: {} K: {}'.format(df_sel['i'][idx], df_sel['j'][idx], df_sel['k'][idx]))

df_sel['matrix'] = matrix_labels

df_sel['CACHE_MISS_PER_L1D'] = (df_sel['cache-misses'] / df_sel['L1-dcache-loads'])
df_sel['CACHE_ACCESS_PER_L1D'] = (df_sel['cache-references'] / df_sel['L1-dcache-loads'])
df_sel['DTLB_MISS_PER_L1D'] = (df_sel['dTLB-load-misses'] / df_sel['L1-dcache-loads'])

# # df_sel['matrix'] = 'I: {} J: {} K: {}'.format(df_sel['i'], df_sel['j'], df_sel['k'])
# df_sel['matrix'] = df_sel['i'] + df_sel['j']

print(df_sel['matrix'])


matmul_description = ""


# df_filtered = df_sel[(df_sel['k'] == K) & (df_sel['bs'] == 64)]
df_filtered = df_sel

# sns.scatterplot(data=df_filtered, x='r', y='DTLB_MISS_PER_L1D')
# plt.show()

g = sns.relplot(
  data=df_filtered, x='r', y='DTLB_MISS_PER_L1D',
  col='matrix', hue='matrix', col_wrap=4
)
g.set(yscale='log')
plt.show()

g = sns.relplot(
  data=df_filtered, x='r', y='CACHE_ACCESS_PER_L1D',
  col='matrix', hue='matrix', col_wrap=4
)
plt.show()

g = sns.relplot(
  data=df_filtered, x='r', y='CACHE_MISS_PER_L1D',
  col='matrix', hue='matrix', col_wrap=4
)
plt.show()

# df_filtered = df_filtered.groupby(['i', 'r'], as_index=False).mean()


# df_select = df_filtered.pivot('i', 'r', 'CACHE_MISS_PER_L1D')
# ax = sns.heatmap(df_select, cmap='YlGnBu')
# ax.set_xlabel('R')
# ax.set_ylabel('I')
# ax.set_title('CACHE_MISS_PER_L1D' + '\n' + matmul_description)
# plt.show()

# df_select = df_filtered.pivot('i', 'r', 'CACHE_ACCESS_PER_L1D')
# ax = sns.heatmap(df_select, cmap='YlGnBu')
# ax.set_xlabel('R')
# ax.set_ylabel('I')
# ax.set_title('CACHE_ACCESS_PER_L1D' + '\n' + matmul_description)
# plt.show()

# df_select = df_filtered.pivot('i', 'r', 'DTLB_MISS_PER_L1D')
# ax = sns.heatmap(df_select, cmap='YlGnBu')
# ax.set_xlabel('R')
# ax.set_ylabel('I')
# ax.set_title('DTLB_MISS_PER_L1D' + '\n' + matmul_description)
# plt.show()

# df_select = df_filtered.pivot('i', 'bs', 'PERF_COUNT_HW_CPU_CYCLES')
# ax = sns.heatmap(df_select, cmap='YlGnBu')
# ax.set_xlabel('Block Size')
# ax.set_ylabel('I')
# ax.set_title('CPU CYCLES' + '\n' + matmul_description)
# plt.show()


# df_select = df_filtered.pivot('i', 'bs', 'PERF_COUNT_HW_CACHE_L1D')
# ax = sns.heatmap(df_select, cmap='YlGnBu')
# ax.set_xlabel('Block Size')
# ax.set_ylabel('I')
# ax.set_title('PERF_COUNT_HW_CACHE_L1D' + '\n' + matmul_description)
# plt.show()

# df_select = df_filtered.pivot('i', 'bs', 'PERF_COUNT_SW_PAGE_FAULTS')
# ax = sns.heatmap(df_select, cmap='YlGnBu')
# ax.set_xlabel('Block Size')
# ax.set_ylabel('I')
# ax.set_title("PERF_COUNT_SW_PAGE_FAULTS" + '\n' + matmul_description)
# plt.show()


# matmul_description = ""


# # df_filtered = df_sel[(df_sel['i'] == 64) & (df_sel['bs'] == 64)]

# df_filtered = df_filtered.groupby(['j', 'k'], as_index=False).mean()

# df_select = df_filtered.pivot('j', 'k', 'CACHE_MISS_PER_L1D')
# ax = sns.heatmap(df_select, cmap='YlGnBu')
# ax.set_xlabel('J')
# ax.set_ylabel('K')
# ax.set_title('CACHE_MISS_PER_L1D' + '\n' + matmul_description)
# plt.show()

# df_select = df_filtered.pivot('j', 'k', 'CACHE_ACCESS_PER_L1D')
# ax = sns.heatmap(df_select, cmap='YlGnBu')
# ax.set_xlabel('J')
# ax.set_ylabel('K')
# ax.set_title('CACHE_ACCESS_PER_L1D' + '\n' + matmul_description)
# plt.show()

# df_select = df_filtered.pivot('j', 'k', 'DTLB_MISS_PER_L1D')
# ax = sns.heatmap(df_select, cmap='YlGnBu')
# ax.set_xlabel('J')
# ax.set_ylabel('K')
# ax.set_title('DTLB_MISS_PER_L1D' + '\n' + matmul_description)
# plt.show()
