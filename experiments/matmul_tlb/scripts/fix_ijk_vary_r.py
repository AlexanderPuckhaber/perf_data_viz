import numpy as np
import subprocess
import pandas as pd
import random
import os
import copy

csv_filename = os.path.join('..', 'data', 'fix_ijk_vary_r-3.csv')
bin_filename = os.path.join('..', '..', '..', 'bin', 'perf_matmul_tlb_runner')

I_sizes = [16, 64, 128, 512]
# I_sizes = np.logspace(1, 24, num=24*2 - 1, base=2)

J_sizes = [16, 64, 128, 512]
# J_sizes = np.logspace(1, 24, num=24*2 - 1, base=2)

# K_sizes = [1024]
K_sizes = np.logspace(1, 24, num=24*2 - 1, base=2)

block_sizes = [1024*1024]
# block_sizes = np.logspace(6, 16, num=11, base=2)

r_amounts = np.logspace(-3, 0, num=24, base=10)

methods = ['tiled']

# perf_configs = ['CFG_L1_LL', 'CFG_BRANCHES', 'CFG_CYCLES_TLB']
perf_configs = ['CFG_LL_TLB']

num_samples = 1

matrix_computations_target = 1024*1024*100


result_dicts = []

params_dicts = []


for i in I_sizes:
  for j in J_sizes:
    k = int(matrix_computations_target / (i * j * 2))
    print(i, j, k)
    if k != 0:
    # for k in K_sizes:
      # matrix_computations = calculate_matrix_computations(i, j, k)
      # if are_matrix_computations_near_target(matrix_computations, matrix_computations_target, 0.2):
      for method in methods:
        for bs in block_sizes:
          for r in r_amounts:
            new_param = {}
            new_param['i'] = int(i)
            new_param['j'] = int(j)
            new_param['k'] = int(k)
            new_param['method'] = method
            new_param['bs'] = int(bs)
            new_param['r'] = float(r)
            params_dicts.append(new_param)

random.shuffle(params_dicts)

print('unique combos:', len(params_dicts))

# params_dicts = params_dicts[:100]

print(len(params_dicts))

new_param_dicts = []

# duplicate for more samples
for param_dict in params_dicts:
  for perf_config in perf_configs:
    for sample in range(num_samples):
      new_param_dict = copy.copy(param_dict)
      new_param_dict['sample'] = sample
      new_param_dict['p'] = perf_config
      new_param_dicts.append(new_param_dict)
      # print(new_param_dict)

random.shuffle(new_param_dicts)
params_dicts = new_param_dicts

results_processed_dicts = []

for param_dict in params_dicts:
  sample = param_dict['sample']
  i = param_dict['i']
  j = param_dict['j']
  k = param_dict['k']
  method = param_dict['method']
  bs = param_dict['bs']
  perf_config = param_dict['p']
  r = param_dict['r']

  # print(i, k)

  result_dict = {}
  result_dict['sample'] = sample
  result_dict['i'] = i
  result_dict['j'] = i
  result_dict['k'] = k
  result_dict['m'] = method
  result_dict['bs'] = bs
  result_dict['p'] = perf_config
  result_dict['r'] = r

  param_dict_lines = []

  # print(a)

  output = subprocess.check_output(['{0} -i {1} -j {2} -k {3} -m {4} -b {5} -p {6} -r {7}'.format(bin_filename, i, j, k, method, bs, perf_config, r)], shell=True, text=True)

  for line in output.splitlines():
    # print(line)
    last_colon_idx = line.rindex(':')
    # print(last_colon_idx)
    output_sliced = line.split(':')
    key = line[:last_colon_idx]
    value = int(output_sliced[-1].strip())
    result_dict[key] = value

    param_dict_line = copy.copy(param_dict)
    param_dict_line['event'] = key
    param_dict_line['count'] = value
    param_dict_lines.append(param_dict_line)
    results_processed_dicts.append(param_dict_line)
  
  # print(result_dict)
  result_dicts.append(result_dict)

  # print(param_dict_lines)
  # results_processed_dicts.append(param_dict_lines)

  # df = pd.DataFrame.from_records(param_dict_lines)
  # print(df)

  print(len(result_dicts), '/', len(params_dicts))          

df = pd.DataFrame.from_records(results_processed_dicts)
print(df)

df.to_csv(csv_filename)
