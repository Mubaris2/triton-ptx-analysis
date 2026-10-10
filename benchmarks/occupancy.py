import math
import pandas as pd

REGS_PER_SM = 65536
MAX_THREADS = 1024
MAX_BLOCKS = 16
MAX_WARPS = 32

def occupancy(n_regs, num_warps):
    regs_alloc = math.ceil(n_regs / 8) * 8
    regs_per_block = regs_alloc * 32 * num_warps
    by_regs = REGS_PER_SM // regs_per_block
    by_threads = MAX_THREADS // (32 * num_warps)
    blocks = min(by_regs, by_threads, MAX_BLOCKS)
    return blocks * num_warps / MAX_WARPS

df = pd.read_csv('results/softmax_warp_tuning.csv')
df['theoretical_occupancy'] = [
    occupancy(r, w) for r, w in zip(df['n_regs'], df['num_warps'])
]
df.to_csv('results/softmax_tuning_with_occupancy.csv', index=False)
print(df[['n_cols', 'num_warps', 'n_regs', 'theoretical_occupancy', 'gbps']])