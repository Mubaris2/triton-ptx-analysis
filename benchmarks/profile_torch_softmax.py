import json
import os
import torch
from torch.profiler import profile, ProfilerActivity

os.makedirs('results', exist_ok=True)
rows = []

for n_cols in [1024, 4096, 8192]:
    x = torch.randn(4096, n_cols, device='cuda')
    for _ in range(3):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    trace_path = f'results/torch_softmax_trace_{n_cols}.json'
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)

    with open(trace_path) as f:
        events = json.load(f)['traceEvents']

    for e in events:
        if e.get('cat') == 'kernel':
            a = e.get('args', {})
            rows.append({
                'n_cols': n_cols,
                'kernel': e['name'][:90],
                'dur_us': e.get('dur'),
                'grid': a.get('grid'),
                'block': a.get('block'),
                'regs_per_thread': a.get('registers per thread'),
                'smem_bytes': a.get('shared memory'),
                'est_occupancy_pct': a.get('est. achieved occupancy %'),
            })

import pandas as pd
df = pd.DataFrame(rows)
pd.set_option('display.max_colwidth', 100, 'display.width', 250)
print(df.to_string(index=False))
df.to_csv('results/torch_softmax_profile.csv', index=False)