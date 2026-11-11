import numpy as np, sys
B='/data/scratch/ann-data/big'
def read_bin(path, dtype, n=None):
    with open(path, 'rb') as f:
        nr, d = np.fromfile(f, np.int32, 2)
        n = nr if n is None else min(n, nr)
        a = np.fromfile(f, dtype, n * d).reshape(n, d)
    return a
N = 10_000_000
x = read_bin(f'{B}/bigann_base100M.u8bin', np.uint8, N); np.save('/data/scratch/ann-data/bigann-10M/base_u8.npy', x)
qq = read_bin(f'{B}/bigann_query.u8bin', np.uint8); np.save('/data/scratch/ann-data/bigann-10M/query_u8.npy', qq)
print('bigann', x.shape, qq.shape, flush=True)
x = read_bin(f'{B}/deep_base100M.fbin', np.float32, N); np.save('/data/scratch/ann-data/deep-10M/base.npy', x)
qq = read_bin(f'{B}/deep_query.fbin', np.float32); np.save('/data/scratch/ann-data/deep-10M/query.npy', qq)
print('deep', x.shape, qq.shape, flush=True)
