# Vectro Comprehensive Benchmark — glove-100-angular (n=20,000, d=100)

_Single-thread. FAISS SIMD build: `generic`. Linux-6.18.5-x86_64-with-glibc2.39_

## ANN Search — Recall@10 vs QPS (single-thread)

| Backend | Build (s) | Index MB | Max R@10 | QPS@R0.90 | QPS@R0.95 | QPS@R0.99 |
|:--|--:|--:|--:|--:|--:|--:|
| vectro-hnsw | 133.61 | 8.56 | 0.998 | 248 | 144 | 46 |
| faiss-hnsw | 6.24 | 10.38 | 1.000 | 10,825 | 5,594 | 2,981 |
| faiss-ivf | 0.12 | 7.82 | 1.000 | 6,697 | 6,697 | 3,436 |
| hnswlib | 4.54 | 10.46 | 1.000 | 9,400 | 7,622 | 2,666 |
| exact-faiss | 0.01 | 7.63 | 1.000 | 2,647 | 2,647 | 2,647 |

## Quantization — encode throughput / compression / quality

| Method | Throughput (vec/s) | Compression | Reconstruction cosine |
|:--|--:|--:|--:|
| vectro-int8 (rust simd) | 10,908,610 | 3.9x | 1.0000 |
| faiss-scalarquantizer-int8 | 4,418,807 | 4.0x | 0.9999 |
| vectro-pq (M=25) | 46,824 | 16.0x | 0.9503 |
| faiss-indexpq (M=25) | 867,430 | 16.0x | 0.9512 |
