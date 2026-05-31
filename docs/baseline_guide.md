# Hướng dẫn chạy các Baseline

> Tài liệu hướng dẫn chi tiết cách chạy 4 mô hình baseline (FS-GNNTR, GCN, GIN, GraphSAGE) trên bộ dữ liệu Tox21 và SIDER.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Yêu cầu môi trường](#2-yêu-cầu-môi-trường)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Bước 0: Smoke Test](#4-bước-0-smoke-test)
5. [Bước 1: Sinh test episodes](#5-bước-1-sinh-test-episodes)
6. [Bước 2: Huấn luyện (Meta-Training)](#6-bước-2-huấn-luyện-meta-training)
7. [Bước 3: Đánh giá (Meta-Testing)](#7-bước-3-đánh-giá-meta-testing)
8. [Bước 4: Tổng hợp kết quả](#8-bước-4-tổng-hợp-kết-quả)
9. [Chạy tất cả baselines cùng lúc](#9-chạy-tất-cả-baselines-cùng-lúc)
10. [Bảng Hyperparameters](#10-bảng-hyperparameters)
11. [Xử lý lỗi thường gặp](#11-xử-lý-lỗi-thường-gặp)

---

## 1. Tổng quan kiến trúc

Có **4 mô hình baseline**, tất cả đều sử dụng phương pháp **FOMAML (First-Order MAML)** cho few-shot learning trên dữ liệu đồ thị phân tử:

| Baseline   | Kiến trúc                                | GNN Backbone    | Transformer |
|------------|------------------------------------------|-----------------|:-----------:|
| **FS-GNNTR** | GNN_prediction (5-layer GIN) + Vision TR + MAML | GIN             | ✅           |
| **GCN**      | GNN_prediction (5-layer GCN) + MAML             | GCN             | ❌           |
| **GIN**      | GNN_prediction (5-layer GIN) + MAML             | GIN             | ❌           |
| **GraphSAGE**| GNN_prediction (5-layer GraphSAGE) + MAML       | GraphSAGE       | ❌           |

### Pipeline tổng thể

```
Smoke Test → Sinh Episodes → Meta-Training → Meta-Testing → Tổng hợp kết quả
```

> **Lưu ý quan trọng:** Tất cả các baseline đều sử dụng **cùng một bộ test episodes** (được sinh sẵn từ file JSON) để đảm bảo đánh giá công bằng. Không có baseline nào tự sinh support/query split riêng khi test.

---

## 2. Yêu cầu môi trường

### Conda environment

```bash
conda activate 3Br_MGD
```

### Thư viện chính

| Thư viện           | Phiên bản  |
|--------------------|-----------|
| Python             | 3.11      |
| PyTorch            | 2.3.0     |
| torch-geometric    | 2.7.0     |
| torch-scatter      | 2.1.2     |
| torch-sparse       | 0.6.18    |
| torch-cluster      | 1.6.3     |
| RDKit              | 2025.9.5  |
| scikit-learn       | 1.8.0     |
| numpy              | 1.26.4    |

Cài đặt (nếu chưa có):

```bash
conda activate 3Br_MGD
pip install -r requirements.txt
```

### Thư mục gốc (Project Root)

**Tất cả các lệnh đều phải được chạy từ thư mục gốc** `D:\3Br_MGD`:

```bash
cd D:\3Br_MGD
```

---

## 3. Cấu trúc thư mục

```
D:\3Br_MGD/
├── 3Br_MGD/Data/                   # Dữ liệu đã tiền xử lý
│   ├── tox21/processed/            #   ├── Tox21
│   └── sider/processed/            #   └── SIDER
├── FS-GNNTR_repo/FS-GNNTR/        # Mã nguồn FS-GNNTR gốc + pretrained weights
│   └── pre-trained/
│       ├── supervised_contextpred.pth         # GIN pretrained
│       ├── gcn_supervised_contextpred.pth     # GCN pretrained
│       └── graphsage_supervised_contextpred.pth  # GraphSAGE pretrained
├── baselines/                      # Mã nguồn baseline framework
│   ├── episode_manager.py          # Sinh & load test episodes
│   ├── gnn_baseline_runner.py      # Runner chung cho GCN/GIN/GraphSAGE
│   ├── graph_adapter.py            # Chuyển đổi đồ thị 3Br-MGD → FS-GNNTR format
│   ├── maml_utils.py               # Các hàm MAML (inner update, meta-step)
│   ├── smoke_test.py               # Kiểm tra imports
│   ├── summarize_results.py        # Tổng hợp kết quả
│   ├── run_all_baselines.py        # Script chạy tất cả baselines
│   ├── seed_utils.py               # Cố định seed
│   ├── episodes_seed42_tox21.json  # Episodes đã sinh sẵn (Tox21)
│   ├── episodes_seed42_sider.json  # Episodes đã sinh sẵn (SIDER)
│   ├── fsgnntr/                    # FS-GNNTR baseline
│   │   ├── config.py
│   │   ├── train.py
│   │   └── test.py
│   ├── gcn/                        # GCN baseline
│   │   ├── config.py
│   │   ├── train.py
│   │   └── test.py
│   ├── gin/                        # GIN baseline
│   │   ├── config.py
│   │   ├── train.py
│   │   └── test.py
│   └── graphsage/                  # GraphSAGE baseline
│       ├── config.py
│       ├── train.py
│       └── test.py
├── checkpoints/                    # Checkpoints được lưu sau training
│   ├── fsgnntr/
│   ├── gcn/
│   ├── gin/
│   └── graphsage/
└── results/                        # Kết quả testing
    ├── tox21/
    └── sider/
```

---

## 4. Bước 0: Smoke Test

Trước khi chạy bất cứ thứ gì, hãy kiểm tra toàn bộ imports hoạt động chính xác:

```bash
conda activate 3Br_MGD
python baselines/smoke_test.py
```

**Kết quả mong đợi:**

```
============================================================
SMOKE TEST — Baseline Framework
============================================================
  [OK]  seed_utils.set_seed
  [OK]  graph_adapter
  [OK]  episode_manager
  [OK]  3Br-MGD data module
  [OK]  BrMGD_train.create_meta_task
  [OK]  FS-GNNTR transformer.GNN_prediction
  [OK]  maml_utils
  [OK]  gnn_baseline_runner
  [OK]  load_all_splits(tox21)
  [OK]  graph_adapter on real SMILES (CCO)
  [OK]  baselines/fsgnntr config
  [OK]  baselines/gcn config
  [OK]  baselines/gin config
  [OK]  baselines/graphsage config

============================================================
RESULT: ALL 11 CHECKS PASSED ✓
============================================================
```

> Nếu có bất kỳ `[ERR]` nào, phải sửa lỗi trước khi tiếp tục.

---

## 5. Bước 1: Sinh test episodes

> **Lưu ý:** File episodes đã được sinh sẵn (`episodes_seed42_tox21.json` và `episodes_seed42_sider.json`) nằm trong `baselines/`. Bạn **chỉ cần chạy lại bước này** nếu muốn thay đổi seed, số lượng shot, hoặc số episodes.

### Tox21

```bash
conda activate 3Br_MGD
python baselines/episode_manager.py ^
    --generate ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 ^
    --shots 5 10 ^
    --n_episodes 30 ^
    --seed 42
```

### SIDER

```bash
conda activate 3Br_MGD
python baselines/episode_manager.py ^
    --generate ^
    --data_dir 3Br_MGD/Data/sider/processed ^
    --dataset sider ^
    --shots 5 10 ^
    --n_episodes 30 ^
    --seed 42
```

### Tham số

| Tham số        | Mô tả                                            | Giá trị mặc định |
|----------------|---------------------------------------------------|:-----------------:|
| `--generate`   | Bắt buộc phải có để kích hoạt sinh episodes       | —                 |
| `--data_dir`   | Đường dẫn tới thư mục data preprocessed           | (bắt buộc)        |
| `--dataset`    | Tên dataset (`tox21` hoặc `sider`)                | `tox21`           |
| `--shots`      | Danh sách K-shot (ví dụ: `5 10`)                  | `5 10`            |
| `--n_episodes` | Số episodes sinh cho mỗi task mỗi K-shot          | `30`              |
| `--seed`       | Random seed                                       | `42`              |
| `--out_dir`    | Thư mục lưu file JSON                             | `baselines`       |

**Output:** File `baselines/episodes_seed42_<dataset>.json`

---

## 6. Bước 2: Huấn luyện (Meta-Training)

### 6.1. Chạy từng baseline riêng lẻ

#### FS-GNNTR (GNN + Transformer)

```bash
conda activate 3Br_MGD
python baselines/fsgnntr/train.py ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 ^
    --shots 5 10 ^
    --max_epochs 200 ^
    --patience 20 ^
    --seed 42
```

#### GCN

```bash
conda activate 3Br_MGD
python baselines/gcn/train.py ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 ^
    --shots 5 10 ^
    --max_epochs 200 ^
    --patience 20 ^
    --seed 42
```

#### GIN

```bash
conda activate 3Br_MGD
python baselines/gin/train.py ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 ^
    --shots 5 10 ^
    --max_epochs 200 ^
    --patience 20 ^
    --seed 42
```

#### GraphSAGE

```bash
conda activate 3Br_MGD
python baselines/graphsage/train.py ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 ^
    --shots 5 10 ^
    --max_epochs 200 ^
    --patience 20 ^
    --seed 42
```

### 6.2. Tham số huấn luyện chung

| Tham số            | Mô tả                                      | Giá trị mặc định |
|--------------------|---------------------------------------------|:-----------------:|
| `--data_dir`       | Đường dẫn tới thư mục data đã xử lý         | (bắt buộc)        |
| `--output_dir`     | Thư mục lưu checkpoint                      | `checkpoints/<model>` |
| `--dataset`        | Tên dataset (`tox21` hoặc `sider`)          | `tox21`           |
| `--shots`          | Danh sách K-shot                            | `5 10`            |
| `--max_epochs`     | Số epoch tối đa                             | `200`             |
| `--patience`       | Ngưỡng early stopping (epochs không cải thiện)| `20`             |
| `--train_episodes` | Số episodes mỗi epoch                       | `100`             |
| `--q_query`        | Số query samples mỗi class khi training     | `256`             |
| `--seed`           | Random seed                                 | `42`              |

### 6.3. Output

Checkpoint được lưu tại:

```
checkpoints/<model>/<model>_<dataset>_<K>-shot_best.pt   # Best (theo query AUC)
checkpoints/<model>/<model>_<dataset>_<K>-shot_last.pt   # Last epoch
```

Ví dụ:

```
checkpoints/gcn/gcn_tox21_5-shot_best.pt
checkpoints/gcn/gcn_tox21_10-shot_best.pt
checkpoints/fsgnntr/fsgnntr_tox21_5-shot_best.pt
```

### 6.4. Huấn luyện trên SIDER

Thay `tox21` bằng `sider` và đổi `--data_dir`:

```bash
python baselines/gcn/train.py ^
    --data_dir 3Br_MGD/Data/sider/processed ^
    --dataset sider ^
    --shots 5 10
```

---

## 7. Bước 3: Đánh giá (Meta-Testing)

> **Yêu cầu:** Phải có checkpoint (từ Bước 2) và file episodes (từ Bước 1).

### 7.1. Chạy từng baseline riêng lẻ

#### FS-GNNTR

```bash
conda activate 3Br_MGD
python baselines/fsgnntr/test.py ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 ^
    --shots 5 10 ^
    --episodes_file baselines/episodes_seed42_tox21.json ^
    --seed 42
```

#### GCN

```bash
conda activate 3Br_MGD
python baselines/gcn/test.py ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 ^
    --shots 5 10 ^
    --episodes_file baselines/episodes_seed42_tox21.json ^
    --seed 42
```

#### GIN

```bash
conda activate 3Br_MGD
python baselines/gin/test.py ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 ^
    --shots 5 10 ^
    --episodes_file baselines/episodes_seed42_tox21.json ^
    --seed 42
```

#### GraphSAGE

```bash
conda activate 3Br_MGD
python baselines/graphsage/test.py ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 ^
    --shots 5 10 ^
    --episodes_file baselines/episodes_seed42_tox21.json ^
    --seed 42
```

### 7.2. Tham số testing

| Tham số            | Mô tả                                       | Giá trị mặc định |
|--------------------|----------------------------------------------|:-----------------:|
| `--data_dir`       | Đường dẫn tới thư mục data                   | (bắt buộc)        |
| `--output_dir`     | Thư mục lưu kết quả & tìm checkpoint        | `checkpoints/<model>` |
| `--dataset`        | Tên dataset                                  | `tox21`           |
| `--shots`          | Danh sách K-shot cần test                    | `5 10`            |
| `--episodes_file`  | Đường dẫn tới file episodes JSON             | (bắt buộc)        |
| `--checkpoint`     | Đường dẫn tới checkpoint (ghi đè mặc định)  | `None` (tự tìm)   |
| `--seed`           | Random seed                                  | `42`              |

### 7.3. Output

Kết quả được lưu dưới dạng JSON:

```
results/<dataset>/results_<dataset>.json      # (nếu dùng --output_dir results/<dataset>)
checkpoints/<model>/results_<dataset>.json    # (nếu dùng output_dir mặc định)
```

**Schema kết quả:**

```json
{
  "model": "GCN",
  "dataset": "tox21",
  "gnn_type": "gcn",
  "seed": 42,
  "n_inner_test": 20,
  "meta_test_tasks": ["task_1", "task_2", "..."],
  "shots": {
    "5-shot": {
      "auc_mean": 0.7123,
      "auc_std": 0.0456,
      "per_task": {
        "task_1": {
          "auc_mean": 0.7200,
          "auc_std": 0.0300,
          "n_episodes": 30,
          "raw_auc": [0.71, 0.73, ...]
        }
      }
    },
    "10-shot": { ... }
  }
}
```

### 7.4. Testing trên SIDER

```bash
python baselines/gcn/test.py ^
    --data_dir 3Br_MGD/Data/sider/processed ^
    --dataset sider ^
    --shots 5 10 ^
    --episodes_file baselines/episodes_seed42_sider.json
```

---

## 8. Bước 4: Tổng hợp kết quả

Sau khi đã test xong tất cả các baselines, chạy lệnh sau để tổng hợp và so sánh:

### Tox21

```bash
conda activate 3Br_MGD
python baselines/summarize_results.py --dataset tox21
```

### SIDER

```bash
conda activate 3Br_MGD
python baselines/summarize_results.py --dataset sider
```

### Cả hai dataset

```bash
conda activate 3Br_MGD
python baselines/summarize_results.py --dataset all
```

**Output mẫu:**

```
======================================================================
  RESULTS: TOX21
======================================================================
  Model                       5-shot              10-shot
  ---------------  ────────────────────  ────────────────────
  fsgnntr          0.7123 ± 0.0456      0.7589 ± 0.0321
  GCN              0.6834 ± 0.0512      0.7201 ± 0.0398
  GIN              0.6912 ± 0.0478      0.7345 ± 0.0367
  GraphSAGE        0.6789 ± 0.0534      0.7156 ± 0.0412
```

Kết quả tổng hợp được lưu tại: `results/<dataset>/summary.json`

---

## 9. Chạy tất cả baselines cùng lúc

Script `run_all_baselines.py` cho phép chạy toàn bộ pipeline tự động.

### Chỉ Training

```bash
conda activate 3Br_MGD
python baselines/run_all_baselines.py ^
    --dataset tox21 ^
    --shots 5 10 ^
    --mode train
```

### Chỉ Testing

```bash
conda activate 3Br_MGD
python baselines/run_all_baselines.py ^
    --dataset tox21 ^
    --shots 5 10 ^
    --mode test
```

### Cả Training + Testing

```bash
conda activate 3Br_MGD
python baselines/run_all_baselines.py ^
    --dataset tox21 ^
    --shots 5 10 ^
    --mode all
```

### Chọn một số baselines

```bash
python baselines/run_all_baselines.py ^
    --dataset tox21 ^
    --shots 5 10 ^
    --mode all ^
    --baselines gcn gin
```

### Tham số

| Tham số        | Mô tả                                      | Giá trị mặc định              |
|----------------|---------------------------------------------|-------------------------------|
| `--dataset`    | Dataset (`tox21` / `sider`)                 | (bắt buộc)                    |
| `--shots`      | Danh sách K-shot                            | `5 10`                        |
| `--mode`       | Chế độ (`train` / `test` / `all`)           | `all`                         |
| `--baselines`  | Chọn baselines cụ thể                       | `fsgnntr gcn gin graphsage`   |
| `--max_epochs` | Số epoch tối đa                             | `200`                         |
| `--patience`   | Early stopping patience                     | `20`                          |
| `--seed`       | Random seed                                 | `42`                          |

---

## 10. Bảng Hyperparameters

### Thông số chung (tất cả baselines)

| Hyperparameter     | Giá trị | Mô tả                                    |
|--------------------|:-------:|-------------------------------------------|
| EMB_SIZE           | 300     | Chiều embedding GNN                       |
| GRAPH_LAYERS       | 5       | Số lớp GNN                                |
| JK                 | last    | Jumping Knowledge aggregation             |
| DROPOUT            | 0.5     | Tỷ lệ dropout                            |
| POOLING            | mean    | Graph pooling                             |
| LR_GNN             | 0.001   | Learning rate meta-optimizer (GNN)        |
| LR_UPDATE          | 0.5     | MAML inner update step size               |
| N_INNER_TRAIN      | 1       | Số bước inner update khi train            |
| N_INNER_TEST       | 20      | Số bước inner update khi test (k_test)    |
| Q_QUERY            | 256     | Số query samples / class khi train        |
| MAX_EPOCHS         | 200     | Số epoch tối đa                           |
| PATIENCE           | 20      | Early stopping patience                   |
| TRAIN_EPISODES     | 100     | Số episodes mỗi epoch                     |
| SEED               | 42      | Random seed                               |

### Thông số riêng cho FS-GNNTR

| Hyperparameter     | Giá trị | Mô tả                          |
|--------------------|:-------:|---------------------------------|
| LR_TR              | 1e-5    | Learning rate cho Transformer   |
| TR_DIM             | 128     | Transformer hidden dimension    |
| TR_DEPTH           | 5       | Số lớp Transformer              |
| TR_HEADS           | 5       | Số attention heads              |
| TR_MLP_DIM         | 256     | MLP dimension trong Transformer |
| TR_PATCH_SIZE      | (30,1)  | Patch size cho Vision TR        |
| POS_WEIGHT_TOX21   | 25.0    | BCELoss pos_weight cho Tox21    |
| POS_WEIGHT_SIDER   | 1.0     | BCELoss pos_weight cho SIDER    |

### Pre-trained weights

| Baseline   | Pre-trained File                              |
|------------|-----------------------------------------------|
| FS-GNNTR   | `FS-GNNTR_repo/FS-GNNTR/pre-trained/supervised_contextpred.pth`          |
| GIN        | `FS-GNNTR_repo/FS-GNNTR/pre-trained/supervised_contextpred.pth`          |
| GCN        | `FS-GNNTR_repo/FS-GNNTR/pre-trained/gcn_supervised_contextpred.pth`      |
| GraphSAGE  | `FS-GNNTR_repo/FS-GNNTR/pre-trained/graphsage_supervised_contextpred.pth`|

---

## 11. Xử lý lỗi thường gặp

### ❌ `ModuleNotFoundError: No module named 'data'`

**Nguyên nhân:** Chạy script không đúng thư mục gốc.

**Giải pháp:** Luôn chạy từ `D:\3Br_MGD`:

```bash
cd D:\3Br_MGD
python baselines/gcn/train.py ...
```

---

### ❌ `Episodes file not found`

**Nguyên nhân:** Chưa sinh test episodes hoặc đường dẫn sai.

**Giải pháp:** Chạy lại Bước 1 (sinh episodes) hoặc kiểm tra file tồn tại:

```bash
dir baselines\episodes_seed42_tox21.json
```

---

### ❌ `No checkpoint found. Using pretrained GNN only.`

**Nguyên nhân:** Chạy test trước khi train.

**Giải pháp:** Phải chạy training (Bước 2) trước khi testing (Bước 3).

---

### ❌ `RuntimeError: CUDA out of memory`

**Giải pháp:** Giảm `--q_query` (ví dụ: `128` thay vì `256`) hoặc giảm `--train_episodes`:

```bash
python baselines/fsgnntr/train.py ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 ^
    --shots 5 10 ^
    --q_query 128 ^
    --train_episodes 50
```

---

### ❌ `ValueError` khi `create_meta_task`

**Nguyên nhân:** Một số task không đủ dữ liệu cho K-shot sampling.

**Giải pháp:** Đây là hành vi bình thường. Script sẽ in warning và bỏ qua episode đó. Không ảnh hưởng tới kết quả tổng thể.

---

## Quick Reference: Quy trình đầy đủ

```bash
# 0. Activate conda environment
conda activate 3Br_MGD
cd D:\3Br_MGD

# 1. Smoke test
python baselines/smoke_test.py

# 2. Sinh episodes (nếu chưa có)
python baselines/episode_manager.py --generate ^
    --data_dir 3Br_MGD/Data/tox21/processed ^
    --dataset tox21 --shots 5 10 --n_episodes 30 --seed 42

python baselines/episode_manager.py --generate ^
    --data_dir 3Br_MGD/Data/sider/processed ^
    --dataset sider --shots 5 10 --n_episodes 30 --seed 42

# 3. Train + Test tất cả baselines trên Tox21
python baselines/run_all_baselines.py --dataset tox21 --shots 5 10 --mode all

# 4. Train + Test tất cả baselines trên SIDER
python baselines/run_all_baselines.py --dataset sider --shots 5 10 --mode all

# 5. Tổng hợp kết quả
python baselines/summarize_results.py --dataset all
```
