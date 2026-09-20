# Dodo / no_dodo image classifier

Repo này huấn luyện một mô hình phân loại ảnh nhị phân:

- `dodo`: ảnh chim dodo;
- `no_dodo`: ảnh các loài chim khác dodo.

Mô hình sử dụng ResNet50 được fine-tune bằng PyTorch. Checkpoint PyTorch có thể được chuyển sang ONNX và chạy bằng ONNX Runtime để so sánh tốc độ, kích thước mô hình và mức sử dụng tài nguyên.

## Cấu trúc thư mục

```text
.
├── .gitattributes                 # Quy tắc Git LFS cho dataset và checkpoint
├── README.md
└── midterm/
    ├── train.py                    # Fine-tune ResNet50
    ├── convert_to_onnx.py          # Chuyển .pth sang ONNX
    ├── demo.py                     # Giao diện Gradio
    ├── benchmark.py                # Benchmark PyTorch/ONNX Runtime
    ├── dataset/
    │   ├── dodo/                   # Ảnh dodo dùng cho train/validation
    │   ├── no_dodo/                # Ảnh chim khác dodo dùng cho train/validation
    │   └── test/
    │       ├── dodo/               # 5 ảnh test
    │       └── no_dodo/            # 5 ảnh test
    └── checkpoints/
        ├── best_resnet50.pth
        ├── last_resnet50.pth
        ├── dodo_classifier_basic.onnx
        ├── dodo_classifier_extend.onnx
        └── dodo_classifier.onnx
```

`train.py` chỉ đọc hai thư mục lớp trực tiếp dưới `dataset/`, nên thư mục `dataset/test/` không bị đưa nhầm vào tập train.

## Cài đặt

Chạy các lệnh sau từ thư mục gốc repo:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision pillow numpy onnx onnxruntime gradio
```

Nếu dùng CUDA, cài phiên bản `torch` và `torchvision` phù hợp với CUDA trên máy theo hướng dẫn của PyTorch. Benchmark trong repo này được thiết kế cho PyTorch CPU và ONNX Runtime `CPUExecutionProvider`.

### Git LFS

Dataset ảnh và các checkpoint lớn được tracking bằng Git LFS. Cài Git LFS một lần trên máy:

```bash
git lfs install
```

Sau khi clone repo, tải đầy đủ các file lớn:

```bash
git lfs pull
```

Các pattern LFS hiện tại gồm ảnh trong `midterm/dataset/` và các file model
`.pth`, `.onnx`, `.pt`, `.ckpt` trong `midterm/checkpoints/`. Kiểm tra danh sách
file đang được quản lý bởi LFS bằng:

```bash
git lfs ls-files
```

## Dataset và preprocessing

Dataset train gồm hai lớp `dodo` và `no_dodo`. `train.py` chia dữ liệu theo stratified split, mặc định dùng `val_ratio=0.2`, seed `42` và batch size `8`.

Ảnh validation, test và ảnh đưa vào demo được xử lý như sau:

1. chuyển sang RGB;
2. resize cạnh ngắn về tối thiểu `256` pixel;
3. center crop về `224 x 224`;
4. chuẩn hóa theo ImageNet: `mean=(0.485, 0.456, 0.406)` và `std=(0.229, 0.224, 0.225)`.

Mapping lớp là:

```text
dodo     -> 0
no_dodo  -> 1
```

## Huấn luyện PyTorch

Lệnh mặc định tạo checkpoint trong `midterm/checkpoints/`:

```bash
python midterm/train.py --epochs 15 --batch-size 8
```

Các file sinh ra:

- `midterm/checkpoints/best_resnet50.pth`: checkpoint có validation accuracy tốt nhất;
- `midterm/checkpoints/last_resnet50.pth`: checkpoint ở epoch cuối;
- `midterm/checkpoints/history.json`: loss, accuracy và learning rate theo epoch.

Một số tùy chọn thường dùng:

```bash
python midterm/train.py \
  --epochs 20 \
  --batch-size 8 \
  --val-ratio 0.2 \
  --freeze-backbone-epochs 1 \
  --patience 5 \
  --output-dir midterm/checkpoints
```

Mặc định mô hình dùng trọng số ImageNet của ResNet50. Dùng `--no-pretrained` nếu không muốn tải trọng số pretrained:

```bash
python midterm/train.py --no-pretrained
```

## Chuyển checkpoint sang ONNX

`convert_to_onnx.py` nhận thêm tham số `--graph-optimization`. Ba mức đang có trong repo được tạo bằng các lệnh sau:

```bash
python midterm/convert_to_onnx.py \
  --checkpoint midterm/checkpoints/best_resnet50.pth \
  --output midterm/checkpoints/dodo_classifier_basic.onnx \
  --graph-optimization basic

python midterm/convert_to_onnx.py \
  --checkpoint midterm/checkpoints/best_resnet50.pth \
  --output midterm/checkpoints/dodo_classifier_extend.onnx \
  --graph-optimization extend

python midterm/convert_to_onnx.py \
  --checkpoint midterm/checkpoints/best_resnet50.pth \
  --output midterm/checkpoints/dodo_classifier.onnx \
  --graph-optimization all
```

`extend` là alias của mức chính thức `extended` trong ONNX Runtime. Các mức chính:

| Mức | Ý nghĩa |
|---|---|
| `basic` | Các tối ưu cơ bản như constant folding và loại bỏ node dư thừa |
| `extend` / `extended` | Thêm các fusion phức tạp, ví dụ `FusedConv` |
| `all` | Bao gồm tối ưu layout; có thể tạo graph phụ thuộc phần cứng CPU |

Script kiểm tra graph bằng ONNX checker và so sánh output ONNX Runtime với PyTorch sau khi export. Input ONNX có dạng `[batch, 3, 224, 224]`, output là hai logits và batch được export ở dạng dynamic mặc định.

## Chạy giao diện demo

Giao diện Gradio cho phép chọn checkpoint có sẵn hoặc upload file `.pth` / `.onnx`:

```bash
python midterm/demo.py --device cpu
```

Sau đó mở `http://127.0.0.1:7860`. Với `.pth`, `--device auto` có thể dùng CUDA nếu PyTorch phát hiện được GPU. File `.onnx` hiện được chạy bằng `onnxruntime` với `CPUExecutionProvider`.

Có thể đổi cổng hoặc thư mục weight:

```bash
python midterm/demo.py \
  --weights-dir midterm/checkpoints \
  --host 127.0.0.1 \
  --port 7860 \
  --device cpu
```

## Benchmark PyTorch và ONNX Runtime

Script [`midterm/benchmark.py`](midterm/benchmark.py) so sánh bốn model:

1. `best_resnet50.pth` chạy bằng PyTorch trên CPU;
2. `dodo_classifier_basic.onnx`;
3. `dodo_classifier_extend.onnx`;
4. `dodo_classifier.onnx` ở mức `all`.

Chạy benchmark mặc định 1.000 lần trên toàn bộ thư mục test:

```bash
python midterm/benchmark.py \
  --iterations 1000 \
  --warmup 20 \
  --threads 1
```

Có 10 ảnh test nên mỗi model sẽ thực hiện `10.000` lượt inference có tính thời gian. Preprocessing được thực hiện trước khi đo; thời gian báo cáo là thời gian gọi model. `--threads 1` đặt cùng số luồng intra-op cho PyTorch và ONNX Runtime để phép so sánh có cùng cấu hình CPU.

Kết quả được lưu trong `midterm/benchmark_results/`:

- `benchmark_report.json`: báo cáo đầy đủ, thông tin môi trường, latency percentiles, CPU/RAM/Flash, accuracy và phân tích graph;
- `benchmark_summary.csv`: một dòng cho mỗi model, thuận tiện mở bằng Excel;
- `benchmark_per_image.csv`: dự đoán và accuracy của từng ảnh.

Các metric gồm:

- inference time trung bình, median, P50, P90, P95, P99, min, max và độ lệch chuẩn;
- throughput theo ảnh/giây và MiB/giây;
- accuracy và số lượng dự đoán đúng;
- CPU time, CPU usage theo một core và theo tổng số core;
- RSS trước/sau khi load và peak RSS trong lúc inference;
- kích thước file model, được dùng làm dung lượng Flash;
- số node ONNX, số node giảm so với graph `basic`, số node fusion và tên node fusion;
- speedup và chênh lệch latency so với PyTorch.

Mặc định benchmark đặt `--onnx-runtime-optimization disable` khi load ONNX để đo graph đã serialize ở ba mức `basic`, `extend` và `all`. Nếu muốn đo thêm một lần tối ưu hóa tại thời điểm load, dùng:

```bash
python midterm/benchmark.py \
  --iterations 1000 \
  --warmup 20 \
  --threads 1 \
  --onnx-runtime-optimization all
```

Mỗi model được chạy trong process riêng để số đo RAM và CPU không bị cộng dồn bởi model hoặc allocator của model trước. Các file ONNX hiện có được tối ưu bằng `CPUExecutionProvider`; khi chuyển sang máy CPU khác, nên benchmark lại trên đúng môi trường triển khai, đặc biệt với graph mức `all`.

## Kiểm tra nhanh

Kiểm tra cú pháp các script:

```bash
python -m py_compile \
  midterm/train.py \
  midterm/convert_to_onnx.py \
  midterm/demo.py \
  midterm/benchmark.py
```

Xem tùy chọn của từng chương trình:

```bash
python midterm/train.py --help
python midterm/convert_to_onnx.py --help
python midterm/demo.py --help
python midterm/benchmark.py --help
```
