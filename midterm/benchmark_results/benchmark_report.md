# Báo cáo benchmark mô hình Dodo / no_dodo

## 1. Kết quả chính

Benchmark được chạy trên cùng một máy, cùng dữ liệu đầu vào và cùng cấu hình
CPU. Trong bốn lựa chọn, ONNX Runtime với graph tối ưu mức `all` có latency
thấp nhất và throughput cao nhất.

| Model | Graph | Kích thước | Mean inference | P95 | P99 | Throughput | Speedup | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch `.pth` CPU | N/A | 269,51 MiB | 187,75 ms | 254,67 ms | 297,37 ms | 5,33 ảnh/s | 1,00x | 100% |
| ONNX `basic` | 122 nodes | 89,61 MiB | 148,45 ms | 197,28 ms | 234,25 ms | 6,74 ảnh/s | 1,26x | 100% |
| ONNX `extend` | 89 nodes | 89,61 MiB | 146,75 ms | 196,75 ms | 235,96 ms | 6,81 ảnh/s | 1,28x | 100% |
| ONNX `all` | 58 nodes | 89,61 MiB | **116,50 ms** | **156,98 ms** | **189,01 ms** | **8,58 ảnh/s** | **1,61x** | 100% |

So với PyTorch CPU, bản ONNX `all` giảm mean latency khoảng `37,95%` và tăng
throughput khoảng `61,15%`. Các giá trị speedup được tính theo:

```text
speedup = mean_latency_pytorch / mean_latency_model
```

## 2. Thiết lập benchmark

| Thành phần | Giá trị |
|---|---|
| Dataset | `midterm/dataset/test` |
| Số ảnh test | 10 ảnh, gồm 5 `dodo` và 5 `no_dodo` |
| Số repetition | 1.000 lần trên toàn bộ test set/model |
| Số inference có đo/model | 10.000 |
| Warm-up | 20 lần trên toàn bộ test set/model |
| Batch size | 1 |
| PyTorch | CPU |
| ONNX Runtime | `CPUExecutionProvider` |
| Số thread runtime | 1 |
| Runtime graph optimization khi load ONNX | `disable` |
| Preprocessing trong latency | Không; ảnh được chuẩn bị trước khi đo |
| Kích thước input | `[1, 3, 224, 224]`, float32 |
| Input mỗi ảnh | 602.112 bytes, khoảng 0,574 MiB |

Preprocessing được dùng giống pipeline inference: RGB, resize cạnh ngắn về 256
pixel, center crop `224 x 224` và chuẩn hóa ImageNet. Latency chỉ bao gồm lời
gọi model và không bao gồm đọc JPEG, resize hoặc crop.

## 3. Latency và throughput

| Model | Median | Std. dev. | Min | Max | Tổng inference | Input throughput |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch CPU | 180,96 ms | 34,58 ms | 130,34 ms | 470,93 ms | 1.877,53 s | 3,06 MiB/s |
| ONNX basic | 142,24 ms | 24,59 ms | 113,05 ms | 329,65 ms | 1.484,53 s | 3,87 MiB/s |
| ONNX extend | 140,19 ms | 24,99 ms | 90,32 ms | 305,81 ms | 1.467,52 s | 3,91 MiB/s |
| ONNX all | **111,12 ms** | **20,75 ms** | **87,26 ms** | **271,14 ms** | **1.165,03 s** | **4,93 MiB/s** |

ONNX `all` có latency thấp hơn ONNX `basic` khoảng `21,52%` và thấp hơn ONNX
`extend` khoảng `20,61%` trên lần chạy này.

## 4. Kích thước model và Flash

| Model | Kích thước bytes | Kích thước MiB | So với PyTorch |
|---|---:|---:|---:|
| PyTorch `best_resnet50.pth` | 282.606.127 | 269,51 MiB | 100% |
| ONNX basic | 93.967.196 | 89,61 MiB | -66,75% |
| ONNX extend | 93.966.220 | 89,61 MiB | -66,75% |
| ONNX all | 93.964.797 | 89,61 MiB | -66,75% |

Các file ONNX nhỏ hơn checkpoint PyTorch khoảng `3,01 lần`. Checkpoint `.pth`
đang chứa cả thông tin phục vụ resume training như optimizer và scheduler,
trong khi ONNX chỉ chứa graph và trọng số phục vụ inference; vì vậy đây là so
sánh dung lượng triển khai giữa checkpoint và inference artifact.

## 5. CPU và RAM

Mỗi model được chạy trong một process riêng. `RSS after load` là RSS sau khi
runtime và model được nạp; `Peak RSS` là RSS lớn nhất quan sát được trong lúc
inference.

| Model | Load time | RSS sau load | Peak RSS | RSS tăng khi load | CPU time | CPU theo 1 core | CPU toàn máy |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch CPU | 11.066,29 ms | 772,30 MiB | 800,17 MiB | 732,63 MiB | 1.891,07 s | 100,70% | 8,39% |
| ONNX basic | 987,63 ms | 156,69 MiB | 180,82 MiB | 117,03 MiB | 1.494,31 s | 100,63% | 8,39% |
| ONNX extend | 323,41 ms | 156,82 MiB | 185,07 MiB | 117,16 MiB | 1.477,31 s | 100,64% | 8,39% |
| ONNX all | 474,48 ms | 156,77 MiB | **172,02 MiB** | 117,11 MiB | 1.172,87 s | 100,63% | 8,39% |

PyTorch sử dụng nhiều RAM hơn rõ rệt trong process benchmark. ONNX `all` có
peak RSS thấp nhất trong ba graph ONNX, khoảng `172,02 MiB`.

CPU theo một core xấp xỉ 100% vì benchmark giới hạn mỗi runtime ở một thread.
CPU toàn máy thấp hơn vì máy có 12 logical CPU.

## 6. Accuracy

Mỗi model thực hiện 10.000 dự đoán có đo thời gian. Kết quả của cả bốn model:

| Lớp | Đúng | Tổng | Accuracy |
|---|---:|---:|---:|
| `dodo` | 5 | 5 | 100% |
| `no_dodo` | 5 | 5 | 100% |
| **Tổng** | **10.000** | **10.000** | **100%** |

Các model cho cùng nhãn dự đoán trên cả 10 ảnh test. Chi tiết theo từng ảnh có
trong [`benchmark_per_image.csv`](benchmark_per_image.csv).

Accuracy 100% ở đây chỉ được đo trên 10 ảnh test hiện có. Việc lặp lại 1.000
lần giúp đo latency ổn định hơn, không làm tập accuracy trở thành 10.000 ảnh
độc lập.

## 7. Phân tích node fusion

Graph ONNX `basic` được dùng làm graph tham chiếu với 122 nodes.

| Graph | Tổng nodes | Nodes giảm so với basic | Tỷ lệ giảm | Explicit fused nodes | Layout nodes |
|---|---:|---:|---:|---:|---:|
| `basic` | 122 | 0 | 0% | 0 | 0 |
| `extend` | 89 | 33 | 27,05% | 33 `FusedConv` | 0 |
| `all` | 58 | 64 | 52,46% | 0 `FusedConv` | 56 |

Mức `extend` thể hiện rõ 33 node `FusedConv`, chủ yếu thay thế các cụm
`Conv + Relu`. Mức `all` không lưu các node dưới type `FusedConv`; graph được
chuyển sang layout NCHWc của CPU và chỉ còn 58 nodes. Vì vậy `64 nodes giảm`
ở `all` bao gồm tác động của fusion, loại bỏ node trung gian và tối ưu layout,
không nên diễn giải toàn bộ con số này như số lượng `FusedConv`.

Phân tích đầy đủ operation type và tên fused node nằm trong
[`benchmark_report.json`](benchmark_report.json).

## 8. Môi trường chạy

| Thành phần | Giá trị |
|---|---|
| CPU | 13th Gen Intel(R) Core(TM) i5-13420H |
| Hệ điều hành | Linux trên WSL2, x86_64 |
| Logical CPU | 12 |
| Python | 3.12.14 |
| PyTorch | 2.14.0 |
| Torchvision | 0.29.0 |
| ONNX Runtime | 1.30.0 |
| ONNX | 1.23.0 |
| NumPy | 2.5.3 |
| Pillow | 12.3.0 |

## 9. Kết luận

Trong cấu hình CPU một thread này, ONNX `all` là lựa chọn tốt nhất cho triển
khai inference:

- latency mean `116,50 ms`, thấp nhất trong bốn model;
- throughput `8,58 ảnh/s`, cao nhất;
- speedup `1,61x` so với PyTorch CPU;
- peak RSS `172,02 MiB`, thấp nhất trong các model ONNX;
- accuracy giữ nguyên ở mức `100%` trên test set.

ONNX `extend` cũng giảm node graph và có throughput nhỉnh hơn `basic`, nhưng
chênh lệch latency giữa hai graph chỉ khoảng `1,15%` trong lần benchmark này.

Các file dữ liệu gốc của benchmark:

- [`benchmark_report.json`](benchmark_report.json)
- [`benchmark_summary.csv`](benchmark_summary.csv)
- [`benchmark_per_image.csv`](benchmark_per_image.csv)
- [`benchmark.log`](benchmark.log)
