# Ôn tập phản biện mã nguồn — ONNX và ONNX Runtime, Nhóm 06

Tài liệu được tổng hợp từ 24 trang của [slide thuyết trình](ONNX_ONNXRuntime_Nhom06.pptx.pdf), đối chiếu với mã nguồn hiện có. Các câu hỏi dưới đây là dự đoán để ôn tập, không phải câu hỏi đã được giảng viên xác nhận. Những đoạn code được rút gọn để tập trung vào ý chính; dùng liên kết nguồn để xem đầy đủ.

## 1. Những đoạn cần nắm trước tiên

| Ưu tiên | Slide | Đoạn mã | Nội dung dễ bị hỏi |
|---|---|---|---|
| Rất cao | 17 | `convert_to_onnx.py:260` | Khôi phục model, `eval()`, export, opset, dynamic batch |
| Rất cao | 10, 17 | `convert_to_onnx.py:166` | Dòng nào thật sự tối ưu và lưu graph offline? |
| Rất cao | 7, 17 | `convert_to_onnx.py:225` | Kiểm chứng cấu trúc và tương đương số học |
| Rất cao | 18 | `benchmark.py:335` | Vì sao tắt graph optimization khi benchmark? |
| Rất cao | 18–19 | `benchmark.py:385` | Vùng đo, warm-up, throughput, accuracy |
| Rất cao | 20 | `benchmark.py:546` | 33 FusedConv, 64 node giảm, 56 node layout |
| Rất cao | 15, 20 | `benchmark.py:239` | RSS, memory arena, memory pattern và CPU cache |
| Cao | 18, 20 | `benchmark.py:625` | Cách ly process và cách tính CPU usage |
| Cao | 16–17 | `demo.py:163` | Demo ONNX có cấu hình giống benchmark không? |
| Bổ sung | 16 | `train.py:125` | Chia dataset, fine-tune và checkpoint |

## 2. Khôi phục model từ checkpoint

**Slide 16–17. Nguồn:** [convert_to_onnx.py](../convert_to_onnx.py#L260).

```python
checkpoint = load_checkpoint(args.checkpoint, device)
state_dict = get_state_dict(checkpoint)
num_classes, class_to_idx = infer_num_classes(checkpoint, state_dict)
model = build_model(num_classes=num_classes, use_pretrained=False)
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()
```

**Câu hỏi:** Tại sao cần tạo ResNet50 trước khi nạp `.pth`?

**Trả lời:** Checkpoint lưu tensor và thông tin huấn luyện. Nhóm tạo kiến trúc tương ứng rồi đưa trọng số vào đúng layer bằng `load_state_dict`. `strict=True` kiểm tra khóa tham số/buffer khớp với model; shape không phù hợp cũng làm nạp thất bại.

**Câu hỏi:** Tại sao `use_pretrained=False` lúc convert?

**Trả lời:** Trọng số fine-tune đã có trong checkpoint. Không cần tải lại ImageNet trước khi ghi đè bằng trọng số checkpoint.

**Câu hỏi:** `state_dict` và `class_to_idx` khác nhau thế nào?

**Trả lời:** `state_dict` chứa weight, bias và buffer như thống kê BatchNorm. `class_to_idx` là mapping `{"dodo": 0, "no_dodo": 1}`, dùng để hiểu ý nghĩa hai đầu ra. Nếu thiếu mapping, converter suy ra số lớp từ shape classifier nhưng chỉ tạo tên lớp dạng số; không thể tự suy ra tên loài.

**Câu hỏi phụ:** Vì sao bỏ tiền tố `module.`?

```python
normalized[key.removeprefix("module.")] = value
```

Checkpoint từ DataParallel có thể thêm tiền tố này; model đơn không có tiền tố nên cần chuẩn hóa khóa.

## 3. `eval()` khác `inference_mode()` thế nào?

**Slide 13, 17–18. Nguồn:** [converter](../convert_to_onnx.py#L266), [benchmark](../benchmark.py#L443).

```python
model.eval()
with torch.inference_mode():
    logits = model(input_value)
```

**Trả lời ngắn:** `eval()` đổi hành vi layer: Dropout ngừng ngẫu nhiên bỏ phần tử, BatchNorm dùng thống kê đã lưu. `inference_mode()` tắt cơ chế autograd phục vụ huấn luyện. Hai thao tác giải quyết hai việc khác nhau, cần dùng cùng nhau trong inference PyTorch.

**Câu hỏi:** Tại sao Conv–BatchNorm fusion cần inference mode của BatchNorm?

BatchNorm inference là phép biến đổi affine với thống kê cố định. Có thể hấp thụ nó vào Conv bằng công thức trên slide 13:

```text
a  = gamma / sqrt(variance + epsilon)
W' = a * W
b' = a * (b - mean) + beta
```

Trong training, thống kê phụ thuộc batch nên không thể dùng một bộ `W'`, `b'` cố định theo cách này.

## 4. Dummy input và dynamic axes

**Slide 5, 7, 17. Nguồn:** [convert_to_onnx.py](../convert_to_onnx.py#L268).

```python
dummy_input = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
dynamic_axes = {"images": {0: "batch"}, "logits": {0: "batch"}}
```

**Câu hỏi:** Dummy input có phải dữ liệu train không?

Không. Đây là đầu vào mẫu cho exporter theo dõi phép toán và shape; nó không huấn luyện hay cập nhật trọng số.

**Câu hỏi:** Chiều nào động? Có nhận ảnh 300 × 300 không?

Chỉ chiều 0 là batch động. Input vẫn cần `[N, 3, 224, 224]`; output `[N, 2]`. Batch 1, 4, 8 có thể dùng cùng artifact, nhưng chiều cao/rộng vẫn cố định. `--static-batch` bỏ dynamic axes và giữ batch theo đầu vào mẫu.

**Câu hỏi:** Graph tĩnh có mâu thuẫn với dynamic batch không?

Không. Cấu trúc phép toán được export là cố định, còn một số chiều tensor có thể mang kích thước ký hiệu. Code hiện dùng tracing nên luồng điều khiển Python phụ thuộc dữ liệu cần được đánh giá riêng khi export.

## 5. Export: opset, constant folding và `dynamo=False`

**Slide 5–7, 11, 17. Nguồn:** [convert_to_onnx.py](../convert_to_onnx.py#L275).

```python
export_kwargs = {
    "input_names": ["images"],
    "output_names": ["logits"],
    "dynamic_axes": dynamic_axes,
    "opset_version": args.opset,
    "do_constant_folding": True,
}
torch.onnx.export(model, dummy_input, str(raw_path), dynamo=False, **export_kwargs)
```

**Câu hỏi:** Opset 17 có phải phiên bản thư viện ONNX không?

Không. Đó là phiên bản bộ định nghĩa toán tử mà graph tham chiếu. Phiên bản package ONNX và opset là hai thông tin khác nhau. Repo chọn mặc định 17 và đã kiểm tra chạy được với runtime của thí nghiệm.

**Câu hỏi:** `dynamo=False` để làm gì?

Chọn đường export TorchScript cũ. Điều này phù hợp với phần tracing được trình bày trên slide; không nên mô tả code hiện tại là dùng exporter `torch.export` mới.

**Câu hỏi:** Constant folding là gì, đã xảy ra trước ORT chưa?

Ví dụ `Transpose(W)` khi `W` cố định có thể tính trước và lưu tensor mới. Converter yêu cầu folding lúc export; tùy phiên bản exporter, folding có thể đã được bật sẵn. Graph đưa vào ORT có thể đã được tối ưu, vì vậy `basic` không phải baseline ONNX hoàn toàn chưa tối ưu.

**Câu hỏi:** Nhóm tự viết thuật toán fusion ở đâu?

Nhóm cấu hình exporter và ONNX Runtime. Các phép chuyển đổi/fusion thực tế do hai thư viện thực hiện, không có thuật toán fusion tự viết trong repo.

## 6. Các mức graph optimization

**Slide 10–14. Nguồn:** [convert_to_onnx.py](../convert_to_onnx.py#L145).

```python
normalized_name = "extended" if name == "extend" else name
levels = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
```

**Trả lời ngắn:** Các mức tích lũy. Basic đơn giản hóa graph; extended thêm các fusion phù hợp với EP; all bật cả tối ưu layout được hỗ trợ. `extend` chỉ là alias do CLI của nhóm đặt.

**Câu hỏi:** `layout` khác `all` thế nào trong code?

Nhánh `layout` tra enum bằng `getattr`, báo lỗi nếu phiên bản runtime không hỗ trợ. Ba artifact của thí nghiệm là basic/extended/all; nhóm không có benchmark riêng cho layout. Không trình bày layout như chỉ bật layout và bỏ các mức trước.

**Câu hỏi:** `all` luôn nhanh nhất không?

Chỉ kết luận nhanh nhất trong cấu hình đã đo. Kết quả có thể đổi theo CPU, shape, datatype, operator và chi phí chuyển layout. Mức tối ưu cao hơn không bảo đảm tăng tốc trên mọi model.

## 7. Dòng nào thực hiện tối ưu offline?

**Slide 10, 17. Nguồn:** [convert_to_onnx.py](../convert_to_onnx.py#L180).

```python
session_options = ort.SessionOptions()
session_options.graph_optimization_level = optimization_level(graph_optimization)
session_options.optimized_model_filepath = str(output_path)
ort.InferenceSession(
    str(raw_path),
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)
```

**Trả lời:** Gán `optimized_model_filepath` chọn nơi lưu; khởi tạo `InferenceSession` thực hiện tối ưu và serialize graph. Ba lần convert chọn ba mức và ba file đích.

**Câu hỏi:** Có thể đem bản all sang Raspberry Pi hoặc GPU ngay không?

Không thể mặc định mọi artifact tối ưu offline đều di động. Bản này được tối ưu cho CPU của môi trường thí nghiệm; có thể chứa operator/layout chuyên biệt. Khi đổi EP hoặc kiến trúc phần cứng, nên export/tối ưu lại cho đích và kiểm chứng. Đặc biệt graph all cần phần cứng tương thích.

## 8. ONNX checker khác kiểm chứng output thế nào?

**Slide 7, 17. Nguồn:** [checker](../convert_to_onnx.py#L206), [verify](../convert_to_onnx.py#L241).

```python
onnx.checker.check_model(onnx_model)

test_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
with torch.inference_mode():
    torch_output = model(test_input).detach().cpu().numpy()
onnx_output = session.run(None, {input_name: test_input.cpu().numpy()})[0]
max_difference = float(np.max(np.abs(torch_output - onnx_output)))
if max_difference > 1e-4:
    raise RuntimeError(...)
```

**Trả lời:** Checker kiểm tra tính hợp lệ cấu trúc/schema. Verify chạy cùng tensor qua hai runtime rồi so sánh logits. Hợp lệ cấu trúc chưa chứng minh tương đương số học hoặc accuracy.

**Câu hỏi:** Code có dùng `assert_allclose(rtol=1e-3, atol=1e-5)` trên slide 7 không?

Không. Slide 7 là ví dụ kiểm chứng. Converter thực tế dùng sai số tuyệt đối lớn nhất với ngưỡng `1e-4`; không dùng relative tolerance.

**Câu hỏi:** Một batch ngẫu nhiên đủ chưa?

Đây là kiểm tra nhanh. Để kiểm chứng rộng hơn cần nhiều ảnh thật, nhiều batch/shape được hỗ trợ, kiểm tra logits và nhãn. Benchmark hiện xác nhận nhãn trên 10 ảnh; nó không lưu toàn bộ logits để chứng minh tương đương trên mọi input. Kiểm tra hiện tại cũng chưa xử lý riêng NaN/Inf.

**Câu hỏi:** Verify có tắt tối ưu khi load như benchmark không?

Không. `verify_with_onnxruntime()` tạo session không truyền SessionOptions nên dùng mức mặc định. Nó kiểm tra graph sau các bước tối ưu runtime có thể áp dụng thêm; benchmark mới chủ động giữ mức `disable`.

## 9. Metadata và cấu trúc ONNX

**Slide 5, 17. Nguồn:** [convert_to_onnx.py](../convert_to_onnx.py#L208).

```python
metadata = {
    "class_to_idx": json.dumps(class_to_idx, sort_keys=True),
    "image_size": str(image_size),
    "input_layout": "NCHW",
    "input_dtype": "float32",
    "normalization_mean": json.dumps(IMAGENET_MEAN),
    "normalization_std": json.dumps(IMAGENET_STD),
}
```

**Câu hỏi:** Metadata có tự preprocess ảnh cho ORT không?

Không. Đây là thông tin mô tả. Ứng dụng cần thực hiện preprocessing; graph hiện nhận tensor đã normalize. Demo đọc class mapping/image size nhưng mean/std đang dùng hằng số trong code, chưa tự đọc mọi cấu hình metadata.

**Câu hỏi:** Graph node và trọng số nằm ở đâu?

`model.graph.node` mô tả phép toán; `model.graph.initializer` thường lưu các tensor trọng số. Có thể kiểm tra bằng ONNX hoặc Netron. Metadata lưu thông tin kèm model, không thay thế node tính toán.

## 10. Vì sao tắt optimization lúc benchmark?

**Slide 17–18. Nguồn:** [benchmark.py](../benchmark.py#L343).

```python
session_options.graph_optimization_level = parse_onnx_runtime_optimization(
    ort, runtime_optimization
)
session_options.intra_op_num_threads = threads
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
```

**Trả lời:** Cấu hình benchmark dùng `disable` để giữ khác biệt giữa các graph đã lưu. Nếu bật all khi load cả basic và extended, runtime có thể tối ưu chúng tiếp khiến phép so sánh ba mức bị thay đổi.

**Câu hỏi:** Disable có tắt memory arena và mọi tối ưu kernel không?

Không. Đây là mức tối ưu graph. Các cơ chế quản lý bộ nhớ và kernel của runtime là cấu hình khác; script không tắt chúng qua biến này.

## 11. Công bằng về CPU và thread

**Slide 18. Nguồn:** [PyTorch](../benchmark.py#L299), [ORT](../benchmark.py#L347).

```python
torch.set_num_threads(threads)
torch.set_num_interop_threads(1)

session_options.intra_op_num_threads = threads
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
providers = ["CPUExecutionProvider"]
```

**Trả lời:** Hai runtime đều dùng CPU, cùng batch/input/precision và cùng số thread intra-op. Intra-op là song song bên trong một phép toán; inter-op liên quan thực thi các phép toán.

**Câu hỏi:** Máy có 12 CPU logic, sao chỉ dùng một thread?

Để so sánh ở một cấu hình cố định, dễ diễn giải. Kết quả này chưa đánh giá throughput khi dùng nhiều core; cần thí nghiệm thêm để chọn cấu hình triển khai.

**Câu hỏi:** Đã cố định mọi yếu tố gây nhiễu như tiêu đề slide 18 chưa?

Nhóm cố định các cấu hình chính. Code chưa đặt CPU affinity, khóa xung nhịp, kiểm soát nhiệt/tải nền, hoặc đổi ngẫu nhiên thứ tự model. Vì vậy tránh khẳng định loại bỏ toàn bộ nhiễu. WSL2 cũng là một phần môi trường đo, không phải thiết bị biên thật.

## 12. Preprocessing và nhãn đầu ra

**Slide 16, 18. Nguồn:** [benchmark.py](../benchmark.py#L215).

```python
array = np.asarray(image, dtype=np.float32) / 255.0
array = (array - IMAGENET_MEAN) / IMAGENET_STD
return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32, copy=True)
```

**Trả lời:** Chia 255 đưa pixel về khoảng 0–1; normalize theo ImageNet; HWC đổi thành CHW; thêm batch thành NCHW. PyTorch dùng `torch.from_numpy` trên input đã chuẩn bị, ONNX nhận NumPy tương ứng.

**Câu hỏi:** Có giống tuyệt đối validation transform không?

Benchmark và demo dùng cùng công thức resize/crop. Train validation dùng torchvision `Resize(256)`/`CenterCrop`, còn benchmark/demo tính kích thước bằng `round`; có thể lệch cách làm tròn vài pixel ở một số ảnh. Có thể nói cùng quy trình tiền xử lý, nhưng chưa chứng minh mọi tensor giống từng phần tử với torchvision.

**Câu hỏi:** Mapping test có luôn khớp checkpoint không?

Code benchmark tạo mapping bằng sắp xếp tên thư mục; dataset hiện tại khớp `dodo=0`, `no_dodo=1`. Script chưa kiểm tra mapping này với metadata của từng model. Nếu thay dataset/model, cần kiểm tra mapping để tránh tính accuracy sai.

## 13. Warm-up và số lần lặp

**Slide 18. Nguồn:** [benchmark.py](../benchmark.py#L420).

```python
for _ in range(warmup):
    for input_value in inputs:
        run_once(input_value)
```

Đoạn trên rút gọn hai nhánh PyTorch/ORT; nhánh PyTorch còn có `inference_mode()`.

**Trả lời:** 20 warm-up là 20 lượt toàn bộ 10 ảnh, tức 200 lời gọi/model. Đo 1.000 lượt là 10.000 lời gọi/model, tổng 40.000 lời gọi đo cho bốn model. Warm-up giúp giảm ảnh hưởng khởi tạo ban đầu.

**Câu hỏi:** Có phải benchmark cold-cache không?

Không. Input đã có trong RAM và model đã warm-up. Đây là trạng thái chạy lặp ổn định; không có bước xóa cache CPU hay đo cold-cache riêng.

## 14. Vùng đo thời gian

**Slide 18–19. Nguồn:** [benchmark.py](../benchmark.py#L446).

```python
start_ns = time.perf_counter_ns()
logits = run_once(input_value)
elapsed_ns = time.perf_counter_ns() - start_ns
latency_ns.append(elapsed_ns)

prediction = int(logits.argmax(dim=1).item())  # nhánh PyTorch
```

**Trả lời:** Chỉ lời gọi runtime nằm trong latency từng ảnh. Đọc ảnh, preprocessing, load model và argmax nằm ngoài vùng đo này. Thời gian vẫn gồm overhead lời gọi Python/API và việc runtime tạo output, không phải chỉ riêng kernel CPU.

**Câu hỏi:** Vì sao không cần CUDA synchronize?

Thí nghiệm hiện chỉ dùng CPU. Nếu mở rộng GPU cần thiết kế timing theo cơ chế thực thi bất đồng bộ; không dùng nguyên vùng đo CPU để kết luận latency GPU.

**Câu hỏi:** P95 là gì?

Khoảng 95% latency quan sát được không vượt giá trị P95. P95/P99 diễn tả đuôi phân bố; median diễn tả trung vị; mean dễ bị kéo lên bởi các lượt chậm.

## 15. Throughput, speedup và giảm latency

**Slide 18–19. Nguồn:** [benchmark.py](../benchmark.py#L497).

```python
measured_throughput = len(latency_ns) / measured_inference_s
wall_throughput = len(latency_ns) / wall_elapsed_s
```

```text
speedup = mean_PyTorch / mean_model
latency_reduction = (mean_PyTorch - mean_model) / mean_PyTorch
```

**Trả lời:** Throughput chính dùng tổng thời gian các lời gọi model. Wall throughput dùng cả thời gian vòng benchmark, gồm argmax, ghi nhận kết quả và lấy mẫu RSS. MiB/s dùng kích thước tensor input × ảnh/s; không phải đo băng thông RAM phần cứng.

**Ví dụ all:** `187,75 / 116,50 ≈ 1,61x`; giảm latency `37,95%`; tăng throughput khoảng `61,16%`. Hai phần trăm khác nhau do mẫu số khác nhau.

**Câu hỏi:** Có thể nói nhanh hơn do fusion chắc chắn bao nhiêu phần trăm không?

Không thể phân rã nhân quả chỉ từ một lần so sánh này. Các graph và runtime thay đổi nhiều yếu tố cùng lúc. Muốn tách lợi ích từng transformer cần thí nghiệm loại bỏ/bật riêng từng tối ưu và profiling.

## 16. RSS và CPU usage

**Slide 20. Nguồn:** [RSS](../benchmark.py#L239), [sampler](../benchmark.py#L260), [CPU](../benchmark.py#L517).

```python
if line.startswith("VmRSS:"):
    return int(line.split()[1]) * 1024

cpu_percent_one_core = 100 * cpu_elapsed_s / wall_elapsed_s
cpu_percent_all_cores = cpu_percent_one_core / logical_cpu_count
```

**Trả lời:** RSS là RAM resident của process, gồm model, runtime, thư viện và vùng cấp phát. Sampler đọc RSS định kỳ mặc định 50 ms; peak là đỉnh quan sát bằng lấy mẫu, có thể bỏ lỡ đỉnh ngắn. Fallback `ru_maxrss` là high-water mark, khác RSS tức thời.

**Câu hỏi:** Vì sao ~100,6% CPU và ~8,39% toàn máy?

Khoảng 100% là một core-equivalent; chia 12 logical CPU được khoảng 8,39%. `process_time()` gồm thời gian CPU của process, có cả thread lấy mẫu/overhead; không phải biểu đồ sử dụng CPU toàn hệ thống hay đo điện năng. Một thread intra-op không có nghĩa process chỉ có đúng một thread.

**Câu hỏi:** RAM giảm từ ~800 xuống ~172 MiB có phải toàn bộ do fusion?

Không. Có cả khác biệt framework, import thư viện, allocator và model. Nên kết luận RSS process ONNX thấp hơn trong phép đo này, không quy tất cả cho tensor trung gian hoặc CPU cache.

## 17. Vì sao mỗi model chạy process riêng?

**Slide 18. Nguồn:** [benchmark.py](../benchmark.py#L633).

```python
context = mp.get_context("spawn")
process = context.Process(target=worker_entry, args=(...))
process.start()
process.join()
```

**Trả lời:** `spawn` tạo process Python mới. Sau một model, process kết thúc rồi mới chạy model tiếp theo; allocator/session model trước không cộng dồn vào RSS process sau. Benchmark chạy tuần tự, không chạy bốn model cạnh tranh CPU cùng lúc.

**Giới hạn:** Cách ly process không xóa page cache OS, cache CPU hoặc bảo đảm cùng nhiệt độ/xung nhịp. Load time còn gồm import runtime/tạo kiến trúc/nạp checkpoint nên không diễn giải nó như thời gian đọc weight thuần túy.

## 18. Đếm fusion/layout có chứng minh gì?

**Slide 12–14, 20. Nguồn:** [benchmark.py](../benchmark.py#L553).

```python
nodes = list(model.graph.node)
op_counts = Counter(node.op_type for node in nodes)
fused_nodes = [
    node for node in nodes
    if "fused" in node.op_type.lower() or "fused" in node.name.lower()
]
layout_nodes = [
    node for node in nodes
    if "_nchwc" in node.name.lower() or node.op_type.lower() == "reorderoutput"
]
```

**Trả lời:** Đây là thống kê artifact theo type/name, chưa phải profiling kernel hoặc lịch sử transformer. Trong kết quả: basic 122 nodes; extended 89 và 33 FusedConv; all 58 với 56 nodes có dấu hiệu layout.

**Câu hỏi:** All có 0 FusedConv nghĩa là không fusion?

Không. Sau biến đổi layout, graph có thể dùng operator/domain khác. Rule tìm chữ fused không bao quát mọi fusion. 64 node giảm không tương đương 64 lần fusion; 56 layout nodes không tương đương 56 phép chuyển layout.

**Câu hỏi:** Node `Conv` của all có giống standard Conv không?

Cần xem cả `node.domain`, attributes và input, không chỉ `op_type`. Code hiện thống kê theo type/name nên có giới hạn. Muốn khẳng định mapping chính xác nên mở graph và kiểm tra thuộc tính hoặc log transformer.

**Câu hỏi:** 122 nodes basic có phải 122 layer ResNet không?

Không. Layer kiến trúc và operator node sau export/tối ưu là hai cách đếm khác nhau. Trọng số initializer cũng không được đếm là operator node trong phép thống kê này.

## 19. Phần tối ưu bộ nhớ trên slide 15 đã thực nghiệm chưa?

**Nguồn:** [load_onnx_model](../benchmark.py#L343).

Code hiện không đặt riêng `enable_cpu_mem_arena`, `enable_mem_pattern`, `enable_mem_reuse`, không gọi `run_with_iobinding` và không đo hardware cache counters.

**Trả lời nên dùng:** “Slide 15 trình bày cơ chế của ONNX Runtime. Thí nghiệm của nhóm đo latency và RSS với cấu hình runtime hiện tại, chưa có thí nghiệm bật/tắt riêng từng cơ chế quản lý bộ nhớ và chưa đo cache miss.”

**Đoạn minh họa nếu thầy hỏi cách bổ sung** — chưa chạy trong benchmark:

```python
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
```

Muốn đánh giá cần thay đổi từng option một, giữ graph/input/thread không đổi, chạy process riêng và đo lại. Không gọi RSS là cache hit/miss. I/O Binding là chủ đề riêng về binding bộ nhớ input/output; demo hiện dùng `session.run()` thông thường.

## 20. Accuracy 100% chứng minh được gì?

**Slide 19–20. Nguồn:** [benchmark.py](../benchmark.py#L451).

```python
prediction = int(logits.argmax(dim=1).item())
expected = int(records[index]["label_index"])
correct += int(prediction == expected)
accuracy = correct / (iterations * len(records))
```

**Trả lời:** Cả bốn cấu hình giữ đúng nhãn trên 10 ảnh hiện có: 5 dodo và 5 no_dodo. Chạy lặp tạo 10.000 dự đoán/model nhưng chỉ có 10 mẫu độc lập; không kết luận tổng quát hóa từ 10.000 ảnh khác nhau.

**Câu hỏi:** Confidence có phải xác suất đã hiệu chuẩn?

Code lấy softmax của logits để hiển thị confidence. Chưa thực hiện probability calibration; confidence không bảo đảm tỷ lệ đúng tương ứng.

**Câu hỏi:** Chứng minh test không trùng train bằng code nào?

`BinaryImageFolder.find_classes()` chỉ đọc hai thư mục train trực tiếp, tránh dùng test như một lớp. Tuy nhiên cấu trúc thư mục không tự chứng minh ảnh không trùng nội dung; code hiện chưa kiểm tra hash/ảnh gần trùng hoặc nguồn dữ liệu trùng giữa train/test.

## 21. Flash và kích thước checkpoint

**Slide 20. Nguồn:** [checkpoint](../train.py#L275), [benchmark](../benchmark.py#L506).

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    # epoch, class mapping, training configuration...
}
model_size_bytes = weight_path.stat().st_size
```

**Trả lời:** 269,51 MiB là training checkpoint có trạng thái optimizer; 89,61 MiB là inference artifact ONNX. Giảm ~66,75% dung lượng file không phải bằng chứng ONNX nén trọng số 3 lần hay giảm precision: các model đang dùng float32.

**Câu hỏi:** Đã đo footprint Flash toàn ứng dụng chưa?

Chưa. Trường Flash đang dùng size file model; chưa cộng runtime, dependencies hoặc đo thiết bị Flash vật lý. Muốn so sánh trọng số công bằng hơn cần thêm checkpoint chỉ chứa state_dict; chưa có số đo đó trong báo cáo.

## 22. Demo có giống benchmark không?

**Slide 17. Nguồn:** [demo.py](../demo.py#L171).

```python
cache_key = (str(path.resolve()), stat.st_mtime_ns)
if cache_key in self.onnx_cache:
    return self.onnx_cache[cache_key]
session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
```

**Trả lời:** Demo tái sử dụng model/session đã load, key theo đường dẫn và thời gian sửa file. Đây là cache đối tượng ứng dụng; khác CPU cache và memory arena. Demo không đặt disable hoặc số thread như benchmark nên tốc độ demo không đại diện trực tiếp cho bảng đo ba graph.

**Câu hỏi:** ONNX đã bỏ hoàn toàn Python/GIL khỏi ứng dụng chưa?

Graph ONNX không chứa pipeline Python huấn luyện, nhưng demo/benchmark vẫn dùng Python để gọi runtime và preprocess. API invocation và pipeline Python vẫn có overhead. Không dùng slide 6 để khẳng định toàn ứng dụng hiện tại không còn Python hoặc luôn gọn vài MB.

## 23. Các đoạn train có thể bị hỏi bổ sung

**Slide 16. Nguồn:** [train.py](../train.py#L125).

```python
val_count = max(1, round(len(shuffled) * val_ratio))
val_count = min(val_count, len(shuffled) - 1)
```

`stratified_split` chia riêng từng lớp, tối thiểu một ảnh train và một ảnh validation/lớp. Với 30 ảnh mỗi lớp và val_ratio 0,2: 24 train + 6 validation mỗi lớp. Seed cố định giúp lặp lại cách chia khi dữ liệu/thứ tự đầu vào không đổi.

```python
for parameter in model.parameters():
    parameter.requires_grad = trainable
for parameter in model.fc.parameters():
    parameter.requires_grad = True
```

`set_backbone_trainable(False)` đóng băng gradient tham số backbone, giữ classifier học. Do training vẫn gọi `model.train()`, BatchNorm running statistics có thể thay đổi; đây chưa phải đóng băng toàn bộ trạng thái backbone.

```python
model.fc = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.fc.in_features, num_classes),
)
```

Classifier có hai logits. Dùng CrossEntropyLoss trực tiếp trên logits; softmax được dùng lúc hiển thị confidence, không thêm trước loss.

## 24. Những câu trên slide cần diễn giải thận trọng

| Slide | Cách nói dễ bị bắt bẻ | Cách trả lời sát thực nghiệm |
|---|---|---|
| 6 | Bỏ toàn bộ Python/GIL, dịch vụ vài MB | Artifact tách khỏi code train; pipeline demo vẫn Python, deployment size chưa được đo |
| 7 | Code dùng assert_allclose trên slide | Đó là ví dụ; converter thực tế dùng max absolute difference ≤ 1e-4 |
| 12 | Fusion triệt tiêu hoàn toàn mọi ghi/đọc RAM | Graph giảm tensor trung gian; chưa đo traffic RAM/cache hoặc mọi buffer nội bộ kernel |
| 12–13 | BatchNorm fusion là riêng extended | Có thể fold ở export/basic trước Conv–Activation fusion extended |
| 15 | Nhóm đã benchmark memory arena/cache | Chưa có thí nghiệm tách riêng các option, cache miss hay I/O Binding |
| 18 | Cố định mọi nhiễu | Cố định cấu hình chính; chưa kiểm soát affinity, nhiệt, xung nhịp và thứ tự chạy |
| 20 | 64 node giảm = 64 fusion | Đây là chênh lệch tổng node giữa basic và all |
| 20 | Dung lượng giảm do nén model | .pth có optimizer; ONNX chỉ inference; không lượng tử hóa trong thí nghiệm |
| 20 | 100% trên 10.000 ảnh | 10 ảnh độc lập lặp 1.000 lần/model |

## 25. Lệnh demo nhanh khi phản biện

Chạy từ thư mục gốc repo. Các lệnh export cần thêm thời gian và sẽ ghi đè artifact đích; ưu tiên dùng artifact đã chuẩn bị khi thuyết trình.

```bash
python midterm/convert_to_onnx.py --help
python midterm/benchmark.py --help
python midterm/demo.py --device cpu
```

Kiểm tra luồng benchmark bằng lượt nhỏ, lưu riêng để không thay báo cáo chính:

```bash
python midterm/benchmark.py --iterations 1 --warmup 1 --threads 1 \
  --output-dir midterm/benchmark_results/demo_run
```

Đọc graph và xác nhận domain/type/count trực tiếp:

```python
from collections import Counter
from pathlib import Path
import onnx

for name in ["dodo_classifier_basic.onnx", "dodo_classifier_extend.onnx", "dodo_classifier.onnx"]:
    model = onnx.load(str(Path("midterm/checkpoints") / name))
    print(name, len(model.graph.node))
    print(Counter((node.domain, node.op_type) for node in model.graph.node))
```

## 26. Câu trả lời mở đầu khi thầy yêu cầu mô tả toàn bộ pipeline

“Nhóm khôi phục ResNet50 hai lớp từ checkpoint và chuyển sang eval. Nhóm export graph ONNX với opset 17, batch động, rồi dùng ONNX Runtime tối ưu offline cho CPU ở ba mức. Khi benchmark, nhóm tắt tối ưu graph bổ sung để giữ ba artifact khác nhau, dùng cùng input float32 và một thread intra-op. Mỗi cấu hình chạy process riêng, warm-up 200 lời gọi rồi đo 10.000 lời gọi. Nhóm báo cáo latency, throughput, RSS, CPU và nhãn dự đoán; kết quả all nhanh hơn PyTorch khoảng 1,61 lần trong môi trường này.”

## Tài liệu đối chiếu

- [Slide Nhóm 06](ONNX_ONNXRuntime_Nhom06.pptx.pdf)
- [Báo cáo benchmark](../benchmark_results/benchmark_report.md)
- [Dữ liệu benchmark JSON](../benchmark_results/benchmark_report.json)
- [ONNX Runtime: mức tối ưu, online/offline và tương thích phần cứng](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [ONNX Runtime Python API: SessionOptions, InferenceSession và I/O Binding](https://onnxruntime.ai/docs/api/python/api_summary.html)
