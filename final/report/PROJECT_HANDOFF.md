# EdgeFace — Tài liệu tổng hợp hệ thống AIoT nhận diện thành viên

**Mục đích:** Tài liệu tổng hợp bài toán, kiến trúc, dữ liệu, phương pháp và kết quả thực nghiệm của EdgeFace, phục vụ việc bàn giao trong nhóm và chuẩn bị báo cáo cuối kỳ 20–35 trang cùng bài thuyết trình. Nội dung được trình bày theo những thành phần đã triển khai và bằng chứng đang lưu trong dự án; các giới hạn đánh giá được nêu cùng kết quả liên quan.

**Mốc tham chiếu:** 21/10/2026. Tại mốc này, Raspberry Pi sử dụng gói mô hình (bundle) **5010dc47644c**, với mã ONNX **1e170d7902b8**. Các phép đo hiệu năng được thực hiện ngày 20/10/2026. Dữ liệu khuôn mặt và checkpoint huấn luyện được lưu cục bộ, không đính kèm tài liệu.

## 1. Giới thiệu đề tài

**EdgeFace — Hệ thống nhận diện thành viên bằng Deep Learning trên Raspberry Pi, điều khiển qua Telegram và giao tiếp MQTT**

EdgeFace sử dụng camera USB để nhận diện các thành viên đã đăng ký. Raspberry Pi thực hiện toàn bộ chuỗi xử lý ảnh, gồm phát hiện khuôn mặt, tiền xử lý và phân loại danh tính. Laptop đảm nhiệm huấn luyện mô hình và vận hành MQTT broker cùng Telegram Bot. Người dùng điều khiển thiết bị và nhận kết quả qua Telegram; MQTT Explorer hỗ trợ quan sát trực tiếp các bản tin trên broker.

## 2. Tổng quan bài toán

### 2.1 Bài toán và bối cảnh thực tế

Từ ảnh camera, hệ thống xác định vùng khuôn mặt rồi phân loại vào một trong năm thành viên đã đăng ký hoặc lớp Unknown. Thiết kế mô phỏng một trạm nhận diện tại cửa phòng học hoặc phòng thí nghiệm, có thể phục vụ điểm danh nhóm và thông báo thành viên. Phạm vi hiện tại dừng ở nhận diện và phản hồi: hệ thống chưa điều khiển khóa cửa hoặc cấp quyền truy cập vật lý, do đó chưa thể thay thế một hệ thống an ninh.

Bài toán chịu ảnh hưởng của ánh sáng, góc mặt và khoảng cách đến camera. Dữ liệu từ video còn có nhiều khung hình gần giống nhau, trong khi số phiên quay độc lập của mỗi thành viên còn ít. Khi triển khai, hệ thống phải xử lý thêm người ngoài nhóm, giới hạn CPU/RAM của Raspberry Pi và sự thay đổi mạng kết nối tại nơi trình diễn.

### 2.2 Mục tiêu hệ thống

Về mô hình, mục tiêu là xây dựng dữ liệu khuôn mặt từ video thành viên, kết hợp phiên thu ảnh trực tiếp bằng camera Pi của Sơn; tinh chỉnh mô hình phân loại đã huấn luyện sẵn trên laptop có GPU; và chọn ngưỡng từ chối Unknown bằng tập xác thực (validation). Mô hình sau đó được xuất sang ONNX để suy luận bằng CPU ngay tại Raspberry Pi.

Về hệ thống, MQTT truyền lệnh điều khiển, trạng thái, số đo vận hành (telemetry) và kết quả nhận diện; video và khung hình không được truyền qua giao thức này. Telegram cung cấp giao diện điều khiển và thông báo, còn MQTT Explorer hỗ trợ giám sát luồng bản tin. Chất lượng phân loại, độ trễ, thông lượng, RAM và kích thước tệp mô hình được đánh giá với phạm vi đo xác định rõ.

### 2.3 Phạm vi phiên bản hiện tại (MVP)

Mô hình sử dụng sáu nhãn nội bộ: Unknown, dung, hung, son, thanh và tuan. Trong phần diễn giải, các thành viên được gọi là Dung, Hùng, Sơn, Thanh và Tuấn. Camera chỉ mở khi có lệnh bắt đầu nhận diện hoặc thu thập dữ liệu, sau đó được giải phóng khi dừng, hoàn tất thu thập hoặc xảy ra lỗi.

Khi nhận diện một thành viên đã đăng ký, Bot gửi lệnh dừng phiên tương ứng và phản hồi kết quả qua Telegram. Nếu dự đoán Unknown, phiên nhận diện tiếp tục mà không gửi tin nhắn hoặc ảnh. Tính năng gửi ảnh hiện sử dụng JPEG có khung mặt (bounding box), tên và độ tin cậy phân loại (confidence); ảnh được tải qua HTTP riêng trong mạng Tailscale. Phiên bản này chưa bật âm thanh.

## 3. Cơ sở lý thuyết và nghiên cứu liên quan

### 3.1 Phát hiện khuôn mặt và phân loại danh tính

Phát hiện khuôn mặt trả lời câu hỏi “khuôn mặt nằm ở đâu trong ảnh”, còn phân loại danh tính xác định người trong vùng ảnh đó. Tách hai nhiệm vụ giúp giảm ảnh hưởng của nền và cho phép nhận biết lỗi ở từng bước. EdgeFace dùng YuNet để dự đoán khung mặt cùng các điểm mốc (landmarks), rồi đưa vùng mặt đã cắt và căn chỉnh vào MobileNetV3-Small.

YuNet được chọn vì hướng tới phát hiện khuôn mặt với mô hình gọn nhẹ, phù hợp với tài nguyên thiết bị biên. Đồ án sử dụng trọng số huấn luyện sẵn từ OpenCV Zoo và thực thi qua OpenCV trên Pi; detector không được tinh chỉnh trong dự án. Cơ sở lựa chọn và mô hình tham chiếu được trình bày trong [bài báo YuNet](https://link.springer.com/article/10.1007/s11633-023-1423-y) và [model card của OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

### 3.2 Học chuyển giao và MobileNetV3-Small

Học chuyển giao (transfer learning) sử dụng trọng số đã học từ một tập dữ liệu lớn làm điểm khởi đầu cho bài toán mới. Khi dữ liệu mục tiêu còn hạn chế, có thể huấn luyện bộ phân loại trước rồi tinh chỉnh một phần mạng trích xuất đặc trưng, thay vì ước lượng toàn bộ trọng số từ đầu. EdgeFace áp dụng cách tiếp cận này cho dữ liệu khuôn mặt của nhóm.

MobileNetV3 hướng đến bài toán thị giác trên thiết bị có tài nguyên hạn chế, kết hợp khối phần dư đảo (inverted residual), tích chập theo chiều sâu (depthwise convolution) và các hàm kích hoạt phù hợp. Đồ án chọn MobileNetV3-Small với trọng số ImageNet, thay lớp tuyến tính cuối bằng lớp sáu đầu ra. Sau thay đổi này, mô hình có **1.524.006 tham số**. Thiết kế MobileNetV3-Small được mô tả trong [Howard và cộng sự, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf).

Trong huấn luyện, mô hình dự đoán sáu giá trị logits. Phép softmax chuyển chúng thành các điểm tin cậy tương đối giữa các lớp. Các điểm này chưa được hiệu chuẩn xác suất: confidence bằng 0,9 không đủ để kết luận rằng 90% trường hợp tương tự sẽ được nhận diện đúng.

### 3.3 Lớp Unknown và bài toán nhận diện tập mở

Phân loại trong tập đóng giả định ảnh kiểm tra thuộc một trong các lớp đã biết. Khi gặp người chưa đăng ký, giả định này không còn phù hợp. EdgeFace bổ sung lớp Unknown và áp dụng ngưỡng từ chối theo quy tắc sau:

1. Nếu lớp có điểm softmax cao nhất là Unknown, trả về Unknown.
2. Nếu lớp cao nhất là thành viên nhưng confidence thấp hơn ngưỡng, cũng trả về Unknown.
3. Trường hợp còn lại trả tên thành viên.

Kết hợp lớp Unknown với ngưỡng từ chối tạo thành một phương án cơ sở cho nhận diện trong tập mở (open-set recognition), nhưng chưa bảo đảm phát hiện mọi người chưa đăng ký. [Scheirer và cộng sự, IEEE TPAMI](https://pubmed.ncbi.nlm.nih.gov/23682001/) đặt vấn đề đánh giá khi dữ liệu vận hành có thể chứa các lớp chưa được biết trong huấn luyện. Trong EdgeFace, các danh tính LFW của lớp Unknown được tách riêng giữa tập huấn luyện, xác thực và kiểm tra; tuy nhiên, tập kiểm tra chỉ có 21 ảnh và khác miền dữ liệu camera Pi.

Một hướng tiếp cận khác là học biểu diễn khuôn mặt (embedding) rồi so sánh khoảng cách trong không gian đặc trưng. Chẳng hạn, FaceNet sử dụng embedding Euclidean và triplet loss, như mô tả trong [Schroff và cộng sự, CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf). EdgeFace hiện dùng bộ phân loại sáu lớp để phù hợp với quy mô nhóm nhỏ và quy trình tinh chỉnh, xuất mô hình, triển khai đã xây dựng.

### 3.4 ONNX và tối ưu suy luận

ONNX biểu diễn đồ thị tính toán của mô hình, giúp tách framework huấn luyện khỏi bộ thực thi khi triển khai. Trong EdgeFace, PyTorch đảm nhiệm tinh chỉnh; bộ phân loại được xuất với opset 17 và đầu vào cố định 1×3×224×224. Raspberry Pi nạp tệp ONNX bằng ONNX Runtime với CPUExecutionProvider.

ONNX Runtime hỗ trợ tối ưu đồ thị khi khởi tạo phiên suy luận, gồm các phép biến đổi như tính trước biểu thức hằng (constant folding), tùy cấu hình và phép toán được hỗ trợ; xem [tài liệu tối ưu của ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/). Quy trình hiện tại bật constant folding khi xuất và ORT_ENABLE_ALL khi chạy. Hai phiên bản đối sánh đều giữ trọng số FP32; dự án chưa áp dụng cắt tỉa (pruning), lượng tử hóa INT8, FP16 hoặc chưng cất tri thức (knowledge distillation).

### 3.5 MQTT và mô hình publish/subscribe

MQTT tổ chức trao đổi bản tin theo mô hình xuất bản/đăng ký (publish/subscribe). Thành phần gửi xuất bản bản tin lên một chủ đề (topic); broker chuyển bản tin đến các thành phần đã đăng ký bộ lọc topic phù hợp. Nhờ đó, Pi và Bot trao đổi qua broker mà không cần duy trì kết nối ứng dụng trực tiếp. MQTT 3.1.1 quy định ba mức chất lượng dịch vụ: QoS 0 — tối đa một lần; QoS 1 — ít nhất một lần, có thể lặp; QoS 2 — đúng một lần trong phạm vi luồng giao thức MQTT, với chi phí bắt tay bổ sung. Căn cứ là [đặc tả OASIS MQTT 3.1.1](https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/mqtt-v3.1.1.html).

Bảo đảm truyền bản tin của QoS 2 không đồng nghĩa với bảo đảm một tác vụ ứng dụng chỉ được thực hiện một lần, cũng không thay thế xác thực hoặc mã hóa. EdgeFace sử dụng QoS 1 và kiểm soát ở tầng ứng dụng bằng request_id, thời điểm gửi, thời hạn hiệu lực (TTL), từ chối lệnh retained, clean session và sổ yêu cầu SQLite trên Pi. Xác thực tài khoản, danh sách kiểm soát truy cập (ACL) của broker và kết nối Tailscale bổ sung giới hạn truy cập; QoS không được coi là cơ chế mật mã chống phát lại.

## 4. Thiết kế kiến trúc hệ thống

### 4.1 Kiến trúc tổng thể

![Kiến trúc EdgeFace và ba đường trao đổi MQTT, HTTP, SSH](assets/figures/system_architecture.png)

**Hình 1.** Kiến trúc triển khai của EdgeFace. Raspberry Pi thực hiện toàn bộ suy luận; laptop vận hành huấn luyện và các dịch vụ điều khiển. MQTT truyền lệnh và metadata, HTTP chuyển ảnh kết quả khi được bật, còn SSH phục vụ triển khai mô hình. Người dùng và Telegram Cloud nằm ngoài hai máy triển khai.

Bản [SVG](assets/figures/system_architecture.svg) và [nguồn Mermaid](assets/figures/system_architecture.mmd) được cung cấp để tái sử dụng hoặc chỉnh sửa sơ đồ.

Kiến trúc tách ba đường trao đổi: MQTT cho lệnh và metadata; SSH cho triển khai mô hình; HTTP có token cho ảnh kết quả. Đường laptop–Pi nằm trong Tailscale, còn Telegram Bot kết nối Telegram Cloud qua Internet. Các khung hình phục vụ suy luận được xử lý ngay trên Pi.

### 4.2 Luồng chuẩn bị dữ liệu và huấn luyện

![Quy trình chuẩn bị dữ liệu, huấn luyện, xác thực và triển khai](assets/figures/data_pipeline.png)

**Hình 2.** Quy trình từ dữ liệu đến mô hình trên Pi. Ảnh thành viên được cắt vùng mặt và lọc chất lượng trước khi huấn luyện. Lớp Unknown được tách theo danh tính; dữ liệu thành viên được chia theo đoạn thời gian có khoảng loại trừ. Checkpoint và ngưỡng chỉ được chọn bằng tập xác thực, sau đó mới đánh giá trên tập kiểm tra.

Bản [SVG](assets/figures/data_pipeline.svg) và [nguồn Mermaid](assets/figures/data_pipeline.mmd) được cung cấp để tái sử dụng hoặc chỉnh sửa sơ đồ.

### 4.3 Luồng nhận diện và phản hồi

![Luồng điều khiển nhận diện, phân nhánh Unknown và tự dừng đúng phiên](assets/figures/recognition_workflow.png)

**Hình 3.** Luồng nhận diện và phản hồi. Sau khi Pi xác nhận lệnh bắt đầu, ảnh được xử lý cục bộ. Unknown không tạo thông báo và phiên tiếp tục. Với thành viên đã đăng ký, Bot gửi lệnh dừng đúng phiên rồi phản hồi văn bản và ảnh; ảnh được tải riêng qua HTTP. Việc không phát hiện mặt hoặc có nhiều mặt được xử lý trước classifier.

Bản [SVG](assets/figures/recognition_workflow.svg) và [nguồn Mermaid](assets/figures/recognition_workflow.mmd) được cung cấp để tái sử dụng hoặc chỉnh sửa sơ đồ.

### 4.4 Thành phần phần mềm

| Thành phần | Nơi chạy | Trách nhiệm |
|---|---|---|
| Telegram Bot | Container trên laptop | Nhận lệnh qua long polling, gửi lệnh lên MQTT, nhận xác nhận/kết quả và phản hồi người dùng |
| Mosquitto | Container trên laptop | Môi giới bản tin; xác thực tài khoản, ACL, LWT và phân phối theo topic |
| MQTT Explorer | Container trên laptop | Giao diện web chỉ đọc để xem topic/payload; mở tại http://localhost:4000 |
| Edge agent | Python native + systemd trên Pi | Điều khiển camera, phát hiện mặt, phân loại ONNX, quản lý trạng thái và telemetry |
| YuNet | OpenCV DNN trên Pi | Phát hiện khuôn mặt và điểm mốc |
| MobileNetV3-Small ONNX | ONNX Runtime CPU trên Pi | Phân loại sáu lớp và từ chối Unknown |
| Huấn luyện, xuất và đánh giá | Python/PyTorch trên laptop | Chia dữ liệu, huấn luyện, hiệu chỉnh ngưỡng, đánh giá và xuất ONNX |
| Endpoint ảnh kết quả | Pi, chỉ gắn địa chỉ Tailnet | Trả JPEG có chú thích và token; giữ trong RAM, không ghi ảnh kết quả ra ổ đĩa |

### 4.5 Mạng và triển khai

Laptop/WSL2 vận hành Mosquitto, Telegram Bot và MQTT Explorer bằng Docker Compose. Trên Pi, chương trình điều khiển (edge agent) chạy Python trực tiếp dưới systemd để truy cập camera và các thư viện hệ thống. Kết nối Tailscale giúp hai máy liên lạc qua địa chỉ trong Tailnet khi địa chỉ Wi-Fi thay đổi, với điều kiện mỗi máy vẫn có mạng và Tailscale hoạt động.

Pi kết nối broker qua cơ chế chuyển tiếp TCP trong Tailscale; cổng broker trên máy chủ chỉ gắn với địa chỉ loopback. Cấu hình tham chiếu không dùng SSH tunnel cho MQTT. SSH qua Tailnet phục vụ triển khai và kiểm tra nhật ký. Telegram Bot dùng long polling nên không cần webhook, địa chỉ IP công khai hoặc tên miền.

## 5. Hạ tầng dữ liệu và giao thức truyền thông

### 5.1 Nguồn dữ liệu

| Nguồn | Nội dung | Số liệu của phiên bản hiện tại |
|---|---|---|
| Video Dung | dataset/videos/dung/dung.MOV, khoảng 60,65 giây | 71 huấn luyện / 25 xác thực / 23 kiểm tra |
| Video Hùng | dataset/videos/hung/hung.mp4, khoảng 67,39 giây | 92 / 26 / 30 |
| Video Sơn | dataset/videos/son/1.mp4 và 2.mp4, khoảng 17,47 và 61,36 giây | Kết hợp vùng mặt từ video và camera |
| Video Thanh | dataset/videos/thanh/thanh.mp4, khoảng 52,55 giây | 71 / 24 / 24 |
| Video Tuấn | dataset/videos/tuan/tuan.MOV, khoảng 49,96 giây | 73 / 18 / 19 |
| Camera Pi | Một phiên thu có sự đồng thuận của Sơn, 150 vùng mặt hợp lệ | Thêm 78 huấn luyện / 19 xác thực / 26 kiểm tra; 27 ảnh bị loại trong purge gap |
| Unknown | 167 vùng mặt từ 60 danh tính thuộc Labeled Faces in the Wild | 120 huấn luyện / 26 xác thực / 21 kiểm tra; tách danh tính giữa các tập |

Video thành viên do nhóm cung cấp. Dữ liệu Unknown lấy từ Labeled Faces in the Wild (LFW), tập dữ liệu khuôn mặt trong điều kiện không kiểm soát, được giới thiệu trong [Huang và cộng sự, UMass](https://people.cs.umass.edu/~elm/papers/Huang_eccv2008-lfw.pdf). Mục đích nghiên cứu ban đầu của LFW là xác minh khuôn mặt; việc sử dụng làm lớp Unknown ở đây là lựa chọn dữ liệu của dự án. Ảnh LFW chưa đại diện đầy đủ cho người lạ trước camera Pi. Tính công khai của nguồn cũng không đồng nghĩa với xác nhận đồng thuận của từng cá nhân, vì vậy ảnh không được phát tán trong Git hoặc bài trình chiếu.

### 5.2 Thu thập và xử lý ảnh

Với video điện thoại, chương trình đọc thông tin hướng quay trước khi giải mã và lấy mẫu khung hình khoảng mỗi 0,25 giây. YuNet phát hiện khuôn mặt; chỉ khung hình có đúng một mặt mới được đưa vào xử lý tiếp. Vùng mặt được căn chỉnh theo mắt khi điểm mốc phù hợp, mở rộng biên cắt và kiểm tra chất lượng trước khi lưu.

Cấu hình lọc video yêu cầu kích thước mặt tối thiểu khoảng 80 px trên ảnh gốc. Độ nét được ước lượng bằng phương sai Laplacian trên ảnh tham chiếu 224×224; ảnh có giá trị dưới 60 bị loại. Bộ lọc độ sáng loại ảnh có giá trị trung bình dưới 35 hoặc trên 220. Khoảng cách Hamming của dHash không quá 2 được dùng để tìm các vùng mặt gần trùng. Các ngưỡng này thuộc cấu hình dự án, cần điều chỉnh nếu thay đổi camera hoặc điều kiện thu.

Mỗi video được chia theo thời gian: 0–50% dành cho huấn luyện, 60–75% cho xác thực và 85–100% cho kiểm tra. Hai khoảng 50–60% và 75–85% bị loại để tạo khoảng cách giữa các tập (purge gap); ảnh gần trùng qua ranh giới tập tiếp tục được loại. Cách chia này hạn chế rò rỉ từ khung hình liền kề so với chia ngẫu nhiên, nhưng ảnh thành viên ở các tập vẫn thuộc cùng bản ghi hoặc phiên thu. Vì vậy manifest ghi independent_known_test=false. Riêng Unknown được chia theo danh tính, bảo đảm không có người trùng giữa ba tập.

Trong phiên thu ảnh Pi của Sơn, camera thương lượng định dạng MJPEG. Picamera2 giải mã thành RGB, trong khi adapter camera trước đó giả định BGR, khiến hai kênh đỏ và xanh bị đảo. Adapter đã được sửa theo định dạng điểm ảnh thực tế. Dự án giữ nguyên 150 ảnh gốc và tạo các ảnh PNG dẫn xuất đã đổi kênh, kèm SHA256 nguồn/đích; phép sửa này chỉ áp dụng cho phiên bị ảnh hưởng.

Ngoài 150 ảnh hợp lệ, phiên thu của Sơn ghi nhận 767 khung hình bị loại: 277 chưa đạt khoảng thời gian lấy mẫu, 406 gần trùng, 79 không có mặt, 4 bị mờ và 1 có mặt quá nhỏ. Ảnh được lưu cục bộ với quyền truy cập hạn chế. Thu thập qua camera yêu cầu xác nhận đồng thuận; người ngoài nhóm phải ra khỏi khung hình và nhãn thu thập không được tự suy từ dự đoán của mô hình.

### 5.3 Bộ dữ liệu hiện tại và lưu trữ

| Lớp | Huấn luyện | Xác thực | Kiểm tra | Tổng |
|---|---:|---:|---:|---:|
| Unknown | 120 | 26 | 21 | 167 |
| Dung | 71 | 25 | 23 | 119 |
| Hùng | 92 | 26 | 30 | 148 |
| Sơn | 214 | 54 | 64 | 332 |
| Thanh | 71 | 24 | 24 | 119 |
| Tuấn | 73 | 18 | 19 | 110 |
| **Tổng** | **641** | **173** | **181** | **995** |

Phiên bản dữ liệu dùng cho lần huấn luyện hiện tại nằm tại dataset/revisions/team_20261020_son_live/processed, với mã SHA256 manifest bắt đầu bằng e49dc56d. Dữ liệu gốc và ảnh dẫn xuất được lưu trong dataset/raw và dataset/revisions; checkpoint nằm trong runs; các gói suy luận nằm trong models/releases. Liên kết models/active chọn gói đang dùng, đồng thời hỗ trợ quay về phiên bản trước khi cần.

Ảnh khuôn mặt được giữ ngoài Git. Cơ sở dữ liệu SQLite tại dataset/requests.sqlite3 trên Pi lưu request_id và phản hồi ACK để hạn chế thực thi lặp, không lưu lịch sử telemetry. Mosquitto cấu hình persistence=false: bản tin retained cung cấp trạng thái gần nhất khi broker đang chạy, nhưng không tạo kho dữ liệu bền vững qua lần khởi động lại broker. Telemetry được gửi định kỳ bằng QoS 0; MQTT Explorer phục vụ quan sát trực tiếp. Hệ thống chưa có kho lịch sử telemetry dài hạn; cũng chưa triển khai các hạ tầng lưu trữ hoặc luồng dữ liệu như InfluxDB/Kafka.

### 5.4 Chủ đề, bản tin và chính sách MQTT

Các topic dùng tiền tố aiot/edgeface. Payload JSON mang device_id và timestamp UTC theo ISO-8601; lệnh điều khiển có request_id theo UUID để liên kết yêu cầu với phản hồi. Ví dụ dưới đây minh họa cấu trúc bản tin, trong đó uuid là giá trị thay thế cho mã yêu cầu thực tế:

~~~json
{
  "request_id": "uuid",
  "device_id": "raspberrypi-01",
  "action": "start",
  "timestamp": "2026-10-20T14:22:16Z"
}
~~~

| Topic | Hướng chính | QoS khi gửi | Retained | Nội dung |
|---|---|---:|---|---|
| aiot/edgeface/command | Bot → Pi | 1 | Không | start, stop, status, members, collect, collect_unknown |
| aiot/edgeface/status | Pi → Bot/UI | 1 | Có cho trạng thái; ACK từng lệnh không retained | online, state, camera, phiên bản mô hình, CPU/RAM/nhiệt độ, lỗi, ACK |
| aiot/edgeface/result | Pi → Bot/UI | 1 | Không | recognition, no_face, multiple_faces; confidence và thời gian suy luận |
| aiot/edgeface/telemetry | Pi → Bot/UI | 0 | Không | số đo định kỳ, mặc định mỗi 10 giây |
| aiot/edgeface/collection | Pi → Bot/UI | 1 | Không | tiến độ và kết quả thu thập dữ liệu |

Lệnh điều khiển, sự kiện nhận diện và tiến độ thu thập dùng QoS 1. Telemetry dùng QoS 0 để ưu tiên số đo mới và tránh dồn bản tin khi mất kết nối. MQTT client của Bot và Pi sử dụng clean session, tự kết nối lại với khoảng chờ tăng dần từ 1 đến 30 giây, keepalive 30 giây và hàng đợi tối đa 100 bản tin. Khi kết nối Pi bị ngắt bất thường, broker phát bản tin báo mất kết nối đã cấu hình trước (Last Will and Testament — LWT), retained với trạng thái offline.

Pi từ chối lệnh retained, payload vượt 16 KiB, action không hợp lệ, device_id không khớp, timestamp hết hạn hoặc lệch đồng hồ quá mức, và lệnh thu thập thiếu xác nhận đồng thuận. TTL mặc định là 60 giây; Bot chờ ACK tối đa 15 giây. Trước khi mở camera hoặc đổi trạng thái, Pi ghi nhận request_id vào SQLite để yêu cầu đã xử lý không được thực thi lại sau khởi động. Cơ chế này xử lý nguy cơ lặp ở tầng ứng dụng, chưa phải chữ ký số hay xác thực mật mã cho từng payload.

Broker không cho truy cập ẩn danh và kiểm tra mật khẩu cùng ACL. Tài khoản edgeface_bot được gửi lệnh và đọc các topic phản hồi; edgeface_pi nhận lệnh và gửi trạng thái, kết quả, telemetry, tiến độ; edgeface_monitor chỉ được đọc aiot/edgeface/#. Thông tin xác thực nằm trong file .env cục bộ và tệp mật khẩu đã băm của broker.

Bản tin nhận diện chỉ chứa tham chiếu ảnh và metadata như image_id, kiểu nội dung, kích thước và khung mặt. Bot dùng token để tải JPEG qua endpoint HTTP trong Tailnet. Như vậy, tính năng gửi ảnh bổ sung một đường dữ liệu riêng mà không đưa ảnh vào MQTT.

## 6. Mô hình AI và tối ưu hóa suy luận biên

### 6.1 Chuỗi suy luận trên Raspberry Pi

~~~text
Frame camera 640x480
  → kiểm tra format và chuyển đúng BGR/RGB
  → YuNet phát hiện mặt và landmarks
  → crop mặt, căn chỉnh khi landmark phù hợp
  → chuyển RGB, resize PIL bilinear 224x224
  → scale về [0,1], chuẩn hóa mean/std ImageNet, NCHW float32
  → MobileNetV3-Small ONNX, sáu logits
  → softmax + lớp Unknown hoặc confidence threshold
  → kết quả MQTT; ảnh annotation tùy chọn qua HTTP/Tailscale
~~~

Bộ phân loại chỉ chạy khi khung hình có đúng một khuôn mặt. Nếu không có mặt hoặc có nhiều mặt, hệ thống phát sự kiện tương ứng; trong chế độ thu thập, khung hình đó bị loại để tránh gán nhãn tùy tiện. Tiền xử lý giữa huấn luyện và suy luận thống nhất: RGB, 224×224, nội suy PIL bilinear, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225], bố trí NCHW và kiểu float32.

### 6.2 Hai mô hình AI

| Mô hình | Vai trò | Huấn luyện trong đồ án | Bộ thực thi |
|---|---|---|---|
| YuNet huấn luyện sẵn | Phát hiện mặt, khung mặt và điểm mốc | Giữ nguyên trọng số | OpenCV DNN CPU trên Pi |
| MobileNetV3-Small khởi tạo từ ImageNet | Phân loại Unknown và năm danh tính | Tinh chỉnh trên vùng mặt đã cắt | ONNX Runtime CPU trên Pi |

YuNet và MobileNetV3-Small đảm nhiệm hai nhiệm vụ nối tiếp. Trong phép đo phát lại khung hình, phát hiện khuôn mặt chiếm phần lớn độ trễ; vì vậy cải thiện riêng bộ phân loại chưa thể được xem là mức cải thiện tương ứng cho toàn bộ chuỗi xử lý.

#### 6.2.1 Kiến trúc YuNet

![Kiến trúc bộ phát hiện khuôn mặt YuNet](assets/figures/yunet_architecture.png)

**Hình 4.** Kiến trúc YuNet tái dựng từ đồ thị ONNX 2023mar đang dùng trong dự án. Mạng gồm backbone gọn nhẹ, phần hợp nhất đặc trưng đa tỉ lệ và ba nhánh dự đoán ở stride 8/16/32. Mỗi nhánh tạo điểm phân loại, điểm hiện diện đối tượng (objectness), khung mặt và năm điểm mốc, tương ứng 10 tọa độ. Kích thước trong hình ứng với đầu vào tham chiếu 640×640 của đồ thị; khi chạy trên Pi, OpenCV FaceDetectorYN nhận kích thước khung hình thực tế 640×480. Ngưỡng phát hiện 0,85 và ngưỡng NMS IoU 0,30 độc lập với ngưỡng từ chối danh tính.

Bản [SVG YuNet](assets/figures/yunet_architecture.svg) hỗ trợ phóng lớn khi in hoặc đưa vào bài trình chiếu. Hình mô tả kiến trúc mô hình, không sử dụng tốc độ công bố trong bài báo YuNet làm kết quả đo trên Pi.

#### 6.2.2 Kiến trúc MobileNetV3-Small

![Kiến trúc MobileNetV3-Small sáu lớp](assets/figures/mobilenetv3_small_architecture.png)

**Hình 5.** Kiến trúc MobileNetV3-Small sau khi thay classifier thành sáu đầu ra. Hình thể hiện kích thước tensor theo từng nhóm khối, phép gộp đặc trưng và classifier 576→1024→6. Khối inverted residual gồm mở rộng kênh, tích chập depthwise, squeeze-and-excitation khi được bật và phép chiếu tuyến tính. Kết nối tắt chỉ xuất hiện khi stride bằng 1 và số kênh phù hợp. Mô hình có 1.524.006 tham số; softmax và ngưỡng từ chối nằm ngoài đồ thị ONNX. Dropout không hoạt động khi suy luận.

Bản [SVG MobileNetV3-Small](assets/figures/mobilenetv3_small_architecture.svg) hỗ trợ tái sử dụng hình. Tệp [provenance.json](assets/figures/provenance.json) ghi nguồn, mã SHA256 của mô hình và dữ liệu tạo hình, cùng kích thước tensor. Các hình được dựng bằng [render_report_figures.py](../scripts/render_report_figures.py) từ mô hình và kết quả đã lưu, không suy ra kiến trúc chỉ từ tên mạng.

### 6.3 Huấn luyện bộ phân loại

Lần huấn luyện đầu khởi tạo MobileNetV3-Small từ trọng số ImageNet và thay lớp phân loại cuối. Bộ trích xuất đặc trưng được đóng băng khi học classifier, sau đó mở các khối cuối để tinh chỉnh. Lần huấn luyện hiện tại tiếp tục từ checkpoint sáu lớp trước đó, bổ sung dữ liệu camera của Sơn và giữ nguyên tiền xử lý.

Lần huấn luyện hiện tại dùng seed 42, batch size 16, AdamW và Cross-Entropy có trọng số để giảm ảnh hưởng của mất cân bằng lớp. Tăng cường dữ liệu gồm lật ngang, biến đổi affine nhẹ và ColorJitter; thống kê trượt của BatchNorm được giữ cố định.

Classifier được huấn luyện trong 3 epoch với learning rate 1e-4, tiếp theo là tối đa 10 epoch tinh chỉnh các khối cuối cùng classifier với learning rate 5e-5 và cơ chế dừng sớm với patience 4.

GPU thực tế là NVIDIA RTX 2050 4 GiB trên WSL2, có CUDA. Quá trình kéo dài 107,56 giây; checkpoint tốt nhất ở epoch 11. Cả trọng số classifier và các khối cuối đều thay đổi, xác nhận đã có cập nhật mô hình.

Trên 173 ảnh xác thực, checkpoint đầu vào đạt loss 9,06209e-5 và accuracy 100%; checkpoint tốt nhất sau tinh chỉnh đạt loss 2,98236e-7 và accuracy 100%. Loss xác thực giảm trong lần huấn luyện này, nhưng accuracy không đổi. Kết quả chưa cho thấy khả năng khái quát tốt hơn ở các phiên thu độc lập.

![Diễn biến loss huấn luyện và xác thực](assets/figures/training_loss.png)

**Hình 6.** Diễn biến loss qua 13 epoch của lần huấn luyện team_20261020_son_live, gồm ba epoch huấn luyện classifier và mười epoch tinh chỉnh. Dấu sao đánh dấu checkpoint tốt nhất ở epoch 11, với loss xác thực 2,98236e-7. Trục tung dùng thang log; mọi điểm đo được giữ nguyên, không làm mượt. Huấn luyện sử dụng tăng cường dữ liệu và Dropout, còn xác thực chạy ở chế độ eval, nên hai đường loss không nhất thiết có cùng độ lớn. Quá trình tiếp tục từ checkpoint đã huấn luyện; tập kiểm tra không tham gia chọn epoch.

Nguồn dữ liệu của Hình 6 là [history.json](experiments/team_20261020_son_live/history.json) và [training_run.json](experiments/team_20261020_son_live/training_run.json); bản [SVG](assets/figures/training_loss.svg) phục vụ in và trình chiếu. Loss thấp cùng accuracy 100% trong các phiên hiện có chưa thay thế kiểm thử bằng phiên thu độc lập.

### 6.4 Hiệu chỉnh ngưỡng từ chối Unknown

Ngưỡng từ chối hiện tại là **0,5833254400**, được chọn trên 173 ảnh xác thực để tối đa hóa độ chính xác cân bằng giữa thành viên và Unknown, với ràng buộc tỷ lệ chấp nhận nhầm người lạ (FAR) trên tập xác thực không quá 5%. Tập kiểm tra không tham gia chọn ngưỡng hoặc checkpoint. Khi vận hành, kết quả là Unknown nếu lớp có điểm cao nhất là Unknown hoặc confidence cao nhất thấp hơn ngưỡng.

Lớp Unknown chỉ bao quát một phần các danh tính ngoài nhóm. Với người lạ chưa gặp, mô hình vẫn có thể dự đoán một thành viên với confidence cao. Do đó, cần kiểm tra bằng người lạ trước camera thực tế, có nhãn đối chứng và sự đồng thuận. Khi phân phối ảnh thay đổi, dữ liệu và ngưỡng cần được cập nhật bằng tập xác thực mới.

### 6.5 Xuất mô hình và tối ưu ONNX

Quy trình xuất mô hình kiểm tra nguồn gốc và sự nhất quán giữa checkpoint, ngưỡng và manifest. Bộ phân loại được xuất với đầu vào cố định 1×3×224×224, opset 17, ONNX IR 8 và constant folding. ONNX checker kiểm tra tính hợp lệ; phép đối chiếu trên cùng đầu vào so sánh logits, argmax và quyết định từ chối giữa PyTorch với ONNX Runtime. Gói triển khai gồm mô hình và các tệp classes.json, preprocessing.json, threshold.json, export_validation.json.

Pi sử dụng ONNX Runtime 1.20.1, CPUExecutionProvider và 2 luồng intra-op. Phép đo hiệu năng cấu hình thực thi tuần tự cùng ORT_ENABLE_ALL. Đối chiếu ONNX trên laptop với ONNX trên Pi bằng 181 ảnh kiểm tra cho thấy các quyết định giống nhau; sai khác logit lớn nhất là 2,81334e-5, đáp ứng allclose với rtol=1e-3 và atol=1e-4. Mô hình đã vượt qua ONNX checker.

Xuất sang ONNX giữ nguyên sáu lớp và ý nghĩa đầu ra. Vì trọng số vẫn FP32, kết quả hiện tại phản ánh việc chuyển bộ thực thi và tối ưu đồ thị, chưa phản ánh lợi ích của lượng tử hóa hoặc cắt tỉa trọng số.

### 6.6 Chẩn đoán sai lệch màu và cập nhật mô hình

Ở phiên bản đầu, hiện tượng nhận Sơn thành Thanh dẫn đến việc kiểm tra đường xử lý màu. Camera USB thương lượng MJPEG, Picamera2 giải mã thành RGB nhưng adapter cũ giả định BGR. Trên 26 vùng mặt cuối của cùng phiên thu, mô hình cũ với màu sai nhận đúng 18/26; sửa màu đạt 24/26; sửa màu kết hợp tinh chỉnh bổ sung đạt 26/26. Đây là phép chẩn đoán trong cùng phiên, chưa phải kiểm thử độc lập; không thể quy toàn bộ mức cải thiện cho tinh chỉnh mô hình.

### 6.7 Đối chứng phương pháp huấn luyện

Lần huấn luyện nhóm ban đầu đối chiếu hai phương án: chỉ huấn luyện classifier trên backbone ImageNet đóng băng (head-only), và tinh chỉnh thêm các khối cuối. Hai phương án dùng cùng phiên bản dữ liệu gồm 563 ảnh huấn luyện, 154 ảnh xác thực, 155 ảnh kiểm tra; ngưỡng từ chối được hiệu chỉnh riêng cho mỗi mô hình.

| Chỉ số trên tập kiểm tra ban đầu | Chỉ huấn luyện classifier | Tinh chỉnh khối cuối |
|---|---:|---:|
| Accuracy | 155/155 | 155/155 |
| Macro F1 | 1,000 | 1,000 |
| Accuracy của thành viên | 134/134 | 134/134 |
| Từ chối Unknown | 21/21 | 21/21 |
| Chấp nhận nhầm Unknown (FAR) | 0/21 | 0/21 |

Cả hai phương án đều đạt 100% trên tập kiểm tra của phiên bản ban đầu, nên chưa có bằng chứng về mức tăng accuracy nhờ tinh chỉnh ở đối chứng này. Cần phân biệt đối chứng huấn luyện đó với đối sánh PyTorch/ONNX sử dụng cùng trọng số và 181 ảnh hiện tại. Bằng chứng được lưu trong [baseline_team_20261020.json](experiments/baseline_team_20261020.json).

## 7. Thiết bị biên và giới hạn phần cứng

| Thiết bị | Thông số đã quan sát | Vai trò |
|---|---|---|
| Laptop/WSL2 | Ubuntu 24.04, x86_64, Python 3.12.14, RAM cấp cho WSL khoảng 7,6 GiB | Huấn luyện, xuất mô hình, backend, broker, Bot, MQTT Explorer |
| GPU laptop | NVIDIA RTX 2050, VRAM 4 GiB; CUDA khả dụng | Tinh chỉnh MobileNetV3 |
| Raspberry Pi | Raspberry Pi 4, Debian 12 Bookworm, ARM64/aarch64, kernel 6.12.25+rpt-rpi-v8; RAM hệ điều hành báo khoảng 3,7 GiB | Thu hình, YuNet và ONNX Runtime CPU |
| Pi CPU | Bốn core; tần số quan sát 1800 MHz khi đo; governor ondemand | Suy luận bằng CPU |
| Camera | USB CyberTrack H7, /dev/video0; driver hỗ trợ chế độ 640×480 @30 FPS; truy cập qua libcamera USB, định dạng thương lượng MJPEG | Thu ảnh và nhận diện cục bộ |
| Mạng | Wi-Fi DHCP có thể đổi địa chỉ khi chuyển hotspot; kết nối máy–Pi qua Tailscale | MQTT, SSH và tải ảnh trong Tailnet |

Dung lượng RAM khả dụng và lưu trữ trống phụ thuộc thời điểm kiểm tra. Trong cấu hình đánh giá, Pi không dùng GPU/NPU riêng cho suy luận; CPU thực thi cả YuNet lẫn bộ phân loại. Hiệu năng và nhiệt độ còn chịu ảnh hưởng của nguồn điện, tản nhiệt, kernel, phiên bản thư viện và độ phân giải xử lý.

Chế độ camera 640×480 ở 30 FPS là khả năng thu hình do driver công bố, không phải tốc độ xử lý AI của ứng dụng. Phép đo phát lại qua detector, cắt mặt và classifier cho thông lượng thấp hơn. Hệ thống chưa có phát hiện sống (liveness detection), nên ảnh in hoặc ảnh trên màn hình vẫn có thể được nhận diện. Các yếu tố chưa đo gồm điện năng, độ bền thẻ SD, tải nhiệt kéo dài và độ trễ mạng/Telegram.

## 8. Đánh giá định lượng và đo hiệu năng

### 8.1 Thiết kế phép đo hiệu năng

Đối sánh chính được thực hiện trên **cùng Raspberry Pi**, cùng checkpoint, tiền xử lý và đầu vào float32, giữa PyTorch CPU ở chế độ eager với ONNX Runtime CPU. Mỗi bộ thực thi chạy trong một tiến trình mới để đo RAM riêng. Sau 30 lượt khởi động làm nóng (warm-up), mỗi tiến trình đo 200 lượt; lặp 3 vòng để thu 600 mẫu thời gian cho mỗi runtime, đồng thời thay đổi thứ tự PyTorch/ONNX giữa các vòng. Phạm vi đo chỉ bao quanh forward hoặc session.run, không gồm đọc ảnh, camera, detector, tiền xử lý và mạng.

Ba vùng mặt từ tập kiểm tra được lặp luân phiên để đo độ trễ; toàn bộ 181 ảnh được đánh giá riêng về chất lượng và độ tương đương đầu ra. Trọng số không thay đổi trong quá trình đo. Dịch vụ nhận diện trên Pi được tạm dừng để tránh cạnh tranh CPU/camera, sau đó bật lại; trạng thái MQTT sau phép đo xác nhận online/idle và camera đã đóng.

### 8.2 Đối sánh bộ phân loại PyTorch và ONNX Runtime

| Chỉ số trên Raspberry Pi | PyTorch CPU FP32 | ONNX Runtime CPU FP32 | Thay đổi quan sát |
|---|---:|---:|---:|
| Độ trễ trung bình | 74,080 ms | 30,732 ms | ONNX nhanh hơn 2,41× |
| Độ trễ P50 | 74,331 ms | 30,720 ms | — |
| Độ trễ P95 | 75,564 ms | 31,371 ms | — |
| Thông lượng classifier | 13,497 FPS | 32,529 FPS | Chỉ classifier |
| Tệp mô hình | 6.218.949 bytes / 5,931 MiB | 6.105.639 bytes / 5,823 MiB | Tệp ONNX nhỏ hơn khoảng 1,82% |
| Mô hình và gói cấu hình | 5,960 MiB | 5,853 MiB | Cách đóng gói khác nhau |
| RSS trung bình sau warm-up | 395,095 MiB | 76,242 MiB | — |
| RSS cực đại khi suy luận | 396,855 MiB | 77,332 MiB | Giảm khoảng 80,51% |
| RSS cao nhất qua nạp/warm-up/suy luận | 397,039 MiB | 77,633 MiB | — |
| CPU, theo đơn vị một core | 192,35% | 201,84% | Khoảng 48,09% / 50,46% tổng năng lực bốn core |
| Accuracy trên tập kiểm tra | 181/181 = 100% | 181/181 = 100% | Cùng quyết định |
| Macro F1 | 1,000 | 1,000 | Cùng quyết định |
| Cross-Entropy kiểm tra không trọng số | 9,73009e-5 | 9,73031e-5 | Rất gần |

Tệp checkpoint PyTorch có kích thước khoảng 5,93 MiB, còn ONNX khoảng 5,82 MiB. Mức giảm nhỏ chủ yếu phản ánh khác biệt cách lưu giữa hai định dạng, chưa phải nén trọng số đáng kể. Các tệp thuộc package torch+torchvision trên Pi chiếm khoảng 428,04 MiB theo kích thước logic và 460,83 MiB theo dung lượng cấp phát; onnxruntime tương ứng khoảng 34,89 và 36,17 MiB. Phạm vi này loại OS và nhiều phụ thuộc dùng chung hoặc gián tiếp. Vì cả hai runtime vẫn được cài trên Pi, không thể xem chênh lệch trên là dung lượng thẻ SD đã được giải phóng.

![Đối sánh runtime, RAM và kích thước](assets/figures/runtime_comparison.png)

**Hình 7.** Đối sánh PyTorch CPU và ONNX Runtime CPU trên cùng Raspberry Pi, với cùng trọng số, đầu vào, FP32 và 2 luồng. Các biểu đồ tách độ trễ, thông lượng classifier, RSS cực đại, kích thước tệp mô hình, các tệp runtime đã cài và loss kiểm tra không trọng số. Nguồn: [comparison.json](experiments/benchmark_team_20261020/comparison.json). Bản [SVG](assets/figures/runtime_comparison.svg) hỗ trợ phóng lớn.

### 8.3 Chất lượng nhận diện và độ tương đương đầu ra

| Chỉ số trên 181 vùng mặt kiểm tra | Giá trị |
|---|---:|
| Accuracy | 181/181 = 100% |
| Macro F1 | 1,000 |
| Accuracy của thành viên | 160/160 = 100% |
| Tỷ lệ từ chối Unknown | 21/21 = 100% |
| Tỷ lệ chấp nhận nhầm Unknown | 0/21 trên tập kiểm tra này |
| Từ chối nhầm thành viên | 0/160 |
| Precision / Recall / F1 từng lớp | 1,000 cho cả sáu lớp trên mẫu test |

Số ảnh kiểm tra theo lớp lần lượt là Unknown 21, Dung 23, Hùng 30, Sơn 64, Thanh 24 và Tuấn 19. Khi chạy trên cùng Pi, PyTorch và ONNX có argmax trùng khớp 100% và quyết định từ chối giống nhau trên 181 mẫu; sai khác logit tuyệt đối lớn nhất là 8,15392e-5. Đây là phép so sánh khác với đối chiếu ONNX laptop–Pi ở mục 6.5.

![Ma trận nhầm lẫn trên tập kiểm tra nội bộ](assets/figures/confusion_matrix.png)

**Hình 8.** Ma trận nhầm lẫn của 181 ảnh kiểm tra nội bộ. Trục dọc là nhãn thực tế, trục ngang là nhãn dự đoán; mỗi ô biểu diễn số ảnh, không phải tỷ lệ phần trăm. Tên hiển thị được Việt hóa, thứ tự lớp và số đếm giữ nguyên theo [evaluation_metrics.json](experiments/team_20261020_son_live/evaluation_metrics.json). Bản [SVG](assets/figures/confusion_matrix.svg) phục vụ tái sử dụng.

**Phạm vi diễn giải:** manifest ghi independent_known_test=false. Ảnh kiểm tra của các thành viên đến từ cùng bản ghi hoặc phiên thu với ảnh huấn luyện, dù đã giữ lại các đoạn thời gian riêng và tạo purge gap. Unknown được tách danh tính nhưng chỉ có 21 ảnh từ 9 danh tính LFW. Vì vậy accuracy 100% phản ánh tập kiểm tra nội bộ hiện tại; chưa đủ căn cứ để suy rộng sang phiên quay, camera hoặc mọi người lạ ngoài tập dữ liệu.

### 8.4 Độ trễ toàn chuỗi trên khung hình phát lại

Phép đo toàn chuỗi thị giác sử dụng một khung hình có sẵn trong video Sơn 1, tại mốc khoảng 16 giây. Khung hình được đổi kích thước giữ tỷ lệ và thêm đệm (letterbox) về 640×480, rồi phát lại trên Pi qua 20 lượt warm-up và 200 lượt đo, với OpenCV 4 luồng và ONNX Runtime 2 luồng. Cả 200 lượt đều phát hiện đúng một khuôn mặt. Bước đầu chỉ sao chép ảnh trong RAM, do đó không đo thời gian camera thu hình vật lý.

| Bước xử lý khi phát lại | Trung bình | P50 | P95 |
|---|---:|---:|---:|
| Sao chép khung hình trong RAM | 0,425 ms | 0,424 ms | 0,488 ms |
| Phát hiện mặt bằng YuNet | 73,847 ms | 73,351 ms | 78,461 ms |
| Cắt/căn chỉnh, tiền xử lý và từ chối | 11,451 ms | 11,433 ms | 11,645 ms |
| Bộ phân loại ONNX | 31,516 ms | 31,426 ms | 31,726 ms |
| Toàn chuỗi từ khung hình đến kết quả | 117,239 ms | 116,641 ms | 122,491 ms |

Thông lượng phát lại đạt **8,528 FPS**, RSS cực đại **121,348 MiB** và mức RSS cao nhất của tiến trình **123,605 MiB**. CPU ghi nhận khoảng 315,11% theo đơn vị một core, tương đương 78,78% tổng năng lực bốn core. Nhiệt độ tại hai thời điểm trước/sau tăng từ 59,9°C lên 70,1°C; vcgencmd báo cờ throttling 0x0 ở cả hai lần đọc. Phép đo không mở camera, không gồm MQTT/HTTP/Telegram, nên 8,528 FPS chưa phải thông lượng nhận diện trực tiếp của ứng dụng.

![Phân rã độ trễ chuỗi xử lý thị giác khi phát lại trên Pi](assets/figures/pipeline_stages.png)

**Hình 9.** Thời gian trung bình của từng bước trong phép phát lại khung hình 640×480 trên Pi. Tổng các bước không bao gồm camera vật lý, gửi bản tin hoặc tải ảnh. Phần YuNet chiếm độ trễ lớn nhất trong cấu hình đo này. Nguồn: [pipeline_replay_opencv4.json](experiments/benchmark_team_20261020/pipeline_replay_opencv4.json). Bản [SVG](assets/figures/pipeline_stages.svg) hỗ trợ tái sử dụng.

Thao tác vẽ khung mặt, tên, confidence và mã hóa JPEG được đo riêng trong 20 lượt: trung bình 15,469 ms, P95 15,937 ms, dung lượng JPEG trung bình 67.361 bytes. Phép đo không bao gồm tải HTTP qua Tailscale hoặc gửi Telegram. Ứng dụng còn cấu hình khoảng lấy mẫu khung hình 0,15 giây, thời gian hạn chế lặp thông báo (cooldown) 30 giây và tự dừng sau khi nhận diện thành viên; vì vậy hành vi vận hành khác vòng đo phát lại liên tục.

### 8.5 Diễn giải đúng các chỉ số

- **Độ trễ classifier** là thời gian riêng của bộ phân loại, không gồm detector, camera và mạng.
- **Thông lượng classifier** tính theo vòng suy luận, không phải nghịch đảo P95 hoặc tốc độ camera trực tiếp.
- **Thông lượng phát lại** gồm phát hiện mặt, tiền xử lý và phân loại; không gồm thu hình vật lý hoặc truyền mạng.
- **Accuracy/F1/loss** được tính trên 181 vùng mặt kiểm tra; cần phiên kiểm tra độc lập trước khi suy rộng kết luận.
- **RAM RSS** là bộ nhớ tiến trình, gồm Python, thư viện, mô hình và bộ đệm; không phải RAM riêng của trọng số hoặc tổng RAM Pi.
- **Kích thước mô hình** là dung lượng tệp; dung lượng runtime là các tệp thuộc package trong phạm vi đã nêu.
- Đối sánh PyTorch/ONNX sử dụng CPU trên cùng Pi, bảo đảm tách khỏi cấu hình GPU huấn luyện trên laptop.

## 9. Ứng dụng, điều khiển và trực quan hóa

### 9.1 Telegram là ứng dụng điều khiển

Telegram đóng vai trò ứng dụng điều khiển và nhận thông báo. Bot trên laptop tiếp nhận lệnh bằng long polling, tạo command có request_id/timestamp rồi gửi qua MQTT. Pi thực thi và trả xác nhận, trạng thái hoặc sự kiện; Bot diễn giải phản hồi thành tin nhắn cho người dùng. Toàn bộ suy luận hình ảnh vẫn diễn ra trên Pi.

| Lệnh | Tác dụng ở mức sử dụng |
|---|---|
| /start | Mở camera và bắt đầu nhận diện cục bộ tại Pi; trả lời “Đang nhận diện...” |
| /stop | Dừng phiên và giải phóng camera |
| /status | Xem trạng thái, camera, phiên bản mô hình, CPU, nhiệt độ, RAM, thời gian suy luận và lỗi |
| /members | Xem các lớp trong mô hình và các thư mục đã thu |
| /collect &lt;name&gt; | Bắt đầu thu ảnh sau bước xác nhận đồng thuận |
| /collect_unknown &lt;anonymous-id&gt; | Thu một phiên Unknown sau xác nhận người tham gia |
| /help | Xem hướng dẫn lệnh |

Kết quả nhận diện thành viên gồm tên hiển thị, confidence, thiết bị và inference_ms, được gửi về cuộc trò chuyện đã khởi tạo yêu cầu. Bot dùng expected_request_id trong lệnh dừng để phản hồi của phiên cũ không dừng nhầm phiên mới. Unknown không tạo tin nhắn/ảnh và không kết thúc phiên; no_face hoặc multiple_faces có thông báo được hạn chế theo cooldown. Token Bot và thông tin xác thực MQTT được giữ trong .env cục bộ.

Khi bật gửi ảnh, Pi vẽ khung mặt YuNet, tên tiếng Việt và confidence lên chính khung hình nhận diện. Bộ đệm RAM giữ tối đa 8 ảnh với TTL mặc định 120 giây. MQTT chỉ mang image_id và metadata; Bot tải JPEG từ endpoint có bearer token trong Tailnet, sau đó gửi tin nhắn văn bản trước, ảnh sau. Lỗi tải ảnh không làm mất kết quả nhận diện. Ảnh huấn luyện tiếp tục được lưu cục bộ; âm thanh/TTS chưa được bật.

Trong cấu hình trình diễn tham chiếu, danh sách Telegram user/chat được phép sử dụng đang để trống theo lựa chọn của chủ dự án. Trước khi chia sẻ Bot rộng hơn, cần cấu hình các ID được phép để giới hạn người điều khiển.

### 9.2 MQTT Explorer: giao diện giám sát

MQTT Explorer chạy trong Docker Compose profile monitor và mở tại **http://localhost:4000** trên laptop. Có thể bật bằng lệnh sau từ thư mục dự án:

~~~bash
docker compose --profile monitor up -d mqtt-explorer
~~~

Thông tin đăng nhập giao diện được cấu hình bằng MQTT_EXPLORER_USERNAME và MQTT_EXPLORER_PASSWORD trong .env. Giao diện tự kết nối Mosquitto với tài khoản edgeface_monitor chỉ đọc aiot/edgeface/#, cho phép theo dõi JSON, trạng thái gần nhất và số đo vận hành:

- status: online/offline, idle/recognizing/collecting/error, camera, model, CPU/RAM/nhiệt độ.
- command: lệnh vừa được Bot gửi; chỉ xem được message mới vì command không retained.
- result: recognition, no_face, multiple_faces.
- collection: tiến độ thu ảnh.
- telemetry: số đo định kỳ; UI không thay thế kho lưu lịch sử dài hạn.

MQTT Explorer hiển thị bản tin và metadata, không hiển thị ảnh hoặc video. Hướng dẫn vận hành và kết nối được lưu tại [README](../README.md) và [MQTT_EXPLORER.md](../docs/MQTT_EXPLORER.md).

### 9.3 Kịch bản trình diễn

1. Mở MQTT Explorer, xác nhận broker và topic status/telemetry của Pi.
2. Gửi /status trên Telegram để cho thấy thiết bị online và model version.
3. Gửi /start; trình bày camera và inference chạy trên Pi. Khi thành viên được nhận diện, quan sát result MQTT, tin nhắn Telegram, auto-stop và ảnh annotation.
4. Thử với người không thuộc nhóm, đã đồng ý tham gia; hệ thống dự kiến im lặng với Unknown và tiếp tục chạy. Ghi nhận ground truth, không giả định mọi người lạ đều bị từ chối.
5. Gửi /stop nếu cần kết thúc phiên và /status để xác nhận camera đóng, state idle.

## 10. Tài liệu tham khảo, minh chứng và hướng phát triển

### 10.1 Tài liệu tham khảo

1. Howard, A. et al. (2019). “Searching for MobileNetV3.” Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314–1324. [Bài báo](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf).
2. Wu, W., Peng, H., & Yu, S. (2023). “YuNet: A Tiny Millisecond-level Face Detector.” Machine Intelligence Research, 20(5), 656–665. [Bài báo](https://link.springer.com/article/10.1007/s11633-023-1423-y), [OpenCV Zoo implementation/model card](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
3. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). “FaceNet: A Unified Embedding for Face Recognition and Clustering.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 815–823. [Bài báo](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf).
4. Scheirer, W. J., de Rezende Rocha, A., Sapkota, A., & Boult, T. E. (2013). “Toward Open Set Recognition.” IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(7), 1757–1772. DOI: 10.1109/TPAMI.2012.256. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/23682001/).
5. Huang, G. B., Mattar, M., Berg, T., & Learned-Miller, E. (2008). “Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.” European Conference on Computer Vision (ECCV). [Bài báo và mô tả dataset](https://people.cs.umass.edu/~elm/papers/Huang_eccv2008-lfw.pdf).
6. OASIS (2014). “MQTT Version 3.1.1.” [Đặc tả chuẩn](https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/mqtt-v3.1.1.html).
7. ONNX Runtime. “Model optimizations” và “Thread management.” [Tối ưu graph/quantization](https://onnxruntime.ai/docs/performance/model-optimizations/), [cấu hình thread](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
8. Open Neural Network Exchange. “ONNX Concepts.” [Tài liệu định dạng ONNX](https://onnx.ai/onnx/intro/concepts.html).

### 10.2 Tài liệu và minh chứng nội bộ

| Nội dung | Tệp trong repository |
|---|---|
| Kiến trúc MVP và triển khai | [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) |
| Sơ đồ kiến trúc, luồng dữ liệu và nhận diện; PNG/SVG/Mermaid | [report/assets/figures/](assets/figures/) |
| Đối chứng head-only/fine-tune, revision trước | [report/experiments/baseline_team_20261020.json](experiments/baseline_team_20261020.json) |
| Sửa pixel format camera và fine-tune hiện tại | [report/experiments/team_20261020_son_live/](experiments/team_20261020_son_live/) |
| Benchmark runtime, replay, RAM và model size | [report/BENCHMARK_20261020.md](BENCHMARK_20261020.md) |
| Theo dõi MQTT Explorer | [docs/MQTT_EXPLORER.md](../docs/MQTT_EXPLORER.md) |
| Ảnh annotation gửi Telegram | [docs/PHOTO_NOTIFICATIONS.md](../docs/PHOTO_NOTIFICATIONS.md) |
| Môi trường đã kiểm tra | [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md) |
| Đồ thị training/validation loss mới, PNG/SVG | [report/assets/figures/training_loss.png](assets/figures/training_loss.png) |
| Ảnh kiến trúc YuNet, MobileNetV3-Small | [report/assets/figures/](assets/figures/) |
| Ma trận nhầm lẫn | [report/assets/figures/confusion_matrix.png](assets/figures/confusion_matrix.png) |
| So sánh PyTorch/ONNX/RAM/model size | [report/assets/figures/runtime_comparison.png](assets/figures/runtime_comparison.png) |
| Phân rã độ trễ chuỗi phát lại | [report/assets/figures/pipeline_stages.png](assets/figures/pipeline_stages.png) |
| Số liệu JSON benchmark | [report/experiments/benchmark_team_20261020/](experiments/benchmark_team_20261020/) |
| Manifest, metrics và provenance model hiện tại | [report/experiments/team_20261020_son_live/](experiments/team_20261020_son_live/) |

Mã nguồn được tổ chức theo trách nhiệm: training/data.py, training/train.py, training/evaluate.py và training/export_onnx.py xử lý dữ liệu/mô hình; edgeface/preprocess.py, edgeface/protocol.py và edgeface/mqtt.py cung cấp thành phần dùng chung; các module trong raspberry_pi đảm nhiệm thiết bị biên; telegram_bot/bot.py điều phối lệnh và phản hồi. Dữ liệu khuôn mặt gốc cùng checkpoint huấn luyện được giữ ngoài Git.

### 10.3 Kết quả có thể khẳng định và giới hạn cần giữ

Dự án đã triển khai chuỗi xử lý từ video thành viên đến mô hình chạy cục bộ trên Raspberry Pi ARM64. Ngoài video, có 150 vùng mặt của Sơn được thu bằng camera Pi; vì vậy cần mô tả đây là dữ liệu kết hợp, không coi toàn bộ dữ liệu là ảnh thu từ Pi. Việc tinh chỉnh đã cập nhật classifier và các khối cuối của backbone bằng GPU laptop.

Trên 181 ảnh kiểm tra nội bộ, PyTorch và ONNX cùng đạt accuracy 100%, macro F1 1,000 và các quyết định trùng khớp 100%. Trên cùng Pi, độ trễ trung bình của ONNX classifier thấp hơn, tương ứng mức tăng tốc 2,41×; RSS cực đại trong suy luận giảm khoảng 80,51%. Các kết quả này có phạm vi cụ thể: dữ liệu thành viên chưa tách theo phiên độc lập, Unknown chỉ có 21 ảnh từ 9 danh tính LFW, và phép đo toàn chuỗi là phát lại ảnh thay vì camera trực tiếp.

Telegram và MQTT Explorer đã cung cấp điều khiển, phản hồi và giám sát. Các phần chưa được triển khai hoặc đo gồm kho lịch sử telemetry dài hạn, cơ cấu khóa cửa, độ trễ mạng/Telegram, công suất, tải nhiệt kéo dài, INT8/pruning và phát hiện sống. Đây là các hướng phát triển có thể kiểm thử sau khi bổ sung dữ liệu và xác định yêu cầu.

Dữ liệu sinh trắc học cần được quản lý theo sự đồng thuận của người tham gia, giới hạn quyền truy cập và hỗ trợ xóa phiên thu. Ảnh thô không được đưa lên Git hoặc bài trình chiếu công khai; việc gửi ảnh kết quả chỉ thực hiện qua tính năng snapshot đã bật.

Khi có các phiên thu mới của thành viên và người lạ đã đồng ý tham gia, bước đánh giá tiếp theo là tạo phiên bản dữ liệu mới với tập kiểm tra độc lập theo phiên, huấn luyện lại, hiệu chỉnh ngưỡng trên validation, xuất và triển khai mô hình mới. Chỉ cập nhật kết luận định lượng sau khi có kết quả đánh giá và đo hiệu năng tương ứng trên Pi.
