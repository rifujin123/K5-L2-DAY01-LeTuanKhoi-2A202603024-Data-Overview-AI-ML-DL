# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
{
"sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
}
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Cả record này nói về cả bức ảnh. Vì ImageNet-1K là một lớp cho cả ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp đến từ taxonomy.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Bởi vì mỗi đối tượng sử dụng cho mỗi mục đích khác nhau. class_id(Dành cho GPU và Mô hình),class_name(Dành cho con người và UI), taxonomy_name(Dành cho hệ thống quản trị).
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Phải có bộ quy tắc rõ như: Ma trận ưu tiên (Con người > Phương tiện > Động vật...), Diện tích và độ nổi bật thị giác (chiếm dụng bao nhiêu % bức ảnh, độ sâu trường ảnh), Trường hợp không xác định (gán nhãn Ambiguous/Other hoặc chuyển qua cho người kiểm duyệt).
6. Vì sao không được lấy `score` của model làm ground truth?
Vì score của model chỉ là giá trị dự đoán, phải lấy của con người làm ground truth để tăng độ tin cậy.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
  {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  }
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
góc trên trái = (385.33, 69.24), góc dưới phải = (498.92,348.92), rộng = 113.58, cao = 279.68
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
đối với ngưỡng 0.20 có 53 prediction, đối với ngưỡng 0.6 thì có 25 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Kiểm tra lại xem model đã label đúng hay chưa và điều chỉnh
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Mỗi vật thể phải được vẽ bounding box hình chữ nhật sát với vật thể, đảm bảo chứa toàn bộ phần nhìn thấy và hạn chế tối đa vùng không liên quan.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Viết thành quy tắc trong guideline, vì guideline là dành cho annotators đọc và làm theo khi làm sai quy tắc thì annotators chỉ cần check lại không cần phải đưa cho lab coach.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
"sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-005",
    "class_id": 60,
    "class_name": "dining table",
    "score": 0.578416,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      0.97,
      234.74,
      345.86,
      423.87
    ],
    "polygon_point_count": 598,
    "polygon_xy": [
      [
        251.0,
        399.0
      ],
      [
        252.0,
        400.0
      ],
      [
        252.0,
        401.0
      ],
      [
        253.0,
        402.0
      ],
      [
        254.0,
        402.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
polygon nhiều thông tin hơn về vật thể so với box, ví dụ box thì phải xài hình chữ nhật bao quanh 1 vật thì có thể dư thừa những phần không cần thiết, còn polygon thì sẽ bao quanh vật thể thông qua nhiều điểm hơn như cắt ảnh trong adobe.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id là id của từng vật thể trong bức ảnh nó không phải là class ID ( ví dụ ảnh có 3 cái tô thì 3 cái tô sẽ có cùng class_id là bowl còn instance sẽ là 1,2,3), không phải tracking_id vì mỗi instance là 1 vật khác nhau nên tracking_id là không chính xác (ví dụ: cả 2 chiếc xe đều có cùng màu thì sẽ có instance_id là 1,2 thì tracking_id không thể nào mà chung được instance_id)
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Bám sát đường viền: biên mask phải đi sát với vật thể
Vật thể bị che khuất: chỉ gán mask cho vật thể nhìn thấy rõ không gán mác và tự suy đoán cho vật thể bị khuất hoặc không thấy rõ
Biên mờ: Khi ranh giới không rõ, chọn đường biên hợp lý nhất dựa trên hình dạng và màu sắc quan sát được.
Vật thể bị cắt khỏi ảnh: Mask kết thúc tại mép ảnh, không vẽ ra ngoài ảnh.
Nhiều vật thể: Mỗi vật thể có một mask riêng, kể cả khi chúng thuộc cùng một class.
Lỗ hổng bên trong vật thể: Những vùng nền nhìn xuyên qua vật thể không được tính vào mask.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
ưu tiên chọn guideline vì quy tắc yêu cầu mask bám sát phần vật thể nhìn thấy và khoog bị lấn sang vật thể khác.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Mỗi ảnh có 1 nhãn class đại diện cho bức ảnh | Ảnh có nhiều vật thể nên khó chọn một lớp đại diện | Xác định chủ thể theo guideline và chọn nhãn phù hợp | Kiểm tra nhãn có đúng chủ đề và tuân thủ guideline hay không |
| Phát hiện vật thể | Mỗi vật thể có 1 boundary box + class_id | có nhiều box xếp chồng nhau có thể bị nhận diện sai lớp | vẽ box sát từng vật theo guideline, bỏ vật thể không nhìn thấy rõ |Kiểm tra kết quả boundary box có tuân thủ theo guideline hay không nếu có vấn đề chưa rõ thì cập nhật guideline  |
| Instance segmentation | Mỗi cá thể có một 1 mask + class_id + instance_id |các vật thể dính liền che khuất nhau khiến ranh giới mask khó xác định  |vẽ mask theo guideline từng vật thể, không tô vùng phần bị che  |Kiểm tra biên mask bị tô nhầm, phần vật thể bị bỏ sót giữa các cá thể |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng dữ liệu công khai hoặc được cấp phép, không chia sẻ thông tin cá nhân hay dữ liệu nhạy cảm, và lưu trữ dữ liệu an toàn.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Lab coach (escaplation)

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
