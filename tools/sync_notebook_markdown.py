"""Align narrative and summary plots with each notebook's own saved evaluations."""
import copy
import json
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT / "DAT301m_DataCentric_AI_Roman_Numerals_Master_AllInOne.ipynb",
         ROOT / "submission.ipynb"]


def output(cell):
    return "\n".join("".join(o.get("text", [])) for o in cell.get("outputs", []))


def update(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    before = copy.deepcopy(nb)
    cells = nb["cells"]
    metrics = []
    for index in (7, 14, 20, 25, 30):
        text = output(cells[index])
        val = re.findall(r"• Validation Accuracy[^:\n]*:\s*([\d.]+)%\s*\| Loss:\s*([\d.]+)", text)
        lb = re.findall(r"• Label Book Accuracy[^:\n]*:\s*([\d.]+)%\s*\| Loss:\s*([\d.]+)", text)
        if not val or not lb:
            raise ValueError(f"Missing saved evaluation: {path.name}, cell {index+1}")
        metrics.append(tuple(map(float, (*val[-1], *lb[-1]))))

    def put(i, text):
        cells[i]["source"] = (text.strip() + "\n").splitlines(keepends=True)

    def result(i):
        v, vl, l, ll = metrics[i]
        return (f"**Kết quả lưu trong notebook:** Validation Accuracy **{v:.2f}%** "
                f"(loss {vl:.4f}); Label Book Accuracy **{l:.2f}%** (loss {ll:.4f}).")

    names = ["Baseline", "Làm sạch", "Chuẩn hóa hình học", "Cân bằng 7.500 ảnh", "Mở rộng 9.500 ảnh"]
    table = "| Stage | Validation Accuracy | Validation Loss | Label Book Accuracy | Label Book Loss |\n|---|---:|---:|---:|---:|\n"
    for i, (v, vl, l, ll) in enumerate(metrics):
        table += f"| {i} — {names[i]} | {v:.2f}% | {vl:.4f} | {l:.2f}% | {ll:.4f} |\n"
    table += "| 5 — Superior | Chưa có kết quả | — | Chưa có kết quả | — |"
    best = max(range(5), key=lambda i: metrics[i][2])
    best_stages = ", ".join(str(i) for i in range(5) if metrics[i][2] == metrics[best][2])
    fixed = "build_superior_dataset" in "".join(cells[33]["source"])
    v0, _, l0, _ = metrics[0]
    v4, _, l4, _ = metrics[4]

    put(0, f'''# 🏛️ ĐỒ ÁN DAT301m: DATA-CENTRIC AI — CHỮ SỐ LA MÃ
## Thực nghiệm với VGG16 tự xây dựng từ đầu

**Nhóm thực hiện:** Team 1 (Tuân, Chiến, Tuất, Huy, Duy).

Notebook khảo sát các bước làm sạch, chuẩn hóa và tăng cường dữ liệu khi giữ cùng hàm xây dựng mô hình VGG16 scratch và optimizer Adam. Stage 4 và Superior sử dụng thêm callback chọn checkpoint/dừng sớm; vì vậy không diễn giải đây là thí nghiệm chỉ thay đổi dữ liệu trong mọi điều kiện.

**Kết quả đã lưu:** Baseline đạt {v0:.2f}% validation / {l0:.2f}% Label Book; Stage 4 đạt {v4:.2f}% validation / {l4:.2f}% Label Book. Các stage đạt Label Book Accuracy cao nhất là **{best_stages}**, cùng mức **{metrics[best][2]:.2f}%**. Superior chưa có output huấn luyện để kết luận hiệu năng.

{table}

Các số trên lấy từ phần **đánh giá sau huấn luyện** của từng stage trong chính file này, không lấy epoch có validation cao nhất nếu stage đó đánh giá trọng số cuối. Các lần chạy có thể cho kết quả khác nhau.

`label_book` có 52 ảnh, là tập đánh giá tham chiếu; không gọi đây là kết quả trên test ẩn của cuộc thi. Validation giữa các stage có cách tạo khác nhau, nên các điểm số không phải phép so sánh trên một tập validation cố định.

### Mục lục
1. [Khởi tạo mô hình](#setup)
2. [Stage 0: Baseline và kiểm tra dữ liệu](#stage0)
3. [Stage 1: Làm sạch](#stage1)
4. [Stage 2: Chuẩn hóa hình học](#stage2)
5. [Stage 3: Cân bằng 7.500 ảnh](#stage3)
6. [Stage 4: Mở rộng 9.500 ảnh](#stage4)
7. [Stage 5: Superior](#stage5_superior)
8. [Đánh giá tổng hợp](#stage6_eval)
''')
    put(2, '''<a id="setup"></a>
## 1. KHỞI TẠO VGG16 SCRATCH

Kiến trúc được viết bằng các lớp Keras: 13 Conv2D, 5 MaxPooling2D và 3 Dense. Không dùng `keras.applications.VGG16` hoặc pretrained weights. Ảnh đầu vào 32 × 32 RGB; mô hình chia giá trị pixel cho 255. Adam có learning rate 1e-4, batch size 32, tối đa 15 epoch mỗi stage.
''')
    put(4, '''<a id="stage0"></a>
## 2. NẠP DỮ LIỆU GỐC

Output ghi nhận **2.067 ảnh train, 813 ảnh validation và 52 ảnh Label Book**, thuộc 10 lớp. Stage 0 dùng cách chia gốc của cuộc thi. Các bước sau ghi ảnh xử lý vào thư mục riêng, không sửa ảnh trong `data/train`, `data/val` hoặc `label_book`.
''')
    put(6, "## 3. HUẤN LUYỆN BASELINE TRONG 15 EPOCH\n\n" + result(0))
    put(8, '''## 4. KIỂM TRA TRÙNG LẶP VÀ ẢNH TRẮNG

Output của cell kiểm tra ghi nhận **49 ảnh trong validation có MD5 trùng với ảnh train** và một ảnh trắng `data/train/vi/aac26b40-ce5d-11eb-b317-38f9d35ea60f.png`. Cell này không định lượng ảnh lệch tâm, phân bố kích thước nét chữ hoặc tỷ lệ sai nhãn.
''')
    put(10, '''<a id="stage1"></a>
# STAGE 1: LÀM SẠCH DỮ LIỆU

Gộp ảnh gốc từ train và validation, bỏ các tên file trong danh sách loại trừ, khử bản sao theo MD5 và bỏ ảnh trắng. Ảnh được sao chép vào `stage1_cleaned_data`; dữ liệu gốc được giữ nguyên.

Output ghi nhận **2.825 ảnh**, chia thành **2.260 train / 565 validation** bằng `validation_split=0.2` và cùng seed. Đây là phép chia ngẫu nhiên của Keras, **không phải stratified split theo lớp**. Danh sách tên file loại trừ là quy tắc đã khai báo trong code, không phải kết quả tự động phát hiện sai nhãn của cell kiểm tra.
''')
    put(11, "## 1. TẠO TẬP DỮ LIỆU LÀM SẠCH")
    put(13, "## 2. CHIA 80/20 VÀ HUẤN LUYỆN LẠI\n\n" + result(1))
    put(15, '''<a id="stage2"></a>
# STAGE 2: CHUẨN HÓA HÌNH HỌC VÀ NỀN

Chuyển ảnh sang grayscale, đưa pixel > 215 về trắng, tìm bounding box của nét chữ, căn giữa trên canvas vuông có padding 15%, tăng tương phản nét tối rồi resize về 32 × 32 bằng bilinear. Label Book được xử lý bằng cùng hàm.

Stage này đọc lại `data/train` và `data/val`, bỏ hai tên file và khử MD5 trong từng lớp; không đọc trực tiếp tập 2.825 ảnh của Stage 1. Output ghi nhận **2.829 ảnh**, chia **2.264 train / 565 validation**. Vì bộ lọc và phép chia khác Stage 1, chênh lệch accuracy không thể quy hoàn toàn cho chuẩn hóa hình học.
''')
    put(17, "## 1. TẠO `stage2_geometric_data` VÀ LABEL BOOK CHUẨN HÓA")
    put(19, "## 2. HUẤN LUYỆN TRÊN DỮ LIỆU CHUẨN HÓA\n\n" + result(2))
    put(21, '''<a id="stage3"></a>
# STAGE 3: CÂN BẰNG LỚP VÀ AUGMENTATION

Dùng ảnh trong `stage2_geometric_data` làm nguồn; giữ ảnh nguồn và sinh thêm đến **750 ảnh/lớp**, tổng **7.500 ảnh**. Augmentation gồm xoay ±8°, dịch ngang/dọc tối đa 8% và zoom 8%; không lật ngang để hạn chế làm thay đổi ý nghĩa ký tự.

Output nạp dữ liệu ghi nhận **6.000 train / 1.500 validation**. Việc chia diễn ra **sau augmentation**, nên các biến thể của cùng ảnh gốc có thể xuất hiện ở cả hai tập. Validation này không bảo đảm độc lập ở cấp ảnh nguồn. Không có output trong notebook chứng minh nhận định cũ về recall 27% của lớp `ii`.
''')
    put(22, "## 1. TẠO TẬP CÂN BẰNG 7.500 ẢNH")
    put(24, "## 2. HUẤN LUYỆN TRÊN 7.500 ẢNH\n\n" + result(3) + "\n\nCell `fit` hiện không truyền `callbacks_s3`; kết quả đánh giá là trọng số sau epoch cuối, không mặc định là checkpoint tốt nhất.")
    put(26, '''<a id="stage4"></a>
# STAGE 4: MỞ RỘNG TẬP DỮ LIỆU LÊN 9.500 ẢNH

Đọc lại ảnh gốc train/validation, bỏ hai tên file, khử MD5 trong từng lớp và chuẩn hóa hình học. Sinh thêm ảnh bằng xoay ±10°, dịch ngang/dọc tối đa 8% và zoom 8%, không lật ngang, để đạt **950 ảnh/lớp**.

Output ghi nhận **9.500 ảnh**, gồm **7.600 train / 1.900 validation**. Stage 4 vẫn chia sau augmentation và có nguy cơ trùng ảnh nguồn giữa hai tập. Huấn luyện tối đa 15 epoch, lưu checkpoint theo `val_accuracy` và dùng EarlyStopping với patience 8; log đã lưu cho thấy hoàn thành 15 epoch và khôi phục trọng số epoch 15.
''')
    put(27, "## 1. TẠO VÀ NẠP 9.500 ẢNH")
    put(29, "## 2. HUẤN LUYỆN VÀ ĐÁNH GIÁ\n\n" + result(4))
    if fixed:
        status = '''Code Superior hiện đã có hàm tạo dữ liệu, nhưng các cell Superior **chưa có output chạy được lưu trong file này**. Vì vậy chưa báo cáo số ảnh thực tế hoặc accuracy của Superior từ lần chạy này.

Pipeline dự kiến: khử ảnh trùng theo pixel, loại ảnh có pixel trùng nhưng khác nhãn, loại ảnh trùng Label Book; chia khoảng 80/20 ảnh gốc trong từng lớp; chỉ augmentation phần train. Tổng train + validation đặt mục tiêu 990 ảnh/lớp (9.900 ảnh); tỷ lệ 80/20 áp dụng cho **ảnh nguồn trước augmentation**, không phải tổng số ảnh cuối.

Ảnh được căn giữa ở kích thước nguồn rồi resize Lanczos về 32 × 32. Augmentation gồm thay đổi độ dày nét, affine với góc ngẫu nhiên ±7°, shear, scale và dịch chuyển. Dữ liệu mới được ghi vào `superior_data_9900/split_v2_...`, gồm `train`, `val`, `label_book` và manifest; dữ liệu gốc chỉ được đọc.

Trên VS Code kết nối Colab, chạy lần lượt cell High-Res → tạo dữ liệu → nạp dữ liệu → huấn luyện. Runtime mới cần chạy cell khởi tạo dữ liệu và cell định nghĩa mô hình/tham số trước.'''
    else:
        status = '''**Trạng thái theo output đã lưu:** cell chuẩn bị báo **0 ảnh** trong `superior_data_9900`; cell nạp dữ liệu dừng với `FileNotFoundError`. Cell huấn luyện Superior chưa có output. Vì vậy chưa có Validation Accuracy hay Label Book Accuracy của Superior trong file này.

Mục tiêu thiết kế là chia ảnh gốc theo lớp trước khi augmentation, giữ validation chỉ gồm ảnh gốc, dùng chuẩn hóa High-Res và resize Lanczos, sau đó tăng cường riêng train để đạt tổng 9.900 ảnh. Tuy nhiên, code đang lưu trong file này chưa tạo tập Superior và vẫn gọi `validation_split` trên thư mục chung, nên chưa thực hiện được thiết kế đó. Không dùng các số 81.95% / 84.62% cũ làm kết quả của Superior.

Các đường dẫn tương đối được tìm trên runtime Colab đang kết nối, không phải ổ C: của máy mở VS Code.'''
    put(31, '<a id="stage5_superior"></a>\n# STAGE 5: SUPERIOR — TRẠNG THÁI VÀ THIẾT KẾ\n\n' + status)
    put(36, '''<a id="stage6_eval"></a>
# PHẦN 6: ĐÁNH GIÁ TỔNG HỢP

Các cell bên dưới chưa có output lưu trong notebook. Cell nạp mô hình chọn trọng số **Stage 4** (`best_vgg16_stage4.weights.h5`) và Label Book chuẩn hóa hình học, không chọn mô hình Superior. Nếu chưa có trọng số, code có nhánh huấn luyện Stage 4 lại; kết quả mới có thể khác bảng tóm tắt đã lưu.

Phần hiện có gồm ma trận nhầm lẫn, classification report và biểu đồ so sánh Stage 0–4. Không có cell riêng xuất hard examples hoặc báo cáo error analysis, nên chưa khẳng định đã hoàn thành các nội dung đó.
''')
    put(37, "## 1. NẠP TRỌNG SỐ STAGE 4 VÀ LABEL BOOK CHUẨN HÓA")
    put(39, "## 2. MA TRẬN NHẦM LẪN VÀ CLASSIFICATION REPORT\n\nChưa có output lưu cho phần này; chỉ diễn giải precision/recall và cặp lớp nhầm lẫn sau khi chạy cell bên dưới.")
    put(41, "## 3. SO SÁNH KẾT QUẢ STAGE 0–4\n\n" + table + "\n\nBảng chép các giá trị đánh giá đã lưu trong file này. Biểu đồ bên dưới ưu tiên biến kết quả trong kernel hiện tại; nếu chưa có biến thì dùng các số đã lưu này. Validation giữa các stage khác nhau, Stage 3–4 chia sau augmentation; Label Book ở Stage 0–1 là ảnh gốc, Stage 2–4 được chuẩn hóa. Đây là so sánh các pipeline, không phải ablation được kiểm soát hoàn toàn. Một ảnh đúng/sai trên 52 ảnh Label Book tương ứng khoảng 1,92 điểm phần trăm.")
    put(43, f'''## 4. NHẬN XÉT TỪ KẾT QUẢ ĐÃ LƯU

1. Label Book Accuracy từ Stage 0 đến Stage 4 thay đổi **{l0:.2f}% → {l4:.2f}%**, tương ứng **{l4-l0:+.2f} điểm phần trăm**. Đây là kết quả của lần chạy đã lưu, chưa có trung bình hoặc độ lệch chuẩn qua nhiều seed.
2. Stage 1 đạt {metrics[1][0]:.2f}% validation / {metrics[1][2]:.2f}% Label Book. Điểm giảm sau làm sạch không tự chứng minh toàn bộ mức giảm do loại rò rỉ, vì tập train/validation cũng đã thay đổi.
3. Stage 2 đạt {metrics[2][0]:.2f}% validation / {metrics[2][2]:.2f}% Label Book. Stage 2 không dùng đúng tập nguồn đã lọc của Stage 1, nên chưa tách riêng được tác động của chuẩn hóa.
4. Stage 3–4 tăng số ảnh và đạt kết quả như bảng; việc chia sau augmentation hạn chế giá trị của validation khi đánh giá khả năng khái quát hóa.
5. Superior hướng đến chia ảnh nguồn trước augmentation, nhưng chưa có kết quả huấn luyện lưu trong notebook để kết luận tốt hơn Stage 4. Label Book đã được dùng để theo dõi nhiều thử nghiệm nên không xem nó là phép đo cuối cùng trên một test ẩn chưa từng tham khảo.
''')

    # The plot must use the same measurements as the narrative, not legacy constants.
    src = "".join(cells[42]["source"])
    vals = [row[0] for row in metrics]
    lbs = [row[2] for row in metrics]
    val_expr = "[round(float(globals().get(f'val_acc_s{i}', saved / 100)) * 100, 2) for i, saved in enumerate(" + repr(vals) + ")]"
    lb_expr = "[round(float(globals().get(f'lb_acc_s{i}', saved / 100)) * 100, 2) for i, saved in enumerate(" + repr(lbs) + ")]"
    src = re.sub(r"^val_accuracies = .*", "val_accuracies = " + val_expr, src, flags=re.M)
    src = re.sub(r"^lb_accuracies = .*", "lb_accuracies = " + lb_expr, src, flags=re.M)
    src = src.replace("Tiến trình Cải thiện Hiệu năng qua 5 Stage Data-Centric AI", "So sánh kết quả các pipeline Stage 0–4")
    src = src.replace("plt.ylim(50, 100)", "plt.ylim(0, 100)")
    put(42, src)
    assert not cells[42].get("outputs"), "Existing plot output requires review before editing."
    for i, (old, new) in enumerate(zip(before["cells"], cells)):
        assert old.get("outputs") == new.get("outputs"), (path.name, i, "outputs changed")
        assert old.get("execution_count") == new.get("execution_count")
        if old["cell_type"] == "code" and i != 42:
            assert old["source"] == new["source"], (path.name, i, "training code changed")
        if new["cell_type"] == "code":
            compile("".join(new["source"]), f"cell_{i+1}", "exec")
    nbformat.validate(nbformat.from_dict(nb))
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path.name, "validation=", vals, "label_book=", lbs)


if __name__ == "__main__":
    for path in PATHS:
        if path.exists():
            update(path)
