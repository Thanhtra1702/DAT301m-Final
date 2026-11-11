"""Add evaluation-only recovery without touching training cells or saved logs."""
import copy
import json
from pathlib import Path
import nbformat

path = Path(__file__).resolve().parents[1] / "submission.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
before = copy.deepcopy(nb)
cells = nb["cells"]

for cell in cells:
    if cell["cell_type"] != "markdown":
        continue
    s = "".join(cell["source"])
    s = s.replace("| 5 — Superior | Chưa có kết quả | — | Chưa có kết quả | — |",
                  "| 5 — Superior | Chưa khôi phục số chính xác | — | Khoảng 88% theo lần chạy người thực hiện quan sát; log chưa lưu | — |")
    s = s.replace("Các stage đạt Label Book Accuracy cao nhất là **3, 4**, cùng mức **82.69%**. Superior chưa có output huấn luyện để kết luận hiệu năng.",
                  "Trong các output còn lưu, Stage 3 và 4 cùng đạt 82.69% Label Book. Người thực hiện cho biết **Superior đã đạt khoảng 88% Label Book**, cao hơn các stage trước, nhưng log này chưa được lưu trong bản trên ổ đĩa. Cần đánh giá lại checkpoint để khôi phục số chính xác và loss; không xem Superior là chưa từng được chạy.")
    s = s.replace("Code Superior hiện đã có hàm tạo dữ liệu, nhưng các cell Superior **chưa có output chạy được lưu trong file này**. Vì vậy chưa báo cáo số ảnh thực tế hoặc accuracy của Superior từ lần chạy này.",
                  "**Kết quả cần khôi phục:** người thực hiện đã quan sát Label Book Accuracy của Superior khoảng **88%**. Output của lần chạy đó hiện không còn trong bản notebook trên ổ đĩa. Đây là kết quả được người thực hiện báo lại; chưa có số thập phân, validation accuracy hoặc loss để chép chính xác. Dùng mục **Khôi phục kết quả Superior** ở cuối notebook để đánh giá lại checkpoint, không huấn luyện lại.")
    s = s.replace("Superior hướng đến chia ảnh nguồn trước augmentation, nhưng chưa có kết quả huấn luyện lưu trong notebook để kết luận tốt hơn Stage 4.",
                  "Superior hướng đến chia ảnh nguồn trước augmentation. Theo người thực hiện, lần chạy Superior đạt khoảng 88% Label Book và là mức cao nhất đã quan sát; cần khôi phục đánh giá từ checkpoint để xác nhận số chính xác.")
    cell["source"] = s.splitlines(keepends=True)

note = '''## Khôi phục kết quả Superior — chỉ đánh giá checkpoint

Chạy cell bên dưới trên **kernel Colab đã dùng để huấn luyện**. Cell không gọi `fit`, không ghi đè checkpoint và không tạo lại log epoch. Nó nạp trọng số tốt nhất rồi tính lại accuracy/loss trên đúng validation và Label Book của Superior; việc này khôi phục được kết quả đánh giá nếu checkpoint và dữ liệu vẫn còn.

Nếu kernel chưa có các biến/hàm cần thiết, chạy lại hai cell khởi tạo ở đầu notebook và các cell High-Res → tạo dữ liệu Superior. **Không chạy cell huấn luyện 15 epoch**. Nếu checkpoint không còn, cell sẽ dừng để tránh báo kết quả của một mô hình khác. Sau khi hiện kết quả, nhấn **Ctrl+S** để lưu output trước khi Reload/Revert.
'''
code = '''# SUPERIOR_RECOVERY: evaluate only; never train or overwrite weights
from pathlib import Path

weight_path = Path("superior_best_model.weights.h5")
if not weight_path.is_file():
    raise FileNotFoundError(
        f"Không thấy checkpoint Superior tại {weight_path.resolve()}. "
        "Kiểm tra đúng kernel/thư mục Colab hoặc bản checkpoint đã tải về; chưa huấn luyện lại."
    )
required = ("tf", "build_fixed_vgg16_model", "SUPERIOR_VAL_DIR", "CLEAN_LB_DIR",
            "CLASS_NAMES", "IMAGE_SIZE", "BATCH_SIZE")
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        "Thiếu biến/hàm: " + ", ".join(missing) +
        ". Chạy cell khởi tạo và các cell High-Res/tạo dữ liệu Superior trước; bỏ qua cell fit."
    )
superior_root = Path(SUPERIOR_VAL_DIR).resolve().parent
if Path(CLEAN_LB_DIR).resolve() != superior_root / "label_book":
    raise ValueError("CLEAN_LB_DIR không thuộc bộ dữ liệu Superior. Chạy lại cell tạo dữ liệu Superior.")

superior_recovery_val = tf.keras.preprocessing.image_dataset_from_directory(
    SUPERIOR_VAL_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, shuffle=False,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
superior_recovery_lb = tf.keras.preprocessing.image_dataset_from_directory(
    CLEAN_LB_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, shuffle=False,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

superior_recovery_model = build_fixed_vgg16_model()
superior_recovery_model.load_weights(str(weight_path))
val_loss, val_acc = superior_recovery_model.evaluate(superior_recovery_val, verbose=0)
lb_loss, lb_acc = superior_recovery_model.evaluate(superior_recovery_lb, verbose=0)
superior_recovered_metrics = dict(val_accuracy=float(val_acc), val_loss=float(val_loss),
                                  label_book_accuracy=float(lb_acc), label_book_loss=float(lb_loss))
print("KẾT QUẢ SUPERIOR — ĐÁNH GIÁ LẠI CHECKPOINT")
print(f"Checkpoint: {weight_path.resolve()}")
print(f"Validation: {Path(SUPERIOR_VAL_DIR).resolve()}")
print(f"Label Book: {Path(CLEAN_LB_DIR).resolve()}")
print(f"Validation Accuracy: {val_acc*100:.2f}% | Loss: {val_loss:.4f}")
print(f"Label Book Accuracy: {lb_acc*100:.2f}% | Loss: {lb_loss:.4f}")
print("\nDòng Markdown để cập nhật bảng tổng kết:")
print(f"| 5 — Superior | {val_acc*100:.2f}% | {val_loss:.4f} | {lb_acc*100:.2f}% | {lb_loss:.4f} |")
print("Nhấn Ctrl+S để lưu output này vào notebook.")
'''
# Escape the literal newline in the generated code string.
code = code.replace('print("\nDòng Markdown', 'print("\\nDòng Markdown')
if not any("SUPERIOR_RECOVERY:" in "".join(c.get("source", [])) for c in cells):
    cells.append(dict(cell_type="markdown", metadata={}, source=note.splitlines(keepends=True)))
    cells.append(dict(cell_type="code", metadata={}, execution_count=None,
                      outputs=[], source=code.splitlines(keepends=True)))

for old, new in zip(before["cells"], cells):
    assert old.get("outputs") == new.get("outputs")
    assert old.get("execution_count") == new.get("execution_count")
    if old["cell_type"] == "code":
        assert old["source"] == new["source"]
for c in cells:
    if c["cell_type"] == "code":
        compile("".join(c["source"]), "notebook", "exec")
nbformat.validate(nbformat.from_dict(nb))
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Updated Superior status and appended evaluation-only recovery cell; existing code/outputs preserved.")
