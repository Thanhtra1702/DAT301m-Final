"""Show all six training stages while retaining saved training outputs."""
import copy
import json
from pathlib import Path

import nbformat

path = Path(__file__).resolve().parents[1] / "submission.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
before = copy.deepcopy(nb)

for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    if cell["cell_type"] == "markdown":
        source = source.replace(
            "Notebook khảo sát các bước làm sạch, chuẩn hóa và tăng cường dữ liệu",
            "Notebook gồm **6 stage huấn luyện, đánh số từ Stage 0 đến Stage 5**: Baseline, làm sạch, chuẩn hóa hình học, cân bằng 7.500 ảnh, mở rộng 9.500 ảnh và Superior 9.900 ảnh. **Phần 6 là đánh giá tổng hợp, không phải stage huấn luyện thứ bảy.**\n\nNotebook khảo sát các bước làm sạch, chuẩn hóa và tăng cường dữ liệu")
        source = source.replace("biểu đồ so sánh Stage 0–4", "biểu đồ so sánh đủ 6 stage (Stage 0–5)")
        source = source.replace("## 3. SO SÁNH KẾT QUẢ STAGE 0–4", "## 3. SO SÁNH ĐỦ 6 STAGE HUẤN LUYỆN: STAGE 0–5")
        source = source.replace(
            "Bảng chép các giá trị đánh giá đã lưu trong file này. Biểu đồ bên dưới ưu tiên biến kết quả trong kernel hiện tại; nếu chưa có biến thì dùng các số đã lưu này.",
            "Bảng và biểu đồ bao gồm đủ **Stage 0–5**. Stage 0–4 dùng kết quả đánh giá đã lưu, hoặc biến kết quả mới trong kernel. Stage 5 (Superior) ưu tiên kết quả đánh giá chính xác từ kernel hoặc cell khôi phục checkpoint. Khi chưa có số chính xác, biểu đồ ghi riêng **≈88% Label Book (theo người thực hiện, chưa khôi phục log)** bằng dấu rỗng; không tự đặt validation accuracy hoặc loss cho Superior.")
        source = source.replace(
            "Các số trên lấy từ phần **đánh giá sau huấn luyện** của từng stage trong chính file này, không lấy epoch có validation cao nhất nếu stage đó đánh giá trọng số cuối.",
            "Các số chính xác của Stage 0–4 lấy từ phần **đánh giá sau huấn luyện** trong chính file này, không lấy epoch có validation cao nhất nếu stage đó đánh giá trọng số cuối. Stage 5 hiện được ghi nhận theo kết quả khoảng 88% người thực hiện đã quan sát, chờ khôi phục log chính xác.")
        source = source.replace(
            "## 4. NHẬN XÉT TỪ KẾT QUẢ ĐÃ LƯU",
            "## 4. TỔNG KẾT 6 STAGE (0–5)")
        source = source.replace(
            "1. Label Book Accuracy từ Stage 0 đến Stage 4 thay đổi",
            "1. **Superior (Stage 5) là stage có Label Book Accuracy cao nhất theo lần chạy người thực hiện quan sát, khoảng 88%**; cần khôi phục log để xác nhận số chính xác và loss. Trong các output còn lưu, Label Book Accuracy từ Stage 0 đến Stage 4 thay đổi")
    elif "superior_model.fit(" in source:
        source = source.replace("KẾT QUẢ SUPERIOR STAGE 6:", "KẾT QUẢ SUPERIOR STAGE 5:")
        if "superior_metrics = dict(" not in source:
            source = source.replace(
                "lb_loss, lb_acc = superior_model.evaluate(clean_lb_ds, verbose=0)",
                "lb_loss, lb_acc = superior_model.evaluate(clean_lb_ds, verbose=0)\n"
                "superior_metrics = dict(val_accuracy=float(val_acc), val_loss=float(val_loss),\n"
                "                        label_book_accuracy=float(lb_acc), label_book_loss=float(lb_loss))")
    cell["source"] = source.splitlines(keepends=True)

plot = '''# So sánh đủ sáu stage; không tự điền kết quả chính xác cho Superior.
import math

stages = ["Stage 0\\nBaseline", "Stage 1\\nClean", "Stage 2\\nGeometric",
          "Stage 3\\n7.500 ảnh", "Stage 4\\n9.500 ảnh", "Stage 5\\nSuperior"]
val_accuracies = [round(float(globals().get(f"val_acc_s{i}", saved / 100)) * 100, 2)
                  for i, saved in enumerate([65.19, 60.18, 84.25, 90.07, 90.84])]
lb_accuracies = [round(float(globals().get(f"lb_acc_s{i}", saved / 100)) * 100, 2)
                 for i, saved in enumerate([48.08, 51.92, 71.15, 82.69, 82.69])]

# Kết quả khôi phục checkpoint hoặc kết quả Superior trong kernel hiện tại.
superior_result = globals().get("superior_recovered_metrics") or globals().get("superior_metrics")
if superior_result is None and "superior_model" in globals():
    superior_result = {"val_accuracy": globals().get("val_acc"),
                       "label_book_accuracy": globals().get("lb_acc")}
superior_result = superior_result or {}
def superior_percent(key):
    value = superior_result.get(key)
    return round(float(value) * 100, 2) if value is not None else float("nan")

val_accuracies.append(superior_percent("val_accuracy"))
lb_accuracies.append(superior_percent("label_book_accuracy"))

fig, ax = plt.subplots(figsize=(12, 6))
x = list(range(6))
ax.plot(x, val_accuracies, marker="o", linewidth=2.5, color="#e74c3c", label="Validation Accuracy (%)")
ax.plot(x, lb_accuracies, marker="s", linewidth=2.5, color="#2ecc71", label="Label Book Accuracy (%)")
for i, (v, lb) in enumerate(zip(val_accuracies, lb_accuracies)):
    if math.isfinite(v):
        ax.annotate(f"{v:.2f}%", (i, v), xytext=(0, 10), textcoords="offset points",
                    ha="center", color="#c0392b")
    if math.isfinite(lb):
        ax.annotate(f"{lb:.2f}%", (i, lb), xytext=(0, -18), textcoords="offset points",
                    ha="center", color="#218c53")
if not math.isfinite(lb_accuracies[5]):
    ax.scatter([5], [88], marker="D", s=90, facecolors="none", edgecolors="#218c53",
               label="Superior ≈88%: theo người thực hiện, chưa khôi phục log")
    ax.annotate("≈88% (chưa khôi phục log)", (5, 88), xytext=(-10, -22),
                textcoords="offset points", ha="right", color="#218c53")
if not math.isfinite(val_accuracies[5]):
    ax.text(5, 10, "Chưa khôi phục\\nvalidation", ha="center", color="#777777")
ax.set_xticks(x, stages)
ax.set_xlim(-0.3, 5.6)
ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy (%)")
ax.set_title("So sánh 6 stage huấn luyện: Stage 0–5", fontweight="bold")
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
plt.show()
'''
nb["cells"][42]["source"] = plot.splitlines(keepends=True)
assert not nb["cells"][42].get("outputs"), "Review existing plot before replacing it."
for old, new in zip(before["cells"], nb["cells"]):
    assert old.get("outputs") == new.get("outputs"), "Saved output changed"
    assert old.get("execution_count") == new.get("execution_count")
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), f"cell_{i+1}", "exec")
nbformat.validate(nbformat.from_dict(nb))
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Updated all six stages; corrected Superior label; saved outputs preserved.")
