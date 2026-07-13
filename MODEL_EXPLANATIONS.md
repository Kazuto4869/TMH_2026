# Mô tả các model

## Baseline

- `nearest`: chọn khách khả thi gần điểm cuối tuyến nhất.
- `deadline`: chọn khách có thời điểm đóng khung sớm nhất.
- `min_deferral`: ưu tiên khách ít cơ hội phục vụ trong các ngày sau, rồi chọn
  vị trí chèn có chi phí nhỏ.

## CP rolling

`cp_rolling` giải bài toán theo từng ngày. Ở ngày `d`, mô hình CP-SAT xét các
khách còn lại trong cửa sổ lăn, cố định lịch ngày `d`, rồi chuyển phần còn lại
sang ngày tiếp theo. Các ràng buộc chính gồm tuyến, thời gian, khung giờ và
quay về kho.

## Regret insertion + local search

Code chạy chính nằm ở `regret_ls/scheduler.py` và được nối vào framework qua
`src/vrp_weekly/models/regret_ls_adapter.py`.

Mỗi ngày, thuật toán tạo tập khách có khung giờ mở. Các khách ở ngày cuối cùng
có thể phục vụ được chèn trước. Với mỗi khách, thuật toán thử các vị trí chèn
khả thi và tính hai chi phí tốt nhất:

```text
regret(i) = cost_second(i) - cost_best(i)
```

Khách có regret lớn được chọn trước; vị trí có chi phí nhỏ nhất được sử dụng.
Sau bước chèn, route được cải thiện bằng 2-opt và Or-opt. Khách chưa chèn được
giữ lại cho ngày sau; khách còn lại sau ngày 7 là `incomplete`.

Chạy:

```bash
python -m vrp_weekly.cli --solver regret_ls --save-results
```

## Inferior-first insertion

Thuật toán chấm điểm độ khó phục vụ dựa trên số khung giờ, độ rộng khung,
deadline và khoảng cách đến kho. Khách khó được xét trước, sau đó chèn vào vị
trí khả thi có chi phí nhỏ.

## Kết quả mới của Regret-LS

Kết quả lưu tại `results/schedules/regret_ls/`. Lần chạy hiện tại giao 299/300
khách, quãng đường 1365.41 km và objective 64994.12.
