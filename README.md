# Weekly VRP with Multiple Time Windows

Mô hình lập lịch giao hàng cho một xe, một kho và 7 ngày. Mỗi khách hàng có
thể có nhiều khung thời gian nhận hàng. Mục tiêu dùng trong evaluator là

```text
F = 1000 * incomplete + 100 * deferral + 10 * distance + waiting
```

## Các solver

| Tên | Vai trò |
|---|---|
| `nearest` | Baseline: chọn khách gần vị trí hiện tại. |
| `deadline` | Baseline: ưu tiên khung giờ đóng sớm. |
| `min_deferral` | Baseline: ưu tiên giảm trì hoãn, sau đó chèn theo chi phí tuyến. |
| `cp_rolling` | CP-SAT theo cửa sổ lăn từng ngày. |
| `regret_dispatch` | Regret insertion theo logic trong `regret_ls/scheduler.py`. |
| `regret_ls` | Tên gọi mới của Regret insertion kèm 2-opt/Or-opt. |
| `inferior_insertion` | Chèn trước các khách khó phục vụ. |

## Chạy thử

```bash
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver regret_ls --save-results
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver cp_rolling --save-results
```

Kết quả được lưu trong `results/schedules/<solver>/`. Dữ liệu đầu vào nằm ở
`data/locations.csv` và `data/time_windows.csv`.

## Cấu trúc chính

```text
src/vrp_weekly/core.py                 dữ liệu và WeeklySchedule
src/vrp_weekly/evaluator.py            tính mục tiêu và kiểm tra khả thi
src/vrp_weekly/model_factory.py        đăng ký solver
src/vrp_weekly/models/                 các model đang dùng
src/vrp_weekly/heuristics/             hàm chèn, đánh giá tuyến, local search
regret_ls/scheduler.py                 Regret insertion + local search
```
