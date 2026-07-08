# Weekly VRP With Multiple Time Windows

Codebase này dùng để mô hình hóa, giải thử nghiệm và so sánh các thuật toán cho bài toán định tuyến giao hàng trong một tuần với nhiều khung thời gian phục vụ cho mỗi khách hàng.

## 1. Mô tả bài toán

Bài toán hiện tại là một biến thể của **Vehicle Routing Problem with Multiple Time Windows (VRPMTW)** trên lịch một tuần. Mỗi khách hàng có một đơn hàng cần giao, nhưng có thể chấp nhận giao trong nhiều khung thời gian khác nhau. Các khung thời gian này có thể nằm ở nhiều ngày trong tuần, và một khách hàng cũng có thể có nhiều khung thời gian trong cùng một ngày.

Trong phiên bản đang được codebase triển khai, hệ thống xét:

- Một kho trung tâm, ký hiệu mặc định là `DEPOT`.
- Một nhân viên giao hàng / một xe.
- Chân trời lập kế hoạch 7 ngày, từ thứ Hai (`day_of_week = 1`) đến Chủ nhật (`day_of_week = 7`).
- Mỗi ngày xe xuất phát từ kho, đi qua một chuỗi khách hàng, rồi quay lại kho.
- Mỗi khách hàng được giao tối đa một lần trong toàn bộ tuần.
- Một đơn chưa giao ở ngày sớm hơn có thể được chuyển sang ngày khả dụng sau đó, nếu khách hàng còn khung thời gian hợp lệ.
- Đơn không giao được sau Chủ nhật được tính là `incomplete`.

### Dữ liệu đầu vào

Dữ liệu gồm hai file CSV chính:

- `data/locations.csv`: tọa độ, nhu cầu và thời gian phục vụ của kho/khách hàng.
- `data/time_windows.csv`: các khung thời gian giao hàng theo khách hàng và ngày trong tuần.

Mỗi location có tọa độ Descartes theo kilomet. Thời gian được lưu bằng định dạng `HH:MM` trong input và được chuyển sang số phút tính từ `00:00` khi xử lý nội bộ. Dữ liệu mẫu hiện có:

- 301 locations, gồm 1 depot và 300 khách hàng.
- 1359 time windows.
- Thời gian phục vụ khách hàng thuộc các giá trị 5, 7 hoặc 10 phút.
- Tổng demand là 818.7 kg.
- Mỗi khách hàng có tối thiểu 2, trung bình khoảng 3.66 và tối đa 6 ngày khả dụng.

### Ràng buộc đang xét

Một lịch giao hàng hợp lệ phải thỏa các ràng buộc chính sau:

- Xe bắt đầu và kết thúc mỗi tuyến ngày tại depot.
- Tuyến mỗi ngày phải quay về depot trước `24:00`.
- Thời gian di chuyển được tính từ khoảng cách Euclidean và vận tốc tối đa 50 km/h.
- Travel time được làm tròn lên bằng `ceil(60 * distance_km / 50)`.
- Có thể chờ nếu xe đến trước khi time window mở.
- Thời điểm bắt đầu phục vụ và kết thúc phục vụ phải nằm trong khung thời gian đã chọn.
- Một khách hàng không được giao lặp lại trong tuần.
- Capacity hiện chưa được kích hoạt vì cấu hình mặc định giả định xe đủ tải cho dữ liệu đang xét.

Depot departure được cho phép linh hoạt. Nếu tuyến ngày bắt đầu bằng một khách hàng có window muộn, hệ thống chọn giờ rời kho hợp lý để tránh tính thời gian chờ giả từ `00:00`.

### Mục tiêu đánh giá

Mục tiêu benchmark ưu tiên theo thứ tự:

1. Giảm số đơn không giao được.
2. Giao càng sớm càng tốt so với ngày khả dụng đầu tiên của từng khách hàng.
3. Giảm tổng quãng đường.
4. Giảm tổng thời gian chờ.
5. Giảm số ngày hoạt động, với trọng số nhỏ hơn.

Hàm mục tiêu báo cáo là:

```text
1_000_000 * incomplete_count
+ 10_000 * total_deferral_days
+ 10 * total_distance_km
+ total_waiting_time_min
+ 100 * active_days
```

Trong đó:

- `incomplete_count`: số khách hàng chưa giao được trong tuần.
- `total_deferral_days`: tổng số ngày bị dời, tính bằng `delivered_day - earliest_available_day`.
- `total_distance_km`: tổng quãng đường của tất cả các tuyến ngày.
- `total_waiting_time_min`: tổng thời gian xe phải chờ tại khách hàng.
- `active_days`: số ngày có ít nhất một điểm giao.

Các kết quả benchmark được sắp xếp theo `incomplete_count`, `total_deferral_days`, `total_distance_km`, rồi `total_waiting_time_min`.

## 2. Tổng quan codebase

Đây là một package Python có tên `vrp_weekly`, đặt trong thư mục `src/`. Codebase tách rõ các phần: đọc dữ liệu, biểu diễn instance, đánh giá nghiệm, solver, benchmark và export kết quả.

```text
.
├── data/                    # Input CSV mẫu
├── params/                  # Bản ghi tham số mặc định phục vụ báo cáo
├── results/                 # Kết quả chạy solver và benchmark
├── src/vrp_weekly/          # Source package chính
├── tests/                   # Unit tests
├── main.py                  # Menu chạy tương tác
├── pyproject.toml           # Metadata package và dependencies
└── README.md
```

### Các module chính

- `src/vrp_weekly/core.py`: dataclass lõi như `Location`, `TimeWindow`, `Instance`, `Stop`, `DailyRoute`, `WeeklySchedule`, `EvaluationMetrics`.
- `src/vrp_weekly/config.py`: hằng số cấu hình mặc định cho horizon, vận tốc, service time, objective, heuristic và CP.
- `src/vrp_weekly/io.py`: đọc `locations.csv`, đọc `time_windows.csv`, nhận diện depot, validate dữ liệu và tạo `Instance`.
- `src/vrp_weekly/distance.py`: tính khoảng cách Euclidean và travel time.
- `src/vrp_weekly/time_utils.py`: parse/format thời gian `HH:MM`.
- `src/vrp_weekly/evaluator.py`: mô phỏng tuyến ngày, chọn time window khả thi, validate lịch tuần và tính metrics.
- `src/vrp_weekly/export.py`: ghi `result.json`, `result.txt`, `daily_schedule.csv`, `incomplete_orders.csv`, run log và biểu đồ benchmark.
- `src/vrp_weekly/model_factory.py`: ánh xạ tên solver sang class tương ứng.
- `src/vrp_weekly/cli.py`: CLI để inspect dữ liệu hoặc chạy một solver.
- `src/vrp_weekly/benchmark.py`: chạy nhiều solver và tạo bảng so sánh.
- `src/vrp_weekly/compare_results.py`: so sánh các kết quả đã lưu mà không chạy lại solver.
- `main.py`: menu terminal để chọn solver và cấu hình nhanh CP-SAT.

## 3. Solver hiện có

Codebase hiện đăng ký các solver sau:

| Solver | Ý nghĩa |
| --- | --- |
| `nearest` | Greedy baseline, chọn khách hàng khả thi gần nhất tiếp theo. |
| `deadline` | Greedy baseline, ưu tiên khách hàng có deadline/time-window-end sớm nhất. |
| `regret` | Rolling-horizon regret insertion, có local search relocate, swap và 2-opt. |
| `cp_full_week` | Mô hình CP-SAT toàn tuần, dùng biến giao/ngày/window/cung và `AddCircuit`; phù hợp để minh họa mô hình hoặc chạy quy mô nhỏ. |
| `cp_rolling` | CP-SAT rolling horizon theo từng ngày; thực tế hơn cho dữ liệu lớn, có giới hạn candidate mỗi ngày. |

Tất cả solver trả về `WeeklySchedule`. Việc validate feasibility và tính metrics được làm tập trung trong `evaluator.py`, giúp so sánh solver nhất quán.

## 4. Cài đặt

Yêu cầu Python `>=3.11`.

Cài package ở chế độ editable:

```bash
python -m pip install -e .
```

Cài thêm gói phục vụ test và benchmark:

```bash
python -m pip install -e ".[all]"
```

Dependency chính:

- `ortools`: dùng cho CP-SAT solver.
- `pytest`: dùng cho test.
- `matplotlib`: dùng để export biểu đồ benchmark.
- `pandas`: được khai báo cho nhóm dependency benchmark, nhưng phần benchmark hiện ghi CSV trực tiếp để tránh xung đột import trong một số môi trường.

## 5. Kiểm tra dữ liệu

In summary của instance:

```bash
python -m vrp_weekly.cli \
  --locations data/locations.csv \
  --time-windows data/time_windows.csv \
  --summary
```

CLI sẽ đọc dữ liệu, validate cột bắt buộc, kiểm tra depot, kiểm tra khách hàng có time window và in thống kê cơ bản.

## 6. Chạy bằng menu tương tác

```bash
python main.py
```

Menu cho phép chọn một solver, nhiều solver hoặc tất cả solver. Nếu chọn CP-SAT, menu sẽ hỏi thêm time limit, số candidate, số worker và tùy chọn in log search của OR-Tools.

Kết quả được ghi vào `results/schedules/{solver}/`.

## 7. Chạy một solver bằng CLI

```bash
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver nearest
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver deadline
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver regret
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver cp_full_week --cp-max-customers 300 --cp-time-limit-sec 60
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver cp_rolling --cp-max-candidates-per-day 80 --cp-time-limit-per-day-sec 10
```

Thêm `--save-results` để ghi `result.json`, `result.txt`, `daily_schedule.csv` và `incomplete_orders.csv`.

## 8. Chạy benchmark

Chạy so sánh nhiều solver:

```bash
python -m vrp_weekly.benchmark \
  --locations data/locations.csv \
  --time-windows data/time_windows.csv \
  --solvers nearest deadline regret cp_rolling \
  --cp-max-candidates-per-day 80 \
  --cp-time-limit-per-day-sec 10
```

Xuất thêm report CSV chi tiết và biểu đồ:

```bash
python -m vrp_weekly.benchmark \
  --locations data/locations.csv \
  --time-windows data/time_windows.csv \
  --solvers nearest deadline regret cp_rolling \
  --cp-max-candidates-per-day 80 \
  --cp-time-limit-per-day-sec 10 \
  --export-report
```

Output chính:

- `results/comparison/benchmark_summary.csv`
- `results/comparison/delivered_count_by_solver.png`
- `results/comparison/incomplete_count_by_solver.png`
- `results/comparison/total_distance_km_by_solver.png`
- `results/comparison/total_waiting_time_min_by_solver.png`
- `results/schedules/{solver}/result.json`
- `results/schedules/{solver}/result.txt`
- `results/schedules/{solver}/daily_schedule.csv`
- `results/schedules/{solver}/incomplete_orders.csv`

## 9. So sánh kết quả đã lưu

Nếu đã có các file `results/schedules/{solver}/result.json`, có thể tạo lại bảng so sánh mà không chạy solver:

```bash
python -m vrp_weekly.compare_results --results-dir results
```

Thêm `--export-plots` để tạo lại biểu đồ:

```bash
python -m vrp_weekly.compare_results --results-dir results --export-plots
```

## 10. Định dạng output

Mỗi solver có thư mục riêng:

```text
results/schedules/{solver}/
├── result.json
├── result.txt
├── daily_schedule.csv
├── incomplete_orders.csv
└── run_log_{solver}_{timestamp}.csv
```

Ý nghĩa các file:

- `result.json`: kết quả đầy đủ dạng machine-readable, gồm solver status, metrics và toàn bộ schedule.
- `result.txt`: report đọc nhanh cho một lần chạy.
- `daily_schedule.csv`: danh sách điểm giao theo ngày, thứ tự, arrival, service start/end, selected window, travel và waiting.
- `incomplete_orders.csv`: khách hàng chưa giao được.
- `run_log_*.csv`: log tóm tắt runtime, status, gap và metrics của lần chạy.

## 11. Tham số mặc định

Nguồn cấu hình chính nằm ở `src/vrp_weekly/config.py`. Bản copy phục vụ báo cáo nằm ở `params/default_params.txt`.

Một số mặc định quan trọng:

- `NUM_DAYS = 7`
- `MAX_SPEED_KMPH = 50.0`
- `USE_CAPACITY = False`
- `ALLOW_MULTIPLE_WINDOWS_PER_DAY = True`
- `ALLOW_WAITING = True`
- `REQUIRE_SERVICE_END_WITHIN_WINDOW = True`
- `REQUIRE_RETURN_TO_DEPOT_EACH_DAY = True`
- `REQUIRE_RETURN_BEFORE_DAY_END = True`
- `FLEXIBLE_DEPOT_DEPARTURE = True`

## 12. Test

Chạy toàn bộ unit test:

```bash
python -m pytest
```

Test hiện tập trung vào logic heuristic/model helper, feasibility và các thao tác cải thiện tuyến.

## 13. Ghi chú giới hạn hiện tại

- Đây là mô hình một xe, chưa kích hoạt ràng buộc capacity.
- `cp_full_week` tạo nhiều biến cung theo ngày nên có thể rất lớn với 300 khách hàng; dùng `--cp-max-customers` khi cần giới hạn.
- `cp_rolling` giải từng ngày và carry-over đơn chưa giao, nên thực tế hơn cho dữ liệu lớn nhưng không phải tối ưu toàn tuần tuyệt đối.
- Các heuristic nhanh hơn và phù hợp baseline, nhưng không cung cấp optimality gap.
- Tất cả so sánh cuối cùng nên dựa trên metrics từ `evaluator.py`, không chỉ dựa vào objective nội bộ của từng solver.
