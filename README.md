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
├── src/vrp_weekly/models/   # Tất cả solver/model đang được đăng ký
├── src/vrp_weekly/heuristics/ # Helper dùng chung cho heuristic và local search
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
- `src/vrp_weekly/export.py`: ghi `result.json`, `result.txt`, `daily_schedule.csv`, `incomplete_orders.csv`, `incomplete_diagnostics.csv`, run log và biểu đồ benchmark.
- `src/vrp_weekly/model_factory.py`: ánh xạ tên solver sang class tương ứng.
- `src/vrp_weekly/models/nearest.py`: baseline nearest feasible customer.
- `src/vrp_weekly/models/deadline.py`: baseline earliest feasible time-window end.
- `src/vrp_weekly/models/min_deferral.py`: baseline giảm deferral bằng feasible insertion, local search phụ và post-fill.
- `src/vrp_weekly/models/inferior_insertion.py`: runnable inferior-first insertion heuristic.
- `src/vrp_weekly/models/regret_dispatch_insertion.py`: runnable dispatch/defer regret insertion heuristic.
- `src/vrp_weekly/models/hybrid_genetic_vns.py`: runnable population-based hybrid genetic heuristic with VNS/local-search repair.
- `src/vrp_weekly/models/cp_rolling_repair.py`: runnable CP-SAT repair solver chạy sau `cp_rolling` trên neighborhood nhiều ngày bị giới hạn.
- `src/vrp_weekly/cp_diagnostics.py`: tiện ích diagnostic cho ablation candidate-cap của `cp_rolling`; không phải solver class.
- `src/vrp_weekly/heuristics/route_eval.py`: helper đánh giá route, insertion và secondary route cost; không chứa solver class.
- `src/vrp_weekly/heuristics/scoring.py`: helper chấm điểm deadline, remaining days, window width và isolation; không chứa solver class.
- `src/vrp_weekly/heuristics/local_search.py`: helper local search dùng chung, gồm relocate, swap, two_opt, remove_reinsert và post_fill; không phải solver độc lập.
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
| `min_deferral` | Baseline ưu tiên giảm deferral: chèn khách hàng cùng ngày theo độ khẩn cấp trước, rồi mới xét chi phí tuyến. |
| `inferior_insertion` | Inferior-first insertion heuristic, ưu tiên khách khó phục vụ: last available day, ít ngày còn lại, window hẹp, deadline sớm và spatial isolation. |
| `inferior_insertion_ls` | `inferior_insertion` cộng local search/post-fill nội ngày. |
| `regret_dispatch` | Dispatch/defer regret insertion heuristic, ưu tiên khách có defer risk và insertion regret cao. |
| `regret_dispatch_ls` | `regret_dispatch` cộng local search/post-fill nội ngày. |
| `hybrid_genetic_vns` | Population-based hybrid genetic heuristic với insertion repair, earliest/latest-day seed population và VNS/local-search improvement. |
| `cp_full_week` | Mô hình CP-SAT toàn tuần, dùng biến giao/ngày/window/cung và `AddCircuit`; phù hợp để minh họa mô hình hoặc chạy quy mô nhỏ. |
| `cp_rolling` | CP-SAT rolling horizon theo từng ngày; thực tế hơn cho dữ liệu lớn, có giới hạn candidate mỗi ngày. |
| `cp_rolling_repair` | Chạy `cp_rolling` trước, sau đó dùng CP-SAT thuần để repair một neighborhood nhiều ngày bị giới hạn và chỉ nhận nếu lịch tuần tốt hơn theo evaluator. |

Tất cả solver trả về `WeeklySchedule`. Việc validate feasibility và tính metrics được làm tập trung trong `evaluator.py`, giúp so sánh solver nhất quán.

Xem thêm [MODEL_EXPLANATIONS.md](MODEL_EXPLANATIONS.md) để đọc giải thích chi tiết về mô hình toán, cách hoạt động và ý nghĩa của từng solver.

### Cách hiểu từng solver

- `nearest`: baseline tham lam đơn giản. Mỗi ngày bắt đầu từ depot, solver thử giao khách khả thi gần nhất tiếp theo. Model này nhanh và dễ kiểm tra dữ liệu, nhưng thường bỏ sót nhiều khách vì chỉ nhìn khoảng cách cục bộ.
- `deadline`: baseline tham lam theo deadline. Solver ưu tiên khách có `time_window.end_minute` sớm nhất trong ngày. Model này tốt hơn `nearest` khi window hẹp/sớm là nguyên nhân chính gây infeasible, nhưng vẫn có thể tạo route dài hoặc bỏ sót khách do không xét deferral toàn tuần đủ mạnh.
- `min_deferral`: heuristic insertion ưu tiên giảm số ngày bị dời. Solver cố giao khách vào ngày sớm khả dụng, dùng evaluator để kiểm tra insertion khả thi, rồi post-fill thêm khách còn chèn được. Đây là baseline thực dụng mạnh cho mục tiêu `incomplete_count` và `total_deferral_days`.
- `inferior_insertion`: heuristic "khách khó trước". Khách được xem là khó nếu đang ở last available day, còn ít ngày khả dụng, window hẹp, deadline sớm hoặc bị cô lập về không gian. Mục tiêu là giữ slot cho các khách dễ mất cơ hội giao.
- `inferior_insertion_ls`: giống `inferior_insertion`, sau đó chạy local search/post-fill nội ngày để cải thiện thứ tự route và thêm khách còn chèn được. Thường tốt hơn bản không `_ls`, đổi lại runtime cao hơn.
- `regret_dispatch`: heuristic dispatch/defer. Mỗi bước so sánh rủi ro nếu hoãn khách sang ngày sau với chi phí chèn khách vào route hôm nay. Khách có defer risk cao hoặc chỉ có ít vị trí chèn tốt sẽ được ưu tiên.
- `regret_dispatch_ls`: giống `regret_dispatch`, có thêm local search/post-fill. Đây là lựa chọn heuristic cân bằng tốt khi muốn giao đủ khách nhưng vẫn giữ deferral/distance hợp lý.
- `hybrid_genetic_vns`: heuristic quần thể. Solver tạo nhiều chromosome ban đầu, mỗi chromosome gồm `day_gene` cho biết khách dự kiến giao ngày nào và `priority_gene` cho biết thứ tự ưu tiên toàn tuần. Sau đó decode bằng feasible insertion, có thể lai ghép/đột biến qua nhiều generation, rồi dùng local search/VNS để repair/cải thiện. Model này không chứng minh tối ưu; chất lượng phụ thuộc seed, population, generations và time limit.
- `cp_full_week`: mô hình CP-SAT toàn tuần. Model này gần với mô hình toán học nhất, nhưng chỉ phù hợp quy mô nhỏ vì số biến/cung tăng nhanh khi đưa cả tuần vào một model.
- `cp_rolling`: CP-SAT rolling horizon. Solver giải từng ngày một bằng `AddCircuit`, time-window variables, optional interval/`NoOverlap` strengthening và candidate filtering. Model này giữ cấu trúc CP thuần, không dùng heuristic fallback, nhưng vì rolling từng ngày nên không chứng minh tối ưu toàn tuần.
- `cp_rolling_repair`: CP-SAT repair sau rolling. Solver giữ nguyên các ngày ngoài neighborhood, bảo toàn mọi khách đã giao ở lịch base, và chỉ accept lịch repaired nếu evaluator xác nhận hard-feasible, không duplicate, không mất khách đã giao và cải thiện theo thứ tự `incomplete_count`, `total_deferral_days`, `total_distance_km`.

### Tham số `hybrid_genetic_vns`

- `--ga-population-size 30`: mỗi generation giữ tối đa 30 nghiệm ứng viên. Population lớn hơn giúp đa dạng nghiệm hơn, nhưng decode/evaluate nhiều hơn nên chạy lâu hơn.
- `--ga-generations 0`: chỉ dùng seed population ban đầu rồi chọn nghiệm tốt nhất sau insertion repair/local search. Đây không phải là "không chạy solver"; solver vẫn tạo 30 nghiệm seed, decode từng nghiệm, chấm điểm và export nghiệm tốt nhất. Với dữ liệu hiện tại, seed mới đã đủ giao `300/300`, nên `generations=0` là profile nhanh và ổn định để lấy nghiệm feasible.
- `--ga-generations 20` hoặc `50`: chạy thêm các vòng lai ghép và đột biến sau seed population. Dùng khi muốn thử giảm distance/deferral thêm, nhưng runtime tăng và không đảm bảo tốt hơn nếu time limit quá thấp.
- `--ga-time-limit-sec`: giới hạn thời gian tổng cho GA. Nếu hết giờ giữa chừng, solver trả nghiệm tốt nhất đã tìm được.
- `--local-search-time-limit-sec`: thời gian cải thiện route cho local search/VNS ở mỗi decode cuối. Tăng giá trị này có thể giúp route tốt hơn nhưng runtime tăng rõ.

Gợi ý chạy genetic hiện tại:

```bash
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver hybrid_genetic_vns --ga-population-size 30 --ga-generations 0 --ga-time-limit-sec 120 --save-results
```

Nếu muốn thử tối ưu thêm sau khi đã có nghiệm giao đủ:

```bash
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver hybrid_genetic_vns --ga-population-size 30 --ga-generations 20 --ga-time-limit-sec 300 --save-results
```

Ghi chú CP:

- `cp_full_week` chỉ là mô hình minh họa/toán học cho instance nhỏ; mặc định chỉ lấy tối đa 40 khách hàng.
- `cp_rolling` là CP-SAT rolling horizon độc lập, giải từng ngày với `AddCircuit` trên tập candidate được chọn. Solver này không dùng fallback, replacement route hoặc hint từ `min_deferral`.
- Default chính thức của `cp_rolling` là `adaptive_daily_deadline=True`, `optimization_mode="full_three_stage"`, `stage2_max_time_fraction=0.10`, `time_limit_per_day_sec=60`, `max_candidates_per_day=80`, `num_workers=4`, `candidate_strategy="hybrid"`, `use_decision_strategy=True`, `use_service_no_overlap=True`, `use_route_interval_no_overlap=True`.
- Chế độ `full_three_stage` là chế độ bình thường/default. Stage 1A tối đa hóa số khách mandatory last-available-day được giao; Stage 1B giữ nguyên mandatory count rồi tối đa hóa tổng số khách giao; Stage 2 giữ nguyên mandatory count và total delivered count rồi tối ưu chất lượng route.
- `service_phases_only` và `mandatory_stage_only` là diagnostic mode. `--cp-phase1-only` chỉ còn là alias deprecated cho `--cp-service-phases-only`, không có nghĩa là chỉ chạy Stage 1A.
- `main.py` không hỏi optimization mode, adaptive/fixed split hay Stage 2 fraction cho `cp_rolling`; các tùy chọn nâng cao này nằm trong CLI.
- `cp_rolling_repair` chạy `cp_rolling` với default `full_three_stage`, rồi repair một neighborhood nhiều ngày bằng CP-SAT thuần. Hints của repair chỉ đến từ lịch CP base đã khả thi, không đến từ nearest/min_deferral/regret/GA/local search.
- `cp_rolling_repair` bảo toàn mọi khách đã hoàn thành ở base schedule và chỉ accept lịch tuần repaired nếu evaluator cho hard-feasible và cải thiện lexicographic.
- Objective khoảng cách trong CP dùng scale theo kilomet: `round(distance_weight * distance_km)`, không dùng meter-scale `distance_km * 1000`.
- Route return time trong output CP được tính lại từ sequence thực tế, không lấy trực tiếp biến `R` nội bộ của CP-SAT.
- `cp_rolling` dùng multiple time-window selection, impossible arc fixing, candidate filtering, two-phase objective mặc định, service `NoOverlap`, round-trip lower bounds, và các ràng buộc tightening như degree linking, arc linking, window-pair cuts, pair conflict cuts, depot-window cuts, dominated-window cuts và precedence cuts.
- `cp_rolling` dùng thêm optional route intervals theo cấu trúc service + travel-to-successor cho từng khách hàng được chọn, một depot interval cho travel từ kho đến khách đầu tiên, và `NoOverlap` trên các route intervals này. Đây là ràng buộc strengthen CP-SAT, không thay thế `AddCircuit`.
- Diagnostics của `cp_rolling` có `route_interval_count`, `depot_interval_enabled`, `no_overlap_route_intervals_enabled`, `total_route_interval_count`, `route_no_overlap_days`, pipeline candidate/stage theo ngày, và final `incomplete_customer_diagnostics`.
- Khi lưu kết quả, `cp_rolling` xuất thêm `results/schedules/cp_rolling/incomplete_diagnostics.csv`. Candidate-cap ablation ghi `results/comparison/cp_rolling_candidate_cap_60s.csv` và `results/comparison/cp_rolling_candidate_cap_300s.csv`; so sánh repair ghi `results/comparison/cp_rolling_repair_60s.csv`.
- Candidate filtering của `cp_rolling` mặc định dùng chiến lược `hybrid`, trộn nhóm urgent/easy/deadline/isolated; có thể đổi về `urgent` bằng `--cp-candidate-strategy urgent`.
- Nếu daily CP không tìm được nghiệm khả thi, hệ thống báo đúng CP status và trả route rỗng cho ngày đó; không vá bằng heuristic.
- `cp_rolling` không claim tối ưu toàn tuần. `global_optimality_claim = False`; nếu tất cả subproblem ngày tối ưu thì status là `ALL_DAYS_OPTIMAL`, còn nghiệm rolling thực tế thường là `FEASIBLE`.
- Column generation, capacity, multi-vehicle, batch splitting không được triển khai trong codebase hiện tại.

Ghi chú heuristic:

- Tất cả runnable solver/model class nằm trong `src/vrp_weekly/models/` và được đăng ký qua `model_factory.py`.
- Helper dùng chung nằm trong `src/vrp_weekly/heuristics/`; các file helper không chứa runnable solver class.
- Local search trong `heuristics/local_search.py` là bước cải thiện route/schedule, không phải standalone constructor hay solver được đăng ký.
- Các heuristic không chứng minh tối ưu; so sánh cuối cùng vẫn dựa trên evaluator metrics và benchmark output.
- Bài toán vẫn là một xe, một route mỗi ngày, capacity ignored.

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

Menu cho phép chọn một solver, nhiều solver hoặc tất cả solver. Nếu chọn `cp_rolling`, menu chỉ hỏi rolling time limit mỗi ngày, số candidate mỗi ngày và số worker; các tùy chọn nâng cao vẫn nằm trong CLI. Nếu chọn `cp_rolling_repair`, menu hỏi thêm repair total time limit và repair maximum days.

Kết quả được ghi vào `results/schedules/{solver}/`.

## 7. Chạy một solver bằng CLI

```bash
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver nearest
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver deadline
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver min_deferral
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver regret_dispatch_ls --save-results
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver hybrid_genetic_vns --ga-population-size 30 --ga-generations 0 --ga-time-limit-sec 120 --save-results
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver cp_full_week --cp-max-customers 40 --cp-time-limit-sec 60
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver cp_rolling --cp-max-candidates-per-day 80 --cp-time-limit-per-day-sec 60 --cp-three-stage --cp-candidate-strategy hybrid
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver cp_rolling_repair --cp-repair-time-limit-sec 300 --cp-repair-max-days 2 --save-results
```

Thêm `--save-results` để ghi `result.json`, `result.txt`, `daily_schedule.csv` và `incomplete_orders.csv`.

## 8. Chạy benchmark

Chạy so sánh nhiều solver:

```bash
python -m vrp_weekly.benchmark \
  --locations data/locations.csv \
  --time-windows data/time_windows.csv \
  --solvers nearest deadline min_deferral inferior_insertion_ls regret_dispatch_ls hybrid_genetic_vns cp_rolling \
  --cp-max-candidates-per-day 80 \
  --cp-time-limit-per-day-sec 10 \
  --cp-two-phase-objective \
  --cp-candidate-strategy hybrid
```

Xuất thêm report CSV chi tiết và biểu đồ:

```bash
python -m vrp_weekly.benchmark \
  --locations data/locations.csv \
  --time-windows data/time_windows.csv \
  --solvers nearest deadline min_deferral cp_rolling \
  --cp-max-candidates-per-day 80 \
  --cp-time-limit-per-day-sec 10 \
  --cp-two-phase-objective \
  --cp-candidate-strategy hybrid \
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

Chạy diagnostic grid riêng cho `cp_rolling`:

```bash
python -m vrp_weekly.benchmark \
  --locations data/locations.csv \
  --time-windows data/time_windows.csv \
  --solvers cp_rolling \
  --cp-diagnostic-grid \
  --export-report
```

Grid này thử các giới hạn candidate `30, 40, 50, 60, 80` và time limit/ngày `30, 60, 120`, rồi ghi `results/comparison/cp_diagnostic_grid.csv`.
File grid được ghi tăng dần sau từng profile; nếu run bị dừng, chạy lại cùng lệnh sẽ bỏ qua profile đã có trong CSV.

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
├── incomplete_diagnostics.csv   # nếu solver có diagnostics CP rolling
└── run_log_{solver}_{timestamp}.csv
```

Ý nghĩa các file:

- `result.json`: kết quả đầy đủ dạng machine-readable, gồm solver status, metrics và toàn bộ schedule.
- `result.txt`: report đọc nhanh cho một lần chạy.
- `daily_schedule.csv`: danh sách điểm giao theo ngày, thứ tự, arrival, service start/end, selected window, travel và waiting.
- `incomplete_orders.csv`: khách hàng chưa giao được.
- `incomplete_diagnostics.csv`: pipeline diagnostic cho khách incomplete của `cp_rolling`/`cp_rolling_repair` khi có trong solver status.
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

Các trọng số cần chỉnh khi tuning:

- Metrics/evaluator P-weights nằm trong `src/vrp_weekly/config.py`: `WEIGHT_INCOMPLETE`, `WEIGHT_DEFERRAL`, `WEIGHT_DISTANCE_KM`, `WEIGHT_WAITING_MIN`, `WEIGHT_ACTIVE_DAY`, `WEIGHT_ROUTE_DURATION_MIN`.
- CP rolling drop penalties nằm trong `src/vrp_weekly/config.py`: `DROP_PENALTY_BY_DAY`.
- CP objective defaults đi qua `src/vrp_weekly/model_factory.py`: `distance_weight=10`, `route_duration_weight=1`, `urgency_weight=100`.
- CP rolling defaults: `adaptive_daily_deadline=True`, `optimization_mode="full_three_stage"`, `stage2_max_time_fraction=0.10`, `time_limit_per_day_sec=60`, `max_candidates_per_day=80`, `num_workers=4`, `candidate_strategy="hybrid"`, `use_service_no_overlap=True`, `use_route_interval_no_overlap=True`, `use_decision_strategy=True`.
- `min_deferral` secondary route-cost defaults nằm trong `src/vrp_weekly/models/min_deferral.py`: `distance_weight=10.0`, `WAITING_WEIGHT`, `duration_weight=0.0`.
- Bản ghi report-friendly nằm trong `params/default_params.txt`.

## 12. Test

Chạy toàn bộ unit test:

```bash
python -m pytest
```

Test hiện tập trung vào logic heuristic/model helper, feasibility và các thao tác cải thiện tuyến.

## 13. Quy ước bảo trì

- Mọi thay đổi liên quan đến hệ thống, solver, CLI, benchmark, output, dữ liệu hoặc cách chạy phải cập nhật README trong cùng lượt thay đổi.
- Solver/model mới phải được đặt trong `src/vrp_weekly/models/` và đăng ký qua `src/vrp_weekly/model_factory.py`.
- Không tạo lại package `src/vrp_weekly/solvers/`; repo hiện dùng `models/` làm nơi chứa solver.
- Solver cũ `regret` đã bị gỡ khỏi code path chính; baseline thứ ba hiện chỉ là `min_deferral`.

## 14. Ghi chú giới hạn hiện tại

- Đây là mô hình một xe, chưa kích hoạt ràng buộc capacity.
- `cp_full_week` tạo nhiều biến cung theo ngày nên có thể rất lớn với 300 khách hàng; dùng `--cp-max-customers` khi cần giới hạn.
- `cp_rolling` giải từng ngày và carry-over đơn chưa giao, nên thực tế hơn cho dữ liệu lớn nhưng không phải tối ưu toàn tuần tuyệt đối.
- `cp_rolling_repair` chỉ tối ưu một neighborhood nhiều ngày bị giới hạn; nó không chứng minh tối ưu toàn tuần và có thể không rescue được đơn nếu neighborhood hoặc time limit quá nhỏ.
- Các heuristic nhanh hơn và phù hợp baseline, nhưng không cung cấp optimality gap.
- Tất cả so sánh cuối cùng nên dựa trên metrics từ `evaluator.py`, không chỉ dựa vào objective nội bộ của từng solver.
