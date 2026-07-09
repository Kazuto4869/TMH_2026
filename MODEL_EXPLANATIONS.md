# Model Explanations

Tài liệu này giải thích ý nghĩa toán học và cách hoạt động của từng solver trong codebase `vrp_weekly`.

Mục tiêu không phải chứng minh tối ưu cho mọi solver. Mục tiêu là làm rõ:

- mỗi model đang biểu diễn bài toán như thế nào;
- biến/quyết định chính là gì;
- thuật toán chọn route ra sao;
- kết quả nên được hiểu như thế nào.

## 1. Bài toán chung

Codebase đang giải bài toán weekly VRP với multiple time windows:

- Một depot, ký hiệu `DEPOT`.
- Một xe / một courier.
- Bảy ngày, từ Monday `1` đến Sunday `7`.
- Mỗi ngày có tối đa một route:

```text
DEPOT -> customer_1 -> customer_2 -> ... -> customer_k -> DEPOT
```

- Mỗi customer có một đơn hàng và được giao tối đa một lần trong tuần.
- Customer có thể có nhiều time window ở nhiều ngày khác nhau.
- Nếu chưa giao hôm nay, customer có thể được dời sang ngày sau nếu còn window.
- Sau Sunday, customer chưa giao được tính là incomplete.
- Capacity đang bị bỏ qua.

Ký hiệu thường dùng:

```text
C          tập customer
D          tập ngày {1, 2, ..., 7}
0          depot
W_{i,d}    tập time window của customer i trong ngày d
s_i        service time của customer i
tau_{i,j}  travel time từ i sang j
d_{i,j}    khoảng cách Euclidean từ i sang j
```

Travel time được tính từ khoảng cách:

```text
tau_{i,j} = ceil(60 * d_{i,j} / 50)
```

Với tốc độ tối đa 50 km/h.

## 2. Feasibility chung

Một lịch hợp lệ phải thỏa:

```text
Mỗi customer giao tối đa một lần trong tuần.
Mỗi route ngày bắt đầu và kết thúc tại depot.
Nếu customer i được giao ở ngày d:
    chọn đúng một window w thuộc W_{i,d}
    service_start_i >= window_start_w
    service_start_i + s_i <= window_end_w
Route phải quay về depot trước 24:00.
Waiting được phép.
Capacity ignored.
```

Trong code, `evaluator.py` là nguồn kiểm tra feasibility chung cho heuristic. Các heuristic không tự viết lại checker riêng; chúng tạo sequence customer rồi gọi evaluator để mô phỏng arrival, waiting, service start, service end và return-to-depot.

## 3. Objective và metric

Benchmark objective đang dùng:

```text
objective =
    WEIGHT_INCOMPLETE * incomplete_count
  + WEIGHT_DEFERRAL * total_deferral_days
  + WEIGHT_DISTANCE_KM * total_distance_km
  + WEIGHT_WAITING_MIN * total_waiting_time_min
  + WEIGHT_ACTIVE_DAY * active_days
  + WEIGHT_ROUTE_DURATION_MIN * total_route_duration_min
```

Ý nghĩa ưu tiên:

1. Giảm `incomplete_count`.
2. Giảm `total_deferral_days`.
3. Giảm `total_distance_km`.
4. Giảm waiting/duration.

Deferral của customer:

```text
deferral_i = delivered_day_i - earliest_available_day_i
```

Không phải `delivered_day_i - 1`.

## 4. Nhóm heuristic greedy baseline

Các solver này xây route từng ngày theo luật tham lam. Chúng nhanh, dễ hiểu, nhưng không chứng minh tối ưu.

### 4.1. `nearest`

File:

```text
src/vrp_weekly/models/nearest.py
```

Ý tưởng:

Tại mỗi bước trong ngày, chọn customer khả thi gần nhất từ vị trí hiện tại.

Quyết định cục bộ:

```text
next = argmin distance(current_node, customer)
       với customer còn chưa giao và chèn tiếp vẫn feasible
```

Cách hoạt động:

1. Duyệt ngày 1 đến 7.
2. Lấy customer chưa giao có window trong ngày.
3. Từ depot, chọn customer gần nhất có thể đi tiếp.
4. Gọi evaluator để kiểm tra sequence mới có hard feasible không.
5. Lặp đến khi không thêm được customer.

Ý nghĩa:

- Dùng làm baseline khoảng cách.
- Rất nhanh.
- Dễ bỏ sót customer window hẹp hoặc customer cần ưu tiên vì chỉ nhìn bước gần nhất.

Nên dùng khi:

- cần kiểm tra dữ liệu/IO nhanh;
- cần baseline thấp để so sánh.

Không nên xem là solver mạnh cho mục tiêu giao đủ.

### 4.2. `deadline`

File:

```text
src/vrp_weekly/models/deadline.py
```

Ý tưởng:

Ưu tiên customer có deadline sớm nhất, tức time-window end nhỏ nhất.

Quyết định cục bộ:

```text
next = argmin earliest_window_end_today(customer)
```

Cách hoạt động:

1. Mỗi ngày lấy các customer chưa giao có window.
2. Sort theo window end sớm.
3. Thử thêm customer theo thứ tự đó.
4. Chỉ nhận nếu evaluator báo route feasible.

Ý nghĩa:

- Tốt hơn `nearest` khi nguy cơ chính là window đóng sớm.
- Vẫn tham lam, chưa cân bằng tốt giữa distance, deferral và khả năng phục vụ tương lai.

Nên dùng khi:

- muốn baseline theo deadline;
- muốn xem độ khó do window sớm/hẹp.

## 5. `min_deferral`

File:

```text
src/vrp_weekly/models/min_deferral.py
```

Ý tưởng:

Ưu tiên giao customer càng sớm càng tốt so với ngày khả dụng đầu tiên. Đây là heuristic insertion mạnh hơn greedy append vì nó thử chèn customer vào nhiều vị trí trong route hiện tại.

Mục tiêu logic:

```text
Ưu tiên chính: giảm incomplete_count
Ưu tiên tiếp: giảm deferral
Ưu tiên phụ: giảm route cost
```

Route secondary cost trong insertion:

```text
route_cost =
    distance_weight * route_distance_km
  + waiting_weight * route_waiting_time_min
  + duration_weight * route_duration_min
```

Cách hoạt động:

1. Duyệt ngày 1 đến 7.
2. Xét customer chưa giao có window hôm nay.
3. Customer khẩn hơn được xét trước, ví dụ last available day.
4. Với mỗi customer, thử tất cả vị trí chèn trong sequence hiện tại.
5. Gọi evaluator cho từng trial sequence.
6. Chọn insertion feasible có incremental cost tốt.
7. Sau khi route cơ bản ổn, post-fill thêm customer còn chèn được.

Ý nghĩa:

- Đây là baseline thực dụng mạnh trong codebase.
- Vì dùng evaluator cho từng insertion, route luôn bám đúng rule time-window.
- Không chứng minh tối ưu nhưng thường tốt cho `incomplete_count` và `total_deferral_days`.

Nên dùng khi:

- cần nghiệm nhanh, ổn định;
- mục tiêu chính là giao đủ hoặc gần đủ.

## 6. Inferior-first insertion

Solver:

```text
inferior_insertion
inferior_insertion_ls
```

File:

```text
src/vrp_weekly/models/inferior_insertion.py
```

Ý tưởng:

"Inferior" ở đây nghĩa là customer khó phục vụ, dễ bị mất cơ hội nếu không xử lý sớm. Solver ưu tiên customer khó trước thay vì customer rẻ trước.

Inferiority score:

```text
score(i, d) =
    1000 * I[d là last available day của i]
  + 200 / max(1, số ngày còn lại của i)
  + 500 / max(1, tổng độ rộng window hôm nay)
  + 50  / max(1, số window hôm nay)
  + 100 * deadline_pressure(i, d)
  + 20  * spatial_isolation(i)
```

Ý nghĩa từng thành phần:

- `last available day`: nếu không giao hôm nay thì customer sẽ incomplete.
- `số ngày còn lại`: càng ít ngày còn lại càng khẩn.
- `tổng độ rộng window`: window càng hẹp càng khó.
- `số window hôm nay`: càng ít lựa chọn càng khó.
- `deadline_pressure`: window đóng càng sớm càng áp lực.
- `spatial_isolation`: customer xa/cô lập dễ làm route khó.

Cách hoạt động:

1. Duyệt từng ngày.
2. Lấy candidate chưa giao có window hôm nay.
3. Nếu giới hạn candidate/ngày được bật, luôn giữ customer last-available-day.
4. Sort candidate theo `inferiority_score` giảm dần.
5. Ở mỗi vòng, thử best feasible insertion cho từng customer.
6. Chọn customer có score khó cao nhất, sau đó mới xét insertion cost.
7. Nếu là bản `_ls`, chạy local search/post-fill sau khi xây route.

Ý nghĩa:

- Giữ slot cho customer khó, tránh bị customer dễ chiếm route.
- Có thể tăng distance/deferral so với solver khác vì ưu tiên feasibility trước chi phí.

Nên dùng khi:

- dữ liệu có nhiều customer window hẹp hoặc last-day;
- muốn giảm rủi ro incomplete.

## 7. Regret dispatch insertion

Solver:

```text
regret_dispatch
regret_dispatch_ls
```

File:

```text
src/vrp_weekly/models/regret_dispatch_insertion.py
```

Ý tưởng:

Mỗi ngày, solver quyết định customer nào nên dispatch hôm nay và customer nào có thể defer. Một customer có priority cao nếu:

- hoãn customer đó nguy hiểm;
- hoặc customer chỉ có ít vị trí chèn tốt trong route.

Defer risk:

```text
defer_risk(i, d) =
    1000 * I[d là last available day của i]
  + 300 / max(1, số ngày còn lại của i)
  + 200 * future_window_loss(i, d)
  + 100 * deadline_pressure(i, d)
  + 500 / max(1, tổng độ rộng window hôm nay)
```

Insertion regret:

```text
nếu không có insertion feasible: -inf
nếu chỉ có một insertion feasible: 100
nếu có nhiều insertion:
    regret = second_best_incremental_cost - best_incremental_cost
```

Dispatch priority:

```text
priority =
    defer_risk_weight * defer_risk
  + regret_weight * insertion_regret
  - insertion_cost_weight * best_insertion_cost
```

Cách hoạt động:

1. Duyệt từng ngày.
2. Với từng customer candidate, liệt kê tất cả insertion feasible.
3. Tính defer risk, insertion regret và best insertion cost.
4. Chọn customer có dispatch priority cao nhất.
5. Apply best insertion.
6. Lặp đến khi không còn insertion feasible.
7. Bản `_ls` chạy thêm local search/post-fill.

Ý nghĩa:

- Cân bằng giữa "khách này có nguy hiểm nếu hoãn không?" và "chèn khách này bây giờ có đang rẻ/hiếm không?".
- Thường cân bằng hơn inferior-first vì vẫn trừ insertion cost.

Nên dùng khi:

- muốn nghiệm heuristic mạnh, giao đủ tốt nhưng không quá hy sinh route quality;
- muốn benchmark cạnh tranh với `min_deferral`.

## 8. Local search / VNS helper

File:

```text
src/vrp_weekly/heuristics/local_search.py
```

Local search không phải solver độc lập. Nó là bước cải thiện route sau khi một solver đã tạo route ban đầu.

Các operator:

```text
relocate          lấy một customer ra và chèn vào vị trí khác
swap              đổi vị trí hai customer
two_opt           đảo một đoạn sequence
remove_reinsert   bỏ một customer rồi chèn lại vào vị trí tốt nhất
post_fill         chèn thêm customer chưa giao nếu route vẫn feasible
```

Acceptance rule:

```text
Chỉ nhận candidate nếu:
    hard_feasible = True
    không có duplicate customer
    với relocate/swap/two_opt/remove_reinsert: giữ nguyên tập customer
    với post_fill: được phép tăng số customer giao

Ưu tiên:
    giao nhiều customer hơn
    distance thấp hơn
    waiting thấp hơn
    duration thấp hơn
```

Ý nghĩa:

- `_ls` solver = constructor heuristic + local search.
- Local search chủ yếu là intra-day, inter-day move/swap mặc định tắt.
- Không chứng minh tối ưu.

## 9. Hybrid Genetic VNS

Solver:

```text
hybrid_genetic_vns
```

File:

```text
src/vrp_weekly/models/hybrid_genetic_vns.py
```

Ý tưởng:

Đây là heuristic quần thể. Thay vì chỉ xây một route từ đầu đến cuối, solver tạo nhiều nghiệm ứng viên, đánh giá chúng, rồi có thể lai ghép/đột biến để tìm nghiệm tốt hơn.

Chromosome:

```text
day_gene:      customer_id -> planned_delivery_day
priority_gene: permutation của toàn bộ customer
```

Ý nghĩa:

- `day_gene[i] = d`: customer `i` được ưu tiên giao vào ngày `d`.
- `priority_gene`: thứ tự ưu tiên khi decode và post-fill.

Decode chromosome:

1. Repair chromosome để day hợp lệ với available days.
2. Duyệt ngày 1 đến 7.
3. Lấy customer có `day_gene = ngày hiện tại`.
4. Chèn từng customer bằng `best_feasible_insertion`.
5. Post-fill customer còn chưa giao nhưng có window hôm nay.
6. Nếu bật local search, cải thiện daily route.
7. Tính weekly score bằng evaluator.

Seed population:

Solver tạo nhiều seed khác nhau:

```text
nearest-like priority      khách gần depot trước
deadline-like priority     deadline sớm trước
min-deferral-like day      ngày khả dụng sớm nhất
inferior-like priority     khách khó trước
regret-like priority       defer risk cao trước
latest-day seeds           đưa một số seed về ngày khả dụng muộn nhất để giữ slot cho khách window muộn
random seeds               tạo đa dạng
```

GA loop:

```text
population = các chromosome seed đã decode và chấm điểm
best = nghiệm tốt nhất hiện tại

for generation in 1..G:
    giữ elite_size nghiệm tốt nhất
    chọn parent bằng tournament selection
    crossover day_gene bằng uniform crossover
    crossover priority_gene bằng order crossover
    mutate chromosome với xác suất mutation_rate
    repair -> decode -> evaluate
    cập nhật best
```

Giải thích tham số:

```text
population_size = số nghiệm ứng viên giữ trong mỗi generation
generations     = số vòng lai ghép/đột biến sau seed population
elite_size      = số nghiệm tốt nhất được giữ nguyên sang generation sau
mutation_rate   = xác suất đột biến child
crossover_rate  = xác suất lai ghép hai parent
time_limit_sec  = giới hạn thời gian tổng
```

`generations = 0` nghĩa là:

```text
Không chạy lai ghép/đột biến.
Nhưng solver vẫn:
    tạo population seed
    decode từng seed thành schedule
    chấm điểm từng schedule
    chọn seed tốt nhất
    chạy local search cuối nếu bật
```

Với dữ liệu hiện tại, seed population đã đủ tạo nghiệm giao `300/300`, nên profile này nhanh và ổn định:

```bash
python -m vrp_weekly.cli --locations data/locations.csv --time-windows data/time_windows.csv --solver hybrid_genetic_vns --ga-population-size 30 --ga-generations 0 --ga-time-limit-sec 120 --save-results
```

Ý nghĩa:

- Có khả năng đa dạng nghiệm hơn greedy.
- Có thể cải thiện nếu tăng generations/time limit.
- Không đảm bảo tốt hơn vì GA phụ thuộc seed, mutation và thời gian.

Nên dùng khi:

- muốn thử nghiệm heuristic population-based;
- muốn tìm nghiệm khác với `min_deferral`/`regret_dispatch`;
- có thời gian chạy dài hơn.

## 10. CP full week

Solver:

```text
cp_full_week
```

File:

```text
src/vrp_weekly/models/cp_full_week.py
```

Đây là model CP-SAT toàn tuần. Nó đưa tất cả ngày và customer vào một model duy nhất.

Biến chính:

```text
y[i,d]       1 nếu customer i được giao ở ngày d
u[i]         1 nếu customer i incomplete
g[i,d,w]     1 nếu chọn window w của customer i trong ngày d
T[i,d]       service start time của customer i trong ngày d
x[i,j,d]     1 nếu route ngày d đi trực tiếp từ i sang j
z[d]         1 nếu ngày d có active route
L[d]         departure time từ depot ngày d
R[d]         return time về depot ngày d
```

Ràng buộc assignment:

```text
sum_d y[i,d] + u[i] = 1
```

Nghĩa là mỗi customer hoặc được giao đúng một ngày, hoặc incomplete.

Ràng buộc window:

```text
sum_w g[i,d,w] = y[i,d]

Nếu g[i,d,w] = 1:
    T[i,d] >= window_start[i,d,w]
    T[i,d] + s_i <= window_end[i,d,w]
```

Ràng buộc route:

CP dùng `AddCircuit` cho từng ngày. Self-loop biểu diễn node không được route đi qua:

```text
x[i,i,d] + y[i,d] = 1
x[DEPOT,DEPOT,d] + z[d] = 1
AddCircuit(x[:,:,d])
```

Nếu customer được giao thì self-loop tắt và node phải nằm trong circuit. Nếu không giao thì self-loop bật.

Ràng buộc thời gian:

```text
Nếu x[i,j,d] = 1:
    T[j,d] >= T[i,d] + s_i + tau[i,j]

Nếu x[DEPOT,j,d] = 1:
    T[j,d] >= L[d] + tau[DEPOT,j]

Nếu x[i,DEPOT,d] = 1:
    R[d] >= T[i,d] + s_i + tau[i,DEPOT]
```

Objective:

```text
minimize
    incomplete_weight * sum_i u[i]
  + deferral_weight * sum_{i,d} (d - earliest_available_day_i) * y[i,d]
  + distance_weight * sum_{i,j,d} distance_cost[i,j] * x[i,j,d]
  + route_duration_weight * sum_d (R[d] - L[d])
```

Ý nghĩa:

- Model gần với mô hình toán học nhất.
- Có thể chứng nhận OPTIMAL/FEASIBLE theo CP-SAT nếu solve được.
- Nhưng quy mô tăng nhanh vì có biến theo customer, day, window và arc.

Nên dùng khi:

- cần mô hình exact cho instance nhỏ;
- cần kiểm tra formulation;
- không nên dùng làm default cho full 300-customer nếu time limit thấp.

## 11. CP rolling horizon

Solver:

```text
cp_rolling
```

File:

```text
src/vrp_weekly/models/cp_rolling_horizon.py
```

Ý tưởng:

Thay vì giải cả tuần một lần, solver giải từng ngày:

```text
undelivered = tất cả customer
for day in 1..7:
    chọn candidate hôm nay từ undelivered
    build CP-SAT daily model
    solve daily model
    remove delivered customer khỏi undelivered
```

Biến daily model:

```text
y[i]          1 nếu customer i được giao hôm nay
g[i,w]        1 nếu chọn window w của customer i hôm nay
T[i]          service start time
x[i,j]        1 nếu route hôm nay đi trực tiếp từ i sang j
z             1 nếu route hôm nay active
L             depot departure time
R             depot return time
next_travel[i]    travel time từ i đến successor
interval_end[i]   T[i] + service_i + next_travel[i]
```

Routing vẫn dùng `AddCircuit`:

```text
x[i,i] + y[i] = 1
x[DEPOT,DEPOT] + z = 1
AddCircuit(x)
```

Window constraints:

```text
sum_w g[i,w] = y[i]

Nếu g[i,w] = 1:
    T[i] >= window_start[i,w]
    T[i] + service_i <= window_end[i,w]
```

Interval strengthening:

Với mỗi customer được chọn:

```text
next_travel[i] = sum_j tau[i,j] * x[i,j]
interval_end[i] = T[i] + service_i + next_travel[i]
route_interval[i] = optional interval:
    start = T[i]
    size  = service_i + next_travel[i]
    end   = interval_end[i]
    present = y[i]
```

Depot interval:

```text
first_travel = sum_j tau[DEPOT,j] * x[DEPOT,j]
depot_interval_end = L + first_travel
route_interval[DEPOT] = optional interval from departure to first customer
```

NoOverlap:

```text
AddNoOverlap([depot_interval] + route_interval[i] for i in candidates)
```

Ý nghĩa:

- Mỗi interval biểu diễn service tại customer cộng travel sang successor.
- `NoOverlap` giúp CP-SAT hiểu các đoạn route không thể chồng lấn thời gian.
- Đây là strengthening, không thay thế `AddCircuit`.

Time propagation:

```text
Nếu x[i,j] = 1:
    interval_end[i] <= T[j]

Nếu x[DEPOT,j] = 1:
    depot_interval_end = T[j]

Nếu x[i,DEPOT] = 1:
    R = interval_end[i]
```

Two-phase objective mặc định:

Phase 1:

```text
maximize sum_i y[i]
```

Tức là giao nhiều customer nhất trong ngày.

Phase 2:

```text
fix delivered_count = phase1_delivered_count
minimize route distance + route duration - urgency bonus
```

Ý nghĩa:

- Ưu tiên không bỏ mất customer trong daily solve.
- Sau khi số customer giao đã cố định, mới tối ưu route quality.

Candidate filtering:

`cp_rolling` không đưa mọi undelivered customer vào mọi daily model nếu số lượng quá lớn. Nó lọc candidate theo chiến lược `hybrid` hoặc `urgent`:

```text
urgent      customer last-day, ít ngày còn lại, deadline sớm
easy        customer dễ chèn/near depot
deadline    window end sớm
isolated    customer xa/cô lập
```

Ý nghĩa:

- Giảm kích thước CP daily model.
- Có thể làm mất một số customer tốt nếu cap quá thấp.
- Vì là rolling, quyết định hôm nay có thể ảnh hưởng ngày sau.

Status cần hiểu:

```text
ALL_DAYS_OPTIMAL      mọi daily subproblem đạt OPTIMAL
FEASIBLE              có nghiệm feasible nhưng không chứng nhận optimal cho mọi ngày
global_optimality_claim = False
```

`cp_rolling` không chứng minh tối ưu toàn tuần.

Nên dùng khi:

- muốn solver CP-SAT thuần cho dữ liệu lớn hơn `cp_full_week`;
- muốn diagnostics CP chi tiết;
- chấp nhận rolling horizon không phải global optimum.

## 12. So sánh nhanh nên dùng solver nào

| Mục tiêu | Solver nên thử |
| --- | --- |
| Kiểm tra dữ liệu rất nhanh | `nearest`, `deadline` |
| Nghiệm thực dụng mạnh, nhanh | `min_deferral` |
| Giữ slot cho khách khó/window hẹp | `inferior_insertion`, `inferior_insertion_ls` |
| Cân bằng defer risk và insertion cost | `regret_dispatch`, `regret_dispatch_ls` |
| Thử nghiệm population-based / GA | `hybrid_genetic_vns` |
| Mô hình exact nhỏ | `cp_full_week` |
| CP-SAT rolling cho dữ liệu lớn | `cp_rolling` |

## 13. Cách đọc kết quả

Quan trọng nhất:

```text
hard_feasible = True
incomplete_count càng thấp càng tốt
total_deferral_days càng thấp càng tốt sau khi incomplete_count bằng nhau
total_distance_km càng thấp càng tốt sau deferral
```

Với heuristic:

```text
status = HEURISTIC_FEASIBLE
gap_percent thường rỗng
```

Điều này bình thường vì heuristic không có lower bound để tính optimality gap.

Với CP:

```text
status, best_bound, gap_percent có ý nghĩa theo CP-SAT subproblem
```

Riêng `cp_rolling`, gap/optimality là theo daily subproblem, không phải chứng minh global optimum toàn tuần.

## 14. Lưu ý giới hạn

- Heuristic không chứng minh tối ưu.
- `hybrid_genetic_vns` cần tuning nếu muốn tối ưu distance/deferral sâu hơn.
- Local search hiện chủ yếu intra-day; inter-day move/swap không bật mặc định.
- Capacity ignored.
- Không có multi-vehicle.
- Không có periodic routing.
- Không có column generation.
- Không dùng OR-Tools RoutingModel.
