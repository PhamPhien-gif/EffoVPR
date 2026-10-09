# EffoVPR-R evaluation trên Modal

Nếu bạn đang dùng **Modal Notebooks trên web**, mở [EffoVPR_R_Evaluation_Modal.ipynb](../EffoVPR_R_Evaluation_Modal.ipynb), upload notebook cùng các file model/index vào `root`, rồi làm theo các cell. Notebook đã nhúng 25 module source của repo nên không cần clone hay upload thêm source. Chạy Preflight với CPU trước; sau khi `PASS`, chọn GPU, restart kernel và bấm Run All. Đặt `LIMIT=3` để smoke run trước, sau đó đổi `LIMIT=0` cho full run.

Nếu muốn chạy từ terminal qua Modal CLI, dùng [modal_evaluation.py](../modal_evaluation.py) theo hướng dẫn bên dưới. Cả hai cách đọc nguyên gallery index đã upload (`gallery.index`, không build lại gallery), và tạo `results.json`, `query_predictions.csv` cùng `failure_cases.csv` như workflow cũ. Split giữ seed 42, tỉ lệ 70/10/10/10, nhóm ảnh trùng nằm cùng split; mặc định đánh giá `test_queries`.

## Modal Notebook trên web

1. Upload `EffoVPR_R_Evaluation_Modal.ipynb` và sáu artifact (`config.json`, `model.safetensors`, `gallery.index`, `gallery_features.npy`, `gallery_metadata.parquet`, `index_config.json`) vào Files > root.
2. Chọn CPU trong Compute profile rồi chạy cell 1 đến hết **Preflight CPU**. Sửa đường dẫn hoặc artifact nếu preflight chưa in `PASS`.
3. Đặt `LIMIT=3`, chọn GPU trong Compute profile, restart kernel, rồi bấm Run All. Dataset sẽ được tải và gallery metadata được đối chiếu với split trước khi model inference bắt đầu.
4. Nếu smoke thành công, đổi `LIMIT=0` và chạy lại notebook để lấy metric test đầy đủ. Các output nằm trong `EffoVPR_R_Evaluation_Modal/artifacts/`.

Notebook dùng dependency pins giống workflow Kaggle hiện tại, giữ nguyên Torch có sẵn trong Modal Notebook image và yêu cầu kernel CUDA cho cell evaluation. Hãy terminate kernel sau khi xong để dừng compute.

## 1. Chuẩn bị các file đầu vào

Trên máy local, gom đúng các file thành cấu trúc sau. `model/` phải là thư mục chứa EffoVPR artifact; `index/` chứa đủ bốn file gallery. Tên file và chữ hoa/chữ thường cần khớp.

```text
modal-input/
  model/
    config.json
    model.safetensors
  index/
    gallery.index
    gallery_features.npy
    gallery_metadata.parquet
    index_config.json
```

Gallery metadata cần có `image_id`, `label`, `dataset_index`. Index phải là `IndexFlatIP`, vector L2-normalized. Index phải được tạo từ cùng `PhamPhien/VN_Attractions` train split, seed 42, gallery split và thứ tự mẫu như notebook cũ. Nếu file `README.md` hoặc file khác có trong artifact thì không cần upload.

## 2. Khai báo Modal và tải input lên

Cài Modal CLI, xác thực tài khoản, rồi tạo hai volume rỗng. Lệnh tạo volume chỉ cần chạy một lần:

```powershell
python -m pip install modal
modal setup
modal volume create effovpr-eval-input
modal volume create effovpr-eval-output
```

Từ thư mục gốc repo, upload từng thư mục vào đúng vị trí:

```powershell
modal volume put --force effovpr-eval-input .\modal-input\model model
modal volume put --force effovpr-eval-input .\modal-input\index index
```

Tên volume và mount path nằm đầu `modal_evaluation.py`. Nếu đổi volume name, đổi các hằng số `INPUT_VOLUME_NAME` / `OUTPUT_VOLUME_NAME` trước khi chạy. Dataset công khai được tải từ Hugging Face bên trong container Modal; không cần đưa ảnh dataset vào volume.

## 3. Chạy preflight trước

```powershell
modal run modal_evaluation.py --stage validate
```

Đây là CPU-only và không gọi model inference. Nó kiểm tra sự tồn tại của đủ sáu file, đọc được safetensors/config, đọc được FAISS và parquet, số vector/metadata, dimension, chuẩn hóa vector, loại index, resolution và dimension model-index. Chỉ tiếp tục nếu in `"status": "PASS"`. Lỗi đường dẫn hoặc hỏng artifact sẽ dừng tại bước này.

Preflight kiểm tra file và cấu trúc index, chưa tải dataset để đối chiếu ID gallery. Bước smoke tiếp theo xác minh gallery IDs/labels/thứ tự so với split thực tế trước khi trích đặc trưng query; nếu không khớp, nó dừng với lỗi cụ thể.

## 4. Smoke run nhỏ để bắt lỗi trước khi chạy hết

```powershell
modal run modal_evaluation.py --stage smoke --split test_queries --limit 3 --batch-size 1 --top-k 100 --gpu T4
```

Smoke tải dataset, tái tạo split, so khớp toàn bộ gallery metadata với split và chạy 3 query. Nó vẫn tìm kiếm trên toàn bộ `gallery.index` upload. Có thể chọn `--split val_queries` để kiểm tra split validation. Kết quả được ghi riêng dưới `artifacts/evaluation/effovpr-r_smoke/` để không ghi đè kết quả full.

Chỉ tính phí GPU cho giai đoạn inference/reranking sau khi Modal khởi tạo container; preflight dùng CPU. Smoke vẫn cần tải model và dataset, do đó có chi phí khởi động và truyền dữ liệu, nhưng xử lý ít query hơn đáng kể. MNN reranking đọc ảnh gallery cho tối đa `top-k` ứng viên trên mỗi query.

## 5. Chọn thông số lần chạy chính

| Tham số | Lựa chọn và ảnh hưởng |
|---|---|
| `--split` | `test_queries` để báo cáo kết quả cuối; `val_queries` cho thử nghiệm tham số. |
| `--batch-size` | Số query encode global cùng lúc. Bắt đầu `1` (an toàn VRAM); tăng `2` hoặc `4` nếu GPU đủ bộ nhớ. Không ảnh hưởng metric. |
| `--top-k` | Số gallery candidate đem MNN rerank. Mặc định theo notebook là `100`; nhỏ hơn sẽ nhanh hơn nhưng có thể đổi metric reranked. |
| `--gpu` | Script cấu hình sẵn `T4` hoặc `A10G`. T4 phù hợp để thử; A10G nhanh hơn trong nhiều trường hợp và có VRAM rộng hơn. GPU Modal còn tùy khả dụng/tài khoản. |
| `--limit` | `0` nghĩa là toàn bộ query. Dùng `3` cho smoke; không đặt limit khi chạy full. |

Lệnh full mặc định dùng T4, test split, batch 1 và top-k 100:

```powershell
modal run modal_evaluation.py --stage full --split test_queries --batch-size 1 --top-k 100 --gpu T4
```

Để dùng A10G hoặc batch lớn hơn:

```powershell
modal run modal_evaluation.py --stage full --split test_queries --batch-size 2 --top-k 100 --gpu A10G
```

Full run tải model và dataset, encode query, sau đó chạy local MNN trên tối đa `query_count × top-k` cặp ảnh. Nếu muốn so sánh top-k hay batch size, trước tiên dùng `val_queries`; giữ nguyên `test_queries` cho lần báo cáo cuối. `resolution`, layer, facet, T1/T2 được lấy từ config EffoVPR đã upload để giữ protocol của model/index.

## 6. Lấy kết quả

Kết quả full nằm trong volume output tại:

```text
artifacts/evaluation/effovpr-r/results.json
artifacts/evaluation/effovpr-r/query_predictions.csv
artifacts/failure_analysis/effovpr-r/failure_cases.csv
```

Smoke có cùng cấu trúc nhưng thư mục mang hậu tố `_smoke`. Tải thư mục kết quả về máy:

```powershell
modal volume get --force effovpr-eval-output artifacts .\modal-results
```

Output được commit vào Modal Volume sau khi chạy xong, nên có thể tải sau khi job đã kết thúc.

## Ghi chú về tính nhất quán

Index upload phải thuộc đúng model checkpoint và gallery ordering mà nó đã được tạo cùng. Script kiểm tra resolution và dimension model/index ở preflight, sau đó xác minh ID và label gallery với dataset split ở runtime. Nó không mã hóa lại gallery hay tạo gallery index mới. Global query embedding vẫn được trích từ model trong lần đánh giá, đúng với luồng truy vấn của notebook cũ.
