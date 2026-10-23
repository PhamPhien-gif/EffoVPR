# EffoVPR trên VN_Attractions

Dự án **truy hồi ảnh theo địa điểm** (visual place retrieval) trên bộ dữ liệu [VN_Attractions](https://huggingface.co/datasets/PhamPhien/VN_Attractions). Hệ thống sử dụng backbone **DINOv2 ViT-L/14 có register tokens** (`facebook/dinov2-with-registers-large`) và hai biến thể mô hình đã huấn luyện:

- **EffoVPR_G**: truy hồi toàn cục bằng độ tương đồng cosine.
- **EffoVPR_R**: có thể xếp hạng lại Top 10 ảnh truy hồi bằng đặc trưng cục bộ và **Mutual Nearest Neighbors (MNN)**.

Bộ dữ liệu có nhãn địa điểm nhưng **không có tọa độ GPS**. Vì vậy, bài toán trong repository này là truy hồi các ảnh cùng nhãn địa điểm, không tái lập giao thức định vị địa lý trong phạm vi 25 mét của bài báo EffoVPR.

## Thành phần chính

| Thành phần | Mô tả |
| --- | --- |
| `app.py` | Ứng dụng Streamlit để so sánh EffoVPR_G và EffoVPR_R trên cùng ảnh truy vấn. |
| `Models/EffoVPR_G/`, `Models/EffoVPR_R/` | Artifact mô hình và gallery index của từng biến thể. |
| `data/examples/` | Ảnh mẫu dùng để truy vấn trong ứng dụng. |
| `src/` | Mã Python dùng chung cho mô hình, tải artifact, truy hồi và MNN reranking. |
| `notebooks/training/` | Notebook Kaggle cho huấn luyện và đánh giá. |
| `notebooks/inference/Use_Model.ipynb` | Notebook cục bộ để dự đoán nhãn cho ảnh trong `data/examples/`. |
| `evaluation/results/` | Các tệp JSON kết quả đánh giá đã lưu. |
| `docs/` | Hướng dẫn chạy, đặc tả triển khai và tài liệu tham khảo. |

## Mô hình và cấu hình

Hai artifact được cấu hình với các đặc điểm chính sau:

- Backbone: `facebook/dinov2-with-registers-large`.
- Embedding toàn cục: 1.024 chiều.
- 49 lớp/nhãn địa điểm.
- Fine-tune 5 layer cuối của backbone.
- Bộ phân loại CosFace với `scale = 30.0` và `margin = 0.4`.
- Độ phân giải suy luận: 504 px.
- Truy hồi bằng FAISS `IndexFlatIP` trên vector đã chuẩn hóa L2.
- MNN reranking dùng layer `n-1`, facet `V`, ngưỡng chọn patch `T1 = 0.05` và ngưỡng ghép `T2 = 0.65`.

## Chạy ứng dụng Streamlit cục bộ

Chạy tại thư mục gốc của repository:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Sau đó mở địa chỉ do Streamlit in ra, thông thường là `http://localhost:8501`.

Ứng dụng cần các bundle mô hình tương ứng dưới đây:

```text
Models/
  EffoVPR_G/
    config.json
    model.safetensors
    index/
      gallery_features.npy
      gallery_metadata.parquet
      index_config.json
  EffoVPR_R/
    config.json
    model.safetensors
    index/
      gallery_features.npy
      gallery_metadata.parquet
      index_config.json
```

`model.safetensors` là bắt buộc để nạp mô hình. `gallery.index` có thể được giữ trong artifact nhưng không được ứng dụng Streamlit sử dụng; ứng dụng tính độ tương đồng trực tiếp từ `gallery_features.npy`.

Xem hướng dẫn chi tiết tại [docs/STREAMLIT.md](docs/STREAMLIT.md).

## Chức năng của ứng dụng

Với cùng một hoặc nhiều ảnh truy vấn, ứng dụng:

1. Chạy EffoVPR_G và EffoVPR_R trên cùng ảnh đầu vào.
2. Hiển thị nhãn dự đoán bởi bộ phân loại CosFace.
3. Hiển thị Top 1, 5 hoặc 10 ảnh gallery có độ tương đồng cosine cao nhất.
4. Xếp hạng các địa điểm theo trung bình độ tương đồng của ba ảnh gallery tốt nhất thuộc mỗi nhãn.
5. Có thể xếp hạng lại Top 10 ảnh của **EffoVPR_R** bằng MNN cục bộ và vẽ các cặp patch tương ứng.
6. Nhận ảnh từ `data/examples/`, ảnh tải lên hoặc ảnh dán từ clipboard bằng `Ctrl+V` trong chế độ **Upload images**.

Các phần trăm Softmax hiển thị cho địa điểm là điểm tương đối giữa toàn bộ nhãn trong gallery, với nhiệt độ 0,05; chúng **không phải xác suất đã hiệu chuẩn**. Tương tự, điểm CosFace chỉ là điểm xếp hạng, không phải xác suất.

Ảnh thu nhỏ từ gallery và MNN reranking cần truy cập bộ dữ liệu Hugging Face `PhamPhien/VN_Attractions` trong lần chạy đầu. Có thể tắt hai tùy chọn này ở sidebar để chỉ xem ID, nhãn và điểm từ gallery index cục bộ.

## Notebook Kaggle

Các notebook thực thi quy trình huấn luyện và đánh giá nằm trong `notebooks/training/`. Notebook chính có các ô `%%writefile` để tạo lại các module `src/` và `scripts/` cần thiết ngay trong Kaggle runtime; chúng không phụ thuộc vào thư mục `scripts/` đã được loại khỏi bản checkout này.

Các notebook đáng chú ý:

- `notebooks/training/kaggle_run_all.ipynb`: quy trình đầu-cuối chính.
- `notebooks/training/EffoVPR_G.ipynb`: thí nghiệm fine-tune biến thể global.
- `notebooks/training/EffoVPR_R.ipynb`: thí nghiệm fine-tune biến thể có local reranking.
- `notebooks/training/EffoVPR_ZS.ipynb` và `DINOv2_Global_ZS.ipynb`: thí nghiệm zero-shot.
- `notebooks/training/Kaggle_FineTune_DINOv2.ipynb`: quy trình fine-tune độc lập trước đó.

Hướng dẫn đầy đủ: [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Suy luận bằng notebook cục bộ

`notebooks/inference/Use_Model.ipynb` là notebook độc lập để dự đoán nhãn cho ảnh trong `data/examples/` bằng artifact EffoVPR_G cục bộ. Notebook này không cung cấp giao diện Streamlit và không phải backend của `app.py`.

## Kết quả và tài liệu

- Kết quả đánh giá được lưu tại `evaluation/results/`:
  - `DINOv2_results.json`
  - `EffoVPR_G_results.json`
  - `EffoVPR_R_results.json`
  - `EffoVPR_ZS_results.json`
- Đặc tả triển khai: [docs/EFFOVPR_IMPLEMENTATION_SPEC.md](docs/EFFOVPR_IMPLEMENTATION_SPEC.md).
- Hướng dẫn đánh giá trên Modal: [docs/MODAL_EVALUATION.md](docs/MODAL_EVALUATION.md).
- Bài báo tham khảo: [docs/references/EffoVPR.pdf](docs/references/EffoVPR.pdf).

## Lưu ý về cấu trúc repository

Các thư mục `scripts/`, `tests/`, `artifacts/` và các thư mục `*_outputs/` cũ đã được loại khỏi bản checkout này. Pipeline trong các notebook Kaggle có thể tái tạo mã cần thiết tại runtime. Một lần chạy pipeline cục bộ chỉ có thể tái tạo `artifacts/` khi mã nguồn tương ứng đã được khôi phục.
