# 124 Cuộc thi MLP - Lê Thanh Tuấn - 21115053120158

## Tổng quan

## Xây dựng một mô hình dự đoán giá nhà ở thành phố Bắc Kinh, Trung Quốc.

## Các bước thực hiện

### 1. Tiền xử lý dữ liệu (Data Pre-processing)

#### 1.1 Load và xử lý dữ liệu

- **Dữ liệu đầu vào** được load từ các tệp CSV:

  - `train_path`: Chứa các đặc trưng (features) và dữ liệu huấn luyện.
  - `target_path`: Chứa biến mục tiêu (`TARGET`), chính là giá nhà.
  - `test_path`: Chứa dữ liệu kiểm tra (test data) không có giá nhà thực tế.

- Cột `ID` không có ý nghĩa trong việc dự đoán nên đã được loại bỏ ngay từ đầu trong cả tập huấn luyện và kiểm tra.

#### 1.2 Xử lý dữ liệu thiếu

- **Điền giá trị thiếu** cho các cột có dữ liệu thiếu:
  - `elevator`: Nếu giá trị bị thiếu thì điền giá trị 0.
  - `subway`: Nếu giá trị bị thiếu thì điền giá trị 0.

#### 1.3 Tạo thêm đặc trưng

- **Chuyển đổi ngày tháng**: Cột `tradeTime` đã được chuyển thành định dạng `datetime` để dễ dàng tính toán.
- **Tuổi của tòa nhà** (`ageOfBuilding`): Tính toán bằng cách lấy năm trong `tradeTime` trừ đi `constructionTime`. Nếu thiếu `constructionTime`, dữ liệu sẽ được xử lý bằng cách loại bỏ hoặc thay thế.
- **Khoảng cách đến thủ đô**: Sử dụng tọa độ vĩ độ (`Lat`) và kinh độ (`Lng`) để tính toán khoảng cách đến thủ đô Bắc Kinh (tọa độ [39.9042, 116.4074]) bằng công thức Haversine.

#### 1.4 Chuyển đổi cột `floor`

- Giá trị của cột `floor` đã được trích xuất từ dữ liệu dạng chuỗi để đảm bảo tất cả các giá trị đều là số nguyên hợp lệ.

#### Cách xử lý dữ liệu được tham khảo tại [đây](https://github.com/eiliaJafari/House-prices-in-Beijing/blob/main/House%20prices%20in%20Beijing.ipynb)

### 2. Chuẩn hóa và Xử lý dữ liệu (Data Transformation)

#### 2.1 Xử lý NaN

- Sử dụng `SimpleImputer`, thay thế giá trị thiếu bằng trung vị (`median`).

#### 2.2 Chuẩn hóa dữ liệu

- Sử dụng `PowerTransformer` để chuẩn hóa dữ liệu, giúp cải thiện hiệu suất của mô hình và làm cho dữ liệu có phân phối gần chuẩn hơn.

### 3. Huấn luyện mô hình

#### 3.1 RandomForest Model

- Sử dụng **RandomForestRegressor**:

  - Số lượng cây quyết định (`n_estimators`) được đặt thành 500.
  - Độ sâu tối đa (`max_depth`) của cây là 15.
  - Chia nhánh tối thiểu của mẫu (`min_samples_split`) là 5, và số lượng mẫu tối thiểu của mỗi lá cây (`min_samples_leaf`) là 2.

#### 3.2 Đánh giá mô hình

- Để đánh giá hiệu suất của mô hình, sử dụng [**RMSE**](https://statisticsbyjim.com/regression/root-mean-square-error-rmse/)
  ```python
  def print_rmse(y_true, y_pred, model_name="Model"):
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      print(f"{model_name} RMSE: {rmse}")
      return rmse
  ```

### 4. Dự đoán kết quả

- Huấn luyện mô hình MLP trên RandomForest

  ```python
  rf_model, val_rmse = train_random_forest(X_train_split y_train_split, X_val_split, y_val_split)
  ```

- Sau đó dự đoán dựa trên kết quả học được từ RandomForest

  ```python
  y_pred_test = rf_model.predict(X_test_scaled)
  ```
