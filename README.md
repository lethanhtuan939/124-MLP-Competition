# 124 MLP Competition - Lê Thanh Tuấn - 21115053120158

## Tổng quan

Xây dựng một mạng MLP dự đoán giá nhà ở thành phố Bắc Kinh, Trung Quốc.

## Các bước thực hiện

### 1. Tiền xử lý dữ liệu (Data Pre-processing)

#### 1.1 Load và xử lý dữ liệu

- **Dữ liệu đầu vào** được load từ các tệp CSV:

  - `train_path`: Chứa các đặc trưng (features) và dữ liệu huấn luyện (`data/X_train.csv`).
  - `target_path`: Chứa biến mục tiêu (`TARGET`), chính là giá nhà (`data/Y_train.csv`).
  - `test_path`: Chứa dữ liệu kiểm tra (test data) không có giá nhà thực tế (`data/X_test.csv`).

- Cột `ID` không có ý nghĩa trong việc dự đoán nên đã được loại bỏ ngay từ đầu trong cả tập huấn luyện.

#### 1.2 Xử lý dữ liệu thiếu

- **Điền giá trị thiếu** cho các cột có dữ liệu thiếu:
  - `elevator`: Nếu giá trị bị thiếu thì điền giá trị 0.
  - `subway`: Nếu giá trị bị thiếu thì điền giá trị 0.

#### 1.3 Chuyển đổi các giá trị danh mục

<table>
<tr>
    <td><b><code>elevator</code></b></td>    
</tr>    
<tr>    
    <td>1</td> <td>'has elevator'</td>
</tr>
<tr>     
    <td>0</td> <td>'no elevator'</td>    
</tr>
</table>

<br>

<table>
<tr>
    <td><b><code>subway</code></b></td>    
</tr>        
<tr>    
<td>1</td> <td>'has subway'</td>
</tr>
<tr>     
<td>0</td> <td>'no subway'</td>    
</tr>        
</table>   
<br>        
<table>
<tr>
<td><b><code>buildingStructure</code></b></td>    
</tr>     
<tr>    
<td>1</td> <td>'unknown'</td>
</tr>
<tr>     
<td>2</td> <td>'mixed'</td>    
</tr>
            
<tr>     
<td>3</td> <td>'brick and wood' </td>    
</tr>
            
<tr>     
<td>4</td> <td>'concrete'</td>    
</tr>
            
<tr>     
<td>5</td> <td>'steel'</td>    
</tr>
            
<tr>     
<td>6</td> <td>'steel-concrete composite'</td>    
</tr>            
</table> 
<br>
        
<table>
<tr>
<td><b><code>renovationCondition</code></b></td>    
</tr>      
<tr>    
<td>1</td> <td>'other'</td>
</tr>
<tr>     
<td>2</td> <td>'rough'</td>    
</tr>
            
<tr>     
<td>3</td> <td>'Simplicity' </td>    
</tr>
            
<tr>     
<td>4</td> <td>'hardcover'</td>    
</tr>
                 
</table>

<br>
<table>
<tr>
<td><b><code>buildingType</code></b></td>    
</tr>      
<tr>    
<td>1</td> <td>'tower'</td>
</tr>
<tr>     
<td>2</td> <td>'bungalow'</td>    
</tr>
            
<tr>     
<td>3</td> <td>'combination of plate and tower' </td>    
</tr>
            
<tr>     
<td>4</td> <td>'plate'</td>    
</tr>
                 
</table>

#### 1.4 Tạo thêm features (Feature Engineering)

- **Chuyển đổi ngày tháng**: Cột `tradeTime` đã được chuyển thành định dạng `datetime` để dễ dàng tính toán.

- **Tuổi của tòa nhà** (`ageOfBuilding`): Tính toán bằng cách lấy năm trong `tradeTime` trừ đi `constructionTime`.

- **Khoảng cách đến thủ đô** (`distanceToCapital`): Sử dụng tọa độ vĩ độ (`Lat`) và kinh độ (`Lng`) để tính toán khoảng cách đến thủ đô Bắc Kinh (tọa độ [39.9042, 116.4074]) bằng công thức Haversine.

#### 1.5 Chuyển đổi cột `floor`

- Giá trị của cột `floor` đã được trích xuất từ dữ liệu dạng chuỗi để đảm bảo tất cả các giá trị đều là số nguyên hợp lệ.

> **_NOTE:_**
> Cách xử lý dữ liệu được tham khảo tại [đây](https://github.com/eiliaJafari/House-prices-in-Beijing/blob/main/House%20prices%20in%20Beijing.ipynb).

### 2. Chuẩn hóa và Xử lý dữ liệu (Data Transformation)

#### 2.1 Xử lý NaN

- Sử dụng `SimpleImputer`, thay thế giá trị thiếu bằng trung vị (`median`).

#### 2.2 Chuẩn hóa dữ liệu

- Sử dụng `PowerTransformer` để chuẩn hóa dữ liệu, giúp cải thiện hiệu suất của mô hình và làm cho dữ liệu có phân phối gần chuẩn hơn.

### 3. Huấn luyện mô hình

#### 3.1 RandomForest Model

- Sử dụng **RandomForestRegressor**:

  - Số lượng cây quyết định (`n_estimators`) được đặt thành 300.
  - Độ sâu tối đa (`max_depth`) của cây là 20.
  - Chia nhánh tối thiểu của mẫu (`min_samples_split`) là 10, và số lượng mẫu tối thiểu của mỗi lá cây (`min_samples_leaf`) là 5.

  ```
  rf_model, rf_rmse = train_random_forest(X_train_split, y_train_split, X_val_split, y_val_split)
  ```

#### 3.2 Sử dụng mô hình RandomForest để tạo các đặc trưng mới

- Dự đoán trên tập huấn luyện (`X_train_split`) và tập validation (`X_val_split`) để lấy các giá trị dự đoán và dùng chúng như một đặc trưng mới cho mô hình MLP.

- Ghép các dự đoán vào tập huấn luyện và validation

  ```
  X_train_split_with_rf = np.hstack((X_train_split, rf_train_predictions))
  X_val_split_with_rf = np.hstack((X_val_split, rf_val_predictions))
  X_test_with_rf = np.hstack((X_test_scaled, rf_test_predictions))
  ```

#### 3.3 Đánh giá mô hình

- Để đánh giá hiệu suất của mô hình, sử dụng [**RMSE**](https://statisticsbyjim.com/regression/root-mean-square-error-rmse/)
  ```python
  def print_rmse(y_true, y_pred, model_name="Model"):
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      print(f"{model_name} RMSE: {rmse}")
      return rmse
  ```

### 4. Dự đoán kết quả

#### 4.1 Huấn luyện MLP Model với đặc trưng từ RandomForest

- Xây dựng và huấn luyện mô hình MLP dựa trên các đặc trưng gốc và đặc trưng từ mô hình `RandomForest`.

```
mlp_model = build_mlp_model(X_train_split_with_rf.shape[1])
mlp_model = compile_and_train_mlp(mlp_model, X_train_split_with_rf, y_train_split, X_val_split_with_rf, y_val_split)
```

#### 4.2 Dự đoán kết quả với MLP Model

- Sử dụng mô hình MLP đã huấn luyện để dự đoán trên tập kiểm định và tính `RMSE`.

```
y_pred_val_mlp = mlp_model.predict(X_val_split_with_rf).flatten()
mlp_rmse = print_rmse(y_val_split, y_pred_val_mlp, model_name="MLP with RF Feature")
```

- Tiến hành dự đoán trên tập test và lưu kết quả vào file `Latest_submission.csv`.

```
y_pred_test_mlp = mlp_model.predict(X_test_with_rf).flatten()
save_submission(y_pred_test_mlp, test_ids, filename='Latest_submission.csv')
```
