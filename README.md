# 📘 Áp dụng thuật toán FP-Growth trong phân loại tin nhắn spam hoặc ham

## 🧠 Giới thiệu

* **Tên môn học**: Khai thác dữ liệu và ứng dụng - Data Mining and Application
* **Mã môn học**: CS313 - **Lớp**: CS313.O23
* **Năm học**: Học kì 2 - Năm học: 2023 - 2024
* **Giảng viên**: TS. Võ Nguyễn Lê Duy
* **Mục tiêu đồ án**: Xây dựng một hệ thống phân loại tin nhắn spam và ham (no spam) bằng cách khai thác đặc trưng theo **Association Rule** sử dụng thuật toán **FP-Growth**, sau đó so sánh hiệu năng với hai thuật toán máy học là **Multinomial Naive Bayes** và **Logistic Regression**

## Thành viên nhóm: 
| STT    | MSSV          | Họ và Tên              |Vai trò    | Email                   |
| ------ |:-------------:| ----------------------:|----------:|-------------------------:
| 1      |22520550       |Lương Anh Huy           |Trưởng nhóm| 22520550@gm.uit.edu.vn|
| 2      |22520521       |Phạm Đông Hưng          |Thành viên| 22520521@gm.uit.edu.vn|
| 3      |22520884       |Phan Công Minh          |Thành viên| 22520884@gm.uit.edu.vn|

## 📂 Cấu trúc thư mục

| Tên file                             | Mô tả                                                                 |
|-------------------------------------|----------------------------------------------------------------------|
| `Final_Model.ipynb`  | Notebook chính, trình bày toàn bộ quá trình từ tiền xử lý, sinh đặc trưng bằng FPGrowth đến huấn luyện và đánh giá mô hình. |
| `fpcommon.py` | Định nghĩa các lớp và hàm cần thiết để xây dựng *FP-Tree* và sinh ra các **Association Rules** từ dữ liệu nhị phân. |

## ⚙️ Các bước thực hiện

1. Tiền xử lý dữ liệu
2. Trích xuất Association Rule bằng FP-Growth
3. Huấn luyện và đánh giá mô hình
4. Kết quả đạt được
   

### 🛠️ Công nghệ và thư viện
Python 3.x
NLTK
Scikit-learn
Pandas
Matplotlib
Logistic Regression
Multinomial Naive Bayes
Torch

#### 📄 Giấy phép
Đồ án học thuật – Không sử dụng vào mục đích thương mại.
© 2025 - Trường Đại học Công nghệ thông tin.

