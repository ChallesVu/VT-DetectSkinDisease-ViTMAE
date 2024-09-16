# VT-DetectSkinDisease-ViTMAE
### The application of Vision Transformer Masked AutoEncoder in detecting skin diseases
### Ứng dụng của Vision Transformer Masked AutoEncoder trong dự đoán các bệnh về da
### Người thực hiện: NGUYỄN VŨ TƯỜNG

# Tóm tắt: 
1. Vấn đề nghiên cứu
Việc dự đoán và phân loại bệnh da là một thách thức lớn trong y học, đặc biệt là với các loại bệnh ác tính như melanoma (ung thư hắc tố), nơi việc phát hiện sớm có thể cứu sống bệnh nhân. Với sự bùng nổ của các kỹ thuật học sâu (deep learning), các hệ thống dự đoán tự động từ hình ảnh da liễu đang trở nên phổ biến, giúp hỗ trợ bác sĩ trong quá trình chẩn đoán.
Mục tiêu của nghiên cứu này là xây dựng một mô hình học sâu để dự đoán các loại bệnh da từ hình ảnh da liễu, sử dụng mô hình ViT-MAE (Vision Transformer - Masked Autoencoder) trên bộ dữ liệu Skin Cancer MNIST: HAM10000.
2. Hướng tiếp cận
ViT-MAE (Vision Transformer - Masked Autoencoder) là một phương pháp mới trong lĩnh vực self-supervised learning (học tự giám sát), trong đó mô hình học cách tái tạo các phần ảnh bị che và sau đó được fine-tune để thực hiện các nhiệm vụ cụ thể, chẳng hạn như phân loại ảnh.
Hướng tiếp cận trong nghiên cứu này bao gồm:
Sử dụng mô hình ViT-MAE: Mô hình ViT-MAE trước tiên được pre-trained để tái tạo ảnh từ các phần bị che, sau đó được fine-tune để dự đoán loại bệnh da trên bộ dữ liệu có nhãn.
Fine-tuning trên bộ dữ liệu Skin Cancer MNIST: HAM10000: Bộ dữ liệu này bao gồm 7 loại bệnh da khác nhau, từ các tổn thương lành tính cho đến các loại ung thư da. Mô hình ViT-MAE được điều chỉnh lại để học các đặc trưng từ ảnh da và các nhãn đi kèm và dự đoán loại bệnh.
3. Cách giải quyết vấn đề
Quy trình giải quyết bài toán dự đoán bệnh da sử dụng ViT-MAE bao gồm các bước chính sau:
a. Chuẩn bị dữ liệu
Bộ dữ liệu Skin Cancer MNIST: HAM10000 gồm các ảnh da và nhãn tương ứng với 7 loại bệnh da. Các hình ảnh được resize về kích thước 224x224 và chuẩn hóa giá trị pixel để phù hợp với đầu vào của mô hình.
Dữ liệu được chia thành tập huấn luyện và tập kiểm tra với tỷ lệ 80-20.
b. Xây dựng và huấn luyện mô hình
ViT-MAE được tải về từ Hugging Face, mô hình này đã được pre-trained trên các bộ dữ liệu lớn để học cách tái tạo các phần ảnh bị che.
Sau đó, một lớp phân loại (fully connected layer) được thêm vào để mô hình có thể dự đoán nhãn của ảnh da.
Mô hình được fine-tune trên bộ dữ liệu Skin Cancer MNIST: HAM10000, với tiêu chí mất mát là Cross-Entropy Loss và tối ưu hóa bằng Adam optimizer.
c. Đánh giá mô hình
Độ chính xác (accuracy) và độ chính xác theo nhãn (precision)
4. Kết quả đạt được
Sau khi hoàn thành quá trình huấn luyện và fine-tuning, các kết quả chính đạt được bao gồm:
Độ chính xác của mô hình:
Mô hình ViT-MAE đạt được độ chính xác cao trên tập kiểm tra, với khoảng 80% accuracy trong việc phân loại đúng loại bệnh da.
Khả năng dự đoán chính xác các loại bệnh:
Mô hình hoạt động tốt trong việc phân loại các loại bệnh da phổ biến như melanocytic nevi (nv) và benign keratosis-like lesions (bkl), nhưng gặp khó khăn với các loại bệnh ít phổ biến hơn như dermatofibroma (df).
Thử nghiệm mô hình:
Mô hình đã được thử nghiệm với một số hình ảnh ngoài, cho kết quả khả quan trong việc dự đoán chính xác các loại bệnh da dựa trên đặc trưng của ảnh.

# Mục lục
CHƯƠNG 1 – CƠ SỞ LÝ THUYẾT	5
1.1 Transformer	5
1.1.1	Self-attention	6
1.1.1.1 Query, Key, and Value (Q, K, V)	7
1.1.1.2 Tính toán attention score	7
1.1.1.3 Softmax để tính trọng số attention	7
1.1.1.4 Tính giá trị đầu ra	7
1.1.1.5 Scaled Dot-Product Attention	8
1.1.2	Multi-head Attention	8
1.1.2.1 Lợi ích của Multi-head Attention	10
1.1.2.2 Nhược điểm của Multi-head Attention	10
1.1.3	Position Embedding	10
1.2 Vision Transfomer	13
1.2.1 Linear Projection of Flattened Patches	14
1.2.2 Classification accuracies	15
1.3 Masked AutoEncoder	16
CHƯƠNG 2 – ỨNG DỤNG CỦA MÔ HÌNH ViT-MAE TRONG VIỆC PHÁT HIÊN/PHÂN LOẠI BỆNH LÝ VỀ DA LIỄU	18
2.1 ViT-MAE và Cơ chế hoạt động trong Lĩnh vực Y tế	18
2.2 Ứng dụng ViT-MAE trong Phát hiện Bệnh Lý Da Liễu	19
2.3 Lợi ích của ViT-MAE trong Y tế	19
2.4 Thách thức và Hướng Phát triển	20
CHƯƠNG 3 – THỰC NGHIỆM VÀ ĐÁNH GIÁ	21
3.1 Thực nghiệm:	21
3.2 Đánh giá	31

# CHƯƠNG 1 – CƠ SỞ LÝ THUYẾT
1.1 Transformer
Cốt lõi của transformer là attension mechanism (cơ chế tập trung), giúp mô hình tập trung vào các phần quan trọng của văn bản để đưa ra dự đoán chính xác hơn.
Transformer được cấu trúc thành hai phần chính là encoder và decoder.
Encoder: Encoder xử lý dữ liệu đầu vào (gọi là "Source") và nén dữ liệu vào vùng nhớ hoặc context mà Decoder có thể sử dụng sau đó.
Decoder: Decoder nhận đầu vào từ đầu ra của Encoder (gọi là "Encoded input") kết hợp với một chuỗi đầu vào khác (gọi là "Target") để tạo ra chuỗi đầu ra cuối cùng.
Mỗi encoder và decoder đều bao gồm nhiều lớp, mỗi lớp chứa các thành phần self-attention và feed-forward neural networks.

![image](https://github.com/user-attachments/assets/b7593156-0194-423c-bc9d-4ba687ea35fa)

