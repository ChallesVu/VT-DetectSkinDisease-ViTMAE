# VT-DetectSkinDisease-ViTMAE
### The application of Vision Transformer Masked AutoEncoder in detecting skin diseases
### Ứng dụng của Vision Transformer Masked AutoEncoder trong dự đoán các bệnh về da
### Người thực hiện: NGUYỄN VŨ TƯỜNG

# link datasets: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

# Tóm tắt: 
1. Vấn đề nghiên cứu
Việc dự đoán và phân loại bệnh da là một thách thức lớn trong y học, đặc biệt là với các loại bệnh ác tính như melanoma (ung thư hắc tố), nơi việc phát hiện sớm có thể cứu sống bệnh nhân. Với sự bùng nổ của các kỹ thuật học sâu (deep learning), các hệ thống dự đoán tự động từ hình ảnh da liễu đang trở nên phổ biến, giúp hỗ trợ bác sĩ trong quá trình chẩn đoán.
Mục tiêu của nghiên cứu này là xây dựng một mô hình học sâu để dự đoán các loại bệnh da từ hình ảnh da liễu, sử dụng mô hình ViT-MAE (Vision Transformer - Masked Autoencoder) trên bộ dữ liệu Skin Cancer MNIST: HAM10000.
2. Hướng tiếp cận
ViT-MAE (Vision Transformer - Masked Autoencoder) là một phương pháp mới trong lĩnh vực self-supervised learning (học tự giám sát), trong đó mô hình học cách tái tạo các phần ảnh bị che và sau đó được fine-tune để thực hiện các nhiệm vụ cụ thể, chẳng hạn như phân loại ảnh.
Hướng tiếp cận trong nghiên cứu này bao gồm:
Sử dụng mô hình ViT-MAE: Mô hình ViT-MAE trước tiên được pre-trained để tái tạo ảnh từ các phần bị che, giúp mô hình học được các chi tiết và đặc trưng nhỏ nhất của da, từ đó dễ dàng để học phân loại hơn. Sau đó được fine-tune để dự đoán loại bệnh da trên bộ dữ liệu có nhãn.
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
Hình 1. 1: The Transformer – model architecture
Nếu như coi mô hình Transformer như một kiểu mô hình encoder-decoder "biến hình". Ta có thể coi mô hình Transformer có N block. Mỗi block chứa ba phần: Encoder, Decoder, Encoder-Decoder Attention.

1.1.1	Self-attention
Một hàm attention có thể được mô tả như một phép ánh xạ từ một truy vấn (query) và một tập hợp các cặp khóa-giá trị (key-value) tới một đầu ra, trong đó truy vấn, các khóa, các giá trị và đầu ra đều là các vector. Đầu ra được tính bằng cách lấy tổng có trọng số của các giá trị, trong đó trọng số được gán cho mỗi giá trị được tính bằng một hàm tương thích giữa truy vấn và khóa tương ứng.
Ví dụ đơn giản, trong câu "The cat sat on the mat", self-attention có thể giúp mô hình hiểu được rằng "cat" là chủ ngữ của hành động "sat", dù chúng có thể cách nhau một vài từ trong câu.
![image](https://github.com/user-attachments/assets/07dd5681-f605-414d-ba0b-c9f603c3cf19)
Hình 1. 2: Minh họa về cơ chế hoạt động của Self-Attention

1.1.1.1 Query, Key, and Value (Q, K, V)
- Đầu tiên, với mỗi phần tử trong chuỗi đầu vào, ta tạo ra ba vector khác nhau: Query (Q), Key (K), và Value (V).
	Query: Được sử dụng để hỏi về mối quan hệ giữa phần tử hiện tại với các phần tử khác trong chuỗi.
	Key: Được sử dụng để trả lời các truy vấn từ các phần tử khác.
	Value: Được sử dụng để tạo ra đầu ra cuối cùng của attention.
- Tất cả các vector Q, K, và V đều có cùng số chiều và được tính toán bằng cách nhân phần tử đầu vào với ba ma trận trọng số khác nhau: W_Q, W_K, và W_V.

1.1.1.2 Tính toán attention score
- Mối quan hệ giữa một phần tử với các phần tử khác trong chuỗi được tính toán bằng cách lấy tích vô hướng (dot product) giữa Query của phần tử hiện tại với Key của các phần tử khác. Kết quả sẽ là một giá trị gọi là attention score.
	Công thức:
attention score(i,j)= Q_i∙K_j
Với Q_i là Query của phần tử thứ i và K_j là Key của phần tử thứ j.

1.1.1.3 Softmax để tính trọng số attention
- Sau khi tính được các attention scores, để chuẩn hóa chúng thành các trọng số (weights) có tổng bằng 1, ta áp dụng hàm softmax. Điều này giúp chuyển đổi các giá trị attention score thành các xác suất cho thấy mức độ quan trọng của các phần tử khác nhau trong chuỗi đối với phần tử hiện tại.
	Công thức:
α_ij=softmax(Q_i∙K_j)
α_ij là trọng số attention giữa phần tử thứ i và j.

1.1.1.4 Tính giá trị đầu ra
- Sau khi tính được các trọng số attention (α), đầu ra của self-attention cho phần tử hiện tại được tính bằng cách lấy tổng có trọng số của tất cả các Value (V) trong chuỗi.
	Công thức:
〖output〗_i= ∑_j▒〖α_ij∙V_j 〗
Tức là, mỗi giá trị Value được nhân với trọng số tương ứng của nó và sau đó cộng dồn lại để tạo ra đầu ra cho phần tử hiện tại.

1.1.1.5 Scaled Dot-Product Attention
- Để tránh việc các giá trị attention score quá lớn khi số chiều của Q và K tăng lên, công thức tính attention score thường được chia cho căn bậc hai của kích thước chiều của vector Key:
Scaled attention score(i,j)=  (Q_i∙K_j)/√(d_k )
Với d_k là số chiều của Key.
	Điều này giúp mô hình ổn định hơn khi số chiều của vector tăng lên.

1.1.2	Multi-head Attention
Vấn đề của Self-attention là attention của một từ sẽ luôn "chú ý" vào chính “nó”. 
![image](https://github.com/user-attachments/assets/8f44a8dc-1489-4df1-8ab5-1d39dd6430ae)
Hình 1. 3: Vấn đề của Self-attention 
Multi-head attention được thực hiện bằng cách nhân bản self-attention thành nhiều "head" và sau đó kết hợp các kết quả này lại bằng cách nối (concatenate) và đưa qua một ma trận trọng số khác.
![image](https://github.com/user-attachments/assets/bc5a5ff6-9dab-4706-bf9e-44f50c3d8970)
Hình 1. 4: Minh họa về Multi-head Attention

1.1.2.1 Lợi ích của Multi-head Attention
- Với mỗi "head", mô hình có thể học các mối quan hệ khác nhau giữa các phần tử trong chuỗi đầu vào. Điều này giúp mô hình mạnh mẽ hơn trong việc nắm bắt các thông tin ngữ cảnh và mối quan hệ trong chuỗi.
- Việc chia nhỏ ma trận Q, K, và V giúp giảm khối lượng tính toán so với việc áp dụng attention lên toàn bộ dữ liệu mà không chia nhỏ.
- Sử dụng nhiều "head" attention giúp mô hình có khả năng học các biểu diễn khác nhau, giúp cải thiện hiệu suất cho các nhiệm vụ như dịch ngôn ngữ, phân loại văn bản, và nhận diện đối tượng.

1.1.2.2 Nhược điểm của Multi-head Attention
- Việc sử dụng nhiều "head" attention đồng thời đòi hỏi khối lượng tính toán và bộ nhớ lớn, đặc biệt với các chuỗi dài hoặc dữ liệu có độ phức tạp cao.
- Việc quyết định số lượng "head" tối ưu có thể cần thử nghiệm nhiều lần. Nếu số lượng "head" quá ít, mô hình có thể bỏ sót thông tin quan trọng. Ngược lại, quá nhiều "head" có thể làm tăng chi phí tính toán mà không mang lại lợi ích tương ứng.

1.1.3	Position Embedding
Word embeddings phần nào cho giúp ta biểu diễn ngữ nghĩa của một từ, tuy nhiên cùng một từ ở vị trí khác nhau của câu lại mang ý nghĩa khác nhau. Đó là lý do Transformers có thêm một phần Positional Encoding để inject thêm thông tin về vị trí của một từ.
![image](https://github.com/user-attachments/assets/8330f2f5-57a0-46a8-8820-8c86685b821a)
Hình 1. 5: Minh họa Position Embedding
Trong sinusoidal position embedding, mỗi phần tử trong chuỗi đầu vào (ví dụ: từ hoặc patch) được gán một vector với các giá trị dựa trên vị trí của nó trong chuỗi.
Công thức để tính position embedding:
PE(pos,2i)=sin⁡(pos/1000^(2i/d) )
PE(pos,2i+1)=cos⁡(pos/1000^(2i/d) )
	pos là vị trí của từ (hoặc phần tử) trong chuỗi
	i là chỉ số của vector embedding
	d là kích thước của vector embedding (số chiều của vector)
Mỗi vị trí (pos) sẽ có một vector với các giá trị khác nhau ở từng chiều (dựa trên hàm sin và cos). Mỗi thành phần của vector (tức là mỗi giá trị sin hoặc cos) đại diện cho một khía cạnh khác nhau của vị trí. Sự kết hợp của các thành phần này tạo ra một vector độc đáo cho từng vị trí.

Ví dụ: Giả sử chúng ta có 2 từ trong chuỗi
	Từ thứ nhất ở vị trí 0
	Từ thứ hai ở vị trí 1
Với d = 4 (tức là vector embedding có 4 chiều), vị trí của hai từ này sẽ được gán hai vector khác nhau:
	Với pos = 0, vector sẽ là:
[sin⁡(0),cos⁡(0),sin⁡(0),cos⁡(0) ]=[0,1,0,1]

	Với pos = 1, vector sẽ là:
[sin⁡(1/1000^((2×0)/4) ),cos⁡(1/1000^((2×1)/4) ),sin⁡(1/1000^((2×2)/4) ),cos⁡(1/1000^((2×3)/4) ) ]=[0.01745,0.9999998477,0.000017453,1]

Điều này tạo ra các vector khác biệt cho các vị trí khác nhau trong chuỗi. Sự khác biệt này giúp mô hình hiểu được thứ tự và khoảng cách giữa các phần tử trong chuỗi.

1.2 Vision Transfomer
![image](https://github.com/user-attachments/assets/ce1f029c-9fb2-45ad-b439-168c5ea446ec)
Hình 1. 6: Minh họa về cách hoạt động của Vision Transformer
Kiến trúc của mô hình gồm 3 thành phần chính:
-	Linear Projection of Flattened Patches
-	Transformer encoder.
-	Classification head.

1.2.1 Linear Projection of Flattened Patches
Bao gồm 3 phần là Patch Embedding và Positional Embedding. Riêng phần Positional Embedding đã được đề cập ở phần 1.1.3 nên em sẽ không nhắc lại.
Patch Embedding:
-	Với mỗi ảnh đầu vào, ViT xử lý bằng cách chia ảnh ra thành các phần có kích thước bằng nhau (patch)
-	Ví dụ như sau: Sau khi chia nhỏ ảnh đầu vào ra ta sẽ có 9 patches tất cả.
![image](https://github.com/user-attachments/assets/6c459bf3-e2b2-4f5d-99c8-4d68f1df9a9d)
Hình 1. 7: Minh họa về Patch Embedding
- Bước tiếp theo, đưa các patches này về dạng vector bằng cách flattend các patches này ra.
![image](https://github.com/user-attachments/assets/bc33914c-3824-4275-9818-6e956f647789)
Hình 1. 8: Linear Projection – là một lớp Dense với đầu vào là Flattend Vector của các patches, đầu ra sẽ là embedding vector tương ứng với từng patch

1.2.2 Classification accuracies
![image](https://github.com/user-attachments/assets/4e145729-e107-4ffe-afef-326d552fef11)
Hình 1. 9: Ba tập dữ liệu cho việc huấn luyện mô hình ViT
Với chiến lược training như trên thì ViT khi so sánh với ResNet đạt kết quả như sau:
![image](https://github.com/user-attachments/assets/c4aa2b0d-90ad-4559-90f5-12f40cbcee41)
Hình 1. 10: Tương quan về Accuracy thu được của ResNet và ViT khi so sánh trên cùng tập dữ liệu
-	Pretrained on ImageNet (small), kết quả kém hơn ResNet
-	Pretrained on ImageNet - 21K (medium), độ chính xác của ViT đạt xấp xỉ bằng ResNet
-	Pretrained on JFT (large), ViT đạt độ chính xác vượt trội hơn so với ResNet

1.3 Masked AutoEncoder
Masked Autoencoder (MAE) là một mô hình tự giám sát (self-supervised). Mô hình này đặc biệt hiệu quả trong việc huấn luyện trên các tập dữ liệu không có gán nhãn (unlabeled data) thông qua việc học tái tạo lại các phần bị che khuất của ảnh. MAE lấy cảm hứng từ các mô hình BERT trong xử lý ngôn ngữ tự nhiên (NLP), nơi mà các từ trong câu được che khuất và mô hình phải dự đoán lại các từ đó.
MAE là một phiên bản của autoencoder - một mô hình học sâu bao gồm hai thành phần chính:
-	Encoder: Nhận đầu vào và chuyển đổi chúng thành một biểu diễn nén.
-	Decoder: Giải nén biểu diễn nén đó để tái tạo lại đầu vào ban đầu.
Tuy nhiên, MAE có một cách tiếp cận khác với autoencoder truyền thống. Dưới đây là các bước cơ bản của cơ chế MAE:
-	Chia nhỏ ảnh thành các patch (Image Patches): 
•	Ảnh đầu vào được chia thành các patch nhỏ (ví dụ 16x16 pixel), tương tự như cách chia trong Vision Transformer (ViT).
•	Mỗi patch được xử lý như một token, và một số lượng lớn các patch sẽ bị che khuất ngẫu nhiên (thường là khoảng 75% số lượng patch). Các patch không bị che sẽ được đưa vào encoder.
-	Encoder chỉ xử lý các patch không bị che:
•	Mô hình chỉ xử lý các patch không bị che để giảm khối lượng tính toán. Điều này giúp tiết kiệm tài nguyên và tập trung vào việc học từ một phần của ảnh.
•	Self-attention được sử dụng để học các mối quan hệ giữa các patch không bị che.
-	Decoder tái tạo lại các patch bị che:
•	Sau khi encoder học được các đặc trưng từ các patch không bị che, đầu ra từ encoder sẽ được đưa vào decoder cùng với thông tin về vị trí của các patch bị che.
•	Decoder sẽ học cách tái tạo lại toàn bộ ảnh, bao gồm cả các phần đã bị che khuất.
•	Nhiệm vụ của MAE là tái tạo các patch bị che càng chính xác càng tốt.
![image](https://github.com/user-attachments/assets/ef486668-3082-4753-917f-c093ed71f168)
Hình 1. 11: Minh họa Masked AutoEncoder

# CHƯƠNG 2 – ỨNG DỤNG CỦA MÔ HÌNH ViT-MAE TRONG VIỆC PHÁT HIÊN/PHÂN LOẠI BỆNH LÝ VỀ DA LIỄU
Tổng quan: Trong lĩnh vực y tế, đặc biệt là da liễu, công nghệ trí tuệ nhân tạo (AI) đang ngày càng đóng vai trò quan trọng trong việc hỗ trợ chẩn đoán và phát hiện sớm các bệnh lý da. Sự kết hợp giữa Vision Transformer (ViT) và Masked Autoencoder (MAE) đã mang đến những tiến bộ đáng kể trong khả năng phát hiện và phân loại các bệnh lý da liễu từ hình ảnh. Mô hình này có thể trích xuất các đặc trưng phức tạp của hình ảnh để học, thậm chí mô hình còn học sự liên quan giữa các patch trong một bức ảnh, giữa nhiều bức ảnh với nhau. Tiếp theo, em sử dụng tập dữ liệu đã chiến thắng ISIC 2018 Challenge: Skin Cancer MNIST HAM10000 để fine tune cho việc học các phân loại các bệnh lý về da liễu, bằng cách thêm một fully connected layer ở cuối mô hình.

2.1 ViT-MAE và Cơ chế hoạt động trong Lĩnh vực Y tế
Vision Transformer (ViT) là một trong những kiến trúc mạng nơ-ron học sâu hiện đại, không phụ thuộc vào lớp convolution truyền thống, mà thay vào đó sử dụng cơ chế self-attention để học các mối quan hệ toàn cục giữa các phần khác nhau của ảnh. Trong bài toán da liễu, hình ảnh da có thể được chia thành các patch nhỏ, từ đó ViT sẽ học cách phát hiện ra các dấu hiệu bất thường liên quan đến các bệnh lý về da như ung thư da, viêm da, hay các tổn thương da khác.
Masked Autoencoder (MAE) được tích hợp với ViT nhằm tăng cường khả năng học đặc trưng. Trong MAE, một phần lớn các patch trong ảnh sẽ bị che khuất và mô hình phải tái tạo lại các phần này dựa trên các thông tin còn lại. Điều này buộc mô hình phải học cách hiểu cấu trúc của hình ảnh da từ các patch bị che, giúp nó học được những đặc trưng tiềm ẩn phức tạp hơn từ ảnh da liễu.

2.2 Ứng dụng ViT-MAE trong Phát hiện Bệnh Lý Da Liễu
Sử dụng mô hình ViT-MAE, việc phát hiện và phân loại các bệnh lý da liễu có thể được thực hiện với độ chính xác cao hơn so với các phương pháp truyền thống. Mô hình này được huấn luyện trên các tập dữ liệu lớn về hình ảnh da và có khả năng tự động phân tích để tìm ra những dấu hiệu bất thường như:
-	Phát hiện ung thư da: Mô hình có thể nhận diện các biểu hiện ban đầu của các loại ung thư da nguy hiểm như melanoma, basal cell carcinoma và squamous cell carcinoma.
-	Phân loại các bệnh da khác nhau: Bên cạnh ung thư da, ViT-MAE còn có khả năng phân loại các bệnh da như keratosis (sừng hóa da), nốt ruồi lành tính (nevus), và các rối loạn da khác dựa trên đặc trưng của hình ảnh da.

Khả năng phân loại này đã được chứng minh qua việc áp dụng trên các tập dữ liệu chuẩn như Skin Cancer MNIST: HAM10000, giúp phân chia hình ảnh da thành các nhóm bệnh lý khác nhau với độ chính xác cao.
2.3 Lợi ích của ViT-MAE trong Y tế
- Học tự giám sát không cần nhãn: Mô hình ViT-MAE có khả năng học từ dữ liệu không nhãn, giúp tận dụng được nguồn dữ liệu lớn không có gán nhãn trong lĩnh vực y tế. Điều này đặc biệt quan trọng khi việc thu thập dữ liệu có nhãn trong y tế thường tốn kém và yêu cầu chuyên môn cao.
- Học các đặc trưng sâu và phức tạp: MAE giúp mô hình học được các đặc trưng sâu từ hình ảnh da thông qua việc tái tạo lại các phần bị che. Điều này cho phép mô hình nhận diện được các dấu hiệu bất thường ngay cả khi chúng có thể không rõ ràng đối với con người.
- Tiết kiệm tài nguyên tính toán: Thay vì xử lý toàn bộ hình ảnh, ViT-MAE chỉ xử lý các phần không bị che của ảnh, từ đó giảm tải khối lượng tính toán nhưng vẫn đảm bảo mô hình có thể học đầy đủ các đặc trưng quan trọng của ảnh.
- Khả năng mở rộng: ViT-MAE có thể được fine-tune dễ dàng trên các tập dữ liệu khác nhau và mở rộng ra nhiều bài toán da liễu khác, không chỉ giới hạn ở ung thư da mà còn trong các bệnh lý phức tạp hơn.

2.4 Thách thức và Hướng Phát triển
Mặc dù ViT-MAE đã cho thấy những tiềm năng lớn trong việc hỗ trợ chẩn đoán bệnh lý da liễu, mô hình này vẫn đối diện với một số thách thức:
-	Dữ liệu y tế: Dữ liệu y tế không dễ thu thập và thường yêu cầu độ chính xác cao. Việc chuẩn hóa và đảm bảo chất lượng dữ liệu để huấn luyện mô hình là một quá trình phức tạp.
-	Fine-tuning cho từng loại bệnh cụ thể: Mặc dù ViT-MAE có thể học từ dữ liệu không nhãn, việc fine-tune mô hình với các bộ dữ liệu có nhãn vẫn là cần thiết để đạt độ chính xác cao trong chẩn đoán.

# CHƯƠNG 3 – THỰC NGHIỆM VÀ ĐÁNH GIÁ
![image](https://github.com/user-attachments/assets/184d48bb-175e-44a1-972e-5d6e1f6185f5)
Thực nghiệm 1: Mô hình dự đoán chính xác

![image](https://github.com/user-attachments/assets/ff95768a-702a-46b1-8022-cbf688ee0679)
Thực nghiệm 2: Mô hình dự đoán chính xác

![image](https://github.com/user-attachments/assets/4e827f7a-b651-4d80-a69a-ac0290880fdf)
Thực nghiệm 3: Mô hình dự đoán chính xác

![image](https://github.com/user-attachments/assets/0b821574-db18-42b6-8106-f1a52c618902)
Thực nghiệm 4: Mô hình dự đoán chính xác

![image](https://github.com/user-attachments/assets/4733ed2b-21c9-43b1-90af-b6447a1a524b)
Dựa trên tập kiểm 20% đã được chia từ tập dữ liệu gốc, độ chính xác của mô hình đạt khoảng 80%
