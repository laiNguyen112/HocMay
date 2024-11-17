import torch
import torch.nn as nn

# Định nghĩa mô hình đơn giản
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

# Khởi tạo mô hình
input_size = 3  # Ví dụ: số lượng cột đầu vào (thay đổi tùy thuộc vào dữ liệu thực tế)
output_size = 2  # Ví dụ: phân loại nhị phân (0 hoặc 1)
model = SimpleModel(input_size, output_size)

# Tạo dữ liệu mẫu và nhãn để huấn luyện (giả lập)
data = torch.randn(10, input_size)  # 10 mẫu, mỗi mẫu có 3 đặc trưng
labels = torch.randint(0, output_size, (10,))  # Nhãn ngẫu nhiên (0 hoặc 1)

# Cấu hình hàm mất mát và bộ tối ưu
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Huấn luyện mô hình
for epoch in range(100):  # Huấn luyện 100 epoch
    optimizer.zero_grad()  # Reset gradient
    outputs = model(data)  # Dự đoán
    loss = criterion(outputs, labels)  # Tính hàm mất mát
    loss.backward()  # Lan truyền ngược
    optimizer.step()  # Cập nhật trọng số

# Lưu trọng số mô hình vào file simple_model.pth
torch.save(model.state_dict(), "simple_model.pth")
print("Model saved as 'simple_model.pth'")
