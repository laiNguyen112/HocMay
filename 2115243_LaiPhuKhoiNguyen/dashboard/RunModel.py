import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
import pandas as pd

class MoHinhDonGian(nn.Module):
    def __init__(self, so_dau_vao, so_dau_ra):
        super(MoHinhDonGian, self).__init__()
        self.fc = nn.Linear(so_dau_vao, so_dau_ra)
    
    def forward(self, x):
        return self.fc(x)

torch.serialization.add_safe_globals({"MoHinhDonGian": MoHinhDonGian})

du_lieu_dau_vao = None
mo_hinh = None
ket_qua_du_doan = None

def tai_file_dau_vao():
    global du_lieu_dau_vao
    duong_dan_file = filedialog.askopenfilename(filetypes=[("Tệp CSV", "*.csv"), ("Tất cả các tệp", "*.*")])
    if duong_dan_file:
        try:
            du_lieu_dau_vao = pd.read_csv(duong_dan_file)
            messagebox.showinfo("Thành công", "Đã tải tệp dữ liệu đầu vào!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải tệp: {e}")

def tai_mo_hinh():
    global mo_hinh
    duong_dan_mo_hinh = filedialog.askopenfilename(filetypes=[("Tệp mô hình PyTorch", "*.pth"), ("Tất cả các tệp", "*.*")])
    if duong_dan_mo_hinh:
        try:
            so_dau_vao = 3
            so_dau_ra = 2
            mo_hinh = MoHinhDonGian(so_dau_vao, so_dau_ra)
            trong_so_mo_hinh = torch.load(duong_dan_mo_hinh, weights_only=True)
            mo_hinh.load_state_dict(trong_so_mo_hinh)
            mo_hinh.eval()
            messagebox.showinfo("Thành công", "Đã tải mô hình thành công!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mô hình: {e}")

def du_doan():
    global mo_hinh, ket_qua_du_doan, du_lieu_dau_vao
    du_lieu_nhap = text_input.get("1.0", tk.END).strip()
    if not mo_hinh:
        messagebox.showerror("Lỗi", "Hãy tải mô hình trước!")
        return
    if not du_lieu_nhap:
        messagebox.showerror("Lỗi", "Hãy nhập dữ liệu đầu vào!")
        return

    try:
        tensor_dau_vao = torch.tensor([float(x) for x in du_lieu_nhap.split()]).unsqueeze(0)
        dau_ra = mo_hinh(tensor_dau_vao)
        ket_qua = dau_ra.argmax(dim=1).item()
        ket_qua_du_doan = f"Kết quả dự đoán: {ket_qua}"
        messagebox.showinfo("Dự đoán", ket_qua_du_doan)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Dự đoán không thành công: {e}")

def hien_thi_ket_qua():
    global ket_qua_du_doan
    if ket_qua_du_doan:
        cua_so_ket_qua = tk.Toplevel(app)
        cua_so_ket_qua.title("Kết quả")
        cua_so_ket_qua.geometry("300x100")
        tk.Label(cua_so_ket_qua, text=ket_qua_du_doan, font=("Arial", 14)).pack(pady=20)
    else:
        messagebox.showerror("Lỗi", "Không có kết quả dự đoán!")

def luu_mo_hinh():
    duong_dan_mo_hinh = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("Tệp mô hình PyTorch", "*.pth")])
    if duong_dan_mo_hinh:
        try:
            so_dau_vao = 3
            so_dau_ra = 2
            mo_hinh = MoHinhDonGian(so_dau_vao, so_dau_ra)
            torch.save(mo_hinh.state_dict(), duong_dan_mo_hinh)
            messagebox.showinfo("Thành công", f"Đã lưu mô hình tại {duong_dan_mo_hinh}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu mô hình: {e}")

app = tk.Tk()
app.title("Giao diện mô hình PyTorch")
app.geometry("600x500")

tk.Button(app, text="Tải tệp dữ liệu", command=tai_file_dau_vao, width=20).pack(pady=10)
tk.Button(app, text="Tải mô hình", command=tai_mo_hinh, width=20).pack(pady=10)
tk.Button(app, text="Lưu mô hình", command=luu_mo_hinh, width=20).pack(pady=10)

tk.Label(app, text="Nhập dữ liệu đầu vào (cách nhau bằng dấu cách):").pack()
text_input = tk.Text(app, height=5, width=50)
text_input.pack(pady=10)

tk.Button(app, text="Dự đoán", command=du_doan, width=20).pack(pady=10)
tk.Button(app, text="Hiển thị kết quả", command=hien_thi_ket_qua, width=20).pack(pady=10)

app.mainloop()
