import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkvideo import tkvideo

recipe=pd.read_csv("recipee.csv")

model = load_model("trainmodel.h5")

class_indices = {
    'Banh canh': 0, 'Banh gio': 1, 'Banh mi': 2, 'Banh pia': 3, 'Banh tet': 4,
    'Banh trang nuong': 5, 'Banh xeo': 6, 'Bun bo Hue': 7, 'Bun dau mam tom': 8, 'Bun rieu': 9,
    'Bun thit nuong': 10, 'Ca kho to': 11, 'Cao lau': 12, 'Chao long': 13, 'Com tam': 14,
    'Goi cuon': 15, 'Hu tieu': 16, 'Mi quang': 17, 'Nem chua': 18, 'Pho': 19,
    'Xoi xeo': 20
}
idx2label = {v: k for k, v in class_indices.items()}

def predict_image_from_PIL(pil_image):
    img = pil_image.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    predicted_label = idx2label[predicted_index]
    confidence = np.max(preds)

    return predicted_label, confidence

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CookLens App")
        self.geometry("1080x630")
        self.resizable(False, False)

        self.page1 = Page1(self)
        self.main_page = MainPage(self)
        self.page3= Page3(self)

        self.page1.pack(fill="both", expand=True)

    def show_page1(self):
        self._hide_all()
        self.page1.pack(fill="both", expand=True)

    def show_main_page(self):
        self._hide_all()
        self.main_page.pack(fill="both", expand=True)

    def show_page3(self, dish_name):
        self._hide_all()
        self.page3.update_recipe(dish_name)  # Gọi hàm để update nội dung
        self.page3.pack(fill="both", expand=True)

    def _hide_all(self):
        for widget in (self.page1, self.page3,self.main_page):
            widget.pack_forget()

    def on_closing(self):
        self.main_page.stop_camera()
        self.destroy()

class Page1(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        bg = Image.open("START (1).png")
        bg_resized = bg.resize((1080, 630), Image.Resampling.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(bg_resized)
        bg_label = tk.Label(self, image=self.bg_image)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.start_img = ImageTk.PhotoImage(Image.open("button.jpg").resize((210, 58)))
        start_btn = tk.Button(self, image=self.start_img, command=master.show_main_page, relief="flat", borderwidth=0)
        start_btn.place(x=668, y=423, width=210, height=58)

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        self.canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bắt sự kiện cuộn chuột
        self.scrollable_frame.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.scrollable_frame.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

class MainPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.cap = None
        self.running = False
        self.last_frame = None
        self.current_image = None

        self.upload_img = ImageTk.PhotoImage(Image.open("b4.jpg").resize((210, 58)))
        self.toggle_img = ImageTk.PhotoImage(Image.open("b3.jpg").resize((210, 58)))
        self.capture_img = ImageTk.PhotoImage(Image.open("b6.jpg").resize((210, 58)))
        self.predict_img = ImageTk.PhotoImage(Image.open("b5.jpg").resize((150, 58)))
        self.back_img = ImageTk.PhotoImage(Image.open("b2.jpg").resize((210, 58)))
        self.recipe_img = ImageTk.PhotoImage(Image.open("b10.jpg").resize((150, 58)))

        scrollable_frame = ScrollableFrame(self)
        scrollable_frame.pack(fill="both", expand=True)

        img1 = Image.open("page2.png").resize((1080, 630))
        img2 = Image.open("START.png").resize((1080, 630))
        self.tk_img1 = ImageTk.PhotoImage(img1)
        self.tk_img2 = ImageTk.PhotoImage(img2)

        total_height = img1.height + img2.height
        canvas = tk.Canvas(scrollable_frame.scrollable_frame, width=1080, height=total_height)
        canvas.pack()

        canvas.create_image(0, 0, anchor="nw", image=self.tk_img1)
        canvas.create_image(0, img1.height, anchor="nw", image=self.tk_img2)

        self.canvas = canvas

        self.video_label = tk.Label(canvas)
        canvas.create_window(300, 700, window=self.video_label, anchor="nw")

        upload_btn = tk.Button(canvas, image=self.upload_img, text="UPLOAD FILE", command=self.upload_file, relief="flat", borderwidth=0)
        upload_btn.place(x=790, y=1070, width=210, height=58)

        self.toggle_btn = tk.Button(canvas, image=self.toggle_img, command=self.toggle_camera, relief="flat", borderwidth=0)
        self.toggle_btn.place(x=80, y=1070, width=210, height=58)

        self.capture_btn = tk.Button(canvas, image=self.capture_img, command=self.capture_image, relief="flat", borderwidth=0)
        self.capture_btn.place(x=435, y=1070, width=210, height=58)

        self.predict_btn = tk.Button(canvas, image=self.predict_img, command=self.predict_dish, relief="flat", borderwidth=0)
        self.predict_btn.place(x=800, y=680, width=150, height=58)

        self.result_label = tk.Label(canvas, text="", font=("Arial", 14), bg="white")
        canvas.create_window(800, 745, window=self.result_label, anchor="nw")

        back_btn = tk.Button(canvas, image=self.back_img, command=master.show_page1, relief="flat", borderwidth=0)
        back_btn.place(x=825, y=33, width=210, height=58)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.png;*.jpeg")])
        if file_path:
            img = Image.open(file_path).resize((400, 300), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk
            self.current_image = Image.open(file_path)

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()
            self.toggle_btn.config(text="Stop Camera")

    def stop_camera(self):
        if self.cap:
            self.running = False
            self.cap.release()
            self.cap = None
            self.video_label.config(image="")
            self.video_label.image = None
            self.toggle_btn.config(text="Start Camera")

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def update_frame(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame).resize((400, 300), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.video_label.config(image=img_tk)
                self.video_label.image = img_tk
                self.last_frame = frame
        if self.running:
            self.after(10, self.update_frame)

    def capture_image(self):
        if self.last_frame is not None:
            captured = self.last_frame.copy()
            self.stop_camera()
            img = Image.fromarray(captured).resize((400, 300), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk
            self.current_image = Image.fromarray(captured)

    def predict_dish(self):
        if self.current_image:
            predicted_label, confidence = predict_image_from_PIL(self.current_image)
            self.result_label.config(text=f"Món ăn: {predicted_label}")
        
            recipe_btn = tk.Button(self.canvas, image=self.recipe_img, relief="flat", borderwidth=0, command=lambda: self.master.show_page3(predicted_label))
            recipe_btn.place(x=800, y=820, width=150, height=58)
        else:
            self.result_label.config(text="Chưa có ảnh!")

class Page3(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both')

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text='Giới thiệu')
        self.notebook.add(self.tab2, text='Công thức')
        self.notebook.add(self.tab3, text='Lưu ý')

        # Tab 1: Giới thiệu
        self.back_img = ImageTk.PhotoImage(Image.open("b2.jpg").resize((150, 58)))

        self.scrollable_frame_tab1 = ScrollableFrame(self.tab1)  # Khung cuộn cho Tab 1
        self.scrollable_frame_tab1.pack(fill="both", expand=True)

        back_btn_tab1 = tk.Button(self.scrollable_frame_tab1.scrollable_frame, image=self.back_img,
                             command=master.show_main_page, relief="flat", borderwidth=0)
        back_btn_tab1.place(x=10, y=10, width=150, height=58)

        label1 = tk.Label(self.scrollable_frame_tab1.scrollable_frame, text="Thông tin giới thiệu món ăn", font=("Arial", 16))
        label1.pack(pady=20)

        self.image_frame = tk.Frame(self.scrollable_frame_tab1.scrollable_frame)
        self.image_frame.pack(pady=10)

        self.image_labels = []
        for _ in range(5):
            lbl = tk.Label(self.image_frame)
            lbl.pack(side="left", padx=5)
            self.image_labels.append(lbl)

        self.desc_label = tk.Label(self.scrollable_frame_tab1.scrollable_frame, text="", font=("Arial", 14), wraplength=800, justify="left")
        self.desc_label.pack(pady=10)

        self.video_label = tk.Label(self.scrollable_frame_tab1.scrollable_frame, text="") 

        self.scrollable_frame = ScrollableFrame(self.tab2)
        self.scrollable_frame.pack(fill="both", expand=True)

        back_btn_tab2 = tk.Button(self.scrollable_frame.scrollable_frame,image=self.back_img,
                             command=master.show_main_page, relief="flat", borderwidth=0)
        back_btn_tab2.place(x=650, y=18, width=150, height=58)

        self.label_title = tk.Label(self.scrollable_frame.scrollable_frame, text="", font=("Arial", 24, "bold"))
        self.label_title.pack(pady=20)

        self.video_label1 = tk.Label(self.scrollable_frame.scrollable_frame, text="")  
        self.video_label1.pack(pady=10)
    
        self.ingredients_text = tk.Label(self.scrollable_frame.scrollable_frame, wraplength=800,
                                        font=("Arial", 14), justify="left")
        self.ingredients_text.pack(padx=20, pady=10)

        scrollable_frame_tab3 = ScrollableFrame(self.tab3)
        scrollable_frame_tab3.pack(fill="both", expand=True)

        tab3_1 = Image.open("page7.png").resize((1080, 630))
        tab3_2 = Image.open("page8.png").resize((1080, 630))
        self.tk_tab3_1 = ImageTk.PhotoImage(tab3_1)
        self.tk_tab3_2 = ImageTk.PhotoImage(tab3_2)

        total_height = tab3_1.height + tab3_2.height
        canvas = tk.Canvas(scrollable_frame_tab3.scrollable_frame, width=1080, height=total_height)
        canvas.pack()

        canvas.create_image(0, 0, anchor="nw", image=self.tk_tab3_1)
        canvas.create_image(0, tab3_1.height, anchor="nw", image=self.tk_tab3_2)

        self.canvas = canvas

        back_btn_tab3 = tk.Button(scrollable_frame_tab3.scrollable_frame,image=self.back_img,
                             command=master.show_main_page, relief="flat", borderwidth=0)
        back_btn_tab3.place(x=800, y=18, width=150, height=58)

    def update_recipe(self, dish_name):
        self.label_title.config(text=f"Công thức: {dish_name}")
        row = recipe[recipe["Tên"] == dish_name]
        if not row.empty:
            for i in range(5):
                col_name = f"Hình {i+1}"
                try:
                    img_path = row.iloc[0][col_name]
                    img = Image.open(img_path).resize((160, 120))
                    photo = ImageTk.PhotoImage(img)
                    self.image_labels[i].config(image=photo)
                    self.image_labels[i].image = photo
                except:
                    self.image_labels[i].config(image='', text='Ảnh lỗi')

            description = row.iloc[0].get("Mô tả", "Không có mô tả.")
            self.desc_label.config(text=description)

            video_file1 = row.iloc[0].get("Video 2", "") 
  
            if video_file1:
                self.video_label1.config(text="")  
                self.video_frame1 = tk.Frame(self.scrollable_frame.scrollable_frame)
                self.video_frame1.pack(pady=10)
                try:
                    video_label1 = tk.Label(self.video_frame1)
                    self.video_label1_widget=video_label1
                    self.video_label1_widget.pack(padx=10, pady=10, fill="both", expand=True)
                    self.videoplayer1 = tkvideo(video_file1, video_label1, loop=0)
                    self.videoplayer1.play()
                except Exception as e:
                    print(f"Error playing video: {e}")
            else:
                self.video_label1.config(text="Không có video minh họa.")  

            ingredients = row.iloc[0]["Nguyên liệu"]
            method = row.iloc[0]["Cách chế biến"]

            self.ingredients_text.config(text= f"🌿 Nguyên liệu:\n{ingredients}\n\n🍳 Cách chế biến:\n{method}")

            self.canvas.delete("nutrition_text") 

            nutrition_info = row.iloc[0].get("Thành phần dinh dưỡng", "")
            if pd.notna(nutrition_info):
                self.canvas.create_text(
                    100, 900,
                    anchor="nw",
                    text=f"Thành phần dinh dưỡng:\n{nutrition_info}",
                    font=("Times New Roman", 20),
                    fill="black",
                    width=980,
                    tags="nutrition_text"
                )

            self.canvas.delete("calo_text") 

            calo_info = row.iloc[0].get("Calories", "")
            if pd.notna(calo_info):
                self.canvas.create_text(
                    200, 650,
                    anchor="nw",
                    text=f"Calories:\n{calo_info}",
                    font=("Times New Roman", 20),
                    fill="black",
                    width=980,
                    tags="calo_text"
                )

        else:
            for lbl in self.image_labels:
                lbl.config(image='', text='')

            self.desc_label.config(text="Không tìm thấy mô tả.")
            self.video_label.config(text="")
            self.ingredients_text.config(text="")

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
