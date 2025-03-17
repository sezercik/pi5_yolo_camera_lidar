import tkinter as tk
from tkinter import ttk, Label, messagebox, Frame, Entry, StringVar
from PIL import Image, ImageTk
import serial
import cv2
from picamera2 import Picamera2
import threading
import logging
import time
from ultralytics import YOLO
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DARK_BG = "#1E1E1E"
LIGHT_BG = "#2D2D2D"
TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#4CAF50"
WARNING_COLOR = "#FF5722"
ALERT_COLOR = "#FFEB3B"
FONT_MAIN = ("Helvetica", 12)
FONT_TITLE = ("Helvetica", 16, "bold")
FONT_WARNING = ("Helvetica", 20, "bold")

class LidarSensor:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.buffer = bytearray()

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, baudrate=self.baudrate, timeout=1)
            logger.info(f"Connected to LIDAR on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to LIDAR: {e}")
            return False

    def disconnect(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Disconnected from LIDAR")

    def read_available(self):
        if not self.serial or not self.serial.is_open:
            logger.warning("LIDAR serial port is not open")
            return []
        try:
            data = self.serial.read(self.serial.in_waiting or 1)
            if data:
                self.buffer.extend(data)
                distances = []
                while len(self.buffer) >= 9:
                    if self.buffer[0] == 0x59 and self.buffer[1] == 0x59:
                        distance = self.buffer[2] + self.buffer[3] * 256
                        distances.append(distance)
                        self.buffer = self.buffer[9:]
                    else:
                        self.buffer = self.buffer[1:]
                return distances
        except Exception as e:
            logger.error(f"Error reading LIDAR data: {e}")
        return []

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Engel Tespit Prototipi")
        self.root.geometry("1280x800")
        self.root.configure(bg=DARK_BG)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', 
                             font=FONT_MAIN, 
                             background=ACCENT_COLOR, 
                             foreground=TEXT_COLOR)
        self.style.map('TButton', 
                       background=[('active', '#388E3C'), ('disabled', '#757575')])
        self.style.configure('TFrame', background=DARK_BG)
        self.style.configure('TEntry', 
                            fieldbackground=LIGHT_BG,
                            foreground=TEXT_COLOR,
                            insertcolor=TEXT_COLOR)
        self.style.configure('TLabel', 
                            background=DARK_BG,
                            foreground=TEXT_COLOR,
                            font=FONT_MAIN)
        
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(main={"size": (640, 480)})
        self.camera.configure(config)
        self.lidar = LidarSensor()
        if not self.lidar.connect():
            messagebox.showerror("Error", "LIDAR bağlantı hatası")
            self.root.destroy()
            return

        self.model = YOLO("nano_ncnn_model", task='segment')
        self.latest_frame = None
        self.latest_distance = 0
        self.running = False
        self.lock = threading.Lock()
        self.segment_results = None
        
        self.min_distance_var = StringVar(value="50")
        self.max_distance_var = StringVar(value="200")
        
        self.min_distance = 50
        self.max_distance = 200
        self.warning_text = ""
        
        self.main_frame = Frame(root, bg=DARK_BG)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.header_frame = Frame(self.main_frame, bg=DARK_BG)
        self.header_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.title_label = Label(self.header_frame, 
                                text="Engel Tespit Sistemi", 
                                font=("Helvetica", 24, "bold"),
                                fg=ACCENT_COLOR, 
                                bg=DARK_BG)
        self.title_label.pack(side=tk.LEFT)
        
        self.settings_frame = Frame(self.header_frame, bg=DARK_BG)
        self.settings_frame.pack(side=tk.RIGHT, padx=10)
        
        self.min_distance_label = ttk.Label(self.settings_frame, 
                                          text="Min Mesafe (cm):", 
                                          style='TLabel')
        self.min_distance_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.min_distance_entry = ttk.Entry(self.settings_frame, 
                                          width=8, 
                                          textvariable=self.min_distance_var, 
                                          style='TEntry')
        self.min_distance_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.max_distance_label = ttk.Label(self.settings_frame, 
                                          text="Max Mesafe (cm):", 
                                          style='TLabel')
        self.max_distance_label.grid(row=0, column=2, padx=5, pady=5)
        
        self.max_distance_entry = ttk.Entry(self.settings_frame, 
                                          width=8, 
                                          textvariable=self.max_distance_var, 
                                          style='TEntry')
        self.max_distance_entry.grid(row=0, column=3, padx=5, pady=5)
        
        self.apply_button = ttk.Button(self.settings_frame, 
                                     text="Uygula", 
                                     command=self.apply_settings)
        self.apply_button.grid(row=0, column=4, padx=5, pady=5)
        
        self.video_frame = Frame(self.main_frame, bg=DARK_BG)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.live_container = Frame(self.video_frame, bg=LIGHT_BG, padx=10, pady=10, bd=2, relief=tk.GROOVE)
        self.live_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.live_label = Label(self.live_container, 
                              text="Canlı Görüntü", 
                              font=FONT_TITLE, 
                              fg=TEXT_COLOR, 
                              bg=LIGHT_BG)
        self.live_label.pack(pady=(0, 10))
        
        self.live_image_label = Label(self.live_container, bg=DARK_BG, bd=1, relief=tk.SUNKEN)
        self.live_image_label.pack(fill=tk.BOTH, expand=True)
        
        self.filtered_container = Frame(self.video_frame, bg=LIGHT_BG, padx=10, pady=10, bd=2, relief=tk.GROOVE)
        self.filtered_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.filtered_label = Label(self.filtered_container, 
                                  text="Tespit Edilen Nesneler", 
                                  font=FONT_TITLE,
                                  fg=TEXT_COLOR, 
                                  bg=LIGHT_BG)
        self.filtered_label.pack(pady=(0, 10))
        
        self.filtered_image_label = Label(self.filtered_container, bg=DARK_BG, bd=1, relief=tk.SUNKEN)
        self.filtered_image_label.pack(fill=tk.BOTH, expand=True)
        
        self.status_frame = Frame(self.main_frame, bg=LIGHT_BG, bd=2, relief=tk.GROOVE)
        self.status_frame.pack(fill=tk.X, pady=20)
        
        self.info_frame = Frame(self.status_frame, bg=LIGHT_BG, padx=15, pady=15)
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.lidar_status_label = Label(self.info_frame, 
                                      text="Lidar Mesafesi: 0 cm", 
                                      font=FONT_MAIN, 
                                      fg=TEXT_COLOR, 
                                      bg=LIGHT_BG)
        self.lidar_status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.range_status_label = Label(self.info_frame,
                                     text=f"Tespit aralığı: {self.min_distance}-{self.max_distance} cm",
                                     font=FONT_MAIN,
                                     fg=ACCENT_COLOR,
                                     bg=LIGHT_BG)
        self.range_status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.warning_frame = Frame(self.status_frame, bg=LIGHT_BG, padx=15, pady=15)
        self.warning_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.warning_label = Label(self.warning_frame, 
                                 text="", 
                                 font=FONT_WARNING, 
                                 fg=WARNING_COLOR, 
                                 bg=LIGHT_BG)
        self.warning_label.pack(side=tk.RIGHT)
        
        self.control_frame = Frame(self.main_frame, bg=DARK_BG, pady=10)
        self.control_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(self.control_frame, 
                                     text="SİSTEMİ BAŞLAT", 
                                     command=self.start, 
                                     style='TButton',
                                     width=20)
        self.start_button.pack(pady=10)
        
        self.progress = ttk.Progressbar(self.control_frame, 
                                      orient="horizontal", 
                                      length=300, 
                                      mode="indeterminate")
        self.progress.pack(pady=(0, 10))
        
        self.setup_image_placeholders()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def apply_settings(self):
        try:
            min_val = int(self.min_distance_var.get())
            max_val = int(self.max_distance_var.get())
            
            if min_val < 0 or max_val < 0:
                messagebox.showerror("Hata", "Mesafe değerleri negatif olamaz")
                return
                
            if min_val >= max_val:
                messagebox.showerror("Hata", "Minimum mesafe maksimumdan küçük olmalı")
                return
                
            self.min_distance = min_val
            self.max_distance = max_val
            
            self.range_status_label.config(text=f"Tespit aralığı: {self.min_distance}-{self.max_distance} cm")
            
            self.show_temp_message("Ayarlar başarıyla güncellendi")
            
        except ValueError:
            messagebox.showerror("Hata", "Mesafe değerleri sayı olmalı")

    def show_temp_message(self, message):
        prev_text = self.warning_text
        prev_color = self.warning_label.cget("fg")
        
        self.warning_label.config(text=message, fg=ACCENT_COLOR)
        
        def restore():
            self.warning_label.config(text=prev_text, fg=prev_color)
        
        self.root.after(2000, restore)

    def setup_image_placeholders(self):
        placeholder = Image.new('RGB', (640, 480), color='black')
        placeholder_text = "Kamera başlatılıyor..."
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(placeholder)
        try:
            font = ImageFont.truetype("Arial", 30)
        except IOError:
            font = ImageFont.load_default()
            
        textwidth, textheight = draw.textbbox((0, 0), placeholder_text, font=font)[2:]
        position = ((640 - textwidth) // 2, (480 - textheight) // 2)
        
        draw.text(position, placeholder_text, fill=(0, 255, 0), font=font)
        
        placeholder_img = ImageTk.PhotoImage(placeholder)
        
        self.live_image_label.config(image=placeholder_img)
        self.live_image_label.image = placeholder_img
        self.filtered_image_label.config(image=placeholder_img)
        self.filtered_image_label.image = placeholder_img

    def start(self):
        if not self.running:
            self.running = True
            self.start_button.config(text="ÇALIŞIYOR...", state="disabled")
            self.progress.start(10)
            self.camera.start()
            threading.Thread(target=self.camera_loop, daemon=True).start()
            threading.Thread(target=self.lidar_loop, daemon=True).start()
            self.update_gui()

    def camera_loop(self):
        while self.running:
            frame = self.camera.capture_array()
            with self.lock:
                self.latest_frame = frame.copy()

    def lidar_loop(self):
        while self.running:
            distances = self.lidar.read_available()
            if distances:
                with self.lock:
                    self.latest_distance = distances[-1]
            time.sleep(0.01)

    def update_gui(self):
        if not self.running:
            return

        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            distance = self.latest_distance

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = self.add_distance_overlay(img, distance)
            imgtk = ImageTk.PhotoImage(image=img)
            self.live_image_label.config(image=imgtk)
            self.live_image_label.imgtk = imgtk

            # Normal durumda burası kullanılmayacak
            # cv2.imwrite("t.jpg", frame)

        self.update_distance_indicator(distance)

        if self.min_distance <= distance <= self.max_distance:
            try:
                self.run_yolo()
            except Exception as e:
                logger.error(f"Error processing image with YOLO: {e}")
                self.warning_text = ""
                self.warning_label.config(text="")
        else:
            self.show_out_of_range_image(distance)
            self.warning_text = ""
            self.warning_label.config(text="")
        
        self.root.after(33, self.update_gui)

    def show_out_of_range_image(self, distance):
        placeholder = Image.new('RGB', (640, 480), color='black')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(placeholder)
        
        try:
            font = ImageFont.truetype("Arial", 24)
            small_font = ImageFont.truetype("Arial", 18)
        except IOError:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        if distance < self.min_distance:
            message = "Engel Bekleniyor"
            reason = f"Mesafe çok yakın: {distance} cm"
            color = (255, 165, 0) 
        else: 
            message = "Engel Bekleniyor"
            reason = f"Mesafe çok uzak: {distance} cm"
            color = (100, 100, 255)
        
        text_bbox = draw.textbbox((0, 0), message, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = ((640 - text_width) // 2, (480 - text_height) // 2 - 20)
        draw.text(position, message, fill=color, font=font)
        
        text_bbox = draw.textbbox((0, 0), reason, font=small_font)
        text_width = text_bbox[2] - text_bbox[0]
        position = ((640 - text_width) // 2, (480 - text_height) // 2 + 20)
        draw.text(position, reason, fill=color, font=small_font)
        
        range_text = f"Geçerli aralık: {self.min_distance}-{self.max_distance} cm"
        text_bbox = draw.textbbox((0, 0), range_text, font=small_font)
        text_width = text_bbox[2] - text_bbox[0]
        position = ((640 - text_width) // 2, (480 - text_height) // 2 + 60)
        draw.text(position, range_text, fill=(200, 200, 200), font=small_font)
        
        placeholder_img = ImageTk.PhotoImage(placeholder)
        self.filtered_image_label.config(image=placeholder_img)
        self.filtered_image_label.image = placeholder_img

    def add_distance_overlay(self, img, distance):
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial", 24)
        except IOError:
            font = ImageFont.load_default()
            
        distance_text = f"{distance} cm"
        text_bbox = draw.textbbox((0, 0), distance_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        padding = 10
        position = (img.width - text_width - padding, padding)
        
        draw.rectangle(
            [position[0] - 5, position[1] - 5, 
             position[0] + text_width + 5, position[1] + text_height + 5], 
            fill=(0, 0, 0, 128)
        )
        
        if distance < self.min_distance:
            text_color = (255, 165, 0)
        elif distance > self.max_distance:
            text_color = (100, 100, 255)
        else:
            text_color = (0, 255, 0)
            
        draw.text(position, distance_text, fill=text_color, font=font)
        return img

    def update_distance_indicator(self, distance):
        if distance < self.min_distance:
            status_text = f"Lidar Mesafesi: {distance} cm (Çok Yakın!)"
            status_color = WARNING_COLOR
        elif distance > self.max_distance:
            status_text = f"Lidar Mesafesi: {distance} cm (Çok Uzak)"
            status_color = TEXT_COLOR
        else:
            status_text = f"Lidar Mesafesi: {distance} cm (Optimal Mesafe)"
            status_color = ACCENT_COLOR
            
        self.lidar_status_label.config(text=status_text, fg=status_color)

    def run_yolo(self):
        results = self.model("t.jpg")
        self.segment_results = results[0]
        
        detected_objects = [self.segment_results.names[int(box.cls)] for box in self.segment_results.boxes]
        
        if detected_objects:
            self.display_segmentation_results()
            
            self.warning_text = f"DİKKAT: {', '.join(detected_objects)} tespit edildi!"
            self.flash_warning()
        else:
            self.show_no_detection_image()
            
            self.warning_text = ""
            self.warning_label.config(text="")

    def display_segmentation_results(self):
        result_img = self.segment_results.plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(result_img_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self.filtered_image_label.config(image=img_tk)
        self.filtered_image_label.image = img_tk

    def show_no_detection_image(self):
        try:
            img = cv2.imread("t.jpg")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(pil_img)
            
            try:
                font = ImageFont.truetype("Arial", 24)
            except IOError:
                font = ImageFont.load_default()
                
            message = "Nesne tespit edilmedi"
            text_bbox = draw.textbbox((0, 0), message, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            position = ((pil_img.width - text_width) // 2, 20)
            
            draw.rectangle(
                [position[0] - 10, position[1] - 5, 
                 position[0] + text_width + 10, position[1] + text_height + 5], 
                fill=(0, 0, 0, 180)
            )
            
            draw.text(position, message, fill=(0, 255, 0), font=font)
            img_tk = ImageTk.PhotoImage(image=pil_img)
            self.filtered_image_label.config(image=img_tk)
            self.filtered_image_label.image = img_tk
            
        except Exception as e:
            logger.error(f"Error showing no detection image: {e}")

    def flash_warning(self):
        if self.running and self.warning_text:
            current_color = self.warning_label.cget("fg")
            new_color = WARNING_COLOR if current_color == ALERT_COLOR else ALERT_COLOR
            self.warning_label.config(text=self.warning_text, fg=new_color)
            self.root.after(500, self.flash_warning)

    def on_closing(self):
        self.running = False
        time.sleep(0.1)
        self.camera.stop()
        self.lidar.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
