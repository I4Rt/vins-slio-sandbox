"""Desktop viewer for one camera and two MaixSense A010 sensors."""

import tkinter as tk
from collections import deque
from threading import Event, Thread
from time import monotonic
from tkinter import ttk

import cv2
from PIL import Image, ImageOps, ImageTk

from capture import StreamError, camera_frames, camera_indices, serial_ports, tof_frames


OFF = "Не выбрано"
BACKGROUND = "#10151e"
PANEL = "#1b2432"
VIEWPORT = "#0b1018"
TEXT = "#edf2f9"
MUTED = "#a8b7c9"


class Stream:
    def __init__(self, source, device):
        self.device = device
        self.stop = Event()
        self.updates = deque(maxlen=1)
        self.thread = Thread(target=self.read, args=(source,), daemon=True)
        self.thread.start()

    def read(self, source):
        try:
            for frame in source(self.device, self.stop):
                self.updates.append(Image.fromarray(frame))
        except (OSError, StreamError, cv2.error) as error:
            if not self.stop.is_set():
                self.updates.append(str(error))


class Preview(ttk.Frame):
    def __init__(self, parent, title, source, on_select, *, is_depth=False):
        super().__init__(parent, padding=8, style="Panel.TFrame")
        self.source = source
        self.is_depth = is_depth
        self.device = None
        self.stream = None
        self.pending = False
        self.frame = None
        self.photo = None
        self.angle = 0
        self.dirty = True

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        header = ttk.Frame(self, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=title, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.selector = ttk.Combobox(header, state="readonly", width=17, values=(OFF,))
        self.selector.set(OFF)
        self.selector.grid(row=0, column=1, padx=(8, 6))
        self.selector.bind("<<ComboboxSelected>>", lambda _event: on_select(self))
        ttk.Button(header, text="↻ 90°", command=self.rotate).grid(row=0, column=2)

        self.canvas = tk.Canvas(self, background=VIEWPORT, highlightthickness=0, width=1, height=1)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.resized)
        self.image_item = self.canvas.create_image(0, 0)
        self.status_item = self.canvas.create_text(0, 0, fill=MUTED, justify="center")
        self.show_status("Выберите устройство")

    def show_status(self, message):
        self.frame = None
        self.dirty = True
        self.canvas.itemconfigure(self.status_item, text=message)

    def set_devices(self, devices):
        choices = [str(device) for device in devices]
        if self.device is not None and str(self.device) not in choices:
            choices.append(str(self.device))
        self.selector.configure(values=[OFF, *choices])

    def select(self, device):
        self.device = device
        if self.stream is not None:
            self.stream.stop.set()
        self.pending = device is not None
        self.show_status("Подключение…" if self.pending else "Выберите устройство")

    def rotate(self):
        self.angle = (self.angle + 90) % 360
        self.dirty = True

    def resized(self, _event):
        self.dirty = True

    def update_stream(self, others, scanning):
        stream = self.stream
        if stream is not None and not stream.stop.is_set():
            try:
                update = stream.updates.popleft()
            except IndexError:
                pass
            else:
                if isinstance(update, str):
                    self.show_status(update)
                else:
                    self.frame = update
                    self.dirty = True

        if stream is not None and not stream.thread.is_alive() and (
            stream.stop.is_set() or not stream.updates
        ):
            self.stream = None

        if self.pending and self.stream is None and (self.is_depth or not scanning):
            busy = any(
                other.stream is not None
                and other.stream.device == self.device
                and other.stream.thread.is_alive()
                for other in others
            )
            if not busy:
                self.stream = Stream(self.source, self.device)
                self.pending = False

    def draw(self):
        if not self.dirty:
            return
        self.dirty = False
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        center = (width // 2, height // 2)
        if self.frame is None:
            self.canvas.itemconfigure(self.image_item, image="")
            self.photo = None
            self.canvas.coords(self.status_item, *center)
            self.canvas.itemconfigure(self.status_item, state="normal", width=max(1, width - 24))
            return

        image = self.frame.rotate(-self.angle, expand=True) if self.angle else self.frame
        resample = Image.Resampling.NEAREST if self.is_depth else Image.Resampling.BILINEAR
        image = ImageOps.contain(image, (width, height), resample)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.coords(self.image_item, *center)
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.canvas.itemconfigure(self.status_item, state="hidden")


class Viewer:
    def __init__(self, root):
        self.root = root
        self.stop_scan = Event()
        self.scan_results = deque(maxlen=1)
        self.scan_thread = None
        self.closing = False

        root.title("Камера и ToF")
        root.geometry("1060x780")
        root.minsize(780, 540)
        root.configure(background=BACKGROUND)
        root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Title.TLabel", background=PANEL, foreground=TEXT, font=("TkDefaultFont", 11, "bold"))
        style.configure("Info.TLabel", background=BACKGROUND, foreground=MUTED)
        style.configure("TButton", padding=(9, 4))

        root.columnconfigure(0, weight=1, uniform="previews")
        root.columnconfigure(1, weight=1, uniform="previews")
        root.rowconfigure(1, weight=1, uniform="previews")
        root.rowconfigure(2, weight=1, uniform="previews")

        toolbar = ttk.Frame(root, padding=(12, 10))
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.refresh = ttk.Button(toolbar, text="Обновить устройства", command=self.scan)
        self.refresh.pack(side="left")
        self.info = ttk.Label(toolbar, style="Info.TLabel")
        self.info.pack(side="left", padx=12)

        self.camera = Preview(root, "Камера", camera_frames, self.select)
        self.camera.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 8))
        self.tofs = [
            Preview(root, f"ToF {index + 1}", tof_frames, self.select, is_depth=True)
            for index in range(2)
        ]
        for index, pane in enumerate(self.tofs):
            padx = (12, 4) if index == 0 else (4, 12)
            pane.grid(row=2, column=index, sticky="nsew", padx=padx, pady=(0, 12))
        self.panes = [self.camera, *self.tofs]

        self.scan()
        self.tick()

    def select(self, pane):
        value = pane.selector.get()
        if value == OFF:
            pane.select(None)
            return
        if pane.is_depth:
            if any(other is not pane and other.device == value for other in self.tofs):
                pane.selector.set(OFF if pane.device is None else pane.device)
                self.info.configure(text=f"Порт {value} уже выбран для другого ToF")
                return
            pane.select(value)
        else:
            pane.select(int(value))

    def scan(self):
        if self.scan_thread is not None:
            return
        self.refresh.state(["disabled"])
        self.camera.selector.state(["disabled"])
        self.info.configure(text="Поиск устройств…")
        stream = self.camera.stream
        active_index = stream.device if stream is not None and stream.thread.is_alive() else None
        self.scan_thread = Thread(target=self.find_devices, args=(active_index,), daemon=True)
        self.scan_thread.start()

    def find_devices(self, active_index):
        try:
            ports = serial_ports()
            cameras = camera_indices(active_index, self.stop_scan)
        except (OSError, cv2.error) as error:
            self.scan_results.append(str(error))
        else:
            self.scan_results.append((cameras, ports))

    def finish_scan(self):
        result = self.scan_results.popleft()
        if isinstance(result, str):
            self.info.configure(text=f"Ошибка поиска. {result}")
        else:
            cameras, ports = result
            self.camera.set_devices(cameras)
            for pane in self.tofs:
                pane.set_devices(ports)
            self.info.configure(text=f"Камеры: {len(cameras)}    Последовательные порты: {len(ports)}")
        self.scan_thread = None
        self.camera.selector.state(["!disabled"])
        self.refresh.state(["!disabled"])

    def tick(self):
        if self.closing:
            return
        if self.scan_results:
            self.finish_scan()
        for pane in self.panes:
            others = [other for other in self.tofs if other is not pane] if pane.is_depth else []
            pane.update_stream(others, scanning=self.scan_thread is not None)
            pane.draw()
        self.root.after(30, self.tick)

    def close(self):
        self.closing = True
        self.stop_scan.set()
        self.root.withdraw()
        threads = [pane.stream.thread for pane in self.panes if pane.stream is not None]
        if self.scan_thread is not None:
            threads.append(self.scan_thread)
        for pane in self.panes:
            if pane.stream is not None:
                pane.stream.stop.set()
        deadline = monotonic() + 1

        def finish():
            if monotonic() < deadline and any(thread.is_alive() for thread in threads):
                self.root.after(30, finish)
            else:
                self.root.destroy()

        finish()


if __name__ == "__main__":
    window = tk.Tk()
    Viewer(window)
    window.mainloop()
