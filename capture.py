"""Capture sources for OpenCV cameras and Sipeed MaixSense A010."""

import sys
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import serial
from serial.tools import list_ports


A010_HEADER = b"\x00\xff"
A010_BAUD = 921600
A010_METADATA_SIZE = 16
A010_SIDES = (25, 50, 100)
A010_PACKET_LENGTHS = {A010_METADATA_SIZE + side * side for side in A010_SIDES}


class StreamError(Exception):
    pass


class A010Decoder:
    """Decode an arbitrary sequence of chunks read from one serial port."""

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, chunk):
        self.buffer.extend(chunk)
        while True:
            start = self.buffer.find(A010_HEADER)
            if start < 0:
                # A lone 0x00 may be the start of a header split across reads.
                self.buffer[:] = self.buffer[-1:] if self.buffer[-1:] == b"\x00" else b""
                return
            del self.buffer[:start]

            if len(self.buffer) < 4:
                return
            length = int.from_bytes(self.buffer[2:4], "little")
            if length not in A010_PACKET_LENGTHS:
                del self.buffer[0]
                continue

            pixel_offset = 4 + A010_METADATA_SIZE
            if len(self.buffer) < pixel_offset:
                return
            rows, columns = self.buffer[14:16]
            if (
                rows not in A010_SIDES
                or rows != columns
                or rows * columns != length - A010_METADATA_SIZE
            ):
                del self.buffer[0]
                continue

            # The vendor's host example counts the checksum and tail after length.
            packet_size = 4 + length + 2
            if len(self.buffer) < packet_size:
                return
            packet = bytes(self.buffer[:packet_size])
            if packet[-1] not in (0xCC, 0xDD) or packet[-2] != (sum(packet[:-2]) & 0xFF):
                del self.buffer[0]
                continue

            del self.buffer[:packet_size]
            pixels = np.frombuffer(packet, dtype=np.uint8, count=rows * columns, offset=pixel_offset)
            yield pixels.reshape(rows, columns)


def serial_ports():
    return [port.device for port in sorted(list_ports.comports())]


def open_camera(index):
    if sys.platform.startswith("linux"):
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if sys.platform == "darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def camera_indices(active_index, stop):
    if sys.platform.startswith("linux"):
        candidates = (
            path.name.removeprefix("video") for path in Path("/dev").glob("video*")
        )
        indices = sorted(int(index) for index in candidates if index.isdecimal())
    else:
        indices = range(10)

    available = set() if active_index is None else {active_index}
    for index in indices:
        if stop.is_set():
            break
        if index == active_index:
            continue
        camera = open_camera(index)
        try:
            if camera.isOpened():
                available.add(index)
        finally:
            camera.release()
    return sorted(available)


def camera_frames(index, stop):
    camera = open_camera(index)
    try:
        if not camera.isOpened():
            raise StreamError(f"Не удалось открыть камеру {index}")
        failures = 0
        while not stop.is_set():
            ok, frame = camera.read()
            if ok:
                failures = 0
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                failures += 1
                if failures >= 20:
                    raise StreamError("Камера перестала передавать кадры")
                stop.wait(0.05)
    finally:
        camera.release()


def tof_frames(port, stop):
    decoder = A010Decoder()
    with serial.Serial(port, A010_BAUD, timeout=0.1, write_timeout=0.5) as connection:
        last_frame = monotonic()
        received_frame = False
        enabled_output = False
        while not stop.is_set():
            chunk = connection.read(min(connection.in_waiting or 1, 16384))
            for depth in decoder.feed(chunk):
                last_frame = monotonic()
                received_frame = True
                # Keep the same color scale across both sensors and all frames.
                color = cv2.applyColorMap(255 - depth, cv2.COLORMAP_TURBO)
                yield cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            elapsed = monotonic() - last_frame
            if elapsed > 1.5 and not received_frame and not enabled_output:
                if stop.is_set():
                    break
                # The sensor may have USB output or its ISP disabled.
                connection.write(b"AT+ISP=1\r")
                if stop.wait(0.1):
                    break
                connection.write(b"AT+DISP=3\r")  # Keep the built-in LCD enabled.
                enabled_output = True
                last_frame = monotonic()
            elif elapsed > 4:
                raise StreamError(f"Нет кадров A010 на порту {port}")
