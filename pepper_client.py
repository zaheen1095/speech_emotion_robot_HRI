# pepper_client.py
import json, queue
from utils import Connection

class PepperClient(object):
    def __init__(self, ip="192.168.0.3", port=7878, timeout=10.0):
        self.ip, self.port, self.timeout = ip, port, timeout
        self.conn = None

    def connect(self):
        if self.conn is None:
            self.conn = Connection(ip=self.ip, port=self.port, type='client')
        return True

    def close(self):
        try:
            if self.conn:
                self.conn.sock.close()
        finally:
            self.conn = None

    def record(self, seconds=5):
        self.conn.send(json.dumps({"command":"record","content":seconds}).encode("utf-8"))
        _ = self.conn.receive()
        try:
            return self.conn.queue.get(timeout=self.timeout)  # WAV bytes
        except queue.Empty:
            raise RuntimeError("No audio returned from Pepper")

    def tts(self, text):
        self.conn.send(json.dumps({"command":"tts","content":text}).encode("utf-8"))
        _ = self.conn.receive()
        return True
