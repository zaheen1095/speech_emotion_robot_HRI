# pepper_client.py
# import json, queue, time
# from util.connection import Connection

# class PepperClient(object):
#     def __init__(self, ip="192.168.0.3", port=7878, timeout=10.0):
#         self.ip, self.port, self.timeout = ip, port, timeout
#         self.conn = None

#     def connect(self):
#         if self.conn is None:
#             self.conn = Connection(ip=self.ip, port=self.port, type='client')
#         return True

#     def close(self):
#         try:
#             if self.conn:
#                 self.conn.sock.close()
#         finally:
#             self.conn = None

#     def record(self, seconds=3, mode="auto"):
#         # mode: "seconds" | "startstop" | "auto"
#         if mode in ("seconds", "auto"):
#             try:
#                 self.conn.send(json.dumps({"command":"record","content":int(seconds)}).encode("utf-8"))
#                 return self._recv_bytes()
#             except Exception:
#                 if mode == "seconds":
#                     raise  # don’t silently switch if explicitly asked for seconds

#         # fallback: start/stop (we’ll stop after `seconds`)
#         self.conn.send(json.dumps({"command":"record","content":"start"}).encode("utf-8"))
#         # server will start recording now
#         time.sleep(max(1, int(round(seconds))))
#         self.conn.send(json.dumps({"command":"record","content":"stop"}).encode("utf-8"))
#         return self._recv_bytes()
    

#     def _recv_bytes(self):
#         _ = self.conn.receive()
#         try:
#             return self.conn.queue.get(timeout=self.timeout) 
#             # data =  self.conn.queue.get(timeout=self.timeout) 
#             # return bytes(data)
#         except queue.Empty:
#             raise RuntimeError("No audio returned from Pepper")

#     def tts(self, text):
#         self.conn.send(json.dumps({"command":"tts","content":text}).encode("utf-8"))
#         return True


import json, queue, time, threading
from util.connection import Connection

class PepperClient(object):
    def __init__(self, ip="192.168.0.3", port=7878, timeout=10.0):
        self.ip, self.port, self.timeout = ip, port, timeout
        self.conn = None
        self._rec_lock = threading.Lock()

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

    # NEW: drain any stale payload left from earlier operations
    def _drain_queue(self):
        try:
            while True:
                self.conn.queue.get_nowait()
        except queue.Empty:
            pass

    def _recv_bytes(self, timeout=None):
        ok = self.conn.receive()
        if not ok:
            raise RuntimeError("Pepper receive failed")
        payload = self.conn.queue.get(timeout=(timeout or self.timeout))
        if isinstance(payload, (bytes, bytearray)) and len(payload) >= 44:
            return payload
        try:
            payload = self.conn.queue.get(timeout=1.0)
            return payload
        except queue.Empty:
            raise RuntimeError("No audio returned from Pepper")

    def record(self, seconds=3, mode="seconds"):
        with self._rec_lock:
            self._drain_queue()  # clear any late/stale audio before a new request
            secs = int(max(1, round(seconds)))
            self.conn.send(json.dumps({"command": "record", "content": secs}).encode("utf-8"))
            # give the server enough time: seconds + small margin
            return self._recv_bytes(timeout=self.timeout + secs + 2)

    def tts(self, text):
        self.conn.send(json.dumps({"command":"tts","content":text}).encode("utf-8"))
        return True


