import sys, os
from PyQt5.QtCore import Qt, QPoint,QEvent
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QToolButton, QSizeGrip, QStyle
)

class ChatBubble(QLabel):
    def __init__(self, text, is_user=False):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMaximumWidth(360)
        self.setFont(QFont("Segoe UI", 10))
        if is_user:
            self.setStyleSheet("background:#DCF8C6; padding:10px; border-radius:12px;")
        else:
            self.setStyleSheet("background:#EAEAEA; padding:10px; border-radius:12px;")

class TitleBar(QFrame):
    """Custom title bar: big icon + title, mic/status, min/max/close, draggable."""
    def __init__(self, window, icon_path: str, title_text: str):
        super().__init__()
        self.window = window
        self._drag_pos = None
        self._mouse_pressed = False

        self.setObjectName("TitleBar")
        ICON = 110
        self.setFixedHeight(110)
        # self.setFixedHeight(70)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 6, 9, 6)
        layout.setSpacing(10)

                # Big icon (you control the size here)
        self.icon_lbl = QLabel()
        self.icon_lbl.setFixedSize(ICON, ICON)
        if os.path.exists(icon_path):
            self.icon_lbl.setPixmap(QPixmap(icon_path).scaled(
                ICON, ICON, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        # Title text
        self.title_lbl = QLabel(title_text)
        self.title_lbl.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        self.title_lbl.setObjectName("TitleText")

        # Optional: mic + status (inside title bar)
        self.mic_lbl = QLabel("🎙️")
        self.mic_lbl.setFont(QFont("Segoe UI Emoji", 20))
        self.status_lbl = QLabel("😐")
        self.status_lbl.setFont(QFont("Segoe UI Emoji", 30))

        # ---------------------------
        # 3-zone alignment
        # Left:  logo + title
        # Center: mic + emoji (centered)
        # Right: window buttons
        # ---------------------------

        # Left group
        left = QHBoxLayout()
        left.setSpacing(8)
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(self.icon_lbl)
        left.addWidget(self.title_lbl)

        left_widget = QWidget()
        left_widget.setLayout(left)

        # Center group
        center = QHBoxLayout()
        center.setSpacing(10)
        center.setContentsMargins(0, 0, 0, 0)
        center.addWidget(self.mic_lbl)
        center.addWidget(self.status_lbl)

        center_widget = QWidget()
        center_widget.setLayout(center)

        # Add groups to main layout
        layout.addWidget(left_widget)                        # left
        layout.addStretch(1)                                 # push center to middle
        layout.addWidget(center_widget, 0, Qt.AlignCenter)    # center
        layout.addStretch(1)                                 # push buttons to right

        # Drag support (clicking logo/title/mic/emoji should drag)
        self.icon_lbl.installEventFilter(self)
        self.title_lbl.installEventFilter(self)
        self.mic_lbl.installEventFilter(self)
        self.status_lbl.installEventFilter(self)

        layout.addSpacing(6)



        # Window buttons (use standard icons)
        self.btn_min = QToolButton()
        self.btn_max = QToolButton()
        self.btn_close = QToolButton()

        st = self.style()
        self.btn_min.setIcon(st.standardIcon(QStyle.SP_TitleBarMinButton))
        self.btn_max.setIcon(st.standardIcon(QStyle.SP_TitleBarMaxButton))
        self.btn_close.setIcon(st.standardIcon(QStyle.SP_TitleBarCloseButton))

        self.btn_min.clicked.connect(self.window.showMinimized)
        self.btn_max.clicked.connect(self._toggle_max_restore)
        self.btn_close.clicked.connect(self.window.close)

        for b in (self.btn_min, self.btn_max, self.btn_close):
            b.setFixedSize(34, 28)
            b.setAutoRaise(True)

        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_close)

        self.setStyleSheet("""
        QFrame#TitleBar {
            background: #f4f6f8;
            border-bottom: 1px solid #e3e7ee;
        }
        QLabel#TitleText { color: #1f2d3d; }
        QToolButton { border-radius: 6px; }
        QToolButton:hover { background: #e9eef6; }
        QToolButton:pressed { background: #dbe6f6; }
        """)
        self._sync_max_button()

    def set_state(self, state: str):
        # Optional helper for your pipeline
        if state == "listening":
            self.mic_lbl.setText("🔴")
            self.status_lbl.setText("🙂")
        elif state == "thinking":
            self.mic_lbl.setText("🎙️")
            self.status_lbl.setText("🤔")
        else:
            self.mic_lbl.setText("🎙️")
            self.status_lbl.setText("😐")

 
    def _toggle_max_restore(self):
        if self.window.windowState() & Qt.WindowMaximized:
            self.window.showNormal()
        else:
            self.window.showMaximized()
        self._sync_max_button()

    def _sync_max_button(self):
        if self.window.windowState() & Qt.WindowMaximized:
            self.btn_max.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))
        else:
            self.btn_max.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))

    # Drag window by title bar
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._mouse_pressed = True
            self._drag_pos = e.globalPos() - self.window.frameGeometry().topLeft()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._mouse_pressed and self._drag_pos is not None and not self.window.isMaximized():
            self.window.move(e.globalPos() - self._drag_pos)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._mouse_pressed = False
        self._drag_pos = None
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._toggle_max_restore()
        super().mouseDoubleClickEvent(e)

    def eventFilter(self, obj, event):
        if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            self._mouse_pressed = True
            self._drag_pos = event.globalPos() - self.window.frameGeometry().topLeft()
            return True

        if event.type() == event.MouseMove and self._mouse_pressed and self._drag_pos is not None:
            if not self.window.isMaximized():
                self.window.move(event.globalPos() - self._drag_pos)
            return True

        if event.type() == event.MouseButtonRelease:
            self._mouse_pressed = False
            self._drag_pos = None
            return True

        return super().eventFilter(obj, event)


class FramelessDemo(QWidget):
    def __init__(self):
        super().__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "robot_logo.png")  # big icon source

        # Frameless window (no native title bar)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setMinimumSize(860, 820)

        # Outer layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)

        # Container with border/rounded corners
        container = QFrame()
        container.setObjectName("Container")
        container.setStyleSheet("""
        QFrame#Container {
            background: white;
            border: 1px solid #dfe6ee;
            border-radius: 15px;
        }
        """)
        outer.addWidget(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title bar (your controlled height/icon size)
        self.titlebar = TitleBar(self, icon_path, "Speech Emotion Application")
        layout.addWidget(self.titlebar)

        # Content area
        content = QFrame()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 8, 12, 8)
        # content_layout.setSpacing(5)

        # Chat area
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)

        chat_container = QWidget()
        chat_container.setLayout(self.chat_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(chat_container)
        scroll.setStyleSheet("""
            QScrollArea { border: 1px solid #e4e7ec; border-radius: 10px; background: #ffffff; }
        """)
        self.scroll = scroll

        # Button
        self.button = QPushButton("🎤  Record & Detect (UI demo)")
        self.button.setMinimumHeight(60)
        self.button.setStyleSheet("""
            QPushButton { background:#0078D7; color:white; font-size:16px; border-radius:10px; }
            QPushButton:pressed { background:#0b5aa0; }
        """)
        self.button.clicked.connect(self.demo_action)

        # Add widgets
        content_layout.addWidget(scroll, 1)
        content_layout.addWidget(self.button)

        layout.addWidget(content)

        # starter message
        self.add_msg("Hi — this is frameless UI. Now your top-left icon can be BIG.", False)

    def add_msg(self, text, is_user=False):
        bubble = ChatBubble(text, is_user=is_user)
        row = QHBoxLayout()
        if is_user:
            row.addStretch(1)
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch(1)
        self.chat_layout.addLayout(row)
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())
    
    def changeEvent(self, e):
        if e.type() == QEvent.WindowStateChange:
            self.titlebar._sync_max_button()
        super().changeEvent(e)

    def demo_action(self):
        self.titlebar.set_state("listening")
        self.add_msg("User: Hello", True)
        self.titlebar.set_state("thinking")
        self.add_msg("Assistant: I’m here. How are you feeling today?", False)
        self.titlebar.set_state("idle")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FramelessDemo()
    w.show()
    sys.exit(app.exec_())
