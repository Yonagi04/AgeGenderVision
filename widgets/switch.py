from PyQt5.QtWidgets import QAbstractButton
from PyQt5.QtCore import Qt, QSize, QPropertyAnimation, pyqtProperty
from PyQt5.QtGui import QPainter, QColor, QBrush

class Switch(QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self.setCheckable(True)
        self._thumb_pos = 2
        self.animation = QPropertyAnimation(self, b"thumb_pos", self)
        self.animation.setDuration(200)
        self.setFixedSize(50, 25)

    def sizeHint(self):
        return QSize(50, 25)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self.isChecked():
            p.setBrush(QBrush(QColor("#00bfa5")))
        else:
            p.setBrush(QBrush(QColor("#ccc")))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 12, 12)

        # 画 thumb
        p.setBrush(QBrush(Qt.white))
        p.drawEllipse(self._thumb_pos, 2, 21, 21)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setChecked(not self.isChecked())
            self.animation.setStartValue(self._thumb_pos)
            self.animation.setEndValue(26 if self.isChecked() else 2)
            self.animation.start()
            self.clicked.emit(self.isChecked())

    @pyqtProperty(int)
    def thumb_pos(self):
        return self._thumb_pos

    @thumb_pos.setter
    def thumb_pos(self, pos):
        self._thumb_pos = pos
        self.update()
