import numpy as np
import datetime
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

app = QApplication([])

data_arr = []

for a in range(0, 100):
    y = np.random.rand(3000)
    x = np.arange(stop = 3000)
    data_arr.append([x, y])

fig = pg.GraphicsLayoutWidget()
ax = fig.addPlot(row=0, col=0)
color = (200, 0, 0, 255)
curves = []
for a in range(0, len(data_arr)):
    curve = ax.plot(pen=pg.mkPen(color=color))
    curves.append(curve)
fig.show()

i = 1
prev_time = datetime.datetime.now()

def update_plot():
    global i 
    global prev_time
    if i >= len(x):
        return   
     
    for a in range(0, len(data_arr)):
        px = data_arr[a][0][:i+1]
        py = data_arr[a][1][:i+1]
        curves[a].setData(px,py)

    i += 1

    if i != 0 and i % 50 == 0:
        print(datetime.datetime.now() - prev_time)
        prev_time = datetime.datetime.now()

    

timer = QTimer()
timer.timeout.connect(update_plot)
timer.start(50)  # 50ms update interval (20 FPS)

app.exec_()