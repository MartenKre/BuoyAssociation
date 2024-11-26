import numpy as np
import time

class LivePlots():
    def __init__(self, parent, event) -> None:
        import pyqtgraph as pg
        self.fig = pg.GraphicsLayoutWidget()
        title="Matching Confidence per Buoy"
        self.refresh = event
        self.parent = parent
        self.plotLive()

    def plotLive(self):
        while True:
            if self.refresh.is_set():
                self.refresh.clear()
                self.redrawPlots()
            time.sleep(0.1)

    def redrawPlots(self):
        import pyqtgraph as pg
        self.fig.clear()

        data = self.parent.matching_confidence_plotting

        size = len([k for k in data])
        if size == 0:
            return 

        for i,k in enumerate(data):
            ax = self.fig.addPlot(row=0, col=i, title='')
            for id in data[k]:
                y = data[k][id]['data']
                color = tuple(x*255 for x in list(data[k][id]['color']))
                x = np.arange(start=0, stop=len(data[k][id]['data']))
                ax.plot(x, y, pen=pg.mkPen(color=color))

        self.fig.show()