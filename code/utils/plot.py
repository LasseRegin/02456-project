
#import numpy as np
import matplotlib.pyplot as plt

PYPLOT_COLORS = ['IndianRed', 'SteelBlue', 'LimeGreen']

class LossPlot:
    def __init__(self, show=False, live=True):
        plt.ion()
        self.show = show
        self.live = live

        # Setup plot
        if self.show:
            self.fig = plt.figure()
            self.axes = self.fig.add_subplot(1,1,1)
            self.axes.set_autoscaley_on(True)
            self.axes.set_xlabel('Epochs')
            self.axes.set_ylabel('Loss')

        # Setup loss
        self.epochs = []
        self.loss = []

    def update(self, epoch, loss):
        self.epochs.append(epoch)
        self.loss.append(loss)

        if self.show:
            self.axes.plot(self.epochs, self.loss, c='IndianRed')
            self.axes.relim()
            self.axes.autoscale_view()

            if self.live:
                self.fig.canvas.draw()
                plt.pause(0.01)


if __name__ == '__main__':

    from random import random
    import numpy as np
    import time

    plot = LossPlot(live=True)

    for epoch in range(0, 100):
        plot.update(epoch=epoch, loss=1 - random() * epoch / 10.0)

        time.sleep(0.2)
