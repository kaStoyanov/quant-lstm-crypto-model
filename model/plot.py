import matplotlib.pyplot as plt


class PlotData:
    final_oof = []
    final_y_trues = []

    def __init__(self,oof,train_losses,validation_losses,y_trues):
        self.final_oof = self.unstack(oof)
        self.final_y_trues = self.unstack(y_trues)
        # self.plot_losses(train_losses, validation_losses)
        # self.plot_oof(y_trues, oof)
    def return_values(self):
        return self.final_oof,self.final_y_trues

    def plot_losses(self, train_losses, validation_losses):
        plt.plot(train_losses, label="Training loss")
        plt.plot(validation_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def plot_oof(self, ground_truth, oof):
        plt.plot(ground_truth, label="Ground truth")
        plt.plot(oof, label="OOF")
        plt.legend()
        plt.title("Out of Fold vs. Ground truth")
        plt.show()
        plt.close()

    def unstack(self, oof):
        preds = []
        for batch_prediction in oof:
            preds.append(batch_prediction.tolist())
        preds = [item for sublist in preds for item in sublist]
        preds = [item for sublist in preds for item in sublist]
        return preds
