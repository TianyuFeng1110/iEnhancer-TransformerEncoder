import numpy as np
import torch
import utils

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,  checkpoints_path, roc_output_dir, patience=7, verbose=False, delta=0, counter=0, best_score=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoints_path = checkpoints_path
        self.roc_output_dir = roc_output_dir
        self.patience = patience
        self.verbose = verbose
        self.counter = counter
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, valid_labels, valid_probas, k):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, valid_labels, valid_probas, k)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print("Early stopping, min loss is: {:.4f}".format(-self.best_score))
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, valid_labels, valid_probas, k)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model, valid_labels, valid_probas, k):
        self.val_loss_min = val_loss

        torch.save({
            'model': model.state_dict()
        }, (self.checkpoints_path +  "best_" + str(k) + "th_model.pth"))

        utils.draw_ROC(utils.stacked_tensor_to_np(valid_labels), utils.stacked_tensor_to_np(valid_probas),
                       self.roc_output_dir, "ROC curve for model" + str(k))