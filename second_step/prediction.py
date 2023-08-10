from torch.utils.data import DataLoader,TensorDataset
from data_process import kmer_encoding, get_kmer_dict
from models.model import model
from torch import nn as nn
from sklearn.model_selection import KFold
from early_stopping import EarlyStopping as Early_Stopping
import utils
import torch
import time
import os, argparse
import numpy as np

class Encoder():
    def __init__(self, batch_size=128, lr=0.0001, num_epochs=500, seq_len=200, model_name='transformer_encoder', hidden=512, k_mer_size=5, \
                num_gpus=1, dropout=0.2, k_fold=20, strong_enhancers_dir='../train_datas/predication/strong_enhancer.txt', weak_enhancers_dir='../train_datas/predication/weak_enhancer.txt'):
        self.model_name = model_name
        self.hidden = hidden
        self.batch_size = int(batch_size)
        self.lr = lr
        self.k_mer_size = k_mer_size
        self.epoch = num_epochs
        self.dropout = dropout
        self.k_fold = k_fold
        self.seq_len = seq_len
        self.weight_decay = 0.01
        self.devices = [utils.try_gpu(i) for i in range(int(num_gpus))]
        self.checkpoints_path = '../saved_models/' + model_name + "/"
        self.output_dir = '../output/' + model_name + "/"
        self.train_datasets, self.valid_datasets = self.get_k_fold_datasets(strong_enhancers_dir, weak_enhancers_dir)
        if not os.path.exists(self.checkpoints_path): os.makedirs(self.checkpoints_path)
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        self.init_model(self.weight_decay)

    def init_model(self, weight_decay):
        print("weight_decay: ", str(weight_decay))
        self.net = model(utils.get_vocab_num(self.k_mer_size), self.seq_len-self.k_mer_size+1, self.hidden, self.dropout)
        # self.optimizer = torch.optim.Adam([{"params":self.net[0].weight,'weight_decay': weight_decay}, {"params":self.net[0].bias}], lr=self.lr)
        self.optimizer = torch.optim.Adam(
            [{'params': (p for name, p in self.net.named_parameters() if 'bias' not in name and 'addnorm' not in name), 'weight_decay': weight_decay},
             {'params': (p for name, p in self.net.named_parameters() if 'bias' in name or 'addnorm' in name)}], lr=self.lr)
        self.net = nn.DataParallel(self.net, device_ids=self.devices).to(self.devices[0]) # DP数据并行

    def get_k_fold_datasets(self, strongs, weaks):
        print("start load dataset...")
        encoded_seq = []
        label = []

        kmer_dict = get_kmer_dict(self.k_mer_size)
        with open(strongs) as file:
            for line in file.readlines():
                if "N" not in line:
                    encoded_seq.append(kmer_encoding(line.upper(), self.k_mer_size, self.seq_len, kmer_dict))
                    label.append(0)

        with open(weaks) as file:
            for line in file.readlines():
                if "N" not in line:
                    encoded_seq.append(kmer_encoding(line.upper(), self.k_mer_size, self.seq_len, kmer_dict))
                    label.append(1)

        self.sample_size = len(encoded_seq)
        print("sample szie: ", self.sample_size)
        encoded_seq = np.array(encoded_seq)

        np.random.seed(1)
        shuffle_indices = np.random.permutation(encoded_seq.shape[0])
        shuffled_data = encoded_seq[shuffle_indices]
        shuffled_labels = np.array(label)[shuffle_indices]

        x = torch.tensor(shuffled_data, dtype=torch.long)
        y = torch.tensor(shuffled_labels, dtype=torch.float32)

        kf = KFold(n_splits=self.k_fold)
        train_datasets, valid_datasets = [],[]
        for train_ind, valid_ind in kf.split(x):
            # 划分训练集和验证集
            train_x, train_y = x[train_ind], y[train_ind]
            valid_x, valid_y = x[valid_ind], y[valid_ind]

            # 构造数据集和数据加载器
            train_dataset = TensorDataset(train_x, train_y)
            valid_dataset = TensorDataset(valid_x, valid_y)
            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)

        return train_datasets, valid_datasets

    def save_models(self, i, train_losses, train_accs, valid_losses, valid_accs, k, counter, best_score):
        if not os.path.exists(self.checkpoints_path + "model" + str(k) + "_for_Kfold/"): os.makedirs(
            self.checkpoints_path + "model" + str(k) + "_for_Kfold/")
        torch.save({
            'epoch': i + 1,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_accs': train_accs,
            'valid_accs': valid_accs,
            'k-fold': k,
            'counter': counter,
            'best_score': best_score,
        }, (self.checkpoints_path + "model" + str(k) + "_for_Kfold/" + self.model_name + "_epoch" + str(i + 1) + ".pth"))

    def load_checkpoints(self, k, epoch):
        checkpoint = torch.load(
            self.checkpoints_path + "model" + k + "_for_Kfold/" + self.model_name + "_epoch" + epoch + ".pth")
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        checkpoint_epoch = checkpoint['epoch']
        print("k:", k, "checkpoint_epoch: ", checkpoint_epoch)
        return checkpoint_epoch, checkpoint

    def train(self):
        checkpoint_epoch, K, is_load_checkpoint, checkpoint = 0, 0, False, {}
        if os.path.exists(self.checkpoints_path + "checkpoint_epoch_num.txt"):
            with open(self.checkpoints_path + 'checkpoint_epoch_num.txt', 'r') as file:
                line = file.read().replace('\n', '').split(",")
                epoch = line[0]
                k = line[1]

                checkpoint_epoch, checkpoint = self.load_checkpoints(k, epoch)
                K = checkpoint["k-fold"]
                if checkpoint_epoch == self.epoch:
                    checkpoint_epoch, K = 0, int(k) + 1
                else:
                    is_load_checkpoint = True

        loss = nn.BCELoss()
        best_labels_for_ROC, best_probas_for_ROC = [], []
        for k in range(K, self.k_fold):
            print("---------------------The " + str(k + 1) + "th model---------------------")
            train_loader = torch.utils.data.DataLoader(self.train_datasets[k], batch_size=self.batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(self.valid_datasets[k], batch_size=self.batch_size, shuffle=False)
            train_total_batch_num, valid_total_batch_num = len(train_loader), len(valid_loader)
            train_losses, valid_losses, train_accs, valid_accs = [], [], [], []
            counter, best_score = 0, None
            best_labels, best_probas = None, None
            if is_load_checkpoint:
                counter, best_score = checkpoint["counter"], checkpoint["best_score"]
                train_losses, valid_losses, train_accs, valid_accs = checkpoint["train_losses"], checkpoint["valid_losses"], checkpoint["train_accs"], checkpoint["valid_accs"]
                is_load_checkpoint = False
            elif k != 0:
                self.init_model(self.weight_decay)
            self.early_stopping = Early_Stopping(self.checkpoints_path, self.output_dir, patience=7, counter=counter, best_score=best_score)
            for i in range(checkpoint_epoch, self.epoch):
                start = time.time()
                train_tps, train_fps, train_tns, train_fns = [], [], [], []
                valid_tps, valid_fps, valid_tns, valid_fns = [], [], [], []
                valid_labels, valid_probas = [], []
                train_loss, valid_loss = 0,0
                self.net.train()
                for batch_num, (x, y) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    x = x.to(self.devices[0])
                    y = y.to(self.devices[0])

                    net_output = self.net(x)
                    l = loss(net_output.reshape(-1), y)
                    # l = loss(net_output.reshape(-1), y) + self.lamda * regularization_loss.item()
                    l.mean().backward()
                    self.optimizer.step()

                    with torch.no_grad():
                        thresholded = torch.where(net_output < 0.5, torch.zeros_like(net_output),torch.ones_like(net_output)).reshape(-1)
                        tp, tn, fp, fn = utils.calculate_metrics1(y.detach().cpu().numpy(), thresholded.detach().cpu().numpy())
                        train_tps.append(tp), train_fps.append(fp), train_tns.append(tn), train_fns.append(fn)
                        train_loss += l.item()

                self.net.eval()
                with torch.no_grad():
                    for batch_num, (x, y) in enumerate(valid_loader):
                        x = x.to(self.devices[0])
                        y = y.to(self.devices[0])

                        net_output = self.net(x)
                        l = loss(net_output.reshape(-1), y)
                        valid_loss += l.item()

                        valid_labels.append(y.clone().detach().cpu())
                        valid_probas.append(net_output.clone().detach().cpu().reshape(-1))

                        thresholded = torch.where(net_output < 0.5, torch.zeros_like(net_output),
                                                  torch.ones_like(net_output)).reshape(-1)
                        # 计算TP FP TN FN
                        tp, tn, fp, fn = utils.calculate_metrics1(y.detach().cpu().numpy(),
                                                                  thresholded.detach().cpu().numpy())
                        valid_tps.append(tp), valid_fps.append(fp), valid_tns.append(tn), valid_fns.append(fn)

                end = time.time()
                valid_sn, valid_sp, valid_mcc, valid_acc = utils.calculate_metrics2(sum(valid_tps), sum(valid_tns), sum(valid_fps), sum(valid_fns))
                train_sn, train_sp, train_mcc, train_acc = utils.calculate_metrics2(sum(train_tps), sum(train_tns), sum(train_fps), sum(train_fns))
                print(
                    "k-fold: {} round, epoch: {}, running time: {:.2f}s, "
                    "Training set loss: {:.4f}, sn:{:.4f}, sp:{:.4f}, mcc:{:.4f}, acc: {:.4f}. "
                    "Validation set loss: {:.4f}, sn:{:.4f}, sp:{:.4f}, mcc:{:.4f}, acc:{:.4f}".format(
                        k + 1, i, (end - start), train_loss / train_total_batch_num, train_sn, train_sp, train_mcc, train_acc,
                        valid_loss / valid_total_batch_num, valid_sn, valid_sp, valid_mcc, valid_acc))
                train_losses.append(train_loss/train_total_batch_num)
                valid_losses.append(valid_loss/valid_total_batch_num)
                train_accs.append(train_acc)
                valid_accs.append(valid_acc)

                if (i != 0 and (i + 1) % 50 == 0) or (i != 0 and i == (self.epoch - 1)):
                    with open(self.checkpoints_path + 'checkpoint_epoch_num.txt', 'w') as file:
                        file.write(str(i + 1) + "," + str(k))
                    self.save_models(i, train_losses, train_accs, valid_losses, valid_accs, k, self.early_stopping.counter, self.early_stopping.best_score)

                is_save_moedl = self.early_stopping(valid_loss / valid_total_batch_num, self.net, valid_labels, valid_probas, k)
                if is_save_moedl:
                    best_labels, best_probas = utils.stacked_tensor_to_np(valid_labels), utils.stacked_tensor_to_np(
                        valid_probas)
                if self.early_stopping.early_stop:  # 达到早停止条件时，early_stop会被置为True
                    break  # 跳出迭代，结束训练

            utils.draw(train_losses, str(k + 1) + "fold_train_loss", self.output_dir,
                       title="The train loss of " + str(k + 1) + " round model")
            utils.draw(train_accs, str(k + 1) + "fold_train_accuracy", self.output_dir,
                       title="Accuracy of the " + str(k + 1) + " round training set")
            utils.draw(valid_losses, str(k + 1) + "fold_valid_loss", self.output_dir,
                       title="The valid loss of " + str(k + 1) + " round model")
            utils.draw(valid_accs, str(k + 1) + "fold_valid_accuracy", self.output_dir,
                       title="Accuracy of the " + str(k + 1) + " round validation set")
            best_labels_for_ROC.append(best_labels), best_probas_for_ROC.append(best_probas)
        utils.draw_ROC_for_every_model(best_labels_for_ROC, best_probas_for_ROC, self.output_dir, "ROC curve")

def main(argv=None):
    parser = argparse.ArgumentParser(description='prediction of enhancers.')
    parser.add_argument("--model_name", default="prediction", help="model name for checkpoint and output")
    parser.add_argument("--num_gpus", default="2", help="Number of GPUs used during training")
    args = parser.parse_args()
    print("init models...")
    encoder = Encoder(model_name=args.model_name, num_gpus=args.num_gpus)
    print("init complete!")
    print("start training...")
    encoder.train()
    print("end!")
if __name__ == "__main__":
    main()