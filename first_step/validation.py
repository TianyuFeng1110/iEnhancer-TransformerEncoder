import torch.nn as nn
import torch
import utils
from data_process import kmer_encoding, get_kmer_dict
from models.model import model
from torch.utils.data import DataLoader,TensorDataset

def get_valid_datas(k_mer_size, seq_len):
    kmer_dict = get_kmer_dict(k_mer_size)
    with open("../train_datas/validation.txt", 'r') as file:
        list, enhancers = [], []
        for line in file.readlines():
            line = line.replace('\n', '').split(" ")
            if len(line[1]) != 200:
                if len(list) != 0: enhancers.append(list)
                list = []
            else:
                list.append(kmer_encoding(line[1].upper(), k_mer_size, seq_len, kmer_dict))
        non_enhancers, strong_enhancers, weak_enhancers = list, enhancers[0], enhancers[1]

    return non_enhancers, strong_enhancers, weak_enhancers

# init net
def init_model(path):
    net = model(utils.get_vocab_num(k_mer_size), seq_len - k_mer_size + 1, hidden, dropout)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    net.load_state_dict(torch.load(path)['model'])
    return net

# set param
k_mer_size, seq_len, hidden, dropout, gpu_num, k_fold_num, batch_size = 5, 200, 512, 0.2, 2, 20, 128
model_name = "classification"
model_path = "../saved_models/classification/"
output_path = "../output/" + model_name
devices = [utils.try_gpu(i) for i in range(int(gpu_num))]

# get dataset
non_enhancers, strong_enhancers, weak_enhancers = get_valid_datas(k_mer_size, seq_len)
datas = strong_enhancers + weak_enhancers + non_enhancers
X = torch.tensor(datas, dtype=torch.long)
Y = torch.cat((torch.zeros(len(strong_enhancers)+len(weak_enhancers)), torch.ones(len(non_enhancers))))
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, drop_last=True)

# validation
average_loss, average_sn, average_sp, average_mcc, average_acc = 0,0,0,0,0
labels_for_roc, probas_for_roc = [],[]
for k in range(k_fold_num):
    path = model_path + "best_"+ str(k) +"th_model.pth"
    valid_labels, valid_probas = [], []
    net = init_model(path)
    loss = nn.BCELoss()
    net.eval()
    with torch.no_grad():
        tps, fps, tns, fns = [], [], [], []  # 每一个元素是一个批量的指标，求和相当于每训练一个epoch，验证时所有验证集的指标
        validation_loss = 0
        for batch_num, (x, y) in enumerate(dataloader):
            x = x.to(devices[0])
            y = y.to(devices[0])

            net_output = net(x)
            l = loss(net_output.reshape(-1), y)
            validation_loss += l

            valid_labels.append(y.clone().detach().cpu())
            valid_probas.append(net_output.clone().detach().cpu().reshape(-1))
            thresholded = torch.where(net_output < 0.5, torch.zeros_like(net_output),
                                      torch.ones_like(net_output)).reshape(-1)
            tp, tn, fp, fn = utils.calculate_metrics1(y.detach().cpu().numpy(), thresholded.detach().cpu().numpy())
            tps.append(tp), fps.append(fp), tns.append(tn), fns.append(fn)

    sn, sp, mcc, acc = utils.calculate_metrics2(sum(tps), sum(tns), sum(fps), sum(fns))
    average_loss += (validation_loss / len(dataloader))
    average_sn += sn
    average_sp += sp
    average_mcc += mcc
    average_acc += acc
    labels_for_roc.append(utils.stacked_tensor_to_np(valid_labels)), probas_for_roc.append(utils.stacked_tensor_to_np(valid_probas))
    print("Validation set,loss: {:.4f} sn:{:.4f}, sp:{:.4f}, mcc:{:.4f}, acc:{:.4f}".format(validation_loss / len(dataloader), sn, sp, mcc, acc))
    # utils.draw_ROC(utils.stacked_tensor_to_np(valid_labels), utils.stacked_tensor_to_np(valid_probas), output_path, "ROC curve of validation for model" + str(k))

utils.draw_ROC_for_every_model(labels_for_roc, probas_for_roc, output_path, "independent dataset ROC curve")
print("average loss: {:.4f} sn:{:.4f}, sp:{:.4f}, mcc:{:.4f}, acc:{:.4f}".format(average_loss/k_fold_num, average_sn/k_fold_num, average_sp/k_fold_num, average_mcc/k_fold_num, average_acc/k_fold_num))
