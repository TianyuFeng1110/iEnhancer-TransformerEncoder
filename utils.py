import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
import multiprocessing as mp
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
import math
import os
import random

def get_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def draw1(y1, y2, path,title=None, legend=None):
    X = list(range(1, (len(y1)+1)))

    plt.plot(X, y1, X, y2)
    plt.legend(legend)
    plt.title(title)
    path = path + "d_loss_components"
    plt.savefig(path)
    plt.show()  # 展示
    plt.close()

def draw(y, legend, path, title=None):
    X = list(range(1, (len(y)+1)))

    plt.plot(X, y)
    plt.legend([legend])
    plt.title(title)
    path = path + str(legend)
    plt.savefig(path)
    plt.show()  # 展示
    plt.close()

def draw_ROC(vaild_labels, probas, path, save_name):
    path = path+"/_ROC_curve/"
    if not os.path.exists(path): os.makedirs(path)

    # 计算TPR和FPR
    fpr, tpr, thresholds = roc_curve(vaild_labels, probas)
    # 计算AUC
    roc_auc = auc(fpr, tpr)
    # 绘制ROC曲线
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(path+save_name)
    plt.show()
    plt.close()

def draw_ROC_for_every_model(true_labels, predicted_labels, path, save_name):
    path = path + "/_ROC_curve/"
    if not os.path.exists(path): os.makedirs(path)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线，表示随机猜测的ROC曲线

    roc_aucs = []
    for i in range(len(true_labels)):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]

        fpr, tpr, _ = roc_curve(true_label, predicted_label)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        plt.plot(fpr, tpr, label='Fold {} (AUC = {:.2f})'.format((i+1),roc_auc))

    mean_auc = np.mean(roc_aucs)
    plt.plot([], [], ' ', label='Mean AUC = %0.2f' % mean_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize='7',bbox_to_anchor=(1.0, 0.0))
    plt.savefig(path + save_name)
    plt.show()
    plt.close()

def max_one(tensor):
    """
    将每一列中最大的元素变为1，其余的变为0。

    Args:
        tensor: 一个形状为 n*4*500 的PyTorch张量。

    Returns:
        一个形状和输入相同的PyTorch张量。
    """
    # 在第二个维度上找到最大值和它们的索引
    batch_size = tensor.shape[0]
    tensor = tensor.permute((1,0,2)).reshape((4,-1))

    # max_vals, max_idxs = torch.max(tensor, dim=1, keepdim=
    max_idxs = torch.argmax(tensor,dim=0)
    # 创建一个全零的张量，与原始张量具有相同的形状
    output = torch.zeros_like(tensor)
    # 将最大值所在的位置设置为1
    output[max_idxs, torch.tensor(np.arange(tensor.shape[1]))] = 1
    output = output.reshape((4,batch_size,-1)).permute((1,0,2))
    return output

def get_best_cpu_num(dataset, batch_size):
    '''打印不同加载数量的CPU加载数据集的时间'''
    for num_workers in range(2, mp.cpu_count(), 2):
        train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size,
                                                   pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

def get_acc(tensor, lable):
    '''
    分类正确的概率
    :param tensor:
    :param lable: 0 G生成的数据， 1 真实数据
    :return:
    '''
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    res = torch.sum(tensor.reshape(-1)).item()/len(tensor)
    if lable==1:
        return res
    else:
        return 1-res

def draw_hist(list,epoch,name):
    plt.hist(list,bins=140)
    path = "./output/" + name+"_"+str(epoch)
    plt.savefig(path)
    # plt.show()
    plt.close()

def gumbel_softmax(logits, temp=1, hard=False, eps=1e-20):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temp: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints: - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, temp=temp, eps=eps)
    if hard:
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def _gumbel_softmax_sample(logits, temp=1, eps=1e-20):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / temp, dims - 1)

def _sample_gumbel(shape, eps=1e-20, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def decode_one_seq(img, letter_dict = {'A':0, 'C':1, 'G':2, 'T':3}):
    seq = ''
    for row in range(len(img)):
        on = np.argmax(img[row,:])
        seq += letter_dict[on]
    return seq

def get_vocab_num(k):
    res = 0
    for i in range(0,k+1):
        res += math.pow(4,i)
    return int(res)

def get_L1_regularization(net):
    regularization_loss = 0
    for param in net.parameters():
        regularization_loss += torch.sum(torch.abs(param))
    return regularization_loss

def stacked_tensor_to_np(tensor_list):
    res = []
    for i in range(len(tensor_list)):
        for j in range(len(tensor_list[i])):
            res.append(tensor_list[i][j])
    return np.array(res)

def count_base_proportion(dna_list):
    """
    输入：字符串列表，每个字符串都是一个DNA序列
    输出：ATCG四种碱基所占比例
    """
    props = []
    for dna_seq in dna_list:
        base_counts = {"A": 0, "T": 0, "C": 0, "G": 0}
        for base in dna_seq:
            if base in base_counts:
                base_counts[base] += 1
        seq_length = len(dna_seq)
        proportions = {}
        for base, count in base_counts.items():
            proportion = round(count / seq_length, 2)
            proportions[base] = proportion
        props.append(proportions)

    res = {"A": 0, "T": 0, "C": 0, "G": 0}
    for prop in props:
        for i in res.keys():
            res[i]+=prop[i]

    for key in res:
        res[key] = round(res[key] / len(dna_list), 4)

    return res

def calculate_metrics1(actual_labels, predicted_labels):
    # 将实际标签和预测标签转换为numpy数组
    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)

    # 计算 True Positives (TP)
    tp = np.sum((actual_labels == 0) & (predicted_labels == 0))
    # 计算 True Negatives (TN)
    tn = np.sum((actual_labels == 1) & (predicted_labels == 1))
    # 计算 False Positives (FP)
    fp = np.sum((actual_labels == 1) & (predicted_labels == 0))
    # 计算 False Negatives (FN)
    fn = np.sum((actual_labels == 0) & (predicted_labels == 1))

    return tp, tn, fp, fn

def calculate_metrics2(tp, tn, fp, fn):
    # 计算敏感度(Sensitivity)
    sn = tp / (tp + fn)
    # 计算特异度(Specificity)
    sp = tn / (tn + fp)
    # 计算Matthews 相关系数(MCC)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    # 计算准确率(Accuracy)
    acc = (tp + tn) / (tp + tn + fp + fn)

    return sn, sp, mcc, acc

def merge_to_file(file_pathA, file_pathB, output_path):
    with open(file_pathA, 'r') as file1:
        content1 = file1.readlines()
    with open(file_pathB, 'r') as file2:
        content2 = file2.readlines()
    content1[len(content1)-1] = content1[len(content1)-1] + "\n"
    # selected_lines_file1 = random.sample(content1, 1000)
    selected_lines_file2 = random.sample(content2, 50000)

    merged_content = content1 + selected_lines_file2
    with open(output_path, 'w') as output_file:
        output_file.writelines(merged_content)
    return output_path