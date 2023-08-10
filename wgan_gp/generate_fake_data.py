import models.models as models
import torch
import numpy as np
import utils
import os
import argparse

def load_model_and_return_fake_seqs(checkpoints_path, hidden, noise_size=128, num=1000):
    bases = ["A", "C", "G", "T", "N"]
    seq_len = 200
    G = models.Generator(seq_len, hidden, noise_size)

    if os.path.exists(checkpoints_path):
        checkpoint = torch.load(checkpoints_path)
        G.load_state_dict(checkpoint['modelG_state_dict'])
        G.eval()
    else:
        print("checkpoints_path is not exist")

    # torch.manual_seed(0)
    random_noise = torch.randn(num, noise_size)

    output_seq = G(random_noise).detach().cpu().numpy()
    res = []
    for i in range(len(output_seq)):
        sequence = ""
        for j in range(output_seq.shape[2]):
            idx = np.argmax(output_seq[i, :, j])
            sequence += bases[idx]
        if "N" not in sequence:
            res.append(sequence)

    return res

def get_valid_datas():
    with open("../train_datas/validation.txt", 'r') as file:
        list, enhancers = [], []
        for line in file.readlines():
            line = line.replace('\n', '').split(" ")
            if len(line[1]) != 200:
                if len(list) != 0: enhancers.append(list)
                list = []
            else:
                list.append(line[1].upper())
        non_enhancers, strong_enhancers, weak_enhancers = list, enhancers[0], enhancers[1]

    return non_enhancers, strong_enhancers, weak_enhancers

def load_model_for_every_checkpoint(model_name, hidden, prop):
    min, which_model = 1, -1
    for i in range(1, 21):
        k = str(i * 100)
        seqs = load_model_and_return_fake_seqs(
            "../saved_models/" + model_name + "/" + model_name + "_epoch" + k + ".pth", hidden, 128, 1000)
        dict = utils.count_base_proportion(seqs)

        result_dict = {}
        for key in list(prop.keys()):
            # 获取对应位置的值
            value1 = dict[key]
            value2 = prop[key]
            # 计算差的绝对值
            absolute_diff = abs(value1 - value2)
            # 存储结果
            result_dict[key] = round(absolute_diff, 4)
        # print(k, dict, result_dict, round(sum(result_dict.values()), 4))
        if min > round(sum(result_dict.values()), 4) and i>=7:
            min = round(sum(result_dict.values()), 4)
            which_model = k
    return min , which_model

def append_list_to_file(data_list, file_path):
    with open(file_path, 'a') as file:
        for item in data_list:
            file.write(str(item) + '\n')

def main(argv=None):
    parser = argparse.ArgumentParser(description='Classification of enhancers.')
    parser.add_argument("--model_name", default="gan_non_enhancer", help="model name for generate fake datas")
    parser.add_argument("--type", default="non", help="Generate data types (strong / weak / non)")
    parser.add_argument("--amount", default="1000", help="The amount of data generated")
    args = parser.parse_args()

    # get dataset
    non_enhancers, strong_enhancers, weak_enhancers = get_valid_datas()
    non_enhancer_proportion = utils.count_base_proportion(non_enhancers)
    enhancer_proportion = utils.count_base_proportion(strong_enhancers + weak_enhancers)
    strong_enhancersr_proportion = utils.count_base_proportion(strong_enhancers)
    weak_enhancers_proportion = utils.count_base_proportion(weak_enhancers)

    # print("enhancer: ", enhancer_proportion)
    print("non_enhancer: ", non_enhancer_proportion)
    print("strong_enhancers: ", strong_enhancersr_proportion)
    print("weak_enhancers: ", weak_enhancers_proportion)

    # append_list_to_file(fake_strong_enhancers, "../train_datas/classification/enhancer1.txt")
    # append_list_to_file(fake_weak_enhancers, "../train_datas/classification/enhancer1.txt")
    # append_list_to_file(fake_non_enhancers, "../train_datas/classification/non_enhancer1.txt")
    # append_list_to_file(fake_weak_enhancers, "../train_datas/predication/weak_enhancer.txt")
    # append_list_to_file(fake_strong_enhancers, "../train_datas/predication/strong_enhancer.txt")

    fake_seqs = []
    if args.type == "non":
        min, k = load_model_for_every_checkpoint(args.model_name, 256, non_enhancer_proportion)
        fake_non_enhancers = load_model_and_return_fake_seqs(
            "../saved_models/" + args.model_name + "/gan_non_enhancer_epoch" + str(k) + ".pth", 256, num=int(args.amount))
        # print(utils.count_base_proportion(fake_non_enhancers))
        fake_seqs = fake_non_enhancers
    elif args.type == "strong":
        min, k = load_model_for_every_checkpoint(args.model_name, 256, strong_enhancersr_proportion)
        fake_strong_enhancers = load_model_and_return_fake_seqs(
            "../saved_models/" + args.model_name + "/gan_strong_enhancer_epoch" + str(k) + ".pth", 256, num=int(args.amount))
        # print(utils.count_base_proportion(fake_strong_enhancers))
        fake_seqs = fake_strong_enhancers
    elif args.type == "weak":
        min, k = load_model_for_every_checkpoint(args.model_name, 256, weak_enhancers_proportion)
        fake_weak_enhancers = load_model_and_return_fake_seqs(
            "../saved_models/" + args.model_name + "/gan_weak_enhancer_epoch" + str(k) + ".pth", 256, num=int(args.amount))
        # print(utils.count_base_proportion(fake_weak_enhancers))
        fake_seqs = fake_weak_enhancers
    else:
        print("Parameter error, please check again.")

    for seq in fake_seqs:
        print(seq)

    # 拿CD HIT评估相似度
    # append_list_to_file(fake_non_enhancers, "../train_datas/evaluation/non_enhancer.fasta")
    # append_list_to_file(fake_strong_enhancers, "../train_datas/evaluation/strong_enhancer.fasta")
    # append_list_to_file(fake_weak_enhancers, "../train_datas/evaluation/weak_enhancer.fasta")

if __name__ == "__main__":
    main()