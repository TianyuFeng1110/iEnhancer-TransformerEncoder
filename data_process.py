import torch
import random
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
from functools import reduce
from collections import defaultdict

# 根据DNA样本获取one-hot编码
def get_one_hot(sample, seq_len):
    x = torch.zeros(5, seq_len)

    chars = list(sample)
    lens = len(chars)
    for i in range(lens):
        # bases = ["A", "C", "G", "T", "P"]
        if chars[i] == 'A' or chars[i]=='a':
            x[0, i] = 1;
        elif chars[i] == 'C' or chars[i]=='c':
            x[1, i] = 1;
        elif chars[i] == 'G' or chars[i]=='g':
            x[2, i] = 1;
        elif chars[i] == 'T' or chars[i]=='t':
            x[3, i] = 1;

    x[4,lens:] = 1
    return x

def get_kmer_dict(k):
    kmer_dict = defaultdict(int)
    all_mer = reduce(lambda x, y: [i + j for i in x for j in y], [['A', 'T', 'C', 'G']] * k)
    for K in range(1, k):
        padding = 'N' * (k - K)
        all_mer.extend(list(
            map(lambda x: x + padding, reduce(lambda x, y: [i + j for i in x for j in y], [['A', 'T', 'C', 'G']] * K))))
    all_N_str = ''
    for K in range(0, k):
        all_N_str += 'N'
    all_mer.append(all_N_str)

    for i in range(len(all_mer)):
        kmer_dict[all_mer[i]] = i

    return kmer_dict

'''k-mer编码'''
def kmer_encoding(dna_seq, k, seq_len, kmer_dict):
    dna_seq = dna_seq.rstrip("\n")
    if len(dna_seq) < seq_len:
        padding = 'N' * (seq_len - len(dna_seq))
        padded_string = dna_seq + padding
    else:
        padded_string = dna_seq[:seq_len]
    dna_seq = padded_string.rstrip("\n")

    # 初始化k-mer计数矩阵
    kmer_count_matrix = np.zeros(seq_len - k + 1)

    for i in range(seq_len - k + 1):
        current_kmer = dna_seq[i:i + k]
        kmer_index = kmer_dict[current_kmer]
        kmer_count_matrix[i] = kmer_index

    return kmer_count_matrix

'''保存增强子的开始和结束序列(只有增强子的部分)'''
def save_cut_idx(file_name, max_length):
    lines = []
    chromosome_name, chromosome, start_idx, end_idx = [], [], [], []
    chromosome_name = ['NC_000001.10', 'NC_000002.11', 'NC_000003.11', 'NC_000004.11', 'NC_000005.9', 'NC_000006.11', 'NC_000007.13', 'NC_000008.10', 'NC_000009.11', 'NC_000010.10', 'NC_000011.9', 'NC_000012.11', 'NC_000013.10', 'NC_000014.8', 'NC_000015.9', 'NC_000016.9', 'NC_000017.10', 'NC_000018.9', 'NC_000019.9', 'NC_000020.10', 'NC_000021.8', 'NC_000022.10', 'NC_000023.10', 'NC_000024.9', 'NC_012920.1']
    with open('./train_datas/hs.bed') as file:
        for line in file:
            line = line.strip().split()
            start = int(line[1])
            end = int(line[2])

            if int(line[2]) - int(line[1]) >= (max_length - 1):
                start = start + random.randint(0, end - start - (max_length-1))
                end = start + max_length - 1

                chromosome.append(line[0])
                start_idx.append(str(start))
                end_idx.append(str(end))

    lines.append(chromosome)
    lines.append(start_idx)
    lines.append(end_idx)

    # with open('./train_datas/GCF_000001405.25_GRCh37.p13_genomic.fna') as file:
    #     for line in file:
    #         if (line[0:4] == '>NC_'):
    #             chromosome_name.append(line.strip().split()[0][1:])

    enhancer_and_NCBI_ID = (chromosome_name, lines)
    with open(file_name, 'wb') as f:
        pickle.dump(enhancer_and_NCBI_ID, f)

    return chromosome_name, lines

'''保存截取增强子的开始和结束序列'''
def save_random_cut_idx(file_name, max_length):
    lines = []
    chromosome_name, chromosome, start_idx, end_idx = [], [], [], []
    chromosome_name = ['NC_000001.10', 'NC_000002.11', 'NC_000003.11', 'NC_000004.11', 'NC_000005.9', 'NC_000006.11', 'NC_000007.13', 'NC_000008.10', 'NC_000009.11', 'NC_000010.10', 'NC_000011.9', 'NC_000012.11', 'NC_000013.10', 'NC_000014.8', 'NC_000015.9', 'NC_000016.9', 'NC_000017.10', 'NC_000018.9', 'NC_000019.9', 'NC_000020.10', 'NC_000021.8', 'NC_000022.10', 'NC_000023.10', 'NC_000024.9', 'NC_012920.1']
    with open('./train_datas/hs.bed') as file:
        for line in file:
            line = line.strip().split()
            start = int(line[1])
            end = int(line[2])
            if int(line[2]) - int(line[1]) < (max_length-1):
                spare = (max_length - 1) - (end - start)  # 除去增强子还有多少空位置
                supplement = random.randint(0, spare)  # 除了增强子外一共要补充多少个碱基

                rand = random.randint(0, supplement)  # 前面补充多少个碱基
                start = 1 if start - rand <= 0 else (start - rand)
                end = end + (supplement - rand)

                chromosome.append(line[0])
                start_idx.append(str(start))
                end_idx.append(str(end))
            else :
                cut_method = random.randint(1, 3)  # 1截后半个增强子，再从后面截一段序列。2是前面截一段序列，再截前半个增强子。3是从增强子中间截取，截取的部分全为增强子。
                if cut_method == 1:
                    start = end - random.randint(100, max_length-1)
                    spare = (max_length-1) - (end - start)
                    end = end + random.randint(0, spare)
                elif cut_method == 2:
                    end = start + random.randint(100, max_length - 1)
                    spare = (max_length - 1) - (end - start)
                    start = start - random.randint(0, spare)
                elif cut_method == 3:
                    cut_len = random.randint(100, max_length-1)
                    start = start + random.randint(0, end-start-cut_len)
                    end = start + cut_len

                chromosome.append(line[0])
                start_idx.append(str(start))
                end_idx.append(str(end))

    lines.append(chromosome)
    lines.append(start_idx)
    lines.append(end_idx)

    # with open('./train_datas/GCF_000001405.25_GRCh37.p13_genomic.fna') as file:
    #     for line in file:
    #         if (line[0:4] == '>NC_'):
    #             chromosome_name.append(line.strip().split()[0][1:])

    enhancer_and_NCBI_ID = (chromosome_name, lines)
    with open(file_name, 'wb') as f:
        pickle.dump(enhancer_and_NCBI_ID, f)

    return chromosome_name, lines

def get_chromosome_name(file_name):
    with open(file_name, 'rb') as f:
        result = pickle.load(f)
    chromosome_name = result[0]
    chromosome = result[1][0]
    start_idx = result[1][1]
    end_idx = result[1][2]

    print(len(chromosome))
    return chromosome_name, chromosome, start_idx, end_idx

def generate_seqkit_bash(chromosome, chromosome_name, start_idx, end_idx, bash_path):
    # 打开文件，如果文件不存在，会自动创建
    with open(bash_path, 'w') as f:
        for i in range(len(chromosome)):
            command = ''
            if 'X' == chromosome[i][3:]:
                command = "seqkit subseq -r " + start_idx[i] + ":" + end_idx[
                    i] + " GCF_000001405.25_GRCh37.p13_genomic.fna --chr " + chromosome_name[22]
            elif 'Y' == chromosome[i][3:]:
                command = "seqkit subseq -r " + start_idx[i] + ":" + end_idx[
                    i] + " GCF_000001405.25_GRCh37.p13_genomic.fna --chr " + chromosome_name[23]
            elif chromosome[i][3:].isdigit():
                command = "seqkit subseq -r " + start_idx[i] + ":" + end_idx[
                    i] + " GCF_000001405.25_GRCh37.p13_genomic.fna --chr " + chromosome_name[
                              int(chromosome[i][3:]) - 1]
            # 向文件中写入内容
            f.write(command+'\n')

def generate_train_data(file_name):
    seqs = []
    seq = ''
    with open(file_name) as file:
        for line in file:
            if line[0].isalpha():
                seq += line[0:len(line)-1]
            else:
                if seq!='':
                    seqs.append(seq)
                seq = ''
        seqs.append(seq)

    with open('./train_datas/gan/hs_non_enhancer.txt', 'a') as f:
        for i in range(len(seqs)):
            f.write(seqs[i] + '\n')

def save_non_cut_idx(file_name, max_length):
    lines = []
    chromosome_name, chromosome, start_idx, end_idx = [], [], [], []
    chromosome_name = ['NC_000001.10', 'NC_000002.11', 'NC_000003.11', 'NC_000004.11', 'NC_000005.9', 'NC_000006.11',
                       'NC_000007.13', 'NC_000008.10', 'NC_000009.11', 'NC_000010.10', 'NC_000011.9', 'NC_000012.11',
                       'NC_000013.10', 'NC_000014.8', 'NC_000015.9', 'NC_000016.9', 'NC_000017.10', 'NC_000018.9',
                       'NC_000019.9', 'NC_000020.10', 'NC_000021.8', 'NC_000022.10', 'NC_000023.10', 'NC_000024.9',
                       'NC_012920.1']
    with open('./train_datas/hs.bed') as file:
        line1 = file.readline()  # 读取第一行
        while line1:  # 当 line1 不为空时
            line2 = file.readline()  # 读取下一行
            if not line2:  # 如果 line2 为空，说明读取已结束
                break
            # 处理 line1 和 line2
            line = line2
            line1 = line1.strip().split()
            line2 = line2.strip().split()

            if line1[0] != line2[0]:
                line1 = line
                continue

            start = int(line1[2])
            end = int(line2[1])

            if (end - start >= max_length - 1):
                start = start + random.randint(0, end - start - (max_length - 1))
                end = start + max_length - 1

                chromosome.append(line1[0])
                start_idx.append(str(start))
                end_idx.append(str(end))

            line1 = line  # 继续读取下一行，作为 line1

    lines.append(chromosome)
    lines.append(start_idx)
    lines.append(end_idx)

    enhancer_and_NCBI_ID = (chromosome_name, lines)
    with open(file_name, 'wb') as f:
        pickle.dump(enhancer_and_NCBI_ID, f)

    return chromosome_name, lines
'''截取非增强子序列的开始结束索引'''
def save_non_enhancer_idx(file_name, max_length):
    lines = []
    chromosome_name, chromosome, start_idx, end_idx = [], [], [], []
    chromosome_name = ['NC_000001.10', 'NC_000002.11', 'NC_000003.11', 'NC_000004.11', 'NC_000005.9', 'NC_000006.11', 'NC_000007.13', 'NC_000008.10', 'NC_000009.11', 'NC_000010.10', 'NC_000011.9', 'NC_000012.11', 'NC_000013.10', 'NC_000014.8', 'NC_000015.9', 'NC_000016.9', 'NC_000017.10', 'NC_000018.9', 'NC_000019.9', 'NC_000020.10', 'NC_000021.8', 'NC_000022.10', 'NC_000023.10', 'NC_000024.9', 'NC_012920.1']
    with open('./train_datas/hs.bed') as file:
        line1 = file.readline()  # 读取第一行
        while line1:  # 当 line1 不为空时
            line2 = file.readline()  # 读取下一行
            if not line2:  # 如果 line2 为空，说明读取已结束
                break
            # 处理 line1 和 line2
            line = line2
            line1 = line1.strip().split()
            line2 = line2.strip().split()

            if line1[0]!=line2[0]:
                line1 = line
                continue

            start = int(line1[2])
            end = int(line2[1])

            if (end-start>=max_length-1):
                cut_len = random.randint(100, max_length - 1)
                start = start + random.randint(0, end - start - cut_len)
                end = start + cut_len

            chromosome.append(line1[0])
            start_idx.append(str(start))
            end_idx.append(str(end))

            line1 = line  # 继续读取下一行，作为 line1

    lines.append(chromosome)
    lines.append(start_idx)
    lines.append(end_idx)

    enhancer_and_NCBI_ID = (chromosome_name, lines)
    with open(file_name, 'wb') as f:
        pickle.dump(enhancer_and_NCBI_ID, f)

    return chromosome_name, lines

def splitting(num ,bash_path, num_lines_per_file):
    with open(bash_path, 'r') as f:
        for i in range(num):
            # 每个文件的名称
            filename = f'./train_datas/gan/bash/non_enhancer_{i}.bash'

            with open(filename, 'w') as out_file:
                for j in range(num_lines_per_file):
                    line = f.readline()
                    out_file.write(line)

def random_lines(input_file_path: str, output_file_path: str, N: int):
    # 打开输入文件和输出文件
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'a') as output_file:
        # 从输入文件的行数中随机选择N行
        num_lines = sum(1 for line in input_file)
        indices = set(random.sample(range(num_lines), N))

        # 将选中的行复制到输出文件中
        input_file.seek(0)  # 将文件指针重新指向开头
        for i, line in enumerate(input_file):
            if i in indices:
                output_file.write(line)

'''简单验证，提取1/k的数据作为验证集'''
def extract_k_fold_data(input_file_path, validation_path, train_path, k):
    # 获取输入文件中的数据
    with open(input_file_path, 'r') as input_file:
        data = input_file.readlines()

    # 计算每份数据的大小
    data_size = len(data)
    subset_size = data_size // k

    # 打乱数据集的顺序
    random.shuffle(data)

    # 将数据集划分为K个随机子集
    subsets = [data[i:i + subset_size] for i in range(0, data_size, subset_size)]

    for i in range(len(subsets)):
        # 分别将每个子集写入不同的文件
        if i == 0:
            with open(validation_path, 'w') as output_file_1:
                output_file_1.writelines(subsets[i])
        else:
            with open(train_path, 'a') as output_file_2:
                output_file_2.writelines(subsets[i])