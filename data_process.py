import torch
import numpy as np
import random


def set_seed(seed=4202):  # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

def load_data():
    # 708,708 药物之间的相似矩阵（根据疾病治疗计算）
    ddad = torch.tensor(np.loadtxt('data/drug_drug_sim_dis.txt')).float()

    # 708,708 药物之间的相似矩阵（化学特征等计算）
    ddcs = torch.tensor(np.loadtxt('data/Similarity_Matrix_Drugs.txt')).float()

    # 708,4192  药物与副作用关联矩阵
    ds = torch.tensor(np.loadtxt('data/mat_drug_se.txt')).long()

    # 4192,4192 副作用之间的相似矩阵(根据副作用关联药物求得)
    ss = torch.tensor(np.loadtxt('data/se_seSmilirity.csv', delimiter=',')).float()
    return ddad, ddcs, ds, ss


# 5 cross  划分数据集 al是药物和副作用的关联矩阵
def split_dataset(al, n):
    rand_index = torch.randperm(al.sum())

    indices = torch.where(al == 1)
    ps = torch.stack(indices).t().index_select(0, rand_index)  # stack将行索引和列索引进行拼接

    indices = torch.where(al == 0)
    ns = torch.stack(indices).t()
    ns = ns.index_select(0, torch.randperm(ns.shape[0]))

    sf = int(ps.shape[0] / n)  # 将正样本划分5份(正样本数少于负样本)
    tri, tei, lda = [], [], []
    for i in range(n):
        ptrn = torch.cat([ps[:(i * sf), :], ps[((i + 1) * sf):(n * sf), :]], dim=0).T
        ntrn = torch.cat([ns[:(i * sf), :], ns[((i + 1) * sf):(n * sf), :]], dim=0).T
        trn = torch.cat([ptrn, ntrn], dim=1)
        pten = torch.cat([ps[(i * sf):((i + 1) * sf), :], ps[(n * sf):, :]], dim=0).T  # ps[(n*sf):,:]]将多余部分全部作为测试集 同下
        nten = torch.cat([ns[(i * sf):((i + 1) * sf), :], ns[(n * sf):, :]], dim=0).T
        ten = torch.cat([pten, nten], dim=1)

        tri.append(trn)  # 训练集
        tei.append(ten)  # 测试集
        ldt = al.clone()
        ldt[pten[0, :], pten[1, :]] = 0  # ldt是遮掩后的关联矩阵
        lda.append(ldt)
    return tri, tei, lda

def cfm(ddad, ddcs, ds, ss):
    r1 = torch.cat([ddad, ds], dim=1)  # 拼接药物疾病距离矩阵和药物副作用矩阵

    r2 = torch.cat([ds.T, ss], dim=1)  # 药物副作用和副作用相似矩阵

    r3 = torch.cat([ddcs, ds], dim=1)  # 药物相似矩阵和药物副作用矩阵
    return torch.cat([r1, r2], dim=0), torch.cat([r3, r2], dim=0)

def a_row(matrix, k):

    matrix1 = matrix.clone()
    mask = matrix1 < k
    matrix1[mask] = 0.

    return matrix1

def g_generator(drug_drug_topk, ds, ss_topk):  # rs遮后的关联
    r1_topk = torch.cat([drug_drug_topk, ds], dim=1)
    r2_topk = torch.cat([ds.T, ss_topk], dim=1)
    h1 = torch.cat([r1_topk, r2_topk], dim=0)  # 超图

    d = torch.diag((torch.sum(h1, dim=1)) ** -1)  # 计算h1各超边的度的-1次方转化为对角矩阵

    b = torch.diag((torch.sum(h1, dim=0)) ** -1)  # 计算节点的度

    h2 = h1.T
    g1 = d @ h1  # 超边度*超边矩阵
    g2 = b @ h2  # 节点度*超边矩阵.T
    return g1, g2, h1


def line_generator(h):
    # 计算矩阵的维度
    cols = h.shape[1]

    # 初始化权值矩阵
    line_g = torch.zeros((cols, cols))

    # 遍历每一对列
    for i in range(cols):
        for j in range(i + 1, cols):  # 避免重复计算和自身比较

            temp1 = np.count_nonzero(h[:, i])
            temp2 = np.count_nonzero(h[:, j])
            temp3 = np.count_nonzero((h[:, i] * h[:, j]))

            # 计算权值
            line_g[i, j] = temp3 / (temp1 + temp2 - temp3)
            line_g[j, i] = line_g[i, j]
    i = torch.eye(cols)
    line_g = line_g + i
    b = torch.diag((torch.sum(line_g, dim=0)) ** -0.5)
    L = b @ line_g @ b
    return L


print('五折划分')
n=5
ddad,ddcs,ds,ss=load_data() # 药物相似，药物相似，关联，副作用
tri,tei,dsa=split_dataset(ds,n)
mads,mcss=[],[]
for i in range(n):
    mad,mcs=cfm(ddad,ddcs,dsa[i],ss)
    mads.append(mad)  # 疾病
    mcss.append(mcs)

print('阈值化')
drug_drug_sim_sub_topk = a_row(ddad, 0.4)
drug_drug_sim_func_topk = a_row(ddcs, 0.4)
se_se_sim_topk = a_row(ss, 0.5)

print('构建超图、线图')
chem_g1s, chem_g2s, chem_hs, chem_lines, dis_g1s, dis_g2s, dis_hs, dis_lines = [], [], [], [], [], [], [], []
for i in range(n):
    chem_g1, chem_g2, chem_h = g_generator(drug_drug_sim_func_topk, dsa[i], se_se_sim_topk)  # h是超边矩阵
    chem_line = line_generator(chem_h)  # 构建线图邻接矩阵并归一化
    dis_g1, dis_g2, dis_h = g_generator(drug_drug_sim_sub_topk, dsa[i], se_se_sim_topk)
    dis_line = line_generator(dis_h)

    chem_g1s.append(chem_g1)
    chem_g2s.append(chem_g2)
    chem_hs.append(chem_h)
    chem_lines.append(chem_line)
    dis_g1s.append(dis_g1)
    dis_g2s.append(dis_g2)
    dis_hs.append(dis_h)
    dis_lines.append(dis_line)

torch.save([n, tri, tei, mads, mcss, ds, chem_g1s, chem_g2s, chem_hs, chem_lines, dis_g1s, dis_g2s, dis_hs, dis_lines],
           'tempdata.pth')