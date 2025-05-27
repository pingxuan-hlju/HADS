import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import random
from model_all import Net

# 准备加载数据
class MyDataset(Dataset):
    def __init__(self, tri, ds):
        self.tri = tri
        self.ds = ds

    def __getitem__(self, idx):
        x, y = self.tri[:, idx]
        label = self.ds[x][y]  # 根据行列索引找到label
        return x, y, label

    def __len__(self):
        return self.tri.shape[1]

def set_seed(seed=4202):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_and_test(epochs = 40, learn_rate = 1e-4, weight_decay = 5e-4, batch = 64, device = 'cpu'):
    n, tri, tei, mads, mcss, ds, chem_g1s, chem_g2s, chem_hs, chem_lines, dis_g1s, dis_g2s, dis_hs, dis_lines = torch.load(
        './tempdata.pth', weights_only=True)
    for cros in range(n):
        print(f'第{cros+1}折')
        torch.cuda.empty_cache()
        mad = mads[cros].float().to(device)  # 疾病
        mcs = mcss[cros].float().to(device)
        chem_g1 = chem_g1s[cros].to(device)
        chem_g2 = chem_g2s[cros].to(device)
        chem_h = chem_hs[cros].to(device)
        chem_line = chem_lines[cros].to(device)
        dis_g1 = dis_g1s[cros].to(device)
        dis_g2 = dis_g2s[cros].to(device)
        dis_h = dis_hs[cros].to(device)
        dis_line = dis_lines[cros].to(device)

        trset = DataLoader(MyDataset(tri[cros], ds), batch, shuffle=True)
        teset = DataLoader(MyDataset(tei[cros], ds), 1024, shuffle=False)

        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), learn_rate, weight_decay=weight_decay)
        fea1 = mad
        fea2 = mcs
        cost = nn.CrossEntropyLoss()
        model.train()
        for epoch in tqdm(range(epochs), desc='epochs'):
            aloss = 0
            preds, ys = [], []
            for x1, x2, y in trset:
                x1, x2, y = x1.long().to(device), (x2 + 708).long().to(device), y.long().to(device)
                out = model(fea1, fea2, chem_g1, chem_g2, chem_h, dis_g1, dis_g2, dis_h, x1, x2, chem_line, dis_line)
                loss = cost(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                aloss += loss
                preds.append(out)
                ys.append(y)
            preds, ys = torch.cat(preds, dim=0), torch.cat(ys)
            print("epoch:" + str(epoch + 1) + " loss:" + str(aloss.item()) + " acc:" + str(
                (preds.argmax(dim=1) == ys).sum() / len(ys)))

            torch.save(model.state_dict(), 'checkpoint%d.pt' % cros)  # 保存模型参数 若使用早停法则将此行注释


    # 测试
    predall, yall = torch.tensor([]), torch.tensor([])
    model.eval()
    model.load_state_dict(torch.load('checkpoint%d.pt' % cros, weights_only=True))
    with torch.no_grad():
        for x1, x2, y in tqdm(teset, desc='test'):
            x1, x2, y = x1.long().to(device), (x2 + 708).long().to(device), y.long().to(device)
            pred = model(fea1, fea2, chem_g1, chem_g2, chem_h, dis_g1, dis_g2, dis_h, x1, x2,chem_line,dis_line).data
            predall = torch.cat([predall, torch.as_tensor(pred, device='cpu')], dim=0)
            yall = torch.cat([yall, torch.as_tensor(y, device='cpu')])
        print("acc:" + str((predall.argmax(dim=1) == yall).sum() / len(yall)))
    torch.save((predall, yall), 'Res%d' % cros)  # 保存预测结果和真实结果

if __name__ == "__main__":

    set_seed()
    epochs = 40
    learn_rate = 1e-4
    weight_decay = 5e-4
    batch = 64
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print('训练开始')
    train_and_test(epochs, learn_rate, weight_decay, batch,device)