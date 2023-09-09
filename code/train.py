from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from model import myModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainDataset(Dataset):

    def __init__(self, args, cm_browse_enterprise, enterprise_browse_cm, cooperation_info, cm_ripple_set,
                 enterprise_ripple_set):

        self.args = args
        self.cm_browse_enterprise = cm_browse_enterprise
        self.enterprise_browse_cm = enterprise_browse_cm
        self.cooperation_info = cooperation_info
        self.num_interactions = len(cm_browse_enterprise)
        self.cm_ripple_set = cm_ripple_set
        self.enterprise_ripple_set = enterprise_ripple_set

        self.cbe_cm = self.cm_browse_enterprise[:, 0]
        self.cbe_e = self.cm_browse_enterprise[:, 1]
        self.cbe_d = self.cm_browse_enterprise[:, 2]
        self.cbe_label = self.cm_browse_enterprise[:, 3]

        self.ebc_e = self.enterprise_browse_cm[:, 0]
        self.ebc_cm = self.enterprise_browse_cm[:, 1]
        self.ebc_label = self.enterprise_browse_cm[:, 2]

        self.coo_cm = self.cooperation_info[:, 0]
        self.coo_e = self.cooperation_info[:, 1]
        self.coo_d = self.cooperation_info[:, 2]
        self.coo_label = self.cooperation_info[:, 3]

    def __getitem__(self, idx):
        cbe_cm = torch.tensor(self.cbe_cm[idx], dtype=torch.long)
        cbe_e = torch.tensor(self.cbe_e[idx], dtype=torch.long)
        cbe_d = torch.tensor(self.cbe_d[idx], dtype=torch.long)
        cbe_label = torch.tensor(self.cbe_label[idx], dtype=torch.long)

        ebc_e = torch.tensor(self.ebc_e[idx], dtype=torch.long)
        ebc_cm = torch.tensor(self.ebc_cm[idx], dtype=torch.long)
        ebc_label = torch.tensor(self.ebc_label[idx], dtype=torch.long)

        coo_cm = torch.tensor(self.coo_cm[idx], dtype=torch.long)
        coo_e = torch.tensor(self.coo_e[idx], dtype=torch.long)
        coo_d = torch.tensor(self.coo_d[idx], dtype=torch.long)
        coo_label = torch.tensor(self.coo_label[idx], dtype=torch.long)

        cbe_cm_memories_h, cbe_cm_memories_r, cbe_cm_memories_t = self.get_ripple_set(cbe_cm, self.cm_ripple_set)
        cbe_e_memories_h, cbe_e_memories_r, cbe_e_memories_t = self.get_ripple_set(cbe_e, self.enterprise_ripple_set)
        ebc_e_memories_h, ebc_e_memories_r, ebc_e_memories_t = self.get_ripple_set(ebc_e, self.enterprise_ripple_set)
        ebc_cm_memories_h, ebc_cm_memories_r, ebc_cm_memories_t = self.get_ripple_set(ebc_cm, self.cm_ripple_set)
        coo_cm_memories_h, coo_cm_memories_r, coo_cm_memories_t = self.get_ripple_set(coo_cm, self.cm_ripple_set)
        coo_e_memories_h, coo_e_memories_r, coo_e_memories_t = self.get_ripple_set(coo_e, self.enterprise_ripple_set)

        return (cbe_cm.to(device), cbe_e.to(device), cbe_d.to(device), cbe_label.to(device), ebc_e.to(device),
                ebc_cm.to(device), ebc_label.to(device), coo_cm.to(device), coo_e.to(device), coo_d.to(device),
                coo_label.to(device), cbe_cm_memories_h, cbe_cm_memories_r, cbe_cm_memories_t, cbe_e_memories_h,
                cbe_e_memories_r, cbe_e_memories_t, ebc_e_memories_h, ebc_e_memories_r, ebc_e_memories_t,
                ebc_cm_memories_h, ebc_cm_memories_r, ebc_cm_memories_t, coo_cm_memories_h, coo_cm_memories_r,
                coo_cm_memories_t, coo_e_memories_h, coo_e_memories_r, coo_e_memories_t)

    def get_ripple_set(self, x, ripple_set):
        memories_h, memories_r, memories_t = [], [], []
        for i in range(self.args.n_hop):
            memories_h.append(torch.LongTensor([ripple_set[int(x)][i][0]]))
            memories_r.append(torch.LongTensor([ripple_set[int(x)][i][1]]))
            memories_t.append(torch.LongTensor([ripple_set[int(x)][i][2]]))
        memories_h = list(map(lambda x: x.to(device), memories_h))
        memories_r = list(map(lambda x: x.to(device), memories_r))
        memories_t = list(map(lambda x: x.to(device), memories_t))
        return memories_h, memories_r, memories_t

    def __len__(self):
        return self.num_interactions


class TestDataset(Dataset):

    def __init__(self, args, dataset, x_ripple_set, y_ripple_set, version):
        self.args = args
        self.dataset = dataset
        self.num_interactions = len(dataset)
        self.x_ripple_set = x_ripple_set
        self.y_ripple_set = y_ripple_set

        self.x = self.dataset[:, 0]
        self.y = self.dataset[:, 1]
        if version == 2:
            self.z = self.dataset[:, 1]
            self.label = self.dataset[:, 2]
        else:
            self.z = self.dataset[:, 2]
            self.label = self.dataset[:, 3]

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.long, device=device)
        y = torch.tensor(self.y[idx], dtype=torch.long, device=device)
        z = torch.tensor(self.z[idx], dtype=torch.long, device=device)
        label = torch.tensor(self.label[idx], dtype=torch.long, device=device)

        x_h, x_r, x_t = self.get_ripple_set(x, self.x_ripple_set)
        y_h, y_r, y_t = self.get_ripple_set(y, self.y_ripple_set)

        return (x, y, z, label, x_h, x_r, x_t, y_h, y_r, y_t)

    def get_ripple_set(self, x, ripple_set):
        memories_h, memories_r, memories_t = [], [], []
        for i in range(self.args.n_hop):
            memories_h.append(torch.LongTensor([ripple_set[int(x)][i][0]]))
            memories_r.append(torch.LongTensor([ripple_set[int(x)][i][1]]))
            memories_t.append(torch.LongTensor([ripple_set[int(x)][i][2]]))
        memories_h = list(map(lambda x: x.to(device), memories_h))
        memories_r = list(map(lambda x: x.to(device), memories_r))
        memories_t = list(map(lambda x: x.to(device), memories_t))
        return memories_h, memories_r, memories_t

    def __len__(self):
        return self.num_interactions


def train(args, train_cm_browse_enterprise, valid_cm_browse_enterprise, train_enterprise_browse_cm,
          valid_enterprise_browse_cm, train_cooperate_info, valid_cooperate_info, vocab_matrix, technique_matrix,
          data_info, cm_ripple_set, enterprise_ripple_set):
    print('device:', device)
    model = myModel(args, data_info['entity_num'], data_info['relation_num'], data_info['cm_vocab_num'],
                    data_info['enterprise_vocab_num'], data_info['demand_vocab_num'],
                    torch.tensor(vocab_matrix.values, dtype=torch.int, device=device),
                    torch.tensor(technique_matrix.values, dtype=torch.int, device=device))
    model = model.to(device)
    for m in model.children():
        if isinstance(m, (nn.Embedding, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    interaction_data_train = TrainDataset(args, train_cm_browse_enterprise.values.astype(int),
                                          train_enterprise_browse_cm.values.astype(int),
                                          train_cooperate_info.values.astype(int), cm_ripple_set, enterprise_ripple_set)
    train_loader = DataLoader(dataset=interaction_data_train, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    for epoch in range(args.epochs):
        loss_list, cbe_loss_list, ebc_loss_list, coo_loss_list, text_loss1_list, text_loss2_list = [], [], [], [], [], []
        model.train()
        first = True
        for data in train_loader:

            loss, cbe_loss, ebc_loss, coo_loss, text_loss1, text_loss2, cbe_score, ebc_score, coo_score = model(*data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.cpu().detach().numpy())
            cbe_loss_list.append(cbe_loss.cpu().detach().numpy())
            ebc_loss_list.append(ebc_loss.cpu().detach().numpy())
            coo_loss_list.append(coo_loss.cpu().detach().numpy())
            text_loss1_list.append(text_loss1.cpu().detach().numpy())
            text_loss2_list.append(text_loss2.cpu().detach().numpy())
            if first:
                cbe_labels, ebc_labels, coo_labels = data[3], data[6], data[10]
                cbe_scores, ebc_scores, coo_scores = cbe_score, ebc_score, coo_score
                first = False
            else:
                cbe_labels = torch.cat((cbe_labels, data[3]), dim=0)
                ebc_labels = torch.cat((ebc_labels, data[6]), dim=0)
                coo_labels = torch.cat((coo_labels, data[10]), dim=0)
                cbe_scores = torch.cat((cbe_scores, cbe_score), dim=0)
                ebc_scores = torch.cat((ebc_scores, ebc_score), dim=0)
                coo_scores = torch.cat((coo_scores, coo_score), dim=0)

        train_loss = float(np.mean(loss_list))
        train_cbe_loss = float(np.mean(cbe_loss_list))
        train_ebc_loss = float(np.mean(ebc_loss_list))
        train_coo_loss = float(np.mean(coo_loss_list))
        train_text_loss1 = float(np.mean(text_loss1_list))
        train_text_loss2 = float(np.mean(text_loss2_list))
        train_cbe_auc = roc_auc_score(y_true=cbe_labels.cpu().detach().numpy(), y_score=cbe_scores.cpu().detach().numpy())
        train_cbe_predictions = [1 if i >= 0.5 else 0 for i in cbe_scores]
        train_cbe_acc = np.mean(np.equal(train_cbe_predictions, cbe_labels.cpu().detach().numpy()))
        train_ebc_auc = roc_auc_score(y_true=ebc_labels.cpu().detach().numpy(), y_score=ebc_scores.cpu().detach().numpy())
        train_ebc_predictions = [1 if i >= 0.5 else 0 for i in ebc_scores]
        train_ebc_acc = np.mean(np.equal(train_ebc_predictions, ebc_labels.cpu().detach().numpy()))
        train_coo_auc = roc_auc_score(y_true=coo_labels.cpu().detach().numpy(), y_score=coo_scores.cpu().detach().numpy())
        train_coo_predictions = [1 if i >= 0.5 else 0 for i in coo_scores]
        train_coo_acc = np.mean(np.equal(train_coo_predictions, coo_labels.cpu().detach().numpy()))

        # Save model to disk
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, args.model)

        print('Training: Epochs: %d, loss: %f, cbe_loss: %f, ebc_loss: %f, coo_loss: %f, text_loss1: %f, '
              'text_loss2: %f, cbe_auc: %f, cbe_acc: %f, ebc_auc: %f, ebc_acc: %f, coo_auc: %f, coo_acc: %f' %
              (epoch + 1, train_loss, train_cbe_loss, train_ebc_loss, train_coo_loss, train_text_loss1,
               train_text_loss2, train_cbe_auc, float(train_cbe_acc), train_ebc_auc, float(train_ebc_acc),
               train_coo_auc, float(train_coo_acc)))
        with open('../result/main_result.txt', 'a') as fp:
            fp.write('Training: Epochs: %d, loss: %f, cbe_loss: %f, ebc_loss: %f, coo_loss: %f, text_loss1: %f, '
                     'text_loss2: %f, cbe_auc: %f, cbe_acc: %f, ebc_auc: %f, ebc_acc: %f, coo_auc: %f, coo_acc: %f\n' %
                     (epoch + 1, train_loss, train_cbe_loss, train_ebc_loss, train_coo_loss, train_text_loss1,
                      train_text_loss2, train_cbe_auc, float(train_cbe_acc), train_ebc_auc, float(train_ebc_acc),
                      train_coo_auc, float(train_coo_acc)))

        (valid_loss, valid_cbe_loss, valid_ebc_loss, valid_coo_loss, valid_text_loss1, valid_text_loss2, valid_cbe_auc,
         valid_cbe_acc, valid_ebc_auc, valid_ebc_acc, valid_coo_auc, valid_coo_acc) = \
            valid(args, valid_cm_browse_enterprise, valid_enterprise_browse_cm, valid_cooperate_info, vocab_matrix,
                    technique_matrix, data_info, cm_ripple_set, enterprise_ripple_set)
        print('Validating: Epochs: %d, loss: %f, cbe_loss: %f, ebc_loss: %f, coo_loss: %f, text_loss1: %f, '
              'text_loss2: %f, cbe_auc: %f, cbe_acc: %f, ebc_auc: %f, ebc_acc: %f, coo_auc: %f, coo_acc: %f'
              % (epoch + 1, valid_loss, valid_cbe_loss, valid_ebc_loss, valid_coo_loss, valid_text_loss1,
                 valid_text_loss2, valid_cbe_auc, float(valid_cbe_acc), valid_ebc_auc, float(valid_ebc_acc),
                 valid_coo_auc, float(valid_coo_acc)))
        with open('../result/main_result.txt', 'a') as fp:
            fp.write('Validating: Epochs: %d, loss: %f, cbe_loss: %f, ebc_loss: %f, coo_loss: %f, text_loss1: %f, '
                     'text_loss2: %f, cbe_auc: %f, cbe_acc: %f, ebc_auc: %f, ebc_acc: %f, coo_auc: %f, coo_acc: %f\n'
                     % (epoch + 1, valid_loss, valid_cbe_loss, valid_ebc_loss, valid_coo_loss, valid_text_loss1,
                        valid_text_loss2, valid_cbe_auc, float(valid_cbe_acc), valid_ebc_auc, float(valid_ebc_acc),
                        valid_coo_auc, float(valid_coo_acc)))


def valid(args, cm_browse_enterprise, enterprise_browse_cm, cooperate_info, vocab_matrix, technique_matrix,
            data_info, cm_ripple_set, enterprise_ripple_set):
    model = myModel(args, data_info['entity_num'], data_info['relation_num'], data_info['cm_vocab_num'],
                    data_info['enterprise_vocab_num'], data_info['demand_vocab_num'],
                    torch.tensor(vocab_matrix.values, dtype=torch.int, device=device),
                    torch.tensor(technique_matrix.values, dtype=torch.int, device=device))
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    cbe = TestDataset(args, cm_browse_enterprise.values.astype(int), cm_ripple_set, enterprise_ripple_set, version=1)
    cbe_loader = DataLoader(dataset=cbe, batch_size=args.batch_size, shuffle=True)
    ebc = TestDataset(args, enterprise_browse_cm.values.astype(int), enterprise_ripple_set, cm_ripple_set, version=2)
    ebc_loader = DataLoader(dataset=ebc, batch_size=args.batch_size, shuffle=True)
    coo = TestDataset(args, cooperate_info.values.astype(int), cm_ripple_set, enterprise_ripple_set, version=3)
    coo_loader = DataLoader(dataset=coo, batch_size=args.batch_size, shuffle=True)

    model.eval()
    with torch.no_grad():
        cbe_loss_list, ebc_loss_list, coo_loss_list, text_loss1_list, text_loss2_list = [], [], [], [], []
        first = True
        for data in cbe_loader:
            cbe_loss, cbe_score = model.compute_loss(*data, 1)
            cbe_loss_list.append(cbe_loss.cpu())
            if first:
                cbe_labels = data[3]
                cbe_scores = cbe_score
                first = False
            else:
                cbe_labels = torch.cat((cbe_labels, data[3]), dim=0)
                cbe_scores = torch.cat((cbe_scores, cbe_score), dim=0)

        first = True
        for data in ebc_loader:
            ebc_loss, ebc_score = model.compute_loss(*data, 2)
            ebc_loss_list.append(ebc_loss.cpu())
            if first:
                ebc_labels = data[3]
                ebc_scores = ebc_score
                first = False
            else:
                ebc_labels = torch.cat((ebc_labels, data[3]), dim=0)
                ebc_scores = torch.cat((ebc_scores, ebc_score), dim=0)

        first = True
        for data in coo_loader:
            coo_loss, coo_score = model.compute_loss(*data, 3)
            text_loss1 = model.compute_text_loss(data[0], data[1], data[3], 1)
            text_loss2 = model.compute_text_loss(data[0], data[2], data[3], 2)
            coo_loss_list.append(coo_loss.cpu())
            text_loss1_list.append(text_loss1.cpu())
            text_loss2_list.append(text_loss2.cpu())
            if first:
                coo_labels = data[3]
                coo_scores = coo_score
                first = False
            else:
                coo_labels = torch.cat((coo_labels, data[3]), dim=0)
                coo_scores = torch.cat((coo_scores, coo_score), dim=0)

    cbe_loss = float(np.mean(cbe_loss_list))
    ebc_loss = float(np.mean(ebc_loss_list))
    coo_loss = float(np.mean(coo_loss_list))
    cbe_auc = roc_auc_score(y_true=cbe_labels.cpu().detach().numpy(), y_score=cbe_scores.cpu().detach().numpy())
    cbe_predictions = [1 if i >= 0.5 else 0 for i in cbe_scores]
    cbe_acc = np.mean(np.equal(cbe_predictions, cbe_labels.cpu().detach().numpy()))
    ebc_auc = roc_auc_score(y_true=ebc_labels.cpu().detach().numpy(), y_score=ebc_scores.cpu().detach().numpy())
    ebc_predictions = [1 if i >= 0.5 else 0 for i in ebc_scores]
    ebc_acc = np.mean(np.equal(ebc_predictions, ebc_labels.cpu().detach().numpy()))
    coo_auc = roc_auc_score(y_true=coo_labels.cpu().detach().numpy(), y_score=coo_scores.cpu().detach().numpy())
    coo_predictions = [1 if i >= 0.5 else 0 for i in coo_scores]
    coo_acc = np.mean(np.equal(coo_predictions, coo_labels.cpu().detach().numpy()))
    loss = args.lc1_weight * cbe_loss + args.lc2_weight * ebc_loss + args.lc3_weight * coo_loss + \
           args.ls1_weight * text_loss1 + args.ls2_weight * text_loss2
    return loss, cbe_loss, ebc_loss, coo_loss, text_loss1, text_loss2, cbe_auc, cbe_acc, ebc_auc, ebc_acc, coo_auc, coo_acc
