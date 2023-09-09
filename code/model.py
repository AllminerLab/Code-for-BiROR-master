import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class myModel(nn.Module):

    def __init__(self, args, n_entity, n_relation, cm_vocab_num, e_vocab_num, d_vocab_num, vocab_matrix,
                 technique_matrix):
        super(myModel, self).__init__()
        self.args = args
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.cm_vocab_num = cm_vocab_num
        self.e_vocab_num = e_vocab_num
        self.d_vocab_num = d_vocab_num
        self.entity_dim = args.entity_dim
        self.pro_dim = args.pro_dim
        self.word_dim = args.word_dim
        self.hidden_dim = args.hidden_dim
        self.n_hop = args.n_hop
        self.n_memory = args.n_memory
        self.technique_update_mode = args.technique_update_mode
        self.using_all_hops = args.using_all_hops

        self.vocab_val_len = vocab_matrix[:, 0]
        self.vocab_matrix = vocab_matrix[:, 1:]
        self.technique_val_len = technique_matrix[:, 0]
        self.technique_matrix = technique_matrix[:, 1:]

        self.entity_emb = nn.Embedding(self.n_entity, self.entity_dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.entity_dim * self.entity_dim)
        self.cm_word_emb = nn.Embedding(self.cm_vocab_num, self.word_dim)
        self.e_word_emb = nn.Embedding(self.e_vocab_num, self.word_dim)
        self.d_word_emb = nn.Embedding(self.d_vocab_num, self.word_dim)
        self.cm_lstm = nn.LSTM(self.word_dim, self.hidden_dim)
        self.e_lstm = nn.LSTM(self.word_dim, self.hidden_dim)
        self.d_lstm = nn.LSTM(self.word_dim, self.hidden_dim)
        self.transform_matrix = nn.Linear(self.entity_dim, self.entity_dim, bias=False)
        self.criterion = nn.BCELoss()

        self.W_u1 = nn.Linear(self.entity_dim + self.hidden_dim, self.pro_dim, bias=False)
        self.W_u2 = nn.Linear(self.entity_dim + self.hidden_dim, self.pro_dim, bias=False)
        self.W_u3 = nn.Linear(self.entity_dim + self.hidden_dim, self.pro_dim, bias=False)
        self.W_v1 = nn.Linear(self.entity_dim + self.hidden_dim, self.pro_dim, bias=False)
        self.W_v2 = nn.Linear(self.entity_dim + self.hidden_dim, self.pro_dim, bias=False)
        self.W_v3 = nn.Linear(self.entity_dim + self.hidden_dim + self.hidden_dim, self.pro_dim, bias=False)

    def forward(self, cbe_cm, cbe_e, cbe_d, cbe_label, ebc_e, ebc_cm, ebc_label, coo_cm, coo_e, coo_d, coo_label,
                cbe_cm_memories_h, cbe_cm_memories_r, cbe_cm_memories_t, cbe_e_memories_h, cbe_e_memories_r,
                cbe_e_memories_t, ebc_e_memories_h, ebc_e_memories_r, ebc_e_memories_t, ebc_cm_memories_h,
                ebc_cm_memories_r, ebc_cm_memories_t, coo_cm_memories_h, coo_cm_memories_r, coo_cm_memories_t,
                coo_e_memories_h, coo_e_memories_r, coo_e_memories_t):
        text_loss1 = self.compute_text_loss(coo_cm, coo_e, coo_label, 1)
        text_loss2 = self.compute_text_loss(coo_cm, coo_d, coo_label, 2)
        cbe_loss, cbe_scores = self.compute_loss(cbe_cm, cbe_e, cbe_d, cbe_label, cbe_cm_memories_h, cbe_cm_memories_r,
                                                 cbe_cm_memories_t, cbe_e_memories_h, cbe_e_memories_r,
                                                 cbe_e_memories_t, 1)
        # the third input is useless
        ebc_loss, ebc_scores = self.compute_loss(ebc_e, ebc_cm, ebc_cm, ebc_label, ebc_e_memories_h, ebc_e_memories_r,
                                                 ebc_e_memories_t, ebc_cm_memories_h, ebc_cm_memories_r,
                                                 ebc_cm_memories_t, 2)
        coo_loss, coo_scores = self.compute_loss(coo_cm, coo_e, coo_d, coo_label, coo_cm_memories_h, coo_cm_memories_r,
                                                 coo_cm_memories_t, coo_e_memories_h, coo_e_memories_r,
                                                 coo_e_memories_t, 3)
        loss = self.args.lc1_weight * cbe_loss + self.args.lc2_weight * ebc_loss + self.args.lc3_weight * coo_loss + \
               self.args.ls1_weight * text_loss1 + self.args.ls2_weight * text_loss2

        return loss, cbe_loss, ebc_loss, coo_loss, text_loss1, text_loss2, cbe_scores, ebc_scores, coo_scores

    def compute_text_loss(self, x, y, labels, version):
        x_vocab_val_len = self.vocab_val_len[x]
        x_sentence = self.vocab_matrix[x]
        y_vocab_val_len = self.vocab_val_len[y]
        y_sentence = self.vocab_matrix[y]

        if version == 1:
            x_text_emb = self.lstm_layer(x_vocab_val_len, x_sentence, 1)
            y_text_emb = self.lstm_layer(y_vocab_val_len, y_sentence, 2)
        else:
            x_text_emb = self.lstm_layer(x_vocab_val_len, x_sentence, 1)
            y_text_emb = self.lstm_layer(y_vocab_val_len, y_sentence, 3)

        predit_scores = torch.sigmoid((x_text_emb * y_text_emb).sum(dim=1))
        loss = self.criterion(predit_scores, labels.float())

        return loss

    def compute_loss(self, x, y, z, labels, x_memories_h, x_memories_r, x_memories_t, y_memories_h, y_memories_r,
                     y_memories_t, version):
        """self, x, y, labels, x_vocab_val_len, x_sentence, y_vocab_val_len, y_sentence, x_technique_val_len,
                     x_technique, y_technique_val_len, y_technique, x_memories_h, x_memories_r, x_memories_t,
                     y_memories_h, y_memories_r, y_memories_t, version"""
        x_vocab_val_len = self.vocab_val_len[x]
        x_sentence = self.vocab_matrix[x]
        y_vocab_val_len = self.vocab_val_len[y]
        y_sentence = self.vocab_matrix[y]
        z_vocab_val_len = self.vocab_val_len[z]
        z_sentence = self.vocab_matrix[z]
        x_technique_val_len = self.technique_val_len[x]
        x_technique = self.technique_matrix[x]
        y_technique_val_len = self.technique_val_len[y]
        y_technique = self.technique_matrix[y]

        x_graph_emb = self.fusion(x_technique_val_len, x_technique, x_memories_h, x_memories_r, x_memories_t)
        y_graph_emb = self.fusion(y_technique_val_len, y_technique, y_memories_h, y_memories_r, y_memories_t)
        if version == 2:
            x_text_emb = self.lstm_layer(x_vocab_val_len, x_sentence, 2)
            y_text_emb = self.lstm_layer(y_vocab_val_len, y_sentence, 1)
        else:
            x_text_emb = self.lstm_layer(x_vocab_val_len, x_sentence, 1)
            y_text_emb = self.lstm_layer(y_vocab_val_len, y_sentence, 2)
            z_text_emb = self.lstm_layer(z_vocab_val_len, z_sentence, 3)

        if version == 1:
            x_emb = torch.cat((x_graph_emb, x_text_emb), dim=1)
            y_emb = torch.cat((y_graph_emb, z_text_emb), dim=1)
            x_emb = self.W_u1(x_emb)
            y_emb = self.W_v1(y_emb)
        elif version == 2:
            x_emb = torch.cat((x_graph_emb, x_text_emb), dim=1)
            y_emb = torch.cat((y_graph_emb, y_text_emb), dim=1)
            x_emb = self.W_v2(x_emb)
            y_emb = self.W_u2(y_emb)
        else:
            x_emb = torch.cat((x_graph_emb, x_text_emb), dim=1)
            y_emb = torch.cat((y_graph_emb, y_text_emb, z_text_emb), dim=1)
            x_emb = self.W_u3(x_emb)
            y_emb = self.W_v3(y_emb)

        predict_scores = torch.sigmoid((x_emb * y_emb).sum(dim=1))
        loss = self.criterion(predict_scores, labels.float())

        """auc = roc_auc_score(y_true=labels.cpu().detach().numpy(), y_score=predict_scores.cpu().detach().numpy())
        predictions = [1 if i >= 0.5 else 0 for i in predict_scores]
        acc = np.mean(np.equal(predictions, labels.cpu().detach().numpy()))

        return loss, auc, acc"""
        return loss, predict_scores

    def lstm_layer(self, val_len, sentence, version):
        if version == 1:
            word_emb = self.cm_word_emb(sentence)
        elif version == 2:
            word_emb = self.e_word_emb(sentence)
        else:
            word_emb = self.d_word_emb(sentence)

        seq_word_emb, seq_lengths, perm_idx = self.sort_batch(word_emb, val_len)
        packed_embed = nn.utils.rnn.pack_padded_sequence(seq_word_emb, seq_lengths, batch_first=True)
        if version == 1:
            packed_out, (last_out, _) = self.cm_lstm(packed_embed)
        elif version == 2:
            packed_out, (last_out, _) = self.e_lstm(packed_embed)
        else:
            packed_out, (last_out, _) = self.d_lstm(packed_embed)
        text_embedding = last_out[-1]
        text_embedding = self.sort_text_embedding(text_embedding, perm_idx)
        return text_embedding

    def sort_text_embedding(self, text_embedding, indexes):
        _, perm_idx = indexes.sort(0, descending=False)
        seq_tensor = text_embedding[perm_idx]
        return seq_tensor

    def sort_batch(self, batch, val_len):
        seq_lengths, perm_idx = val_len.sort(0, descending=True)
        seq_batch = batch[perm_idx]
        return seq_batch, seq_lengths.cpu(), perm_idx

    def fusion(self, val_lens, technique, memories_h, memories_r, memories_t):
        emb = self.entity_emb(technique)
        mask = torch.arange((emb.shape[1]), dtype=torch.float32, device=device)[None, :] < val_lens[:, None]
        emb[~mask] = 0
        weights = (1 / val_lens).unsqueeze(-1).unsqueeze(-1)
        technique_embeddings = torch.sum(emb * weights, dim=1)

        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.entity_dim, self.entity_dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, technique_embeddings)
        emb = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                emb += o_list[i]
        return emb

    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, technique_embeddings):

        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = torch.unsqueeze(h_emb_list[hop], -1).squeeze(1)

            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))

            # [batch_size, dim, 1]
            v = torch.unsqueeze(technique_embeddings, -1)

            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))

            # [batch_size, n_memory]
            try:
                probs_normalized = F.softmax(probs, dim=1)
            except:
                probs = probs.unsqueeze(0)
                probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, -1)

            # [batch_size, dim]
            o = (t_emb_list[hop].squeeze(1) * probs_expanded).sum(dim=1)

            technique_embeddings = self._update_technique_embedding(technique_embeddings, o)
            o_list.append(o)
        return o_list

    def _update_technique_embedding(self, technique_embeddings, o):
        if self.technique_update_mode == "replace":
            technique_embeddings = o
        elif self.technique_update_mode == "plus":
            technique_embeddings = technique_embeddings + o
        elif self.technique_update_mode == "replace_transform":
            technique_embeddings = self.transform_matrix(o)
        elif self.technique_update_mode == "plus_transform":
            technique_embeddings = self.transform_matrix(technique_embeddings + o)
        else:
            raise Exception("Unknown technique updating mode: " + self.technique_update_mode)
        return technique_embeddings

    def get_hrt(self, ripple_set, data):
        memories_h, memories_r, memories_t = [], [], []
        for i in range(self.args.n_hop):
            memories_h.append(torch.LongTensor([ripple_set[int(user)][i][0] for user in data]))
            memories_r.append(torch.LongTensor([ripple_set[int(user)][i][1] for user in data]))
            memories_t.append(torch.LongTensor([ripple_set[int(user)][i][2] for user in data]))
        memories_h = list(map(lambda x: x.to(device), memories_h))
        memories_r = list(map(lambda x: x.to(device), memories_r))
        memories_t = list(map(lambda x: x.to(device), memories_t))
        return memories_h, memories_r, memories_t

    def predict(self, cm, demand_enterprise, cm_num, enterprise_num, demand_num, cm_ripple_set, enterprise_ripple_set):
        enterprise = torch.arange(cm_num, cm_num + enterprise_num)
        demand = torch.arange(cm_num + enterprise_num, cm_num + enterprise_num + demand_num)
        cm_memories_h, cm_memories_r, cm_memories_t = self.get_hrt(cm_ripple_set, cm)
        enterprise_memories_h, enterprise_memories_r, enterprise_memories_t = self.get_hrt(enterprise_ripple_set,
                                                                                           enterprise)
        cm_vocab_val_len = self.vocab_val_len[cm]
        cm_sentence = self.vocab_matrix[cm]
        enterprise_vocab_val_len = self.vocab_val_len[enterprise]
        enterprise_sentence = self.vocab_matrix[enterprise]
        demand_vocab_val_len = self.vocab_val_len[demand]
        demand_sentence = self.vocab_matrix[demand]
        cm_technique_val_len = self.technique_val_len[cm]
        cm_technique = self.technique_matrix[cm]
        enterprise_technique_val_len = self.technique_val_len[enterprise]
        enterprise_technique = self.technique_matrix[enterprise]

        cm_graph_emb = self.fusion(cm_technique_val_len, cm_technique, cm_memories_h, cm_memories_r, cm_memories_t)
        enterprise_graph_emb = self.fusion(enterprise_technique_val_len, enterprise_technique, enterprise_memories_h,
                                           enterprise_memories_r, enterprise_memories_t)
        demand_graph_emb = enterprise_graph_emb[demand_enterprise]
        cm_text_emb = self.lstm_layer(cm_vocab_val_len, cm_sentence, 1)
        enterprise_text_emb = self.lstm_layer(enterprise_vocab_val_len, enterprise_sentence, 2)
        demand_text_emb = self.lstm_layer(demand_vocab_val_len, demand_sentence, 3)

        cm_emb = torch.cat((cm_graph_emb, cm_text_emb), dim=1)
        enterprise_emb = torch.cat((enterprise_graph_emb, enterprise_text_emb), dim=1)
        demand_emb = torch.cat((demand_graph_emb, demand_text_emb), dim=1)

        cbe_matrix = torch.sigmoid(torch.mm(self.W_u1(cm_emb), self.W_v1(demand_emb).T))
        ebc_matrix = torch.sigmoid(torch.mm(self.W_v2(enterprise_emb), self.W_u2(cm_emb).T))
        return cbe_matrix, ebc_matrix



