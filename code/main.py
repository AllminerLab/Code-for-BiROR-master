import numpy as np
import pandas as pd
import argparse
import collections
import pickle
import random
from train import train, valid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity_dim', type=int, default=128, help='dimension of entity and relation embeddings')
    parser.add_argument('--pro_dim', type=int, default=128, help='dimension of projection embeddings')
    parser.add_argument('--word_dim', type=int, default=32, help='dimension of word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=32, help='dimension of hidden state of LSTM')
    parser.add_argument('--n_hop', type=int, default=4, help='maximum hops')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--ls1_weight', type=float, default=1, help='weight of the ls1')
    parser.add_argument('--ls2_weight', type=float, default=1, help='weight of the ls2')
    parser.add_argument('--lc1_weight', type=float, default=1, help='weight of the lc1')
    parser.add_argument('--lc2_weight', type=float, default=1, help='weight of the lc2')
    parser.add_argument('--lc3_weight', type=float, default=2, help='weight of the lc3')
    parser.add_argument('--l2_reg', type=float, default=0.001, help='weight of the l2 regularization term')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=64, help='size of ripple set for each hop')
    parser.add_argument('--technique_update_mode', type=str, default='plus_transform',
                        help='how to update technique at the end of each hop')
    parser.add_argument('--re_get_ripple_set', type=bool, default=True, help='whether to re get ripple set')
    parser.add_argument('--model', type=str, default='model.pt', help='name to save or load model from')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')

    return parser.parse_args()


def load_data(args, cm_browse_enterprise_file, enterprise_browse_cm_file, cooperate_info_file, vocab_matrix_file,
              technique_matrix_file, kg_file, data_info_file, re_get_ripple_set, cm_ripple_set_file,
              enterprise_ripple_set_file, cm_info_file, enterprise_info_file, enterprise_demand_file,
              entity_to_ix_file, ix_to_entity_file):
    """
    load data
    """
    data_ix_prefix = '../data/data_ix/'
    data_info_prefix = '../data/data_info/'
    ix_mapping_prefix = '../data/ix_mapping/'
    raw_data_prefix = '../data/raw_data/'

    with open(data_ix_prefix + 'train_' + cm_browse_enterprise_file) as f:
        train_cm_browse_enterprise = pd.read_table(f, sep='\t', header=None)
    with open(data_ix_prefix + 'valid_' + cm_browse_enterprise_file) as f:
        valid_cm_browse_enterprise = pd.read_table(f, sep='\t', header=None)
    with open(data_ix_prefix + 'test_' + cm_browse_enterprise_file) as f:
        test_cm_browse_enterprise = pd.read_table(f, sep='\t', header=None)
    with open(data_ix_prefix + 'train_' + enterprise_browse_cm_file) as f:
        train_enterprise_browse_cm = pd.read_table(f, sep='\t', header=None)
    with open(data_ix_prefix + 'valid_' + enterprise_browse_cm_file) as f:
        valid_enterprise_browse_cm = pd.read_table(f, sep='\t', header=None)
    with open(data_ix_prefix + 'test_' + enterprise_browse_cm_file) as f:
        test_enterprise_browse_cm = pd.read_table(f, sep='\t', header=None)
    with open(data_ix_prefix + 'train_' + cooperate_info_file) as f:
        train_cooperate_info = pd.read_table(f, sep='\t', header=None)
    with open(data_ix_prefix + 'valid_' + cooperate_info_file) as f:
        valid_cooperate_info = pd.read_table(f, sep='\t', header=None)
    with open(data_ix_prefix + 'test_' + cooperate_info_file) as f:
        test_cooperate_info = pd.read_table(f, sep='\t', header=None)
    with open(data_ix_prefix + vocab_matrix_file) as f:
        vocab_matrix = pd.read_csv(f, header=None)
    with open(data_ix_prefix + technique_matrix_file) as f:
        technique_matrix = pd.read_csv(f, header=None)
    with open(data_info_prefix + data_info_file, 'rb') as handle:
        data_info = pickle.load(handle)
    with open(ix_mapping_prefix + entity_to_ix_file, 'rb') as handle:
        entity_to_ix = pickle.load(handle)
    with open(ix_mapping_prefix + ix_to_entity_file, 'rb') as handle:
        ix_to_entity = pickle.load(handle)
    with open(raw_data_prefix + enterprise_demand_file, encoding='utf-8') as f:
        enterprise_demand = pd.read_csv(f)

    # get ripple set for cm and enterprise
    if re_get_ripple_set:
        cm_ripple_set, enterprise_ripple_set = get_ripple_set(args, kg_file, cm_info_file, enterprise_info_file,
                                                              entity_to_ix, cm_ripple_set_file,
                                                              enterprise_ripple_set_file)
    else:
        with open(data_ix_prefix + cm_ripple_set_file, 'rb') as handle:
            cm_ripple_set = pickle.load(handle)
        with open(data_ix_prefix + enterprise_ripple_set_file, 'rb') as handle:
            enterprise_ripple_set = pickle.load(handle)

    return (train_cm_browse_enterprise, valid_cm_browse_enterprise, test_cm_browse_enterprise,
            train_enterprise_browse_cm, valid_enterprise_browse_cm, test_enterprise_browse_cm, train_cooperate_info,
            valid_cooperate_info, test_cooperate_info, vocab_matrix, technique_matrix, data_info, entity_to_ix,
            ix_to_entity, cm_ripple_set, enterprise_ripple_set, enterprise_demand)


def get_ripple_set(args, kg_file, cm_info_file, enterprise_info_file, entity_to_ix, cm_ripple_set_file,
                   enterprise_ripple_set_file):
    """
    get ripple set for cm and enterprise
    """
    print('constructing ripple set ...')

    kg_prefix = '../data/kg/'
    raw_data_prefix = '../data/raw_data/'
    data_ix_prefix = '../data/data_ix/'
    with open(raw_data_prefix + cm_info_file, encoding='utf-8') as f:
        cm_info = pd.read_csv(f)
        cm_set = set(cm_info['code'])
    with open(raw_data_prefix + enterprise_info_file, encoding='utf-8') as f:
        enterprise_info = pd.read_csv(f)
        enterprise_set = set(enterprise_info['code'])
    with open(kg_prefix + kg_file, 'rb') as handle:
        kg = pickle.load(handle)

    # cm -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    # enterprise -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    cm_ripple_set = do_get_ripple_set(args, cm_set, kg, entity_to_ix, 'cm')
    enterprise_ripple_set = do_get_ripple_set(args, enterprise_set, kg, entity_to_ix, 'enterprise')

    with open(data_ix_prefix + cm_ripple_set_file, 'wb') as handle:
        pickle.dump(cm_ripple_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(data_ix_prefix + enterprise_ripple_set_file, 'wb') as handle:
        pickle.dump(enterprise_ripple_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cm_ripple_set, enterprise_ripple_set


def do_get_ripple_set(args, Set, kg, entity_to_ix, type):
    ripple_set = collections.defaultdict(list)
    for head in Set:
        head = entity_to_ix[(head, type)]
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = [head]
                # tails_of_last_hop = list(set(np.array(kg[head])[:, 0]))
            else:
                tails_of_last_hop = ripple_set[head][-1][2]

            for entity in tails_of_last_hop:
                if entity in kg:
                    for tail_and_relation in kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given entity is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0
            if len(memories_h) == 0:
                ripple_set[head].append(ripple_set[head][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[head].append((memories_h, memories_r, memories_t))
    return ripple_set


def main():
    """
    Main function for model training and testing
    """
    print("Main Loaded")
    random.seed(1000)
    args = parse_args()

    (train_cm_browse_enterprise, valid_cm_browse_enterprise, test_cm_browse_enterprise, train_enterprise_browse_cm,
     valid_enterprise_browse_cm, test_enterprise_browse_cm, train_cooperate_info, valid_cooperate_info,
     test_cooperate_info, vocab_matrix, technique_matrix, data_info, entity_to_ix, ix_to_entity, cm_ripple_set,
     enterprise_ripple_set, enterprise_demand) = load_data(args, 'cm_browse_enterprise.txt', 'enterprise_browse_cm.txt',
                                                           'cooperation_info.txt', 'vocab_matrix.csv',
                                                           'technique_matrix.csv', 'kg_final.dict', 'data_info.dict',
                                                           args.re_get_ripple_set, 'cm_ripple_set.dict',
                                                           'enterprise_ripple_set.dict', 'cm_info.csv',
                                                           'enterprise_info.csv', 'enterprise_demand.csv',
                                                           'entity_to_ix.dict', 'ix_to_entity.dict')
    train(args, train_cm_browse_enterprise, valid_cm_browse_enterprise, train_enterprise_browse_cm,
          valid_enterprise_browse_cm, train_cooperate_info, valid_cooperate_info, vocab_matrix, technique_matrix,
          data_info, cm_ripple_set, enterprise_ripple_set)
    (test_loss, test_cbe_loss, test_ebc_loss, test_coo_loss, test_text_loss1, test_text_loss2, test_cbe_auc,
     test_cbe_acc, test_ebc_auc, test_ebc_acc, test_coo_auc, test_coo_acc) = valid(args, test_cm_browse_enterprise,
                                                                                   test_enterprise_browse_cm,
                                                                                   test_cooperate_info, vocab_matrix,
                                                                                   technique_matrix, data_info,
                                                                                   cm_ripple_set,
                                                                                   enterprise_ripple_set)
    print('Testing: loss: %f, cbe_loss: %f, ebc_loss: %f, coo_loss: %f, text_loss1: %f, text_loss2: %f, cbe_auc: %f, '
          'cbe_acc: %f, ebc_auc: %f, ebc_acc: %f, coo_auc: %f, coo_acc: %f'
          % (test_loss, test_cbe_loss, test_ebc_loss, test_coo_loss, test_text_loss1, test_text_loss2, test_cbe_auc,
             test_cbe_acc, test_ebc_auc, test_ebc_acc, test_coo_auc, test_coo_acc))
    with open('../result/main_result.txt', 'a') as fp:
        fp.write('Testing: loss: %f, cbe_loss: %f, ebc_loss: %f, coo_loss: %f, text_loss1: %f, text_loss2: %f, '
                 'cbe_auc: %f, cbe_acc: %f, ebc_auc: %f, ebc_acc: %f, coo_auc: %f, coo_acc: %f\n'
                 % (test_loss, test_cbe_loss, test_ebc_loss, test_coo_loss, test_text_loss1, test_text_loss2,
                    test_cbe_auc, test_cbe_acc, test_ebc_auc, test_ebc_acc, test_coo_auc, test_coo_acc))


if __name__ == '__main__':
    main()
