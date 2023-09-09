import pandas as pd
import numpy as np
import re
from os import mkdir
import pickle
import jieba
import logging
jieba.setLogLevel(logging.INFO)

def data_preprocess(cm_info_file, enterprise_info_file, enterprise_demand_file, cm_browse_enterprise_file,
                    enterprise_browse_cm_file, cooperation_info_file):
    """
    remove noisy data
    """
    data_prefix = '../data/raw_data/'

    # Get information for researchers, enterprises and demands
    with open(data_prefix + cm_info_file, encoding='utf-8') as f:
        cm_info = pd.read_csv(f)
    with open(data_prefix + enterprise_info_file, encoding='utf-8') as f:
        enterprise_info = pd.read_csv(f)
    with open(data_prefix + enterprise_demand_file, encoding='utf-8') as f:
        enterprise_demand = pd.read_csv(f)

    # Get three interaction datasets
    with open(data_prefix + cm_browse_enterprise_file, encoding='utf-8') as f:
        cm_browse_enterprise = pd.read_csv(f)
    with open(data_prefix + enterprise_browse_cm_file, encoding='utf-8') as f:
        enterprise_browse_cm = pd.read_csv(f)
    with open(data_prefix + cooperation_info_file, encoding='utf-8') as f:
        cooperation_info = pd.read_csv(f)

    # Get researcher set, enterprise set and demand set
    cm_set = cm_info['code'].unique()
    enterprise_set = enterprise_info['code'].unique()
    demand_set = enterprise_demand['code'].unique()

    # Keep the latest information in the researcher information
    save_list = []
    max_ix = int()
    str = ''
    for code in cm_set:
        for i in range(len(cm_info[cm_info['code'] == code])):
            if i == 0:
                str = cm_info[cm_info['code'] == code].iloc[i]['examine_date']
                max_ix = cm_info[cm_info['code'] == code].iloc[i].name
            else:
                if str < cm_info[cm_info['code'] == code].iloc[i]['examine_date']:
                    str = cm_info[cm_info['code'] == code].iloc[i]['examine_date']
                    max_ix = cm_info[cm_info['code'] == code].iloc[i].name
        save_list.append(max_ix)
    cm_info = cm_info.loc[save_list]
    cm_info.to_csv(data_prefix + cm_info_file, index=False, encoding='utf-8')

    # Keep the latest information in the enterprise information
    save_list = []
    for code in enterprise_set:
        for i in range(len(enterprise_info[enterprise_info['code'] == code])):
            if i == 0:
                str = enterprise_info[enterprise_info['code'] == code].iloc[i]['examine_date']
                max_ix = enterprise_info[enterprise_info['code'] == code].iloc[i].name
            else:
                if str < enterprise_info[enterprise_info['code'] == code].iloc[i]['examine_date']:
                    str = enterprise_info[enterprise_info['code'] == code].iloc[i]['examine_date']
                    max_ix = enterprise_info[enterprise_info['code'] == code].iloc[i].name
        save_list.append(max_ix)
    enterprise_info = enterprise_info.loc[save_list]
    enterprise_info.to_csv(data_prefix + enterprise_info_file, index=False, encoding='utf-8')

    # remove noisy data from interaction datasets
    drop_list = []
    for i in range(len(cm_browse_enterprise)):
        if cm_browse_enterprise.iloc[i]['cm_code'] not in cm_set or cm_browse_enterprise.iloc[i]['enterprise_code'] \
                not in enterprise_set or cm_browse_enterprise.iloc[i]['demand_code'] not in demand_set:
            drop_list.append(i)
    cm_browse_enterprise = cm_browse_enterprise.drop(drop_list)
    cm_browse_enterprise.to_csv(data_prefix + cm_browse_enterprise_file, index=False, encoding='utf-8')
    drop_list = []
    for i in range(len(enterprise_browse_cm)):
        if enterprise_browse_cm.iloc[i]['cm_code'] not in cm_set or enterprise_browse_cm.iloc[i][
            'enterprise_code'] not in enterprise_set:
            drop_list.append(i)
    enterprise_browse_cm = enterprise_browse_cm.drop(drop_list)
    enterprise_browse_cm.to_csv(data_prefix + enterprise_browse_cm_file, index=False, encoding='utf-8')
    drop_list = []
    for i in range(len(cooperation_info)):
        if cooperation_info.iloc[i]['cm_code'] not in cm_set or cooperation_info.iloc[i]['enterprise_code'] \
                not in enterprise_set or cooperation_info.iloc[i]['demand_code'] not in demand_set:
            drop_list.append(i)
    cooperation_info = cooperation_info.drop(drop_list)
    cooperation_info.to_csv(data_prefix + cooperation_info_file, index=False, encoding='utf-8')


def train_valid_test_split(data_file):
    """
    Divide the dataset, train : valid : test = 6:2:2
    """
    data_prefix = '../data/raw_data/'
    if data_file == 'cooperation_info.csv':
        with open(data_prefix + data_file, encoding='utf-8') as f:
            data_all = pd.read_csv(f)
            data_all = data_all[['cm_code', 'enterprise_code', 'demand_code']]
            data_all['relation'] = ['cm_cooperate_enterprise'] * len(data_all)
            data_all = data_all[['cm_code', 'relation', 'enterprise_code', 'demand_code']]
            data = data_all.drop_duplicates()
            print('cooperation unique number: ', len(data))
    elif data_file == 'cm_browse_enterprise.csv':
        with open(data_prefix + data_file, encoding='utf-8') as f:
            data_all = pd.read_csv(f)
            data_all = data_all[['cm_code', 'enterprise_code', 'demand_code']]
            data_all['relation'] = ['cm_browse_enterprise'] * len(data_all)
            data_all = data_all[['cm_code', 'relation', 'enterprise_code', 'demand_code']]
            data = data_all.drop_duplicates()
            print('cm browse enterprise unique number: ', len(data))
    else:
        with open(data_prefix + data_file, encoding='utf-8') as f:
            data_all = pd.read_csv(f)
            data_all = data_all[['enterprise_code', 'cm_code']]
            data_all['relation'] = ['enterprise_browse_cm'] * len(data_all)
            data_all = data_all[['enterprise_code', 'relation', 'cm_code']]
            data = data_all.drop_duplicates()
            print('enterprise browse cm unique number: ', len(data))

    # 6:2:2
    valid_test = np.random.choice(len(data), size=int(0.4 * len(data)), replace=False)
    valid_test_idx = np.zeros(len(data), dtype=bool)
    valid_test_idx[valid_test] = True
    data_valid_test = data[valid_test_idx]
    data_train = data[~valid_test_idx]

    num_data_valid_test = data_valid_test.shape[0]
    test = np.random.choice(num_data_valid_test, size=int(0.50 * num_data_valid_test), replace=False)
    test_idx = np.zeros(num_data_valid_test, dtype=bool)
    test_idx[test] = True
    data_test = data_valid_test[test_idx]
    data_valid = data_valid_test[~test_idx]

    train_ix = []
    valid_ix = []
    test_ix = []
    for i in range(len(data_all)):
        if np.sum(np.sum(data_all.values[i] == data_train.values, axis=1) == data_all.shape[1]) > 0:
            train_ix.append(i)
        elif np.sum(np.sum(data_all.values[i] == data_valid.values, axis=1) == data_all.shape[1]) > 0:
            valid_ix.append(i)
        elif np.sum(np.sum(data_all.values[i] == data_test.values, axis=1) == data_all.shape[1]) > 0:
            test_ix.append(i)
        else:
            print("################# Data Error! #################")
    data_train_all = data_all.iloc[train_ix]
    data_valid_all = data_all.iloc[valid_ix].drop_duplicates()
    data_test_all = data_all.iloc[test_ix].drop_duplicates()

    data_train_all.to_csv(data_prefix + 'raw_train_' + data_file.split('.')[0] + '.txt', index=False, header=None,
                          sep='\t')
    data_valid_all.to_csv(data_prefix + 'raw_valid_' + data_file.split('.')[0] + '.txt', index=False, header=None,
                          sep='\t')
    data_test_all.to_csv(data_prefix + 'raw_test_' + data_file.split('.')[0] + '.txt', index=False, header=None,
                         sep='\t')


def create_directory(dir):
    """
    create a directory
    """
    print("Creating directory %s" % dir)
    try:
        mkdir(dir)
    except FileExistsError:
        print("Directory already exists")


def raw_kg_construct(export_dir, kg_file, cm_info_file, enterprise_info_file, enterprise_demand_file,
                     cm_browse_enterprise_file, enterprise_browse_cm_file, cooperation_info_file):
    """
    construct knowledge graph
    """
    raw_data_prefix = '../data/raw_data/'
    raw_train_prefix = '../data/raw_data/raw_train_'
    writer = open(export_dir + kg_file, 'w', encoding='utf-8')
    with open(raw_data_prefix + cm_info_file, encoding='utf-8') as f:
        cm_info = pd.read_csv(f)
    with open(raw_data_prefix + enterprise_info_file, encoding='utf-8') as f:
        enterprise_info = pd.read_csv(f)
    with open(raw_data_prefix + enterprise_demand_file, encoding='utf-8') as f:
        enterprise_demand = pd.read_csv(f)
    with open(raw_train_prefix + cm_browse_enterprise_file, encoding='utf-8') as f:
        cm_browse_enterprise_all = pd.read_table(f, sep='\t', header=None)
        cm_browse_enterprise_all.columns = ['cm_code', 'cm_browse_enterprise', 'enterprise_code', 'demand_code']
        cm_browse_enterprise = cm_browse_enterprise_all.drop_duplicates()
    with open(raw_train_prefix + enterprise_browse_cm_file, encoding='utf-8') as f:
        enterprise_browse_cm = pd.read_table(f, sep='\t', header=None).drop_duplicates()
        enterprise_browse_cm.columns = ['enterprise_code', 'enterprise_browse_cm', 'cm_code']
    with open(raw_train_prefix + cooperation_info_file, encoding='utf-8') as f:
        cooperation_info_all = pd.read_table(f, sep='\t', header=None)
        cooperation_info_all.columns = ['cm_code', 'cm_cooperate_enterprise', 'enterprise_code', 'demand_code']
        cooperation_info = cooperation_info_all.drop_duplicates()

    for i in range(len(cm_info)):
        writer.write('%s\t%s\t%s\n' % ((cm_info.iloc[i]['code'], 'cm'), 'cm_major', cm_info.iloc[i]['major']))
        for keyword in re.split('[；、，, ]', cm_info.iloc[i]['keyword']):
            writer.write('%s\t%s\t%s\n' % ((cm_info.iloc[i]['code'], 'cm'), 'cm_keyword', keyword))
        writer.write('%s\t%s\t%s\n' % ((cm_info.iloc[i]['code'], 'cm'), 'cm_type', cm_info.iloc[i]['type']))
        writer.write('%s\t%s\t%s\n' % ((cm_info.iloc[i]['code'], 'cm'), 'cm_academic', cm_info.iloc[i]['academic']))
        for technique in re.split('[ ]', cm_info.iloc[i]['technique']):
            writer.write('%s\t%s\t%s\n' % ((cm_info.iloc[i]['code'], 'cm'), 'cm_technique', technique))

    for i in range(len(enterprise_info)):
        writer.write('%s\t%s\t%s\n' % ((enterprise_info.iloc[i]['code'], 'enterprise'), 'enterprise_property',
                                       enterprise_info.iloc[i]['properties']))
        writer.write('%s\t%s\t%s\n' % ((enterprise_info.iloc[i]['code'], 'enterprise'), 'enterprise_type',
                                       enterprise_info.iloc[i]['type']))
        for field in re.split('[ ]', enterprise_info.iloc[i]['fields']):
            writer.write('%s\t%s\t%s\n' % ((enterprise_info.iloc[i]['code'], 'enterprise'), 'enterprise_field', field))
        for product in re.split('[；、，, ]', enterprise_info.iloc[i]['product_synopsis']):
            writer.write('%s\t%s\t%s\n' % ((enterprise_info.iloc[i]['code'], 'enterprise'), 'enterprise_product',
                                           product))

    for i in range(len(enterprise_demand)):
        writer.write('%s\t%s\t%s\n' % ((enterprise_demand.iloc[i]['code'], 'demand'), 'demand_belong_enterprise',
                                       (enterprise_demand.iloc[i]['enterprise_code'], 'enterprise')))
        writer.write('%s\t%s\t%s\n' % ((enterprise_demand.iloc[i]['enterprise_code'], 'enterprise'),
                                       'enterprise_has_demand', (enterprise_demand.iloc[i]['code'], 'demand')))
        for keyword in re.split('[；、，, ]', enterprise_demand.iloc[i]['keyword']):
            writer.write('%s\t%s\t%s\n' % ((enterprise_demand.iloc[i]['code'], 'demand'), 'demand_keyword', keyword))
        for field in re.split('[；、，, ]', enterprise_demand.iloc[i]['fields']):
            writer.write('%s\t%s\t%s\n' % ((enterprise_demand.iloc[i]['code'], 'demand'), 'demand_field', field))
        writer.write('%s\t%s\t%s\n' % ((enterprise_demand.iloc[i]['code'], 'demand'), 'demand_type',
                                       enterprise_demand.iloc[i]['type']))
    for i in range(len(cm_browse_enterprise)):
        writer.write('%s\t%s\t%s\n' % ((cm_browse_enterprise.iloc[i]['cm_code'], 'cm'), 'cm_browse_enterprise',
                                       (cm_browse_enterprise.iloc[i]['enterprise_code'], 'enterprise')))
        writer.write('%s\t%s\t%s\n' % ((cm_browse_enterprise.iloc[i]['cm_code'], 'cm'), 'cm_browse_demand',
                                       (cm_browse_enterprise.iloc[i]['demand_code'], 'demand')))
    for i in range(len(enterprise_browse_cm)):
        writer.write('%s\t%s\t%s\n' % ((enterprise_browse_cm.iloc[i]['enterprise_code'], 'enterprise'),
                                       'enterprise_browse_cm', (enterprise_browse_cm.iloc[i]['cm_code'], 'cm')))
    for i in range(len(cooperation_info)):
        writer.write('%s\t%s\t%s\n' % ((cooperation_info.iloc[i]['cm_code'], 'cm'), 'cm_cooperate_enterprise',
                                       (cooperation_info.iloc[i]['enterprise_code'], 'enterprise')))
        writer.write('%s\t%s\t%s\n' % ((cooperation_info.iloc[i]['enterprise_code'], 'enterprise'),
                                       'enterprise_cooperate_cm', (cooperation_info.iloc[i]['cm_code'], 'cm')))
        writer.write('%s\t%s\t%s\n' % ((cooperation_info.iloc[i]['cm_code'], 'cm'), 'cm_finish_demand',
                                       (cooperation_info.iloc[i]['demand_code'], 'demand')))
    writer.close()


def sample_train_data(cm_browse_enterprise_file, enterprise_browse_cm_file, cooperation_info_file):
    """
    sample the training data(cm_browse_enterprise、enterprise_browse_cm data、cooperation_info) to the same size
    """
    data_prefix = '../data/raw_data/raw_train_'
    with open(data_prefix + cm_browse_enterprise_file) as f:
        cm_browse_enterprise = pd.read_table(f, sep='\t', header=None)
    with open(data_prefix + enterprise_browse_cm_file) as f:
        enterprise_browse_cm = pd.read_table(f, sep='\t', header=None)
    with open(data_prefix + cooperation_info_file) as f:
        cooperation_info = pd.read_table(f, sep='\t', header=None)

    max_num = max(len(cm_browse_enterprise), len(enterprise_browse_cm), len(cooperation_info))
    if len(cm_browse_enterprise) < max_num:
        cm_browse_enterprise = cm_browse_enterprise.iloc[np.random.choice(len(cm_browse_enterprise), size=max_num,
                                                                          replace=True)]
    if len(enterprise_browse_cm) < max_num:
        enterprise_browse_cm = enterprise_browse_cm.iloc[np.random.choice(len(enterprise_browse_cm), size=max_num,
                                                                          replace=True)]
    if len(cooperation_info) < max_num:
        cooperation_info = cooperation_info.iloc[np.random.choice(len(cooperation_info), size=max_num, replace=True)]

    cm_browse_enterprise.to_csv(data_prefix + cm_browse_enterprise_file, index=False, header=None, sep='\t',
                                encoding='utf-8')
    enterprise_browse_cm.to_csv(data_prefix + enterprise_browse_cm_file, index=False, header=None, sep='\t',
                                encoding='utf-8')
    cooperation_info.to_csv(data_prefix + cooperation_info_file, index=False, header=None, sep='\t', encoding='utf-8')


def sample_neg(cm_info_file, enterprise_demand_file, cm_browse_enterprise_file, enterprise_browse_cm_file,
               cooperation_info_file):
    """
    sample the negative data for training and testing
    """
    raw_data_prefix = '../data/raw_data/'
    with open(raw_data_prefix + cm_info_file, encoding='utf-8') as f:
        cm_info = pd.read_csv(f)
    cm_set = set(cm_info['code'])
    with open(raw_data_prefix + enterprise_demand_file, encoding='utf-8') as f:
        enterprise_demand = pd.read_csv(f)
    demand_set = set(enterprise_demand['code'])

    do_sample(demand_set, enterprise_demand, cm_browse_enterprise_file, 'cm_browse_enterprise')
    do_sample(cm_set, enterprise_demand, enterprise_browse_cm_file, 'enterprise_browse_cm')
    do_sample(demand_set, enterprise_demand, cooperation_info_file, 'cm_cooperate_enterprise')


def do_sample(Set, enterprise_demand, data_file, relation):
    data_prefix = '../data/raw_data/raw_'
    for version in ['train', 'valid', 'test']:
        if relation == 'enterprise_browse_cm':
            with open(data_prefix + version + '_' + data_file) as f:
                data = pd.read_table(f, sep='\t', header=None)
                data.columns = ['A_code', relation, 'B_code']
                data = data[['A_code', 'B_code', relation]]
                data[data == relation] = 1
        else:
            with open(data_prefix + version + '_' + data_file) as f:
                data = pd.read_table(f, sep='\t', header=None)
                data.columns = ['A_code', relation, 'B_code', 'C_code']
                data = data[['A_code', 'B_code', 'C_code', relation]]
                data[data == relation] = 1
        for A in set(data['A_code']):
            num = len(data[data['A_code'] == A])
            if relation == 'enterprise_browse_cm':
                neg_set = Set - set(data[data['A_code'] == A]['B_code'])
                if num < len(neg_set):
                    for B in np.random.choice(list(neg_set), size=num, replace=False):
                        data = data.append([{'A_code': A, 'B_code': B, relation: 0}], ignore_index=True)
                else:
                    for B in np.random.choice(list(neg_set), size=num, replace=True):
                        data = data.append([{'A_code': A, 'B_code': B, relation: 0}], ignore_index=True)
            else:
                neg_set = Set - set(data[data['A_code'] == A]['C_code'])
                if num < len(neg_set):
                    for C in np.random.choice(list(neg_set), size=num, replace=False):
                        B = enterprise_demand[enterprise_demand['code'] == C]['enterprise_code'].values[0]
                        data = data.append([{'A_code': A, 'B_code': B, 'C_code': C, relation: 0}], ignore_index=True)
                else:
                    for C in np.random.choice(list(neg_set), size=num, replace=True):
                        B = enterprise_demand[enterprise_demand['code'] == C]['enterprise_code'].values[0]
                        data = data.append([{'A_code': A, 'B_code': B, 'C_code': C, relation: 0}], ignore_index=True)

        data.to_csv(data_prefix + 'sample_' + version + '_' + data_file, sep='\t', header=None, index=False,
                    encoding='utf-8')


def ix_mapping(mapping_export_dir, data_info_export_dir, cm_info_file, enterprise_info_file, enterprise_demand_file,
               kg_dir, kg_file, relation_to_ix_file, entity_to_ix_file, cm_vocab_to_ix_file,
               enterprise_vocab_to_ix_file, demand_vocab_to_ix_file, ix_to_relation_file, ix_to_entity_file,
               ix_to_cm_vocab_file, ix_to_enterprise_vocab_file, ix_to_demand_vocab_file, sentence_len, technique_num):
    """
    construct the mapping dictionary which is used to mapping id to index
    """
    raw_data_prefix = '../data/raw_data/'
    with open(kg_dir + kg_file, encoding='utf-8') as f:
        kg = pd.read_table(f, header=None, on_bad_lines='skip')
        kg.columns = ['head', 'relation', 'tail']
    with open(raw_data_prefix + cm_info_file, encoding='utf-8') as f:
        cm_info = pd.read_csv(f)
        cm_info = cm_info[['code', 'research']]
    with open(raw_data_prefix + enterprise_info_file, encoding='utf-8') as f:
        enterprise_info = pd.read_csv(f)
        enterprise_info = enterprise_info[['code', 'synopsis']]
    with open(raw_data_prefix + enterprise_demand_file, encoding='utf-8') as f:
        enterprise_demand = pd.read_csv(f)
        enterprise_demand = enterprise_demand[['code', 'synopsis']]

    cm_set = cm_info['code'].unique()
    enterprise_set = enterprise_info['code'].unique()
    demand_set = enterprise_demand['code'].unique()

    # Id-ix mappings
    entity_set = set(kg['head']) | set(kg['tail'])
    entity_set.add('<PAD/>')
    entity_num = len(entity_set)
    relation_set = set(kg['relation'])
    relation_num = len(relation_set)
    cm_vocab = set()
    cm_vocab.add('<PAD/>')
    for i in range(len(cm_info)):
        cm_vocab = cm_vocab | set(jieba.cut(cm_info.iloc[i]['research']))
    cm_vocab_num = len(cm_vocab)
    enterprise_vocab = set()
    enterprise_vocab.add('<PAD/>')
    for i in range(len(enterprise_info)):
        enterprise_vocab = enterprise_vocab | set(jieba.cut(enterprise_info.iloc[i]['synopsis']))
    enterprise_vocab_num = len(enterprise_vocab)
    demand_vocab = set()
    demand_vocab.add('<PAD/>')
    for i in range(len(enterprise_demand)):
        demand_vocab = demand_vocab | set(jieba.cut(enterprise_demand.iloc[i]['synopsis']))
    demand_vocab_num = len(demand_vocab)

    data_info = {'entity_num': entity_num, 'relation_num': relation_num, 'cm_vocab_num': cm_vocab_num,
                 'enterprise_vocab_num': enterprise_vocab_num, 'demand_vocab_num': demand_vocab_num,
                 'sentence_len': sentence_len, 'technique_num': technique_num}
    with open(data_info_export_dir + 'data_info.dict', 'wb') as handle:
        pickle.dump(data_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    entity_to_ix = {(cm, 'cm'): ix for ix, cm in enumerate(cm_set)}
    entity_to_ix.update({(enterprise, 'enterprise'): ix + len(cm_set) for ix, enterprise in enumerate(enterprise_set)})
    entity_to_ix.update({(demand, 'demand'): ix + len(cm_set) + len(enterprise_set) for ix, demand in enumerate(demand_set)})
    rep_num = 0
    for ix, entity in enumerate(entity_set):
        try:
            entity = eval(entity)
        except:
            entity = entity
        if entity in entity_to_ix:
            rep_num += 1
            continue
        else:
            entity_to_ix.update({entity: ix + len(cm_set) + len(enterprise_set) + len(demand_set) - rep_num})
    relation_to_ix = {relation: ix for ix, relation in enumerate(relation_set)}
    cm_vocab_to_ix = {word: ix for ix, word in enumerate(cm_vocab)}
    enterprise_vocab_to_ix = {word: ix for ix, word in enumerate(enterprise_vocab)}
    demand_vocab_to_ix = {word: ix for ix, word in enumerate(demand_vocab)}

    # Ix-id mappings
    ix_to_relation = {v: k for k, v in relation_to_ix.items()}
    ix_to_entity = {v: k for k, v in entity_to_ix.items()}
    ix_to_cm_vocab = {v: k for k, v in cm_vocab_to_ix.items()}
    ix_to_enterprise_vocab = {v: k for k, v in enterprise_vocab_to_ix.items()}
    ix_to_demand_vocab = {v: k for k, v in demand_vocab_to_ix.items()}

    # Export mappings
    with open(mapping_export_dir + relation_to_ix_file, 'wb') as handle:
        pickle.dump(relation_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + entity_to_ix_file, 'wb') as handle:
        pickle.dump(entity_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + cm_vocab_to_ix_file, 'wb') as handle:
        pickle.dump(cm_vocab_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + enterprise_vocab_to_ix_file, 'wb') as handle:
        pickle.dump(enterprise_vocab_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + demand_vocab_to_ix_file, 'wb') as handle:
        pickle.dump(demand_vocab_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + ix_to_relation_file, 'wb') as handle:
        pickle.dump(ix_to_relation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + ix_to_entity_file, 'wb') as handle:
        pickle.dump(ix_to_entity, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + ix_to_cm_vocab_file, 'wb') as handle:
        pickle.dump(ix_to_cm_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + ix_to_enterprise_vocab_file, 'wb') as handle:
        pickle.dump(ix_to_enterprise_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + ix_to_demand_vocab_file, 'wb') as handle:
        pickle.dump(ix_to_demand_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_to_ixs(entity_to_ix_file, relation_to_ix_file, export_dir, kg_file, kg_ix_file, cm_browse_enterprise_file,
                   enterprise_browse_cm_file, cooperation_info_file):
    """
    mapping id to index
    """
    mapping_data_prefix = '../data/ix_mapping/'
    kg_data_prefix = '../data/kg/'

    writer = open(kg_data_prefix + kg_ix_file, 'w')
    with open(mapping_data_prefix + entity_to_ix_file, 'rb') as handle:
        entity_to_ix = pickle.load(handle)
    with open(mapping_data_prefix + relation_to_ix_file, 'rb') as handle:
        relation_to_ix = pickle.load(handle)
    with open(kg_data_prefix + kg_file, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            try:
                head = eval(line[0])
            except:
                head = line[0]
            try:
                tail = eval(line[2])
            except:
                tail = line[2]
            writer.write('%d\t%d\t%d\n' % (entity_to_ix[head], relation_to_ix[line[1]], entity_to_ix[tail]))
    writer.close()

    do_convert(cm_browse_enterprise_file, export_dir, entity_to_ix, 'cbe')
    do_convert(enterprise_browse_cm_file, export_dir, entity_to_ix, 'ebc')
    do_convert(cooperation_info_file, export_dir, entity_to_ix, 'coo')


def do_convert(data_file, export_dir, entity_to_ix, type):
    data_prefix = '../data/raw_data/raw_sample_'
    for version in ['train', 'valid', 'test']:
        with open(data_prefix + version + '_' + data_file, encoding='utf-8') as f:
            writer = open(export_dir + version + '_' + data_file, 'w')
            for line in f.readlines():
                line = line.strip('\n').split('\t')
                if type == 'cbe':
                    writer.write('%d\t%d\t%d\t%d\n' % (entity_to_ix[(line[0], 'cm')],
                                                       entity_to_ix[(line[1], 'enterprise')],
                                                       entity_to_ix[(line[2], 'demand')], int(line[3])))
                elif type == 'ebc':
                    writer.write('%d\t%d\t%d\n' % (entity_to_ix[(line[0], 'enterprise')], entity_to_ix[(line[1], 'cm')],
                                                   int(line[2])))
                elif type == 'coo':
                    writer.write('%d\t%d\t%d\t%d\n' % (entity_to_ix[(line[0], 'cm')],
                                                       entity_to_ix[(line[1], 'enterprise')],
                                                       entity_to_ix[(line[2], 'demand')], int(line[3])))
                else:
                    print("################# Type Error! #################")
            writer.close()


def construct_kg_ix(kg_file, kg_final):
    """
    mapping the id to index in knowledge graph
    """
    print('constructing knowledge graph ...')

    # reading kg file
    data_prefix = '../data/kg/'
    kg = {}
    with open(data_prefix + kg_file) as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            if int(line[0]) not in kg:
                kg[int(line[0])] = []
            kg[int(line[0])].append((int(line[2]), int(line[1])))
    with open(data_prefix + kg_final, 'wb') as handle:
        pickle.dump(kg, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_vocab_matrix(ix_to_entity_file, cm_vocab_to_ix_file, cm_info_file, enterprise_vocab_to_ix_file,
                     enterprise_info_file, demand_vocab_to_ix_file, enterprise_demand_file, sentence_len, export_dir):
    """
    get the vocab matrix for cm and enterprise, one row represent one cm/enterprise
    """
    mapping_data_prefix = '../data/ix_mapping/'
    raw_data_prefix = '../data/raw_data/'
    with open(mapping_data_prefix + ix_to_entity_file, 'rb') as f:
        ix_to_entity = pickle.load(f)
    with open(mapping_data_prefix + cm_vocab_to_ix_file, 'rb') as f:
        cm_vocab_to_ix = pickle.load(f)
    with open(mapping_data_prefix + enterprise_vocab_to_ix_file, 'rb') as f:
        enterprise_vocab_to_ix = pickle.load(f)
    with open(mapping_data_prefix + demand_vocab_to_ix_file, 'rb') as f:
        demand_vocab_to_ix = pickle.load(f)
    with open(raw_data_prefix + cm_info_file, encoding='utf-8') as f:
        cm_info = pd.read_csv(f)
    with open(raw_data_prefix + enterprise_info_file, encoding='utf-8') as f:
        enterprise_info = pd.read_csv(f)
    with open(raw_data_prefix + enterprise_demand_file, encoding='utf-8') as f:
        enterprise_demand = pd.read_csv(f)
    vocab_matrix = np.empty((0, sentence_len + 1))

    cm_num = len(cm_info['code'].unique())
    enterprise_num = len(enterprise_info['code'].unique())
    demand_num = len(enterprise_demand['code'].unique())

    for i in range(cm_num):
        arr = np.zeros(sentence_len + 1)
        code = ix_to_entity[i][0]
        word_list = list(jieba.cut(cm_info[cm_info['code'] == code]['research'].values[0]))
        true_len = len(word_list)
        if sentence_len <= true_len:
            word_list = word_list[:sentence_len]
            arr[0] = sentence_len
        else:
            arr[0] = len(word_list)
            num_padding = sentence_len - arr[0]
            word_list = word_list + ['<PAD/>'] * int(num_padding)
        for i in range(sentence_len):
            arr[i + 1] = cm_vocab_to_ix[word_list[i]]
        vocab_matrix = np.row_stack((vocab_matrix, arr))

    for i in range(enterprise_num):
        arr = np.zeros(sentence_len + 1)
        code = ix_to_entity[i + cm_num][0]
        word_list = list(jieba.cut(enterprise_info[enterprise_info['code'] == code]['synopsis'].values[0]))
        true_len = len(word_list)
        if sentence_len <= true_len:
            word_list = word_list[:sentence_len]
            arr[0] = sentence_len
        else:
            arr[0] = len(word_list)
            num_padding = sentence_len - arr[0]
            word_list = word_list + ['<PAD/>'] * int(num_padding)
        for i in range(sentence_len):
            arr[i + 1] = enterprise_vocab_to_ix[word_list[i]]
        vocab_matrix = np.row_stack((vocab_matrix, arr))

    for i in range(demand_num):
        arr = np.zeros(sentence_len + 1)
        code = ix_to_entity[i + cm_num + enterprise_num][0]
        word_list = list(jieba.cut(enterprise_demand[enterprise_demand['code'] == code]['synopsis'].values[0]))
        true_len = len(word_list)
        if sentence_len <= true_len:
            word_list = word_list[:sentence_len]
            arr[0] = sentence_len
        else:
            arr[0] = len(word_list)
            num_padding = sentence_len - arr[0]
            word_list = word_list + ['<PAD/>'] * int(num_padding)
        for i in range(sentence_len):
            arr[i + 1] = demand_vocab_to_ix[word_list[i]]
        vocab_matrix = np.row_stack((vocab_matrix, arr))

    vocab_matrix = vocab_matrix.astype(np.int64)
    np.savetxt(export_dir + 'vocab_matrix.csv', vocab_matrix, delimiter=',')


def get_technique_matrix(entity_to_ix_file, relation_to_ix_file, kg_file, technique_num, export_dir):
    """
    get the technique matrix for cm and enterprise, one row represent one cm/enterprise
    """
    mapping_data_prefix = '../data/ix_mapping/'
    kg_data_prefix = '../data/kg/'
    with open(mapping_data_prefix + entity_to_ix_file, 'rb') as f:
        entity_to_ix = pickle.load(f)
    with open(mapping_data_prefix + relation_to_ix_file, 'rb') as f:
        relation_to_ix = pickle.load(f)
    with open(kg_data_prefix + kg_file, encoding='utf-8') as f:
        kg = pd.read_table(f, sep='\t', header=None)
        kg.columns = ['head', 'relation', 'tail']
    technique_matrix = np.empty((0, technique_num + 1))
    cm_technique = kg[kg['relation'] == relation_to_ix['cm_technique']]
    enterprise_field = kg[kg['relation'] == relation_to_ix['enterprise_field']]
    cm_num = len(cm_technique['head'].unique())
    enterprise_num = len(enterprise_field['head'].unique())

    for i in range(cm_num):
        arr = np.zeros(technique_num + 1)
        true_num = len(cm_technique[cm_technique['head'] == i])
        if true_num <= technique_num:
            arr[0] = true_num
            for j in range(technique_num):
                if j < true_num:
                    arr[j+1] = cm_technique[cm_technique['head'] == i].iloc[j]['tail']
                else:
                    arr[j + 1] = entity_to_ix['<PAD/>']
        else:
            arr[0] = technique_num
            ix = np.arange(true_num)
            np.random.shuffle(ix)
            ix = ix[:technique_num]
            for j in range(technique_num):
                arr[j + 1] = cm_technique[cm_technique['head'] == i].iloc[ix[j]]['tail']
        technique_matrix = np.row_stack((technique_matrix, arr))

    for i in range(enterprise_num):
        arr = np.zeros(technique_num + 1)
        true_num = len(enterprise_field[enterprise_field['head'] == i + cm_num])
        if true_num <= technique_num:
            arr[0] = true_num
            for j in range(technique_num):
                if j < true_num:
                    arr[j+1] = enterprise_field[enterprise_field['head'] == i + cm_num].iloc[j]['tail']
                else:
                    arr[j + 1] = entity_to_ix['<PAD/>']
        else:
            arr[0] = technique_num
            ix = np.arange(true_num)
            np.random.shuffle(ix)
            ix = ix[:technique_num]
            for j in range(technique_num):
                arr[j + 1] = enterprise_field[enterprise_field['head'] == i + cm_num].iloc[ix[j]]['tail']
        technique_matrix = np.row_stack((technique_matrix, arr))

    technique_matrix = technique_matrix.astype(np.int64)
    np.savetxt(export_dir + 'technique_matrix.csv', technique_matrix, delimiter=',')


if __name__ == '__main__':

    kg_dir = '../data/kg/'
    ix_mapping_dir = '../data/ix_mapping/'
    data_info_dir = '../data/data_info/'
    data_ix_dir = '../data/data_ix/'
    sentence_len = 50
    technique_num = 10
    create_directory(kg_dir)
    create_directory(ix_mapping_dir)
    create_directory(data_info_dir)
    create_directory(data_ix_dir)

    data_preprocess('cm_info.csv', 'enterprise_info.csv', 'enterprise_demand.csv', 'cm_browse_enterprise.csv',
                    'enterprise_browse_cm.csv', 'cooperation_info.csv')
    train_valid_test_split('cm_browse_enterprise.csv')
    train_valid_test_split('enterprise_browse_cm.csv')
    train_valid_test_split('cooperation_info.csv')
    raw_kg_construct(kg_dir, 'raw_kg.txt', 'cm_info.csv', 'enterprise_info.csv', 'enterprise_demand.csv',
                     'cm_browse_enterprise.txt', 'enterprise_browse_cm.txt', 'cooperation_info.txt')
    sample_train_data('cm_browse_enterprise.txt', 'enterprise_browse_cm.txt', 'cooperation_info.txt')
    sample_neg('cm_info.csv', 'enterprise_demand.csv', 'cm_browse_enterprise.txt', 'enterprise_browse_cm.txt',
               'cooperation_info.txt')
    ix_mapping(ix_mapping_dir, data_info_dir, 'cm_info.csv', 'enterprise_info.csv', 'enterprise_demand.csv', kg_dir,
               'raw_kg.txt', 'relation_to_ix.dict', 'entity_to_ix.dict', 'cm_vocab_to_ix.dict',
               'enterprise_vocab_to_ix.dict', 'demand_vocab_to_ix.dict', 'ix_to_relation.dict', 'ix_to_entity.dict',
               'ix_to_cm_vocab.dict', 'ix_to_enterprise_vocab.dict', 'ix_to_demand_vocab.dict', sentence_len,
               technique_num)
    convert_to_ixs('entity_to_ix.dict', 'relation_to_ix.dict', data_ix_dir, 'raw_kg.txt', 'kg_ix.txt',
                   'cm_browse_enterprise.txt', 'enterprise_browse_cm.txt', 'cooperation_info.txt')
    construct_kg_ix('kg_ix.txt', 'kg_final.dict')
    get_vocab_matrix('ix_to_entity.dict', 'cm_vocab_to_ix.dict', 'cm_info.csv', 'enterprise_vocab_to_ix.dict',
                     'enterprise_info.csv', 'demand_vocab_to_ix.dict', 'enterprise_demand.csv', sentence_len,
                     data_ix_dir)
    get_technique_matrix('entity_to_ix.dict', 'relation_to_ix.dict', 'kg_ix.txt', technique_num, data_ix_dir)

