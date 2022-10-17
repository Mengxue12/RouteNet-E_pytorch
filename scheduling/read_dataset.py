
import networkx as nx
import numpy as np
import torch
from datanetAPI import DatanetAPI  

POLICIES = np.array(['WFQ', 'SP', 'DRR'])


def sample_to_dependency_graph(sample):
    G = nx.DiGraph(sample.get_topology_object())
    R = sample.get_routing_matrix()
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()

    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                D_G.add_node('p_{}_{}'.format(src, dst),
                             traffic=T[src, dst]['Flows'][0]['AvgBw'],
                             packets=T[src, dst]['Flows'][0]['PktsGen'],
                             tos=int(T[src, dst]['Flows'][0]['ToS']),
                             source=src,
                             destination=dst,
                             drops=float(P[src, dst]['AggInfo']['PktsDrop']) / float(T[src, dst]['Flows'][0]['PktsGen']),
                             delay=float(P[src, dst]['AggInfo']['AvgDelay']),
                             jitter=float(P[src, dst]['AggInfo']['Jitter']))

                if G.has_edge(src, dst):
                    D_G.add_node('l_{}_{}'.format(src, dst),
                                 capacity=G.edges[src, dst]['bandwidth'],
                                 policy=np.where(G.nodes[src]['schedulingPolicy'] == POLICIES)[0][0])
                for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                    D_G.add_edge('p_{}_{}'.format(src, dst), 'l_{}_{}'.format(h_1, h_2))
                    if 'schedulingWeights' in G.nodes[h_1]:
                        q_w = str(G.nodes[h_1]['schedulingWeights']).split(',')
                    else:
                        q_w = ['-']
                    if 'tosToQoSqueue' in G.nodes[h_1]:
                        map = [m.split(',') for m in str(G.nodes[h_1]['tosToQoSqueue']).split(';')]
                    else:
                        map = [['0'], ['1'], ['2']]
                    q_n = 0
                    for q in range(G.nodes[h_1]['levelsQoS']):
                        D_G.add_node('q_{}_{}_{}'.format(h_1, h_2, q),
                                     priority=q_n,
                                     weight=float(q_w[q]) if q_w[0] != '-' else 0)
                        D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'q_{}_{}_{}'.format(h_1, h_2, q))
                        if str(int(T[src, dst]['Flows'][0]['ToS'])) in map[q]:
                            D_G.add_edge('p_{}_{}'.format(src, dst), 'q_{}_{}_{}'.format(h_1, h_2, q))
                            D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q), 'p_{}_{}'.format(src, dst))
                        q_n += 1

    D_G.remove_nodes_from([node for node, out_degree in D_G.out_degree() if out_degree == 0])

    n_q = 0
    n_p = 0
    n_l = 0
    mapping = {}
    for entity in list(D_G.nodes()):
        if entity.startswith('q'):
            mapping[entity] = ('q_{}'.format(n_q))
            n_q += 1
        elif entity.startswith('p'):
            mapping[entity] = ('p_{}'.format(n_p))
            n_p += 1
        elif entity.startswith('l'):
            mapping[entity] = ('l_{}'.format(n_l))
            n_l += 1

    D_G = nx.relabel_nodes(D_G, mapping)
    return D_G, n_q, n_p, n_l


def generator(data_dir, label, shuffle=False):
    tool = DatanetAPI(data_dir, [], shuffle)
    it = iter(tool)
    for sample in it:

        D_G, n_q, n_p, n_l = sample_to_dependency_graph(sample)

        link_to_path = np.array([], dtype='int32')
        queue_to_path = np.array([], dtype='int32')
        l_p_s = np.array([], dtype='int32')
        l_q_p = np.array([], dtype='int32')
        path_ids = np.array([], dtype='int32')
        for i in range(n_p):
            l_s_l = 0
            q_s_l = 0
            for elem in D_G['p_{}'.format(i)]:
                if elem.startswith('l_'):
                    link_to_path = np.append(link_to_path, int(elem.replace('l_', '')))
                    l_s_l += 1
                elif elem.startswith('q_'):
                    queue_to_path = np.append(queue_to_path, int(elem.replace('q_', '')))
                    q_s_l += 1
            path_ids = np.append(path_ids, [i] * q_s_l)
            l_p_s = np.append(l_p_s, range(l_s_l))
            l_q_p = np.append(l_q_p, range(q_s_l))

        path_to_queue = np.array([], dtype='int32')
        sequence_queues = np.array([], dtype='int32')
        for i in range(n_q):
            seq_len = 0
            for elem in D_G['q_{}'.format(i)]:
                path_to_queue = np.append(path_to_queue, int(elem.replace('p_', '')))
                seq_len += 1
            sequence_queues = np.append(sequence_queues, [i] * seq_len)

        queue_to_link = np.array([], dtype='int32')
        sequence_links = np.array([], dtype='int32')
        l_q_l = np.array([], dtype='int32')
        for i in range(n_l):
            seq_len = 0
            for elem in D_G['l_{}'.format(i)]:
                queue_to_link = np.append(queue_to_link, int(elem.replace('q_', '')))
                seq_len += 1
            sequence_links = np.append(sequence_links, [i] * seq_len)
            l_q_l = np.append(l_q_l, range(seq_len))

        if 0 in list(nx.get_node_attributes(D_G, 'jitter').values()) or 0 in list(nx.get_node_attributes(D_G, 'delay').values()):
            continue

        yield {"traffic": list(nx.get_node_attributes(D_G, 'traffic').values()),
               "packets": list(nx.get_node_attributes(D_G, 'packets').values()),
               "capacity": list(nx.get_node_attributes(D_G, 'capacity').values()),
               "policy": list(nx.get_node_attributes(D_G, 'policy').values()),
               "priority": list(nx.get_node_attributes(D_G, 'priority').values()),
               "weight": [w / 100 for w in list(nx.get_node_attributes(D_G, 'weight').values())],
               "link_to_path": link_to_path,
               "queue_to_path": queue_to_path,
               "path_to_queue": path_to_queue,
               "queue_to_link": queue_to_link,
               "sequence_queues": sequence_queues,
               "sequence_links": sequence_links,
               "path_ids": path_ids,
               "l_p_s": l_p_s,
               "l_q_p": l_q_p,
               "l_q_l": l_q_l,
               "n_queues": n_q,
               "n_links": n_l,
               "n_paths": n_p,
               }, list(nx.get_node_attributes(D_G, label).values())


# torch dataset
class GenDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label, shuffle, sample=4000, transform=None):
        super(GenDataset).__init__()
        self.data_dir = data_dir
        self.label = label
        self.shuffle = shuffle
        self.transform = transform
        self.len = sample
        self.data_gen = generator(data_dir, label, shuffle)
    def __getitem__(self, idx):
        tempx, tempy = next(self.data_gen)
        # for name in tempx:              
        #     tempx[name] = torch.tensor(tempx[name])
        tempx['traffic'] = torch.tensor(tempx['traffic'],dtype=torch.float32) # list of float
        tempx['packets'] = torch.tensor(tempx['packets'],dtype=torch.float32)
        tempx['capacity'] = torch.tensor(tempx['capacity'],dtype=torch.float32)
        tempx['policy'] = torch.tensor(tempx['policy'],dtype=torch.int64)# index
        tempx['priority'] = torch.tensor(tempx['priority'],dtype=torch.int64) 
        tempx['weight'] = torch.tensor(tempx['weight'],dtype=torch.float32) 

        tempx['link_to_path'] = torch.tensor(tempx['link_to_path'],dtype=torch.int64)
        tempx['queue_to_path'] = torch.tensor(tempx['queue_to_path'],dtype=torch.int64) 
        tempx['path_to_queue'] = torch.tensor(tempx['path_to_queue'],dtype=torch.int64)
        tempx['queue_to_link'] = torch.tensor(tempx['queue_to_link'],dtype=torch.int64)

        tempx['sequence_links'] = torch.tensor(tempx['sequence_links'],dtype=torch.int32)
        tempx['sequence_queues'] = torch.tensor(tempx['sequence_queues'],dtype=torch.int64)
        tempx['path_ids'] = torch.tensor(tempx['path_ids'],dtype=torch.int32)
        tempx['l_p_s'] = torch.tensor(tempx['l_p_s'],dtype=torch.int64)
        tempx['l_q_p'] = torch.tensor(tempx['l_q_p'],dtype=torch.int64) 
        tempx['l_q_l'] = torch.tensor(tempx['l_q_l'],dtype=torch.int64) 
        tempx['n_queues'] = torch.tensor(tempx['n_queues'],dtype=torch.int32)
        tempx['n_links'] = torch.tensor(tempx['n_links'],dtype=torch.int32) 
        tempx['n_paths'] = torch.tensor(tempx['n_paths'],dtype=torch.int32)
        tempy = torch.tensor(tempy)
        if self.transform:
            tempx, tempy = self.transform(tempx, tempy)
        return tempx, tempy
    def map(self, map_func):
        x, y = next(self.data_gen)
        return map_func(x, y)
    def __len__(self):
        return self.len