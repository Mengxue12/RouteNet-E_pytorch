import copy
import numpy as np

import networkx as nx
from datanetAPI import DatanetAPI
import torch

EXTERNAL_DISTRIBUTIONS = ['AR1-0', 'AR1-1']


def generator(data_dir, label, shuffle=False):
    tool = DatanetAPI(data_dir, shuffle=shuffle)
    it = iter(tool)
    num_samples = 0
    for sample in it:
        try:
            HG = network_to_hypergraph(sample=sample)
            num_samples += 1
            n_p = 0
            n_l = 0
            mapping = {}
            for entity in list(HG.nodes()):
                if entity.startswith('p'):
                    mapping[entity] = ('p_{}'.format(n_p))
                    n_p += 1
                elif entity.startswith('l'):
                    mapping[entity] = ('l_{}'.format(n_l))
                    n_l += 1

            D_G = nx.relabel_nodes(HG, mapping)

            link_to_path = []
            path_ids = []
            sequence_path = []
            for i in range(n_p):
                seq_len = 0
                for elem in D_G['p_{}'.format(i)]:
                    link_to_path.append(int(elem.replace('l_', '')))
                    seq_len += 1
                path_ids.extend(np.full(seq_len, i))
                sequence_path.extend(range(seq_len))

            path_to_link = []
            sequence_links = []
            for i in range(n_l):
                seq_len = 0
                for elem in D_G['l_{}'.format(i)]:
                    path_to_link.append(int(elem.replace('p_', '')))
                    seq_len += 1
                sequence_links.extend(np.full(seq_len, i))

            if 0 in list(nx.get_node_attributes(D_G, 'jitter').values()) or 0 in list(
                    nx.get_node_attributes(D_G, 'delay').values()):
                continue

            yield {"traffic": list(nx.get_node_attributes(D_G, 'traffic').values()),
                   "packets": list(nx.get_node_attributes(D_G, 'packets').values()),
                   "time_dist_params": list(nx.get_node_attributes(D_G, 'time_dist_params').values()),
                   "capacity": list(nx.get_node_attributes(D_G, 'capacity').values()),
                   "link_to_path": link_to_path,
                   "path_to_link": path_to_link,
                   "path_ids": path_ids,
                   "sequence_links": sequence_links,
                   "sequence_path": sequence_path,
                   "n_links": n_l,
                   "n_paths": n_p
                   }, list(nx.get_node_attributes(D_G, label).values())

        except Exception as e:
            pass


def network_to_hypergraph(sample):
    G = nx.DiGraph(sample.get_topology_object())
    R = sample.get_routing_matrix()
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()

    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:

                time_dist_params = [0] * 12
                flow = sample.get_traffic_matrix()[src, dst]['Flows'][0]
                if flow['TimeDist'].value != 6:
                    time_dist_params[flow['TimeDist'].value] = 1
                else:
                    time_dist_params[flow['TimeDist'].value + EXTERNAL_DISTRIBUTIONS.index(
                        flow['TimeDistParams']['Distribution'])] = 1

                idx = 7
                for k in flow['TimeDistParams']:
                    if isinstance(flow['TimeDistParams'][k], int) or isinstance(flow['TimeDistParams'][k], float):
                        time_dist_params[idx] = flow['TimeDistParams'][k]
                        idx += 1

                D_G.add_node('p_{}_{}'.format(src, dst),
                             traffic=T[src, dst]['Flows'][0]['AvgBw'],
                             packets=T[src, dst]['Flows'][0]['PktsGen'],
                             source=src,
                             destination=dst,
                             time_dist_params=time_dist_params,
                             drops=float(P[src, dst]['AggInfo']['PktsDrop']) / float(
                                 T[src, dst]['Flows'][0]['PktsGen']),
                             delay=float(P[src, dst]['AggInfo']['AvgDelay']),
                             jitter=float(P[src, dst]['AggInfo']['Jitter']))

                if G.has_edge(src, dst):
                    D_G.add_node('l_{}_{}'.format(src, dst),
                                 capacity=G.edges[src, dst]['bandwidth'])

                for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                    D_G.add_edge('p_{}_{}'.format(src, dst), 'l_{}_{}'.format(h_1, h_2))
                    D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'p_{}_{}'.format(src, dst))

    D_G.remove_nodes_from([node for node, out_degree in D_G.out_degree() if out_degree == 0])

    return D_G


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
        tempx, tempy = copy.deepcopy(next(self.data_gen))
        tempx['traffic'] = torch.tensor(tempx['traffic'], dtype=torch.float32)
        tempx['packets'] = torch.tensor(tempx['packets'], dtype=torch.float32)
        tempx['time_dist_params'] = torch.tensor(tempx['time_dist_params'], dtype=torch.float32)
        tempx['capacity'] = torch.tensor(tempx['capacity'], dtype=torch.float32)
        tempx['link_to_path'] = torch.tensor(tempx['link_to_path'], dtype=torch.int64)
        tempx['path_to_link'] = torch.tensor(tempx['path_to_link'], dtype=torch.int64)
        tempx['path_ids'] = torch.tensor(tempx['path_ids'], dtype=torch.int64)  # list of int
        tempx['sequence_links'] = torch.tensor(tempx['sequence_links'], dtype=torch.int64)
        tempx['sequence_path'] = torch.tensor(tempx['sequence_path'], dtype=torch.int32)
        tempx['n_links'] = torch.tensor(tempx['n_links'], dtype=torch.int32)  # integer
        tempx['n_paths'] = torch.tensor(tempx['n_paths'], dtype=torch.int32)  # integer
        tempy = torch.tensor(tempy, dtype=torch.float32)
        if self.transform:
            tempx, tempy = self.transform(tempx, tempy)
        return tempx, tempy

    def map(self, map_func):
        x, y = next(self.data_gen)
        return map_func(x, y)
    def __len__(self):
        return self.len
