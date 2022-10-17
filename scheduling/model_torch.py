
from __future__ import print_function
import torch
import time

SEED=12245
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 'cpu') #

class GNN_Model(torch.nn.Module):
    """ Init method for the custom model.
    Args:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        output_units (int): Output units for the last readout's layer.
    Attributes:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        link_update (GRUCell): Link GRU Cell used in the Message Passing step.
        path_update (GRUCell): Path GRU Cell used in the Message Passing step.
        queue_update (GRUCell): Queue GRU Cell used in the Message Passing step.
        readout (Keras Model): Readout Neural Network. It expects as input the
                               path states and outputs the per-path delay.
    """

    def __init__(self, config, output_units=1):
        super(GNN_Model, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # GRU Cells used in the Message Passing step
        self.path_update_gru = torch.nn.GRU(
            int(self.config['HYPERPARAMETERS']['queue_state_dim']) +
            int(self.config['HYPERPARAMETERS']['link_state_dim']),
            int(self.config['HYPERPARAMETERS']['path_state_dim']))
        self.link_update_gru = torch.nn.GRU(
            int(self.config['HYPERPARAMETERS']['queue_state_dim']),
            int(self.config['HYPERPARAMETERS']['link_state_dim']))
        self.queue_update = torch.nn.GRUCell(
            int(self.config['HYPERPARAMETERS']['path_state_dim']),
            int(self.config['HYPERPARAMETERS']['queue_state_dim']))

        # self.masking = tf.keras.layers.Masking()
        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(
                int(self.config['HYPERPARAMETERS']['path_state_dim']),
                int(self.config['HYPERPARAMETERS']['readout_units'])),
            torch.nn.ReLU(),
            torch.nn.Linear(
                int(self.config['HYPERPARAMETERS']['readout_units']),  
                int(self.config['HYPERPARAMETERS']['readout_units'])),
            torch.nn.ReLU(),
            torch.nn.Linear(
                int(self.config['HYPERPARAMETERS']['readout_units']),output_units)
        )

    def forward(self, inputs, training=False):
        """This function is execution each time the model is called

        Args:
            inputs (dict): Features used to make the predictions.
            training (bool): Whether the model is train or not. If False, the
                             model does not update the weights.

        Returns:
            tensor: A tensor containing the per-path delay.
        """
        traffic = torch.unsqueeze(torch.squeeze(inputs['traffic']), dim=1).to(device)
        packets = torch.unsqueeze(torch.squeeze(inputs['packets']), dim=1).to(device)
        path_ids = torch.squeeze(inputs['path_ids']).to(device)
        l_q_p = torch.squeeze(inputs['l_q_p']).to(device)
        l_p_s = torch.squeeze(inputs['l_p_s']).to(device)
        l_q_l = torch.squeeze(inputs['l_q_l']).to(device)
        n_paths = inputs['n_paths'].to(device)
        n_links = inputs['n_links'].to(device)
        n_queues = inputs['n_queues'].to(device)


        # Compute the shape for the  all-zero tensor for link_state
        path_shape = [
            n_paths,
            torch.tensor(int(self.config['HYPERPARAMETERS']['path_state_dim']) -
            2).to(device)
        ]

        # Initialize the initial hidden state for links
        path_state = torch.concat([
            traffic, 
            packets,
            torch.zeros(path_shape).to(device)
        ], dim=1)

        # Compute the shape for the  all-zero tensor for path_state
        link_shape =[
            n_links,
            torch.tensor(int(self.config['HYPERPARAMETERS']['link_state_dim']) -
            int(self.config['DATASET']['num_policies']) -
            1).to(device)
        ]

        # Initialize the initial hidden state for paths
        link_state = torch.concat([
            inputs['capacity'].T.to(device), 
            torch.nn.functional.one_hot(torch.squeeze(inputs['policy']), 3).to(device),
            torch.zeros(link_shape).to(device)
        ], axis=1)

        # Compute the shape for the  all-zero tensor for path_state
        queue_shape = [
            n_queues,
            int(self.config['HYPERPARAMETERS']['queue_state_dim']) -
            int(self.config['DATASET']['max_num_queues']) - 1
        ]

        # Initialize the initial hidden state for paths
        queue_state = torch.concat([
            torch.nn.functional.one_hot(torch.squeeze(inputs['priority']).to(device), 3).to(device),
            inputs['weight'].T.to(device),
            torch.zeros(queue_shape).to(device)
        ], axis=1)

        # Iterate t times doing the message passing
        for it in range(int(self.config['HYPERPARAMETERS']['t'])):
            ###################
            #  LINK AND QUEUE #
            #     TO PATH     #
            ###################
            link_gather = torch.squeeze(link_state[torch.squeeze(inputs['link_to_path'])]) 
            queue_gather = torch.squeeze(queue_state[torch.squeeze(inputs['queue_to_path'])]) 

            ids_q = torch.stack([path_ids, l_q_p], dim=1)
            ids_l = torch.stack([path_ids, l_p_s], dim=1)
            max_len = torch.max(l_q_p) + 1
            link_input =  torch.zeros((n_paths, max_len, 
                int(self.config['HYPERPARAMETERS']['path_state_dim']))).to(device) 
            queue_input =  torch.zeros((n_paths, max_len, 
                int(self.config['HYPERPARAMETERS']['path_state_dim']))).to(device)
            link_input[ids_l.T[0],ids_l.T[1]] = link_gather 
            queue_input[ids_q.T[0],ids_q.T[1]] = queue_gather

            concated_input = torch.concat([queue_input, link_input], axis=2)
            lens = []
            for j in concated_input:
                nonzero_line = torch.sum(j.ne(torch.tensor(0.)),1)
                len_batch = torch.sum(nonzero_line.ne(torch.tensor(0.)))
                lens.append(len_batch)
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                concated_input, 
                lens, 
                batch_first=True, 
                enforce_sorted=False)
            path_state = torch.unsqueeze(path_state,0)
            _, path_state = self.path_update_gru(packed_input, path_state)  
            path_state = torch.squeeze(path_state, dim=0)

            ###################
            #  PATH TO QUEUE  #
            ###################
            path_gather = torch.squeeze(path_state[inputs['path_to_queue']]) 
            path_sum = torch.zeros(
                n_queues, path_gather.shape[1], dtype=path_gather.dtype).to(device).scatter_add_(
                    0, inputs['sequence_queues'].T.to(device).repeat(1, path_gather.shape[1]), path_gather)
            queue_state = self.queue_update(path_sum, queue_state)

            ###################
            #  QUEUE TO LINK  #
            ###################
            queue_gather = torch.squeeze(queue_state[inputs['queue_to_link']]) 
            ids_q = torch.stack(
                [torch.squeeze(inputs['sequence_links']).to(device), 
                l_q_l], dim=1)
            max_len = torch.max(l_q_l) + 1
            queue_input =  torch.zeros((n_links, max_len, 
                int(self.config['HYPERPARAMETERS']['link_state_dim']))).to(device)
            queue_input[ids_q.t()[0],ids_q.t()[1]] = queue_gather 
            
            lens = []
            for j in queue_input:
                nonzero_line = torch.sum(j.ne(torch.tensor(0.)),1)
                len_batch = torch.sum(nonzero_line.ne(torch.tensor(0.)))
                lens.append(len_batch)
            
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                queue_input, 
                lens, 
                batch_first=True, 
                enforce_sorted=False)
            
            link_state = torch.unsqueeze(path_state,0)
            _, link_state = self.link_update_gru(packed_input, link_state)  
            link_state = torch.squeeze(link_state, dim=0)

        # Call the readout ANN and return its predictions
        r = self.readout(path_state)
        r = torch.squeeze(r)                       # dataset
        r = torch.unsqueeze(torch.squeeze(r),dim=0) # dataloader 
        return r