import torch
import psutil

SEED = 1224
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GNN_Model(torch.nn.Module):

    def __init__(self, config, output_units=1):
        super(GNN_Model, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # GRU used in the Message Passing step
        self.link_update = torch.nn.GRUCell(
            int(self.config['HYPERPARAMETERS']['path_state_dim']),
            int(self.config['HYPERPARAMETERS']['link_state_dim']))
        self.path_update_gru = torch.nn.GRU(
            int(self.config['HYPERPARAMETERS']['link_state_dim']),
            int(self.config['HYPERPARAMETERS']['path_state_dim']), batch_first=True)

        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        # torch sequential model
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
                int(self.config['HYPERPARAMETERS']['readout_units']), output_units)
        )

    def forward(self, inputs):
        traffic = torch.unsqueeze(torch.squeeze(inputs['traffic']), dim=1).to(device)
        packets = torch.unsqueeze(torch.squeeze(inputs['packets']), dim=1).to(device)
        time_dist_params = torch.squeeze(inputs['time_dist_params']).to(device)
        capacity = torch.unsqueeze(torch.squeeze(inputs['capacity']), dim=1).to(device)
        link_to_path = torch.squeeze(inputs['link_to_path']).to(device)
        path_to_link = torch.squeeze(inputs['path_to_link']).to(device)
        path_ids = torch.squeeze(inputs['path_ids']).to(device)
        sequence_path = torch.squeeze(inputs['sequence_path']).to(device)
        sequence_links = torch.unsqueeze(torch.squeeze(inputs['sequence_links']), dim=1).to(device)
        n_links = inputs['n_links'].to(device)
        n_paths = inputs['n_paths'].to(device)
        # "path_state_dim" - 14 (traffic (1)+packets (1)+time_dist_params (12) = 14). 

        # Compute the shape for the  all-zero tensor for path_state
        link_shape = [
            n_links,
            torch.tensor(int(self.config['HYPERPARAMETERS']['link_state_dim']) - 1).to(device)]

        # Initialize the initial hidden state for paths
        link_state = torch.concat(
            [capacity,
             torch.zeros(link_shape).to(device)],
            dim=1)

        path_shape = [
            n_paths,
            torch.tensor(int(self.config['HYPERPARAMETERS']['link_state_dim']) - 14).to(device)]

        # Initialize the initial hidden state for links
        path_state = torch.concat(
            [traffic,
             packets,
             time_dist_params,
             torch.zeros(path_shape).to(device)], dim=1)

        for _ in range(int(self.config['HYPERPARAMETERS']['t'])):
            # The following lines generate a tensor of dimensions [n_paths, max_len_path, dimension_link] with all 0
            # but the link hidden states
            link_gather = torch.squeeze(link_state[link_to_path])

            ids = torch.stack([
                path_ids,
                sequence_path], dim=1)
            max_len = torch.max(sequence_path) + 1

            # Generate the aforementioned tensor [n_paths, max_len_path, dimension_link]
            link_inputs = torch.zeros(
                (n_paths,
                 max_len,
                 int(self.config['HYPERPARAMETERS']['link_state_dim']))).to(device)
            link_inputs[ids.T[0], ids.T[1]] = link_gather

            # get the length of batch in the link_input to pack
            lens = []
            for j in link_inputs:
                nonzero_line = torch.sum(j.ne(torch.tensor(0.)), 1)
                len_batch = torch.sum(nonzero_line.ne(torch.tensor(0.)))
                lens.append(len_batch)

            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                link_inputs,
                lens,
                batch_first=True,
                enforce_sorted=False)

            # unbidirectional 1 layer gru: dim0: 1*1=1
            path_state = torch.unsqueeze(path_state, 0)

            _, path_state = self.path_update_gru(
                packed_input, path_state)
            # output, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            path_state = torch.squeeze(path_state, dim=0)

            # For each link, gather and sum the sequence of hidden states of the paths that contain it
            path_gather = torch.squeeze(path_state[path_to_link])

            # Second message passing: update the link_state
            path_sum = torch.zeros(
                n_links, path_gather.shape[1], dtype=path_gather.dtype).to(device).scatter_add_(
                0, sequence_links.repeat(1, path_gather.shape[1]), path_gather)

            link_state = self.link_update(path_sum, link_state)

        # Call the readout ANN and return its predictions
        r = self.readout(path_state)
        r = torch.squeeze(r)  # dataset
        r = torch.unsqueeze(torch.squeeze(r), dim=0)  # dataloader [batch_size, length]
        return r
