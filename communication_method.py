from Statistical_method import *
import torch


def communication(mode, server_model, models, client_weights, clients_in_comm):
    with torch.no_grad():
        # aggregate params
        if mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(clients_in_comm)):
                        temp += client_weights[client_idx] * models[clients_in_comm[client_idx]].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(clients_in_comm)):
                        models[clients_in_comm[client_idx]].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif mode.lower() == 'silobn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(clients_in_comm)):
                        temp += client_weights[client_idx] * models[clients_in_comm[client_idx]].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(clients_in_comm)):
                        models[clients_in_comm[client_idx]].state_dict()[key].data.copy_(server_model.state_dict()[key])
                else:
                    #  统计参数和“num_batches_trached”训练时用来统计训练时的forward过的min-batch数目,和第一个client保持一致即可
                    if 'num_batches_trached' or 'running_mean' or 'running_var' in key:
                        server_model.state_dict()[key].data.copy_(models[clients_in_comm[0]].state_dict()[key])
                    elif 'weight' or 'bias' in key:
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        for client_idx in range(len(clients_in_comm)):
                            temp += client_weights[client_idx] * models[clients_in_comm[client_idx]].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        for client_idx in range(len(clients_in_comm)):
                            models[clients_in_comm[client_idx]].state_dict()[key].data.copy_(
                                server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(len(clients_in_comm)):
                    temp += client_weights[client_idx] * models[clients_in_comm[client_idx]].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)

    return server_model, models
