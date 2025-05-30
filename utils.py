"""
    Some handy functions for pytroch model training ...
"""
import torch
import logging
from sklearn.metrics import pairwise_distances
import numpy as np
import requests


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)

def construct_user_relation_graph_via_item(round_user_params, neighborhood_users, item_num, latent_dim, similarity_metric):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(neighborhood_users), item_num * latent_dim), dtype='float32')
    
    for i in range(len(neighborhood_users)):
        item_embedding[i] = round_user_params[neighborhood_users[i]]['embedding_item.weight'].numpy().flatten()
    # construct the user relation graph.
    adj = pairwise_distances(item_embedding, metric=similarity_metric)
    if similarity_metric == 'cosine':
        return adj
    else:
        return -adj

def select_topk_neighboehood(user_realtion_graph, neighborhood_size, neighborhood_threshold):
    topk_user_relation_graph = np.zeros(user_realtion_graph.shape, dtype='float32')
    if neighborhood_size > 0:
        for user in range(user_realtion_graph.shape[0]):
            user_neighborhood = user_realtion_graph[user]
            topk_indexes = user_neighborhood.argsort()[-neighborhood_size:][::-1]
            for i in topk_indexes:
                topk_user_relation_graph[user][i] = 1/neighborhood_size
    else:
        # similarity_threshold = np.mean(user_realtion_graph)*neighborhood_threshold
        similarity_threshold = np.sum(user_realtion_graph) / (user_realtion_graph.shape[0] * user_realtion_graph.shape[1] - user_realtion_graph.shape[0])
        for i in range(user_realtion_graph.shape[0]):
            high_num = np.sum(user_realtion_graph[i] > similarity_threshold)
            if high_num > 1:
                for j in range(user_realtion_graph.shape[1]):
                    if user_realtion_graph[i][j] > similarity_threshold:
                        topk_user_relation_graph[i][j] = 1
                topk_user_relation_graph[i][i] = 1
            else:
                topk_user_relation_graph[i][i] = 1
            topk_user_relation_graph[i][i] = 1

    return topk_user_relation_graph

def gaussian_attention_scores(Q, K, topk_user_relation_graph, sigma=1.0):
    Q_norm = np.sum(Q**2, axis=1, keepdims=True)  # (seq_len, 1)
    Q_norm = np.tile(Q_norm, (1, Q.shape[0]))
    K_norm = np.sum(K**2, axis=1) 
    K_norm = K_norm.reshape((1, Q.shape[0]))                # (seq_len,)
    K_norm = np.tile(K_norm, (Q.shape[0], 1))
    dot_product = np.matmul(Q,  K.T)
    distances = (Q_norm + K_norm - 2 * dot_product) / (2 * sigma ** 2)  # (seq_len, seq_len)
    distances = np.maximum(distances, 0.0)
    result = np.exp(-distances)
    result = result * topk_user_relation_graph
    row_sums = result.sum(axis = 1, keepdims=True)
    row_sums[row_sums == 0] = 1
    result = result / row_sums
    del dot_product
    del Q_norm
    del K_norm
    del row_sums
    result = np.nan_to_num(result, nan=0.0)
    
    return result

    


def MP_on_graph(round_user_params, neighborhood_users, item_num, latent_dim, topk_user_relation_graph, layers):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(neighborhood_users), item_num*latent_dim), dtype='float32')
    for i in range(len(neighborhood_users)):
        item_embedding[i] = round_user_params[neighborhood_users[i]]['embedding_item.weight'].numpy().flatten()
    
    # aggregate item embedding via message passing.
    # sum_layers = item_embedding
    score = gaussian_attention_scores(item_embedding, item_embedding, topk_user_relation_graph)
    aggregated_item_embedding = np.matmul(score, item_embedding)
    # aggregated_item_embedding = item_embedding
    # for layer in range(layers):
    #     aggregated_item_embedding = np.matmul(topk_user_relation_graph, aggregated_item_embedding)
    #     score = gaussian_attention_scores(item_embedding, aggregated_item_embedding)
    #     sum_vec += np.matmul(score, aggregated_item_embedding)
    #     # sum_layers += aggregated_item_embedding
    # sum_vec /= 2
    # aggregated_item_embedding = 1 / (layers + 1) * sum_layers
    

    # reconstruct item embedding.
    item_embedding_dict = {}
    for i in range(len(neighborhood_users)):
        item_embedding_dict[neighborhood_users[i]] = torch.from_numpy(aggregated_item_embedding[i].reshape(item_num, latent_dim))
    avg = np.mean(aggregated_item_embedding, axis=0)
    item_embedding_dict['global'] = torch.from_numpy(avg.reshape(item_num, latent_dim))
    del item_embedding
    del aggregated_item_embedding
    # del sum_vec
    return item_embedding_dict



def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def compute_regularization(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction='mean')
    for name, param in model.named_parameters():
        if name == 'embedding_item.weight':
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss
        
def send_webhook_message(webhook_url, message, username=None):
    data = {"content": message}
    
    if username:
        data["username"] = username
    
    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            print("Message sent successfully!")
        else:
            print(f"Failed to send message: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")