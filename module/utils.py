import random
from collections import Counter

import networkx as nx
import numpy as np


def generate_BA_network(n: int, m: int) -> nx.Graph:
    if m > 1:
        G_init = nx.barabasi_albert_graph(m, 1, seed=random)
        G = nx.barabasi_albert_graph(n, m, seed=random, initial_graph=G_init)
    else:
        G = nx.barabasi_albert_graph(n, m, seed=random)
    return G


def get_degree_sequence(G: nx.Graph, directed: bool = False) -> list[int]:
    edges = list(G.edges())
    from_nodes = []
    to_nodes = []
    for j in range(len(edges)):
        if edges[j][1] > edges[j][0]:
            from_nodes.append(edges[j][1])
            to_nodes.append(edges[j][0])
        else:
            from_nodes.append(edges[j][0])
            to_nodes.append(edges[j][1])

    data = {}
    for i in range(len(from_nodes)):
        from_node = from_nodes[i]
        to_node = to_nodes[i]
        if not directed:
            if from_node in data:
                data[from_node].add(to_node)
            else:
                data[from_node] = set()
                data[from_node].add(to_node)
        if to_node in data:
            data[to_node].add(from_node)
        else:
            data[to_node] = set()
            data[to_node].add(from_node)

    degrees = []
    for i in data:
        degrees.append(len(data[i]))
    return degrees


def get_normal_rank_distribution(G: nx.Graph, directed: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_degree_sequence = sorted(get_degree_sequence(G, directed), reverse=True)
    rank = np.arange(1, len(sorted_degree_sequence) + 1)
    count = np.array(sorted_degree_sequence)
    prob = count / np.sum(count)
    return rank, count, prob


def get_merged_rank_distribution(G: nx.Graph, directed: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    degree_sequence = get_degree_sequence(G, directed)
    total_degree = np.sum(degree_sequence)
    sorted_data = list(sorted(Counter(degree_sequence).items(), reverse=True))
    rank, range_, count, prob = [], [], [], []

    r_tmp = 0
    for i in range(len(sorted_data)):
        k, v = sorted_data[i]
        r_s = r_tmp + 1
        r_e = r_tmp + v
        r_c = np.sqrt(r_s * r_e)
        rank.append(r_c)
        range_.append((r_s, r_e))
        count.append(k)
        prob.append(k / total_degree)
        r_tmp = r_e

    return np.array(rank), np.array(range_), np.array(count), np.array(prob)


def calculate_error(pdf_data, pdf_model):
    error = np.abs(np.array(pdf_data) - np.array(pdf_model)) / np.sqrt(np.array(pdf_model) * np.array(pdf_data))
    return error


def relative_error(pdf_data, pdf_model):
    error = np.abs(np.array(pdf_data) - np.array(pdf_model)) / np.array(pdf_model)
    return error


def calculate_constant(x, px, alpha) -> float:
    """
    Calculate the constant C of the power law distribution p(x)=Cx^{-alpha}.
    :param x: A sequence of different observations x.
    :param px: The probabilities of x.
    :param alpha: A negative value, representing the power law exponent.
    :return: The constant of the power law distribution.
    """
    if alpha <= 0:
        raise RuntimeError("Power-law exponent must >0.")
    x_log = np.log(x)
    px_log = np.log(px)
    const = np.mean(px_log) + alpha * np.mean(x_log)
    return np.exp(const)


def calculate_pdf_hat(x: list, C: float, alpha: float) -> list:
    """
    Calculate the predicted probability dense function p(x) of the given power law model p(x)=Cx^{-alpha} and x
    :param x: Input values of the power-law model.
    :param C: The constant of the power-law model.
    :param alpha: The power-law exponent.
    :return: A list of p(x) for each element in x.
    """
    if alpha <= 0:
        raise RuntimeError("Power-law exponent must >0.")
    return list(map(lambda X: C * X ** (-alpha), x))


def calculate_cdf(pdf):
    return np.cumsum(pdf)


def calculate_ks_D(cdf_data, cdf_model):
    ks_d = abs(np.array(cdf_data) - np.array(cdf_model))
    ks_d_2 = abs(np.array(cdf_data[:-1]) - np.array(cdf_model[1:]))
    return max(max(ks_d), max(ks_d_2))


def KS_test(prob, prob_hat, N, start_index: int = None, end_index: int = None):
    if len(prob) != len(prob_hat):
        raise RuntimeError("length of data and model must be same.")

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(prob) - 1

    cdf = calculate_cdf(prob[start_index:end_index + 1])
    cdf_hat = calculate_cdf(prob_hat[start_index:end_index + 1])
    ks_statistic = calculate_ks_D(cdf, cdf_hat)
    ks_thres = 1.36 / np.sqrt(N)
    ks_result = "accept" if ks_statistic < ks_thres else "reject"

    return ks_statistic, ks_thres, ks_result


def calculate_sample_count(range_all, count, start_index: int = None, end_index: int = None):
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(range_all) - 1
    range_all = np.array(range_all)
    return np.sum((range_all[start_index:end_index + 1, 1] - range_all[start_index:end_index + 1, 0] + 1) * count[start_index:end_index + 1])
