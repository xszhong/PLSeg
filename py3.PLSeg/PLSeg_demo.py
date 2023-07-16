import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from scipy.linalg import lstsq
from GA2019.powerlaw import *
from GA2019.powerlaw_fit import *

def powerlaw_fit(x, y):
    length = len(x)
    B = np.ones([length, 1])
    X = np.column_stack([np.log(x), B])
    Y = np.column_stack([np.log(y)])
    w = lstsq(X, Y)
    yhat = np.dot(X, w[0])
    return w, yhat

def calculate_constant(key, prob, alpha):
    key_log = np.log(key)
    prob_log = np.log(prob)
    key_log_mean = np.mean(key_log)
    prob_log_mean = np.mean(prob_log)
    const = prob_log_mean - alpha * key_log_mean
    return np.exp(const)

def calculate_pdf_hat(key, K, alpha):
    return list(map(lambda X: K * X ** alpha, key))

def calculate_cdf(pdf):
    return np.cumsum(pdf)

def calculate_ks_D(cdf_data, cdf_model):
    ks_d = abs(np.array(cdf_data) - np.array(cdf_model))
    ks_d_2 = abs(np.array(cdf_data[:-1]) - np.array(cdf_model[1:]))
    return max(max(ks_d), max(ks_d_2))

def calculate_error(pdf_data, pdf_model):
    error = abs((np.array(pdf_data) - np.array(pdf_model))/np.array(pdf_data))
    return error

### From Barabasi's networkx ###
def _random_subset(seq, m, rng):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets

def barabasi_albert_graph(n, m, seed=random, initial_graph=None):
    """Returns a random graph using BarabásiAlbert preferential attachment

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    initial_graph : Graph or None (default)
        Initial network for BarabásiAlbert algorithm.
        It should be a connected graph for most use cases.
        A copy of `initial_graph` is used.
        If None, starts from a star graph on (m+1) nodes.

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``, or
        the initial graph number of nodes m0 does not satisfy ``m <= m0 <= n``.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"BarabásiAlbert network must have m >= 1 and m < n, m = {m}, n = {n}"
        )

    if initial_graph is None:
        # Default initial graph : star graph on (m + 1) nodes
        G = nx.generators.classic.star_graph(m)
    else:
        if len(initial_graph) < m or len(initial_graph) > n:
            raise nx.NetworkXError(
                f"BarabásiAlbert initial graph needs between m={m} and n={n} nodes"
            )
        G = initial_graph.copy()

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    # Start adding the other n - m0 nodes.
    source = len(G)
    while source < n:
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)

        source += 1
    return G
###

def generate_BA_network(n, m):
    if m > 1:
        G_init = barabasi_albert_graph(m, 1, seed=random)
        G = barabasi_albert_graph(n, m, seed=random, initial_graph = G_init)
    else:
        G = barabasi_albert_graph(n, m, seed=random)

    edges = list(G.edges())
    from_nodes = []
    to_nodes = []
    for j in range(len(edges)):
        if edges[j][1]>edges[j][0]:
            from_nodes.append(edges[j][1])
            to_nodes.append(edges[j][0])
        else:
            from_nodes.append(edges[j][0])
            to_nodes.append(edges[j][1])
    return from_nodes, to_nodes

def calculate_occurrence(x):
    x_data = {}
    for i in range(len(x)):
        count = 1
        if x[i] in x_data:
            count += x_data[x[i]]
        x_data[x[i]] = count

    data_items = x_data.items()
    return data_items

def get_rank_norm_data(degrees):
    rank = np.arange(1,len(degrees)+1)
    prob = sorted(degrees,reverse=True)/np.sum(degrees)
    count = sorted(degrees,reverse=True)
    return rank, prob, count

def get_rank_merge_data(data_items, x_total):
    sorted_data = list(sorted(data_items, reverse=True))
    rank = []
    value = []
    prob = []
    range_all = []
    rank_current = 0

    for i in range(len(sorted_data)):
        item = sorted_data[i]
        key = item[0]
        count = item[1]
        prob_tem = key * 1.0 / x_total

        rank_start = rank_current + 1
        rank_end = rank_current + count
        rank_tem = np.sqrt(rank_start * rank_end)
        range_tem = (rank_start,rank_end)

        rank.append(rank_tem)
        value.append(key)
        prob.append(prob_tem)
        range_all.append(range_tem)

        rank_current = rank_end

    return rank, range_all, prob, value

def PLSeg_fit(rank, range_all, prob):
    for i in range(len(range_all)):
        merge_range = range_all[i]
        if merge_range[0] != merge_range[1]:
            first_merge_index = i
            break
    x1_index = 0
    for i in range(len(range_all)-1,-1,-1):
        merge_range = range_all[i]
        if merge_range[0] == merge_range[1]:
            x1_index = i
            break

    # learn
    x_first_index = first_merge_index
    x_last_index = len(rank) - 1

    beta = 0
    beta_delta = 10e-4
    counter = 0

    error_threshold = 0.1

    while True:
        beta_values = []
        for i in list(range(x1_index + 1, x_last_index + 1)):
            rank_tem = rank[x_first_index : i + 1]
            prob_tem = prob[x_first_index : i + 1]
            w_tem, prob_tem_hat = powerlaw_fit(rank_tem, prob_tem)
            beta_tem = w_tem[0][0][0]
            beta_values.append(beta_tem)
        
        beta_mean = np.mean(beta_values)
        C_current = calculate_constant(rank[x_first_index : x_last_index + 1], prob[x_first_index : x_last_index + 1], beta_mean)
        
        prob_current_hat = calculate_pdf_hat(rank, C_current, beta_mean)
        error = calculate_error(prob, prob_current_hat)

        x_first_current_index = x_first_index
        if error[x_first_index] <= error_threshold:
            for i in range(x_first_index, -1, -1):
                if error[i] <= error_threshold:
                    x_first_current_index  = i
                else:
                    break 
        else:
            for i in range(x_first_index, x1_index):
                if error[i] > error_threshold:
                    x_first_current_index  = i
                else:
                    break

        x_last_current_index = len(rank) - 1
        for i in range(len(error)-1, x1_index, -1):
            if error[i] > error_threshold:
                x_last_current_index  = i
            else:
                break
        
        if abs(beta - beta_mean) <= beta_delta or counter>=100:
            x_first_index = x_first_current_index
            x_last_index = x_last_current_index
            beta = beta_mean
            break
        
        x_first_index = x_first_current_index
        x_last_index = x_last_current_index
        beta = beta_mean
        counter += 1

    return beta, x_first_index, x_last_index


def LS_avg_fit(rank,prob):
    beta_values = []
    for i in range(2, len(rank)):
        rank_tem = rank[: i + 1]
        prob_tem = prob[: i + 1]
        
        #get alpha
        w_tem, prob_tem_hat = powerlaw_fit(rank_tem, prob_tem)
        beta_tem = w_tem[0][0][0]
        beta_values.append(beta_tem)

    beta_mean = np.mean(beta_values)
    C = calculate_constant(rank, prob, beta_mean)
    prob_hat = calculate_pdf_hat(rank, C, beta_mean)
    error = np.mean(calculate_error(prob, prob_hat))
    cdf = calculate_cdf(prob)
    prob_hat_cdf=calculate_cdf(prob_hat)
    ks_D = calculate_ks_D(cdf,prob_hat_cdf)
    return beta_mean,ks_D,error,prob_hat


def GA2019_fit(rank,prob):
    result = fit_power_disc_sign(np.array(rank), np.array(prob),xmax =len(rank))
    beta = -result['alpha']
    C = calculate_constant(rank, prob, beta)
    prob_hat = calculate_pdf_hat(rank, C, beta)
    error = np.mean(calculate_error(prob, prob_hat))
    cdf = calculate_cdf(prob)
    prob_hat_cdf=calculate_cdf(prob_hat)
    ks_D = calculate_ks_D(cdf,prob_hat_cdf)
    return beta,ks_D,error,prob_hat


n = 100000
m = 3
is_directed = False


### Sampling from the BA model ###
print("Generating BA network with n=1e{}".format(str(int(np.log10(n))))+", m={}...".format(str(m)))
from_nodes, to_nodes = generate_BA_network(n,m)
print("Done.")
###


### Data preprocessing ###
print("Generating merged rank data...")
data = {}
for i in range(len(from_nodes)):
    from_node = from_nodes[i]
    to_node = to_nodes[i]
    if not is_directed:
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
total_degree = np.sum(degrees)

data_items = calculate_occurrence(degrees)
rank, range_all, prob, count = get_rank_merge_data(data_items,total_degree)
rank_norm, prob_norm, count_norm = get_rank_norm_data(degrees)
print("Done.")
###


### Fitting ###
print("Calculating beta...")
beta, xfirst_index, xlast_index = PLSeg_fit(rank, range_all, prob) # PLSeg
beta_lsavg, ks_lsavg, error_lsavg,prob_hat_lsavg = LS_avg_fit(rank, prob)  #LS_avg
beta_ga, ks_ga, error_ga, prob_hat_ga = GA2019_fit(rank_norm, prob_norm)   # GA2019
print("Done.")
###


### KS test ###
# PLSeg
C_mid = calculate_constant(rank[xfirst_index:xlast_index+1], prob[xfirst_index:xlast_index+1], beta)
prob_hat_mid = calculate_pdf_hat(rank, C_mid, beta)

cdf_prob = calculate_cdf(prob[xfirst_index:xlast_index+1])
cdf_prob_hat = calculate_cdf(prob_hat_mid[xfirst_index:xlast_index+1])
D_mid = calculate_ks_D(cdf_prob, cdf_prob_hat)
error_avg = np.mean(calculate_error(prob[xfirst_index:xlast_index+1],prob_hat_mid[xfirst_index:xlast_index+1]))

N_mid = 0
for i in range(xfirst_index, xlast_index+1):
    N_mid += (range_all[i][1]-range_all[i][0]+1)*count[i]
KS_thres = 1.36/np.sqrt(N_mid)

if D_mid<=KS_thres:
    decision = "accept"
else:
    decision = "reject" 

# LS_avg
decision_lsavg = "accept" if ks_lsavg<=1.36/np.sqrt(total_degree) else "reject"

# GA2019
decision_ga = "accept" if ks_ga<=1.36/np.sqrt(total_degree) else "reject"
### 

### Plotting ###
# labels
label_data = r'Rank (n=$10^' + str(int(np.log10(n))) + '$, m=' + str(int(m)) + ', d=' + str(is_directed).lower() + ')'
label_ls = r'PLSeg ($\hat{\beta}$=' + str(-np.round(beta, 4))+', E='+str(np.round(error_avg, 4))+', '+decision+')'
label_lsavg = r'LS$_{avg}$ ($\hat{\beta}$=' + str(-np.round(beta_lsavg, 4))+', E='+str(np.round(error_lsavg, 4))+', '+decision_lsavg+')'
label_ga = r'GA ($\hat{\beta}$=' + str(-np.round(beta_ga, 4))+', E='+str(np.round(error_ga, 4))+', '+decision_ga+')'

plt.figure(figsize=(5.5, 4.5))

# norm data points
plt.plot(rank_norm, prob_norm, "o", label=label_data)

# merge data points
plt.plot(rank, prob, "b*", label="Merged Rank")

# GA2019 fit result
plt.plot(rank_norm, prob_hat_ga, "k-", label=label_ga)

# ls_avg fit result
plt.plot(rank, prob_hat_lsavg, "m--", label=label_lsavg)

# PLSeg fit result
plt.plot(rank[xfirst_index], prob[xfirst_index], "go")
plt.plot(rank[xlast_index], prob[xlast_index], "yo")
plt.plot(rank[:xlast_index+1], prob_hat_mid[:xlast_index+1], "r+-", label=label_ls)

plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left', frameon=False, numpoints=1)
plt.show()
###
