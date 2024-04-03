import matplotlib.pyplot as plt

from module.GA2019.powerlaw_fit import *
from module.fit import PLSeg_fit, LS_avg_fit, GA2019_fit
from module.utils import get_merged_rank_distribution, get_normal_rank_distribution, generate_BA_network, KS_test, calculate_error

if __name__ == '__main__':
    n = 1000000
    m = 1
    directed = True

    MIN_INLIERS_PERCENT = 0.5

    # Sample from BA model
    print(f"Generating BA network with n=1e{int(np.log10(n))}, m={m}...")
    G = generate_BA_network(n, m)
    print("Done.")

    # Preprocessing
    print("Generating normal and merged rank distribution...")
    rank_norm, count_norm, prob_norm = get_normal_rank_distribution(G, directed)
    rank, range_, count, prob = get_merged_rank_distribution(G, directed)
    total_degree = np.sum(count_norm)
    print("Done.")

    # Fitting
    print("Fitting...")
    # PLSeg
    beta, prob_hat, ts_index, te_index = PLSeg_fit(rank, range_, prob)
    # KS test for PLSeg
    N_OP = np.sum((range_[ts_index:te_index + 1, 1] - range_[ts_index:te_index + 1, 0] + 1) * count[ts_index:te_index + 1])
    _, _, decision = KS_test(prob, prob_hat, N_OP, ts_index, te_index)
    decision = decision if (te_index - ts_index + 1) / len(rank) >= MIN_INLIERS_PERCENT else "reject"
    error = np.mean(calculate_error(prob[ts_index:te_index + 1], prob_hat[ts_index:te_index + 1]))
    # baseline: LS_avg
    beta_LS_avg, ks_LS_avg, error_LS_avg, prob_hat_LS_avg = LS_avg_fit(rank, prob)
    decision_LS_avg = "accept" if ks_LS_avg <= 1.36 / np.sqrt(total_degree) else "reject"
    # baseline: GA2019
    beta_GA, ks_GA, error_GA, prob_hat_GA = GA2019_fit(rank_norm, prob_norm)
    decision_GA = "accept" if ks_GA <= 1.36 / np.sqrt(total_degree) else "reject"
    print("Done.")

    # Plot
    label_data = rf'Rank(n=$10^{int(np.log10(n))}$, m={int(m)}, d={str(directed).lower()})'
    label_PLSeg = r'PLSeg ($\hat{\beta}$=' + f'{beta:.4f}, E={error:.4f}, {decision})'
    label_LS_avg = r'LS$_{avg}$ ($\hat{\beta}$=' + f'{beta_LS_avg:.4f}, E={error_LS_avg:.4f}, {decision_LS_avg})'
    label_GA = r'GA2019 ($\hat{\beta}$=' + f'{beta_GA:.4f}, E={error_GA:.4f}, {decision_GA})'
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(rank_norm, prob_norm, "o", label=label_data)
    plt.plot(rank, prob, "b*", label="Merged Rank")
    plt.plot(rank[ts_index], prob[ts_index], "c*", label=r"$t_s$")
    plt.plot(rank[te_index], prob[te_index], "y*", label=r"$t_e$")
    plt.plot(rank_norm, prob_hat_GA, "k-", label=label_GA)
    plt.plot(rank, prob_hat_LS_avg, "m--", label=label_LS_avg)
    plt.plot(rank[:te_index + 1], prob_hat[:te_index + 1], "r+-", label=label_PLSeg)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower left', frameon=False, numpoints=1)
    plt.show()
