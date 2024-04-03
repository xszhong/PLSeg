from scipy.linalg import lstsq
from sklearn.linear_model import LinearRegression

from module.GA2019.powerlaw import pdf_power_disc
from module.GA2019.powerlaw_fit import fit_power_disc_sign
from module.utils import *


def PLSeg_fit(rank, range_all, prob, error_threshold: float = 0.1, delta_beta: float = 10e-4, max_iter: int = 100):
    range_all = np.array(range_all)
    tf_index = np.where(range_all[:, 0] != range_all[:, 1])[0][0]
    tk1_index = np.where(range_all[:, 0] == range_all[:, 1])[0][-1]

    ts_index = tf_index
    te_index = len(rank) - 1

    beta = 0
    counter = 0

    X = np.log10(rank).reshape(-1, 1)
    y = np.log10(prob)
    weight = np.array(prob) ** 2

    while True:
        model = LinearRegression()
        model.fit(X[ts_index:te_index + 1], y[ts_index:te_index + 1], sample_weight=weight[ts_index:te_index + 1])
        beta_current = abs(model.coef_[0])
        prob_hat = np.power(10, model.predict(X))
        error = relative_error(prob, prob_hat)

        ts_index_new = ts_index
        if error[ts_index] <= error_threshold:
            for i in range(ts_index, -1, -1):
                if error[i] <= error_threshold:
                    ts_index_new = i
                else:
                    break
        else:
            for i in range(ts_index, te_index + 1):
                if error[i] > error_threshold:
                    ts_index_new = i
                else:
                    break

        te_index_new = len(rank) - 1
        for i in range(len(error) - 1, tk1_index, -1):
            if error[i] > error_threshold:
                te_index_new = i
            else:
                break

        if abs(beta - beta_current) <= delta_beta or counter >= max_iter:
            ts_index = ts_index_new
            te_index = te_index_new
            break

        ts_index = ts_index_new
        te_index = te_index_new
        beta = beta_current
        counter += 1

    # last fit
    model = LinearRegression()
    model.fit(X[ts_index:te_index + 1], y[ts_index:te_index + 1], sample_weight=weight[ts_index:te_index + 1])
    beta = abs(model.coef_[0])
    prob_hat = np.power(10, model.predict(X))

    return beta, prob_hat, ts_index, te_index


def LS_avg_fit(rank, prob):
    def powerlaw_fit(x, y):
        length = len(x)
        B = np.ones([length, 1])
        X = np.column_stack([np.log(x), B])
        Y = np.column_stack([np.log(y)])
        w = lstsq(X, Y)
        yhat = np.dot(X, w[0])
        return w, yhat

    beta_values = []
    for i in range(2, len(rank)):
        w_tem, _ = powerlaw_fit(rank[: i + 1], prob[: i + 1])
        beta_values.append(w_tem[0][0][0])
    beta = abs(float(np.mean(beta_values)))

    C = calculate_constant(rank, prob, beta)
    prob_hat = calculate_pdf_hat(rank, C, beta)
    error = np.mean(calculate_error(prob, prob_hat))
    cdf, cdf_hat = calculate_cdf(prob), calculate_cdf(prob_hat)
    ks_D = calculate_ks_D(cdf, cdf_hat)
    return beta, ks_D, error, prob_hat


def GA2019_fit(rank, prob):
    result = fit_power_disc_sign(rank, prob, xmin=min(rank), xmax=max(rank))
    beta = abs(result['alpha'])
    prob_hat = pdf_power_disc(rank, xmin=min(rank), xmax=max(rank), gamma=beta)
    error = np.mean(calculate_error(prob, prob_hat))
    cdf, cdf_hat = calculate_cdf(prob), calculate_cdf(prob_hat)
    ks_D = calculate_ks_D(cdf, cdf_hat)
    return beta, ks_D, error, prob_hat
