"""
Metrics

"""
import numpy as np

#hit rate
def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1

def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)

#precision
def precision(recommended_list, bought_list):
    recommended_list = np.array(recommended_list)
    bought_list = np.array(bought_list)
    flags = np.isin(recommended_list, bought_list)
    precision = flags.sum() / len(recommended_list)

    return precision

def precision_at_k(recommended_list, bought_list, k=5):
    recommended_list = np.array(recommended_list[:k])
    bought_list = np.array(bought_list)
    flags = np.isin(recommended_list, bought_list) * 1
    precision = flags.sum() / len(recommended_list)

    return precision


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = np.array(recommended_list[:k])
    bought_list = np.array(bought_list)
    prices_recommended = np.array(prices_recommended[:k])
    flags = np.isin(recommended_list, bought_list) * 1
    precision = prices_recommended @ flags / sum(prices_recommended)

    return precision

#recall
def recall(recommended_list, bought_list):
    recommended_list = np.array(recommended_list)
    bought_list = np.array(bought_list)
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)

    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    recommended_list = np.array(recommended_list[:k])
    bought_list = np.array(bought_list)
    flags = np.isin(bought_list, recommended_list) * 1
    recall = flags.sum() / len(bought_list)

    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)
    flags = np.isin(recommended_list, bought_list)
    
    return np.dot(flags, prices_recommended).sum() / prices_bought.sum()

#AP@k
def ap_k(recommended_list, bought_list, k=5):
    recommended_list = np.array(recommended_list[:k])
    bought_list = np.array(bought_list)
    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0
    amount_relevant = len(relevant_indexes)
    sum_ = sum([precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])

    return sum_ / amount_relevant

#MAP@k
def map_k(recommended_list_n_users, bought_list_n_users, k=5):
    result = np.mean([ap_k(recommended_list_n_users[i], bought_list_n_users[i], k) for i in range(len(recommended_list_n_users))])

    return result

#nDCG@k
def ndcg_at_k(recommended_list, bought_list, k=5):
    recommended_list = np.array(recommended_list[:k])
    bought_list = np.array(bought_list)
    flags = np.isin(recommended_list, bought_list) * 1  # какие из рекомендованных товаров куплены
    discount = []
    for i in range(len(recommended_list)):
        if i + 1 <= 2:
            discount.append(i + 1)
        else:
            discount.append(np.log2(i + 1))
    discount_i = np.array([1 / disc for disc in discount])
    dcg = (1 / len(recommended_list)) * (flags @ discount_i)
    ideal_dcg = (1 / len(recommended_list)) * (np.array([1] * len(recommended_list)) @ discount_i)
    ndcg = dcg / ideal_dcg

    return ndcg


def n_ndcg_at_k(recommended_list_n_users, bought_list_n_users, k=5):
    result = np.mean([ndcg_at_k(recommended_list_n_users[i], bought_list_n_users[i], k) for i in
                      range(len(recommended_list_n_users))])

    return result

#MRR@k
def reciprocal_rank(recommended_list, bought_list, k=1):
    recommended_list = np.array(recommended_list[:k])
    bought_list = np.array(bought_list)
    # номер (ранк) первого купленного товара среди рекомендуемых
    try:
        relevant_rank = np.nonzero(np.isin(recommended_list, bought_list))[0][0] + 1
    except IndexError:
        relevant_rank = 0
    try:
        # reciprocal rank
        rank = 1 / relevant_rank
    except ZeroDivisionError:
        rank = 0

    return rank


def mrr(recommended_list_n_users, bought_list_n_users, k=5):
    mrr = np.mean([reciprocal_rank(recommended_list_n_users[i], bought_list_n_users[i], k) for i in
                   range(len(recommended_list_n_users))])

    return mrr