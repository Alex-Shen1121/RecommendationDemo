import numpy as np
import pandas as pd
from collections import defaultdict
import random
import tensorflow as tf


def gen_test(user_ratings):
    """
    对每一个用户u，在user_ratings中随机找到他评分过的一部电影i,保存在user_ratings_test，
    后面构造训练集和测试集需要用到。
    """
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test


def gen_train_batch(user_ratings, user_ratings_test, item_list, batch_size=512):
    """
    构造训练用的三元组
    对于随机抽出的用户u，i可以从user_ratings随机抽出，而j也是从总的电影集中随机抽出，当然j必须保证(u,j)不在user_ratings中
    """
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]

        j = random.sample(item_list, 1)[0]
        while j in user_ratings[u]:
            j = random.sample(item_list, 1)[0]
        t.append([u, i, j])
    return np.asarray(t)


def gen_test_batch(user_ratings, user_ratings_test, item_list):
    """
    对于每个用户u，它的评分电影i是我们在user_ratings_test中随机抽取的，它的j是用户u所有没有评分过的电影集合，
    比如用户u有1000部电影没有评分，那么这里该用户的测试集样本就有1000个
    """
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in item_list:
            if not (j in user_ratings[u]):
                t.append([u, i, j])
        yield np.asarray(t)


def bpr_mf(user_count, item_count, hidden_dim):
    """
    hidden_dim为矩阵分解的隐含维度k。user_emb_w对应矩阵W, item_emb_w对应矩阵H
    """
    u = tf.compat.v1.placeholder(tf.int32, [None])
    i = tf.compat.v1.placeholder(tf.int32, [None])
    j = tf.compat.v1.placeholder(tf.int32, [None])

    user_emb_w = tf.compat.v1.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))
    item_emb_w = tf.compat.v1.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))

    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)

    # MF predict: u_i > u_j
    # 第一部分的i 和 j的差值计算
    x = tf.compat.v1.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)

    # AUC for one user:
    # reasonable iff all (u,i,j) pairs are from the same user
    # average AUC = mean( auc for each user in test set)
    mf_auc = tf.reduce_mean(tf.compat.v1.to_float(x > 0))

    # 第二部分的正则项
    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])

    # 整个loss
    regulation_rate = 0.0001
    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.compat.v1.log(tf.sigmoid(x)))

    # 梯度上升
    train_op = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(bprloss)
    return u, i, j, mf_auc, bprloss, train_op


if __name__ == "__main__":
    df = pd.read_csv("../datasets/ml-100k/u.data", sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    user_list = df['user_id'].unique().tolist()
    item_list = df['item_id'].unique().tolist()
    user_count = len(user_list)
    item_count = len(item_list)
    # print(user_count, item_count)

    user_ratings = defaultdict(set)
    for index, row in df.iterrows():
        u = row['user_id']
        i = row['item_id']
        user_ratings[u].add(i)

    user_ratings_test = gen_test(user_ratings)

    with tf.compat.v1.Session() as sess:
        """
        这里k取了20， 迭代次数3， 主要是为了快速输出结果。
        如果要做一个较好的BPR算法，需要对k值进行选择迭代，并且迭代次数也要更多一些。
        """
        u, i, j, mf_auc, bprloss, train_op = bpr_mf(user_count, item_count, 20)
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(1, 4):
            _batch_bprloss = 0
            for k in range(1, 5000): # uniform samples from training set
                uij = gen_train_batch(user_ratings, user_ratings_test, item_list)
                _bprloss, _train_op = sess.run([bprloss, train_op],
                                               feed_dict={u: uij[:, 0], i: uij[:, 1], j: uij[:, 2]})

                _batch_bprloss += _bprloss

            print("epoch:", epoch)
            print("bprloss:", _batch_bprloss / k)
            print("_train_op")

            user_count = 0
            _auc_sum = 0.0

            for t_uij in gen_test_batch(user_ratings, user_ratings_test, item_list):
                _auc, _test_bprloss = sess.run([mf_auc, bprloss],
                                               feed_dict={u: t_uij[:, 0], i: t_uij[:, 1], j: t_uij[:, 2]}
                                               )
                user_count += 1
                _auc_sum += _auc
            print("test_loss: ", _test_bprloss, "test_auc: ", _auc_sum / user_count)
            print("")
        variable_names = [v.name for v in tf.compat.v1.trainable_variables()]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)

    """
    现在已经得到了W,H矩阵，就可以对任意一个用户u的评分排序了。注意输出的W,H矩阵分别在values[0]和values[1]中。
    """
    # 0号用户对这个用户对所有电影的预测评分
    session1 = tf.compat.v1.Session()
    u1_dim = tf.expand_dims(values[0][0], 0)
    u1_all = tf.matmul(u1_dim, values[1], transpose_b=True)
    result_1 = session1.run(u1_all)
    print(result_1)

    print("以下是给用户0的推荐：")
    p = np.squeeze(result_1)
    p[np.argsort(p)[:-5]] = 0
    for index in range(len(p)):
        if p[index] != 0:
            print(index, p[index])