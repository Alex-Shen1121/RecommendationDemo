from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy import linalg as la
from scipy.sparse import diags


def read_data(path):
    # Read data from file
    data = pd.read_csv(path, sep='\t', names=[
                       'user_id', 'item_id', 'rating', 'timestamp'])
    userNum = 943
    itemNum = 1682

    R = np.zeros((userNum, itemNum))
    # 将u2.base的数据存入矩阵
    for row in data.itertuples():
        userID,  itemID, rating = row[1]-1, row[2]-1, row[3]
        R[userID, itemID] = rating

    testData = pd.read_csv('./data/u2.test', sep='\t',
                           names=['user_id', 'item_id', 'rating', 'timestamp'])

    return R, testData, userNum, itemNum


def cal_basic_stat(R, userNum, itemNum):
    # 一些统计量及计算公式

    # 是否有评分
    # y_ui[i][j] --- 用户i对物品j是否有评分
    y_ui = np.zeros((userNum, itemNum))
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] != 0:
                y_ui[i][j] = 1

    # 物品平均评分
    r = np.sum(y_ui * R) / np.sum(y_ui)

    # 用户平均评分
    # r_u[i, 0] --- 用户i的平均评分
    r_u = np.zeros((userNum, 1))
    for i in range(userNum):
        if np.sum(y_ui[i, :]) == 0:
            r_u[i] = r
        else:
            r_u[i] = np.sum(y_ui[i, :] * R[i, :]) / np.sum(y_ui[i, :])

    # 物品平均评分
    # r_i[i, 0] --- 物品i的平均评分
    r_i = np.zeros((itemNum, 1))
    for i in range(itemNum):
        if np.sum(y_ui[:, i]) == 0:
            r_i[i] = r
        else:
            r_i[i] = np.sum(R[:, i]) / np.sum(y_ui[:, i])

    # 用户对物品的评分偏差
    # b_u[i, 0] --- 用户i对所有物品的评分偏差
    b_u = np.zeros((userNum, 1))
    for i in range(userNum):
        if np.sum(y_ui[i, :]) == 0:
            continue
        b_u[i] = np.sum(y_ui[i, :] * (R[i, :] - r_i[:, 0])) / \
                        np.sum(y_ui[i, :])

    # 物品对用户的评分偏差
    # b_i[i, 0] --- 物品i对所有用户的评分偏差
    b_i = np.zeros((itemNum, 1))
    for i in range(itemNum):
        if np.sum(y_ui[:, i]) == 0:
            continue
        b_i[i] = np.sum(y_ui[:, i] * (R[:, i] - r_u[:, 0])) / \
                        np.sum(y_ui[:, i])

    return y_ui, r, r_u, r_i, b_u, b_i


def SVDplusplus(R, testData, userNum, itemNum, y_ui, r, r_u, r_i, b_u, b_i, K, alpha, lamda, epoch):
    # SVD++算法
    # 初始化参数
    p_u = np.random.rand(userNum, K)
    q_i = np.random.rand(itemNum, K)
    y_i = np.random.rand(itemNum, K)
    b_u = np.zeros((userNum, 1))
    b_i = np.zeros((itemNum, 1))
    b = r

    # 训练
    for step in tqdm(range(epoch)):
        for i in range(userNum):
            for j in range(itemNum):
                if y_ui[i][j] == 1:
                    # e_uij = R[i][j] - b - b_u[i] - b_i[j] - np.dot(p_u[i, :], q_i[j, :].T) - np.sum(y_i[j, :] * (p_u[i, :] - np.sum(y_ui[i, :] * p_u[i, :], axis=0) / np.sum(y_ui[i, :])))
                    e_uij = R[i][j] - b - b_u[i] - b_i[j] - np.dot(p_u[i, :], q_i[j, :].T) - \
                        np.dot(np.average(np.dot(diags(y_ui[i, :], shape=(
                            itemNum, itemNum)).toarray(), y_i), axis=0), q_i[j, :].T)
                    b = b + alpha * (e_uij - lamda * b)
                    b_u[i] = b_u[i] + alpha * (e_uij - lamda * b_u[i])
                    b_i[j] = b_i[j] + alpha * (e_uij - lamda * b_i[j])
                    p_u[i, :] = p_u[i, :] + alpha * (e_uij * q_i[j, :] - lamda * p_u[i, :])
                    q_i[j, :] = q_i[j, :] + alpha * (e_uij * (p_u[i, :] - np.dot(np.average(np.dot(diags(y_ui[i, :], shape=(itemNum, itemNum)).toarray(), y_i), axis=0), q_i[j, :].T)))
                    y_i[j, :]=y_i[j, :] + alpha * (e_uij * (p_u[i, :] - np.dot(np.average(np.dot(diags(y_ui[i, :], shape=(itemNum, itemNum)).toarray(), y_i), axis=0), q_i[j, :].T)))
        alpha=alpha * 0.9

    # 预测
    # R_hat[i][j] --- 用户i对物品j的预测评分
    R_hat=np.zeros((userNum, itemNum))
    for i in range(userNum):
        for j in range(itemNum):
            R_hat[i][j]=r + r_u[i] + r_i[j] + b_u[i] + \
                b_i[j] + np.dot(p_u[i, :], q_i[j, :].T)

    return R_hat

def var_name(var, all_var=locals()):
    return [var_name for var_name in all_var if all_var[var_name] is var][0]

def MAE(matrix, test):
    sum=0
    for row in test.itertuples():
        userID,  itemID, rating=row[1]-1, row[2]-1, row[3]
        sum += abs(matrix[userID, itemID] - rating)
    print(var_name(matrix), 'MAE: ', sum / len(test))

if __name__ == "__main__":
    # 读取数据
    R, testData, userNum, itemNum=read_data("./data/u2.base")

    # 计算常规统计量
    y_ui, r, r_u, r_i, b_u, b_i=cal_basic_stat(R, userNum, itemNum)

    SVDplusplus_Res=SVDplusplus(R, testData, userNum, itemNum, y_ui,
                                r, r_u, r_i, b_u, b_i, K=10, alpha=0.01, lamda=0.01, epoch=100)
    MAE(SVDplusplus_Res, testData)
