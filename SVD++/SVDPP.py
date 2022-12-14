import numpy as np
import random
from tqdm import tqdm
import math


class SVDPP:
    def __init__(self, mat, d=20):
        self.mat = np.array(mat)
        # 潜在特征向量的纬度 d = 20
        self.d = d
        # 物品偏置 {iid: bi}
        self.bi = {}
        # 用户偏置 {uid: bu}
        self.bu = {}
        # 物品特征向量 {iid: qi}
        self.qi = {}
        # 用户特征向量 {uid: pu}
        self.pu = {}
        # 所有样本的平均打分
        self.avg = np.mean(self.mat[:, 2])
        self.y = {}
        # 用户评价过的物品 {uid:[iid1,iid2,...]}
        self.u_dict = {}
        for i in range(self.mat.shape[0]):
            uid = self.mat[i, 0]
            iid = self.mat[i, 1]
            self.u_dict.setdefault(uid, [])
            self.u_dict[uid].append(iid)
            self.bi.setdefault(iid, 0)
            self.bu.setdefault(uid, 0)
            self.qi.setdefault(iid, np.random.random(
                (self.d, 1))/10*np.sqrt(self.d))
            self.pu.setdefault(uid, np.random.random(
                (self.d, 1))/10*np.sqrt(self.d))
            self.y.setdefault(iid, np.zeros((self.d, 1))+.1)

    def predict(self, uid, iid):  # 预测评分的函数
        # setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu及用户评价过的物品u_dict，并设置初始值为0
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros((self.d, 1)))
        self.pu.setdefault(uid, np.zeros((self.d, 1)))
        self.y.setdefault(uid, np.zeros((self.d, 1)))
        self.u_dict.setdefault(uid, [])
        u_impl_prf, sqrt_Nu = self.getY(uid, iid)
        rating = self.avg+self.bi[iid]+self.bu[uid] + \
            np.sum(self.qi[iid]*(self.pu[uid]+u_impl_prf))  # 预测评分公式
        # 由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    # 计算sqrt_Nu和∑yj
    def getY(self, uid, iid):
        # 用户评价过的物品列表Nu
        Nu = self.u_dict[uid]
        I_Nu = len(Nu)
        sqrt_Nu = np.sqrt(I_Nu)
        y_u = np.zeros((self.d, 1))
        if I_Nu == 0:
            u_impl_prf = y_u
        else:
            for i in Nu:
                y_u += self.y[i]
            u_impl_prf = y_u / sqrt_Nu

        return u_impl_prf, sqrt_Nu

    def train(self, steps=30, gamma=0.01, Lambda=0.01):  # 训练函数，step为迭代次数。
        print('train data size', self.mat.shape)
        for step in tqdm(range(steps)):
            # print('step',step+1,'is running')
            KK = np.random.permutation(
                self.mat.shape[0])  # 随机梯度下降算法，kk为对矩阵进行随机洗牌
            mae = 0.0
            for i in range(self.mat.shape[0]):
                # 随机选取一个样本
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                rating = self.mat[j, 2]
                
                predict = self.predict(uid, iid)
                
                u_impl_prf, sqrt_Nu = self.getY(uid, iid)
                eui = rating-predict
                mae += math.fabs(eui)
                self.bu[uid] += gamma*(eui-Lambda*self.bu[uid])
                self.bi[iid] += gamma*(eui-Lambda*self.bi[iid])
                self.pu[uid] += gamma*(eui*self.qi[iid]-Lambda*self.pu[uid])
                self.qi[iid] += gamma * \
                    (eui*(self.pu[uid]+u_impl_prf)-Lambda*self.qi[iid])
                for j in self.u_dict[uid]:
                    self.y[j] += gamma * \
                        (eui*self.qi[j]/sqrt_Nu-Lambda*self.y[j])
            # 减小学习率
            gamma = 0.9*gamma
            print('mae is', mae/self.mat.shape[0])

    def test(self, test_data):  # gamma以0.9的学习率递减

        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        mae = 0.0
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating-self.predict(uid, iid)
            mae += math.fabs(eui)
        print('mae of test data is', mae/test_data.shape[0])


def getMLData():  # 获取训练集和测试集的函数
    import re
    f = open("./data/u2.base", 'r')
    lines = f.readlines()
    f.close()
    data = []
    for line in lines:
        list = re.split('\t|\n', line)
        if int(list[2]) != 0:
            data.append([int(i) for i in list[:3]])
    train_data = data
    f = open("./data/u2.test", 'r')
    lines = f.readlines()
    f.close()
    data = []
    for line in lines:
        list = re.split('\t|\n', line)
        if int(list[2]) != 0:
            data.append([int(i) for i in list[:3]])
    test_data = data

    return train_data, test_data


if __name__ == '__main__':
    train_data, test_data = getMLData()
    a = SVDPP(train_data, 30)
    a.train()
    a.test(test_data)
