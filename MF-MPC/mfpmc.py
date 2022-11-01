import numpy as np
import random
from tqdm import tqdm
import math


class MF_MPC:
    def __init__(self):
        self.d = 20  # 潜在特征向量的维度
        self.num_rating_types = 10  # 评分类型数 0.5/1/.../5
        # todo
        self.flagGraded = True
        # todo
        self.alpha_u = 0.01
        self.alpha_v = 0.01
        self.alpha_g = 0.01

        # todo
        self.beta_u = 0.01
        self.beta_v = 0.01

        # 学习率
        self.gamma = 0.01
        self.gamma_decay = 0.9

        # 文件路径
        self.fnTrainData = "../datasets/ml-100k/u2.base"
        self.fnTestData = "../datasets/ml-100k/u2.test"

        self.n = 943  # 用户数量
        self.m = 1682  # 物品数量
        self.num_train = 0  # 训练条数

        self.num_test = 0

        self.MinRating = 0.5  # 评分最小值（ML10M：0.5，Netflix：1）
        self.MaxRating = 5  # 评分最大值

        self.num_iterations = 100  # 迭代次数

        self.Train_ExplicitFeedbacksGraded = {}

        # === 训练数据
        self.indexUserTrain = None
        self.indexItemTrain = None
        self.ratingTrain = None

        # === 测试数据
        self.ratingTest = None
        self.indexItemTest = None
        self.indexUserTest = None

        # === 一些统计量
        self.userRatingSumTrain = np.zeros(self.n + 1)
        self.itemRatingSumTrain = np.zeros(self.m + 1)
        self.userRatingNumTrain = np.zeros(self.n + 1)
        self.itemRatingNumTrain = np.zeros(self.m + 1)

        self.user_graded_rating_number = np.zeros([self.n + 1, self.num_rating_types + 1])

        # === 模型参数
        self.U = np.zeros([self.n + 1, self.d])
        self.V = np.zeros([self.m + 1, self.d])

        self.G = np.zeros([self.m + 1, self.num_rating_types + 1, self.d])  # 关于评分为r的物品i'的潜在特征向量

        self.g_avg = 0  # 全局平均得分

        self.biasU = np.zeros(self.n + 1)
        self.biasV = np.zeros(self.m + 1)

    def readData(self):
        print("-------------- Reading data... --------------")
        import re
        # === 读取训练数据
        f = open(self.fnTrainData, 'r')
        lines = f.readlines()
        f.close()

        self.num_train = len(lines)

        # === 训练数据
        self.indexUserTrain = np.zeros(self.num_train)
        self.indexItemTrain = np.zeros(self.num_train)
        self.ratingTrain = np.zeros(self.num_train)

        id_case = 0
        ratingSum = 0
        for line in lines:
            li = re.split('\t|\n', line)
            userID = int(li[0])
            itemID = int(li[1])
            rating = float(li[2])
            self.indexUserTrain[id_case] = int(userID)
            self.indexItemTrain[id_case] = int(itemID)
            self.ratingTrain[id_case] = float(rating)
            id_case += 1

            # ===
            self.userRatingSumTrain[userID] += rating
            self.itemRatingSumTrain[itemID] += rating
            self.userRatingNumTrain[userID] += 1
            self.itemRatingNumTrain[itemID] += 1

            ratingSum += rating

            if self.flagGraded:
                g = int(rating * 2)
                if self.Train_ExplicitFeedbacksGraded.get(userID) is not None:
                    g2itemSet = self.Train_ExplicitFeedbacksGraded[userID]
                    if g2itemSet.get(g) is not None:
                        g2itemSet[g].add(itemID)
                    else:
                        g2itemSet[g] = {itemID}
                    self.Train_ExplicitFeedbacksGraded[userID] = g2itemSet
                else:
                    self.Train_ExplicitFeedbacksGraded[userID] = {g: {itemID}}
                self.user_graded_rating_number[userID][g] += 1
        print("Finished reading the target training data")

        self.g_avg = float(ratingSum) / self.num_train

        # === 读取测试数据
        f = open("../datasets/ml-100k/u2.test", 'r')
        lines = f.readlines()
        f.close()

        self.num_test = len(lines)

        # === 测试数据
        self.indexUserTest = np.zeros(self.num_test)
        self.indexItemTest = np.zeros(self.num_test)
        self.ratingTest = np.zeros(self.num_test)

        id_case = 0
        data = []
        for line in lines:
            li = re.split('\t|\n', line)
            userID = int(li[0])
            itemID = int(li[1])
            rating = float(li[2])
            self.indexUserTest[id_case] = int(userID)
            self.indexItemTest[id_case] = int(itemID)
            self.ratingTest[id_case] = float(rating)
            id_case += 1

        print("Finished reading the target testing data")

    def initializeModel(self):
        # === initialization of U, V, P, N, G
        print("-------------- Initializing model... --------------")
        self.U = (np.random.rand(self.n + 1, self.d) - 0.5) * 0.01
        self.U[0] = np.zeros(self.d)
        self.V = (np.random.rand(self.m + 1, self.d) - 0.5) * 0.01
        self.V[0] = np.zeros(self.d)

        if self.flagGraded:
            self.G = (np.random.rand(self.m + 1, self.num_rating_types + 1, self.d) - 0.5) * 0.01
            self.G[0] = np.zeros([self.num_rating_types + 1, self.d])

        self.biasU = np.zeros(self.n + 1)
        for u in range(1, self.n + 1):
            if self.userRatingNumTrain[u] > 0:
                self.biasU[u] = (self.userRatingSumTrain[u] - self.g_avg * self.userRatingNumTrain[u]) \
                                / self.userRatingNumTrain[u]

        self.biasV = np.zeros(self.m + 1)
        for i in range(1, self.m + 1):
            if self.itemRatingNumTrain[i] > 0:
                self.biasV[i] = (self.itemRatingSumTrain[i] - self.g_avg * self.itemRatingNumTrain[i]) \
                                / self.itemRatingNumTrain[i]

    def train(self):
        print("-------------- Training model... --------------")
        for iter in tqdm(range(self.num_iterations)):
            # === testing
            self.test()

            # === training
            for iter_rand in range(self.num_train):
                # === randomly sample a training instance
                rand_case = np.random.randint(0, self.num_train)
                userID = int(self.indexUserTrain[rand_case])
                itemID = int(self.indexItemTrain[rand_case])
                rating = self.ratingTrain[rand_case]

                tilde_Uu_g = np.zeros(self.d)
                tilde_Uu = np.zeros(self.d)

                if self.flagGraded:
                    for g in range(1, self.num_rating_types + 1):
                        if self.user_graded_rating_number[userID][g] > 0:
                            itemSet = self.Train_ExplicitFeedbacksGraded[userID][g]
                            explicit_feefback_num_u_sqrt = 0
                            if itemID in itemSet \
                                    and len(itemSet) > 1:
                                explicit_feefback_num_u_sqrt = \
                                    np.sqrt(self.user_graded_rating_number[userID][g] - 1)
                            else:
                                explicit_feefback_num_u_sqrt = \
                                    math.sqrt(self.user_graded_rating_number[userID][g])

                            if explicit_feefback_num_u_sqrt > 0:
                                for itemID2 in itemSet:
                                    if itemID2 != itemID:
                                        for f in range(self.d):
                                            tilde_Uu_g[f] += self.G[itemID2][g][f]

                                for f in range(self.d):
                                    tilde_Uu_g[f] /= explicit_feefback_num_u_sqrt
                                    tilde_Uu[f] += tilde_Uu_g[f]
                                    tilde_Uu_g[f] = 0

                # 预测与误差
                pred = self.g_avg + self.biasU[userID] + self.biasV[itemID] \
                       + np.dot(self.U[userID], self.V[itemID]) \
                       + np.dot(tilde_Uu, self.V[itemID])
                err = rating - pred

                # === 更新参数
                self.g_avg -= self.gamma * (-err)
                self.biasU[userID] -= self.gamma * (-err - self.beta_u * self.biasU[userID])
                self.biasV[itemID] -= self.gamma * (-err - self.beta_v * self.biasV[itemID])

                V_before_update = np.zeros(self.d)
                U_before_update = np.zeros(self.d)

                for f in range(self.d):
                    V_before_update[f] = self.V[itemID][f]
                    U_before_update[f] = self.U[userID][f]

                    grad_U_f = -err * self.V[itemID][f] + self.alpha_u * self.U[userID][f]
                    grad_V_f = -err * (self.U[userID][f] + tilde_Uu[f]) + self.alpha_v * self.V[itemID][f]
                    self.U[userID][f] -= self.gamma * grad_U_f
                    self.V[itemID][f] -= self.gamma * grad_V_f

                if self.flagGraded:
                    for g in range(1, self.num_rating_types + 1):
                        if self.user_graded_rating_number[userID][g] > 0:
                            itemSet = self.Train_ExplicitFeedbacksGraded[userID][g]
                            explicit_feefback_num_u_sqrt = 0
                            if itemID in itemSet \
                                    and len(itemSet) > 1:
                                explicit_feefback_num_u_sqrt = \
                                    np.sqrt(self.user_graded_rating_number[userID][g] - 1)
                            else:
                                explicit_feefback_num_u_sqrt = \
                                    math.sqrt(self.user_graded_rating_number[userID][g])

                            if explicit_feefback_num_u_sqrt > 0:
                                for itemID2 in itemSet:
                                    if itemID2 != itemID:
                                        for f in range(self.d):
                                            self.G[itemID2][g][f] -= self.gamma * \
                                                                     (-err * V_before_update[f] /
                                                                      explicit_feefback_num_u_sqrt +
                                                                      self.alpha_g * self.G[itemID2][g][f])
            self.gamma *= self.gamma_decay

    def test(self):
        mae, rmse = 0, 0

        tilde_Uu_g = np.zeros(self.d)
        tilde_Uu = np.zeros([self.n + 1, self.d])

        for userID in range(1, self.n + 1):
            if self.flagGraded:
                for g in range(1, self.num_rating_types + 1):
                    if self.user_graded_rating_number[userID][g] > 0:
                        explicit_feedback_num_u_sqrt = \
                            math.sqrt(self.user_graded_rating_number[userID][g])
                        itemSet = self.Train_ExplicitFeedbacksGraded[userID][g]
                        for itemID in itemSet:
                            for f in range(self.d):
                                tilde_Uu_g[f] += self.G[itemID][g][f]

                        # === 正则化
                        for f in range(self.d):
                            tilde_Uu[userID][f] += tilde_Uu_g[f] / explicit_feedback_num_u_sqrt
                            tilde_Uu_g[f] = 0

        for t in range(self.num_test):
            userID = int(self.indexUserTest[t])
            itemID = int(self.indexItemTest[t])
            rating = float(self.ratingTest[t])

            # === 计算预测值
            pred = self.g_avg + self.biasU[userID] + self.biasV[itemID]

            for f in range(self.d):
                pred += self.U[userID][f] * self.V[itemID][f] + tilde_Uu[userID][f] * self.V[itemID][f]

            if pred < self.MinRating:
                pred = self.MinRating
            if pred > self.MaxRating:
                pred = self.MaxRating

            err = rating - pred
            mae += abs(err)
            rmse += err ** 2

        MAE = mae / self.num_test
        RMSE = math.sqrt(rmse / self.num_test)

        print(f"MAE: {MAE}, RMSE: {RMSE}")


if __name__ == '__main__':
    a = MF_MPC()
    a.readData()
    a.initializeModel()
    a.train()
