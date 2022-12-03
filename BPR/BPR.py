import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import score
from tqdm import tqdm


class BPR:
    # 用户数
    user_count = 943
    # 项目数
    item_count = 1682
    # k个主题,k数
    latent_factors = 20
    # 步长α
    lr = 0.01
    # 参数λ
    reg = 0.01
    # 训练次数
    train_count = 500
    # 训练集
    train_data_path = '../datasets/ml-100k/u1.base.OCCF'
    # 测试集
    test_data_path = '../datasets/ml-100k/u1.test.OCCF'
    # U-I的大小
    size_u_i = user_count * item_count
    # 随机设定的U，V矩阵(即公式中的Wuk和Hik)矩阵
    U = None  # 大小无所谓
    V = None
    biasV = None

    # 生成一个用户数*项目数大小的全0矩阵
    test_data = np.zeros((user_count, item_count))
    # 生成一个一维的全0矩阵
    test = np.zeros(size_u_i)
    # 再生成一个一维的全0矩阵
    predict_ = np.zeros(size_u_i)

    # 获取U-I数据对应
    def load_data(self, path):
        user_ratings = defaultdict(set)
        linecnt = 0
        with open(path, 'r') as f:
            for line in f.readlines():
                linecnt += 1
                u, i, _, _ = line.split("\t")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
        return user_ratings, linecnt

    # 获取测试集的评分矩阵
    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split('\t')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1

    def train(self, user_ratings_train):
        all_pairs = [(k, val) for k, v in user_ratings_train.items() for val in v]
        for user in range(self.user_count):
            # # 随机获取一个用户
            # u = random.randint(1, self.user_count)
            # # 训练集和测试集的用于不是全都一样的,比如train有948,而test最大为943
            # if u not in user_ratings_train.keys():
            #     continue
            # # 从用户的U-I中随机选取1个Item
            # i = random.sample(user_ratings_train[u], 1)[0]
            (u, i) = random.choice(all_pairs)
            # 随机选取一个用户u没有评分的项目
            j = random.randint(1, self.item_count)
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count)
            # python中的取值从0开始
            u = u - 1
            i = i - 1
            j = j - 1
            # BPR
            r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
            r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
            r_uij = r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_uij))
            # 更新2个矩阵
            self.U[u] += -self.lr * (loss_func * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (loss_func * self.U[u] + self.reg * self.V[i])
            self.V[j] += -self.lr * (loss_func * (-self.U[u]) + self.reg * self.V[j])
            # 更新偏置项
            self.biasV[i] += -self.lr * (loss_func + self.reg * self.biasV[i])
            self.biasV[j] += -self.lr * (-loss_func + self.reg * self.biasV[j])

    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    # 主函数
    def main(self):
        # 获取U-I的{1:{2,5,1,2}....}数据
        user_ratings_train, all = self.load_data(self.train_data_path)
        # 获取测试集的评分矩阵
        self.load_test_data(self.test_data_path)

        # 初始化参数
        # 随机设定的U，V矩阵(即公式中的Wuk和Hik)矩阵
        self.U = (np.random.rand(self.user_count, self.latent_factors) - 0.5) * 0.01  # 大小无所谓
        self.V = (np.random.rand(self.item_count, self.latent_factors) - 0.5) * 0.01
        self.biasV = np.zeros(self.item_count)
        mu = all / (self.user_count * self.item_count)
        for i in range(self.item_count):
            y = 0
            for j in user_ratings_train.values():
                if i in j:
                    y += 1
            self.biasV[i] = y / self.user_count - mu

        # 将test_data矩阵拍平
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0

        # 训练
        for i in tqdm(range(self.train_count)):
            self.train(user_ratings_train)  # 训练1000次完成
        predict_matrix = self.predict(self.U, self.V)  # 将训练完成的矩阵內积
        # 预测
        self.predict_ = predict_matrix.getA().reshape(-1)  # .getA()将自身矩阵变量转化为ndarray类型的变量
        self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)
        # Top-K evaluation
        score.topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count)


def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict


if __name__ == '__main__':
    # 调用类的主函数
    bpr = BPR()
    bpr.main()
