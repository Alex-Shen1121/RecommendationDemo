# 来源：https://blog.csdn.net/Hemk340200600/article/details/88633646
import random
import math
import pandas as pd
import numpy as np


class RSVD():
    def __init__(self, allfile, trainfile, testfile, latentFactorNum=20,alpha_u=0.01,alpha_v=0.01,beta_u=0.01,beta_v=0.01,learning_rate=0.01):
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        # all data file
        allData = pd.read_table(allfile, names=data_fields)
        # training set file
        self.train_df = pd.read_table(trainfile, names=data_fields)
        # testing set file
        self.test_df=pd.read_table(testfile, names=data_fields)
        # get factor number
        self.latentFactorNum = latentFactorNum
        # get user number
        self.userNum = len(set(allData['user_id'].values))
        # get item number
        self.itemNum = len(set(allData['item_id'].values))
        # learning rate
        self.learningRate = learning_rate
        # the regularization lambda
        self.alpha_u=alpha_u
        self.alpha_v=alpha_v
        self.beta_u=beta_u
        self.beta_v=beta_v
        # initialize the model and parameters
        self.initModel()

    # initialize all parameters
    def initModel(self):
        self.mu = self.train_df['rating'].mean()

        self.bu = np.zeros(self.userNum)
        self.bi = np.zeros(self.itemNum)
        self.U = np.mat(np.random.rand(self.userNum,self.latentFactorNum))
        self.V = np.mat(np.random.rand(self.itemNum,self.latentFactorNum))

        # self.bu = [0.0 for i in range(self.userNum)]
        # self.bi = [0.0 for i in range(self.itemNum)]
        # temp = math.sqrt(self.latentFactorNum)
        # self.U = [[(0.1 * random.random() / temp) for i in range(self.latentFactorNum)] for j in range(self.userNum)]
        # self.V = [[0.1 * random.random() / temp for i in range(self.latentFactorNum)] for j in range(self.itemNum)]

        print("Initialize end.The user number is:%d,item number is:%d" % (self.userNum, self.itemNum))

    def train(self, iterTimes=100):
        print("Beginning to train the model......")
        preRmse = 10000.0
        for iter in range(iterTimes):
            for index in self.train_df.index:
                if index % 20000 == 0 :
                    print("第%s轮进度：%s%%" %(iter,index/len(self.train_df.index)*100))
                user = int(self.train_df.loc[index]['user_id'])-1
                item = int(self.train_df.loc[index]['item_id'])-1 
                rating = float(self.train_df.loc[index]['rating'])
                pscore = self.predictScore(self.mu, self.bu[user], self.bi[item], self.U[user], self.V[item])
                eui = rating - pscore
                # update parameters bu and bi(user rating bais and item rating bais)
                self.mu= -eui
                self.bu[user] += self.learningRate * (eui - self.beta_u * self.bu[user])
                self.bi[item] += self.learningRate * (eui - self.beta_v * self.bi[item])

                temp = self.U[user]
                self.U[user] += self.learningRate * (eui * self.V[user] - self.alpha_u * self.U[user])
                self.V[item] += self.learningRate * (temp * eui - self.alpha_v * self.V[item])

                # for k in range(self.latentFactorNum):
                #     temp = self.U[user][k]
                #     # update U,V
                #     self.U[user][k] += self.learningRate * (eui * self.V[user][k] - self.alpha_u * self.U[user][k])
                #     self.V[item][k] += self.learningRate * (temp * eui - self.alpha_v * self.V[item][k])
                #
            # calculate the current rmse
            curRmse = self.test(self.mu, self.bu, self.bi, self.U, self.V)
            print("Iteration %d times,RMSE is : %f" % (iter + 1, curRmse))
            if curRmse > preRmse:
                break
            else:
                preRmse = curRmse
        print("Iteration finished!")

    # test on the test set and calculate the RMSE
    def test(self, mu, bu, bi, U, V):
        cnt = self.test_df.shape[0]
        rmse = 0.0

        buT=bu.reshape(bu.shape[0],1)
        predict_rate_matrix = mu + np.tile(buT,(1,self.itemNum))+ np.tile(bi,(self.userNum,1)) +  self.U * self.V.T

        for i in self.test_df.index:
            user = int(self.test_df.loc[i]['user_id']) - 1
            item = int(self.test_df.loc[i]['item_id']) - 1
            score = float(self.test_df.loc[i]['rating'])
            #pscore = self.predictScore(mu, bu[user], bi[item], U[user], V[item])
            pscore = predict_rate_matrix[user,item]
            rmse += math.pow(score - pscore, 2)
        RMSE=math.sqrt(rmse / cnt)
        return RMSE


    # calculate the inner product of two vectors
    def innerProduct(self, v1, v2):
        result = 0.0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result

    def predictScore(self, mu, bu, bi, U, V):
        #pscore = mu + bu + bi + self.innerProduct(U, V)
        pscore = mu + bu + bi + np.multiply(U,V).sum()
        if pscore < 1:
            pscore = 1
        if pscore > 5:
            pscore = 5
        return pscore


if __name__ == '__main__':
    s = RSVD("../datasets/ml-100k/u.data", "../datasets/ml-100k/u1.base", "../datasets/ml-100k/u1.test")
    s.train()

