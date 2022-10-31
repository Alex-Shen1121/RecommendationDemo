import random
import math
import pandas as pd
import numpy as np
import math

class SVDplusplus():
    def __init__(self, allfile, trainfile, testfile, latentFactorNum=20,alpha_u=0.01,alpha_v=0.01,alpha_w=0.01,beta_u=0.01,beta_v=0.01,learning_rate=0.01):
        data_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        # all data file
        allData = pd.read_table(allfile, names=data_fields)
        user_list=sorted(set(allData['user_id'].values))
        item_list=sorted(set(allData['item_id'].values))
        ua_base=allData.sample(n=90570,replace=False)
        self.test_df=allData.drop(ua_base.index,axis=0)
        #ua_base_implict only have (user_id,item_id)
        self.ua_base_implcit=ua_base.sample(frac=0.5,replace=False)
        self.ua_base_explict=ua_base.drop(self.ua_base_implcit.index,axis=0)
        self.implict=self.ua_base_implcit.pivot(index='user_id', columns='item_id', values='rating')
        print(self.test_df.shape)
        print(self.ua_base_explict.shape)
        print(self.ua_base_implcit.shape)

        data_df = pd.DataFrame(index=user_list, columns=item_list)
        rating_matrix=self.ua_base_explict.pivot(index='user_id', columns='item_id', values='rating')
        data_df.update(rating_matrix)
        self.rating_matrix=data_df
        # training set file
        #self.train_df = pd.read_table(trainfile, names=data_fields)
        # testing set file
        #self.test_df=pd.read_table(testfile, names=data_fields)
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
        self.alpha_w=alpha_w
        self.beta_u=beta_u
        self.beta_v=beta_v
        # initialize the model and parameters
        self.initModel()

    # initialize all parameters
    def initModel(self):
        self.mu = self.ua_base_explict['rating'].mean()
        self.bu=(self.rating_matrix-self.mu).sum(axis=1)/self.rating_matrix.count(axis=1)
        self.bu=self.bu.values#dataFrame转numpy
        print(self.bu.shape)
        self.bi = (self.rating_matrix - self.mu).sum() / self.rating_matrix.count()
        self.bi = self.bi.values  # dataFrame转numpy
        self.bi[np.isnan(self.bi)]=0 #填充缺失值

        # r = (np.random.random(1)[0]-0.05)*0.01
        # np.mat((np.random.rand(self.userNum, self.latentFactorNum)-0.05)*0.01)
        self.U = np.mat((np.random.rand(self.userNum, self.latentFactorNum)-0.05)*0.01)
        self.V = np.mat((np.random.rand(self.itemNum, self.latentFactorNum)-0.05)*0.01)
        self.W = np.mat((np.random.rand(self.itemNum, self.latentFactorNum)-0.05)*0.01)
        # self.bu = [0.0 for i in range(self.userNum)]
        # self.bi = [0.0 for i in range(self.itemNum)]
        # temp = math.sqrt(self.latentFactorNum)
        # self.U = [[(0.1 * random.random() / temp) for i in range(self.latentFactorNum)] for j in range(self.userNum)]
        # self.V = [[0.1 * random.random() / temp for i in range(self.latentFactorNum)] for j in range(self.itemNum)]

        print("Initialize end.The user number is:%d,item number is:%d" % (self.userNum, self.itemNum))

    def train(self, iterTimes=100):
        print("Beginning to train the model......")
        preRmse = 10000.0
        temp_count = 0
        for iter in range(iterTimes):
            count=0
            for index in self.ua_base_explict.index:
                user = int(self.ua_base_explict.loc[index]['user_id'])-1
                item = int(self.ua_base_explict.loc[index]['item_id'])-1
                rating = float(self.ua_base_explict.loc[index]['rating'])
                pscore = self.predictScore(self.mu, self.bu[user], self.bi[item], self.U[user], self.V[item],self.W[item],user+1)
                eui = rating - pscore
                # update parameters bu and bi(user rating bais and item rating bais)
                self.mu= -eui
                self.bu[user] += self.learningRate * (eui - self.beta_u * self.bu[user])
                self.bi[item] += self.learningRate * (eui - self.beta_v * self.bi[item])

                temp_Uuser = self.U[user]
                temp_Vitem = self.V[item]

                if user+1 in self.implict.index:
                    temp = self.implict.loc[user+1][self.implict.loc[user+1].isnull() == False]
                    U_bar = self.W[temp.index-1].sum()/temp.count()
                else:
                    U_bar = np.zeros(self.latentFactorNum)
                self.U[user] += self.learningRate * (eui * self.V[user] - self.alpha_u * self.U[user])
                self.V[item] += self.learningRate * ((temp_Uuser+U_bar) * eui - self.alpha_v * self.V[item])
                if user+1 in self.implict.index:
                    self.W[item] += self.learningRate * (eui * temp_Vitem / math.sqrt(self.implict.loc[user+1].count())- self.alpha_w * self.W[item])
                else:
                    self.W[item] += self.learningRate * (eui * temp_Vitem - self.alpha_w * self.W[item])
                # for k in range(self.latentFactorNum):
                #     temp = self.U[user][k]
                #     # update U,V
                #     self.U[user][k] += self.learningRate * (eui * self.V[user][k] - self.alpha_u * self.U[user][k])
                #     self.V[item][k] += self.learningRate * (temp * eui - self.alpha_v * self.V[item][k])
                #
                count += 1
                if count  % 5000 == 0 :
                    print("第%s轮进度：%s/%s" %(iter+1,count,len(self.ua_base_explict.index)))
                    # calculate the current rmse
            self.learningRate = self.learningRate * 0.9 # 缩减学习率
            curRmse = self.test()
            print("Iteration %d times,RMSE is : %f" % (iter + 1, curRmse))
            if curRmse > preRmse:
                break
            else:
                preRmse = curRmse
        print("Iteration finished!")

    # test on the test set and calculate the RMSE
    def test(self):
        cnt = self.test_df.shape[0]
        rmse = 0.0

        # buT=bu.reshape(bu.shape[0],1)
        # predict_rate_matrix = mu + np.tile(buT,(1,self.itemNum))+ np.tile(bi,(self.userNum,1)) +  self.U * self.V.T
        cur = 0
        for i in self.test_df.index:
            cur +=1
            if cur % 1000 == 0:
                print("测试进度:%s/%s" %(cur,len(self.test_df.index)))
            user = int(self.test_df.loc[i]['user_id']) - 1
            item = int(self.test_df.loc[i]['item_id']) - 1
            score = float(self.test_df.loc[i]['rating'])
            pscore = self.predictScore(self.mu,self.bu[user], self.bi[item], self.U[user], self.V[item],self.W[item],user+1)
            # pscore = predict_rate_matrix[user,item]
            rmse += math.pow(score - pscore, 2)
            #print(score,pscore,rmse)
        RMSE=math.sqrt(rmse / cnt)
        return RMSE


    # calculate the inner product of two vectors
    def innerProduct(self, v1, v2):
        result = 0.0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result

    def predictScore(self, mu, bu, bi, U, V, W ,user_id):
        #pscore = mu + bu + bi + self.innerProduct(U, V)
        if user_id in self.implict.index:
            temp = self.implict.loc[user_id][self.implict.loc[user_id].isnull() == False]
            U_bar = self.W[temp.index-1].sum() / temp.count()
        else:
            U_bar = np.zeros(self.latentFactorNum)
        pscore = mu + bu + bi + np.multiply(U,V).sum() +np.multiply(U_bar,V).sum()
        if np.isnan(pscore):
            print("!!!!")
            print(mu,bu,bi,np.multiply(U,V).sum(),np.multiply(U_bar,V).sum(),U_bar)
        if pscore < 1:
            pscore = 1
        if pscore > 5:
            pscore = 5
        return pscore


if __name__ == '__main__':
    s = SVDplusplus("./data/u.data", "./data/u2.base", ".../data/u2.test")
    s.train()

