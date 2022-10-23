## Average Filling

## Average Filling

**AF算法使用二维矩阵保存用户** `user`对于物品 `item`的喜爱程度。

**其中** `mat[i][j] = k`代表的 `用户i`对于 `物品j`的喜爱程度为k

![image-20221024022934221](https://oss.codingshen.com/uPic/image-20221024022934221.png)

## 符号

![image-20221024032056603](https://oss.codingshen.com/uPic/image-20221024032056603.png)

![image-20221024032104568](https://oss.codingshen.com/uPic/image-20221024032104568.png)

## 问题定义

![image-20221024025049276](https://oss.codingshen.com/uPic/image-20221024025049276.png)

**根据输入的矩阵，填写未观察到的rating值，并且进行评估**

## 其他统计量

1. **平均得分**
2. **用户的平均打分**
3. **物品的平均得分**

![image-20221024025141562](https://oss.codingshen.com/uPic/image-20221024025141562.png)

4. **用户的偏移值**
5. **物品的偏移值**

![image-20221024025235534](https://oss.codingshen.com/uPic/image-20221024025235534.png)

## 预测原则

**可以通过如下的填充方式，对未观察到的值进行填充**

![image-20221024025327081](https://oss.codingshen.com/uPic/image-20221024025327081.png)

## 结论

**对于MovieLen100K数据集进行计算，发现MAE最小的是user bias and item average方法**

![image-20221024025430154](https://oss.codingshen.com/uPic/image-20221024025430154.png)

## 资料

1. **Slide：**[https://csse.szu.edu.cn/staff/panwk/recommendation/IRT-Ch02-CF-Slides/AF.pdf](https://csse.szu.edu.cn/staff/panwk/recommendation/IRT-Ch02-CF-Slides/AF.pdf)
2. **Code：**[https://github.com/Alex-Shen1121/RecommendationDemo/tree/master/Average%20Fil](https://github.com/Alex-Shen1121/RecommendationDemo/tree/master/Average%20Fill)

AF算法使用二维矩阵保存用户 `user`对于物品 `item`的喜爱程度。

其中 `mat[i][j] = k`代表的 `用户i`对于 `物品j`的喜爱程度为k

![image-20221024022934221](https://oss.codingshen.com/uPic/image-20221024022934221.png)

## 符号

- user number: $n$
- item number: $m$
- user ID: $u ∈ {1, 2, . . . , n}$
- item ID: $i ∈ {1, 2, . . . , m} $
- grade score set (or rating range): $G, e.g., G = {1, 2, 3, 4, 5} $
- observed rating of user u on item i: $r_{ui} ∈ G $
- predicted rating of user u on item $i: \hat{r}_{ui}$
- Training data
  - indicator variable: $y_{ui} = \left\{\begin{array}{**lr**}  1, & if\ (u,i,r_{ui})\ is\ observed  \\ 0, & if\ (u,i,r_{ui})\ is \ not\ observed \end{array}\right. $
  - rating records: $R = \{(u, i, r_{ui})\}$
  - rating matrix: $R ∈ \{G∪?\}^{n×m}$
  - number of observed ratings: $p = \sum _{u,i}y_{ui} = |R|$
  - density (or sometimes called sparsity): $\frac{p}{nm}$
- Test data
  - rating records: $R^{te} = \{(u, i, r_{ui})\}$

## 问题定义

![image-20221024025049276](https://oss.codingshen.com/uPic/image-20221024025049276.png)

根据输入的矩阵，填写未观察到的rating值，并且进行评估

## 其他统计量

1. 平均得分
2. 用户的平均打分
3. 物品的平均得分

![image-20221024025141562](https://oss.codingshen.com/uPic/image-20221024025141562.png)

4. 用户的偏移值
5. 物品的偏移值

![image-20221024025235534](https://oss.codingshen.com/uPic/image-20221024025235534.png)

## 预测原则

可以通过如下的填充方式，对未观察到的值进行填充

![image-20221024025327081](https://oss.codingshen.com/uPic/image-20221024025327081.png)

## 结论

对于MovieLen100K数据集进行计算，发现MAE最小的是user bias and item average方法

![image-20221024025430154](https://oss.codingshen.com/uPic/image-20221024025430154.png)

## 资料

1. Slide：https://csse.szu.edu.cn/staff/panwk/recommendation/IRT-Ch02-CF-Slides/AF.pdf
2. Code：https://github.com/Alex-Shen1121/RecommendationDemo/tree/master/Average%20Fill
