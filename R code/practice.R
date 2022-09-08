### practice #####
##### multiple regression####

setwd("D:\\KARASU\\Graduated\\数据挖掘\\R任务数据 （出版社）\\任务21线性回归分析——新能源上市企业市盈率的影响因素分析")
data1=read.csv("data.csv")

######因子分析###
install.packages("psych")
library(psych)
setwd("D:\\KARASU\\Graduated\\数据挖掘\\R任务数据 （出版社）\\任务22因子分析——上市公司财务数据的因子分析")
data1=read.csv("data0406v9.csv")
round(head(data1),2) #查看前6行数据，并保留小数点两位
corFactor=as.matrix(data1[,-1]) #删除第一列，第一列不参与因子分析
corFactor_s=scale(corFactor)#做标准化，去除数据规模
round(cor(corFactor_s),2) #计算相关系数
summary(data1) #描述性统计
KMO(corFactor_s)  ##KMO检验大于0.5可以进行因子分析
fa.parallel(corFactor_s,fm="pa",n.iter=100,show.legend=F,fa="fa")

fa=fa(corFactor_s,nfactors = 4,rotate = "none",fm="pa",scores = "regression")
fa

fa=fa(corFactor_s,nfactors = 4,rotate = "varimax",fm="pa",scores = "regression")
fa

round(fa$weights,3)

head(round(fa$scores,3))  #查看前6条因子得分，保留3位小数，不用head可以查看全部##
weight=c(0.35,0.14,0.23,0.18) #设置权重
final_score=fa$scores%*%weight  #矩阵运算
head(final_score)  ##查看前6条


####### k-means#####
install.packages("devtools")
library("devtools") 
library("ggplot2")
devtools::install_github("ricardo-bion/ggradar")
library("ggradar")
setwd("D:\\KARASU\\Graduated\\数据挖掘\\R任务数据 （出版社）\\任务23聚类分析——基于能力指标的基金经理人分类")
fundma=read.csv("经理人能力.csv")
fundma.sample=fundma[1:5,c(1,4:8)]##筛选作图用数据
##自定义归一化函数 （投影到[0,1])
myNormalize<-function(target){
  (target-min(target))/(max(target)-min(target))
}
##对数据做逐列归一化处理
fundma.sample[,-1]=apply(fundma.sample[,-1],2,myNormalize)
ggradar(fundma.sample)

## 在r中使用kmeans函数进行k均值聚类
#centers参数用来设置分类个数，nstart参数用来设置取随机中心的次数
#其默认值为1，但是取较多次可以改善聚类效果
#model$cluster 可以用来提取每个样本所属的类别
set.seed(123) #选123组随机数
fundma=read.csv("经理人能力.csv")
fundma.sample=fundma[,c(4:8)] #筛选数据
model=kmeans(fundma.sample,centers = 3,nstart = 10,iter.max = 10) #聚类
model$center

##聚类分析的可视化
dist.e=dist(fundma.sample,method = 'euclidean') #计算距离矩阵
mds=cmdscale(dist.e,k=2,eig=T) #距离矩阵的二维化过程 用于可视化处理
x=mds$points[,1]#x轴 
y=mds$points[,2] #y轴
class=factor(model$cluster) #存储聚类结果 #分类数据转成因子点
library(ggplot2)
p=ggplot(data.frame(x,y),aes(x,y)) #构造画布
p+geom_point(size=3,alpha=0.8,aes(colour=class,shape=class))


#####逻辑回归#####glm（）#########
##预测结果是目标变量y=1的概率
setwd("D:\\KARASU\\Graduated\\数据挖掘\\R任务数据 （出版社）\\任务24逻辑回归——网贷平台信用风险影响因素与识别")
p2pdata<-read.csv("问题P2P平台影响因素.csv")
str(p2pdata) ##查看变量类型
p2pdata$正常平台<-as.factor(p2pdata$正常平台) ##将目标变量设置为因子
p2pdata<-p2pdata[,-1] ##删除第一列平台名称
p2plogit<-glm(正常平台~.,data = p2pdata,family = binomial(link = "logit")) #相应变量服从二项分布，连接函数为logit，即为logistic回归
summary(p2plogit) ##查看模型
p2plogit.step=step(p2plogit) ##逐步回归 
summary(p2plogit.step) #查看模型

data.predict=predict(p2plogit.step,p2pdata[,-6],type = "response") #对训练集进行预测目标变量 去掉第6列因变量
data.class=ifelse(data.predict>0.7,1,0) ##预测分类结果，type=response直接返回预测的概率值0~1之间
table(data.class,p2pdata[,6]) ##构造混淆矩阵

############决策树##############
##基于CART算法的银行贷款风险识别##Rpart包
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
setwd("D:\\KARASU\\Graduated\\数据挖掘\\R任务数据 （出版社）\\任务25决策树——银行贷款风险识别")
bankloan=read.csv("bankloan.csv",stringsAsFactors = F)
summary(bankloan)
apply(bankloan,MARGIN = 2,class) ##查看各个变量属性
bankloan$违约=as.factor(bankloan$违约) #将违约变量改为因子类型 分类变
ind=sample(2,nrow(bankloan),replace = TRUE,prob=c(0.7,0.3))
bankloan.train<-bankloan[ind==1,]
bankloan.test<-bankloan[ind==2,] ##将数据集分为训练集喝测试集
table(bankloan.train$违约)/nrow(bankloan.train)
table(bankloan.test$违约)/nrow(bankloan.test) ##查看训练集和测试集的数据，分割的很成功 7：3
###建立决策树
ct<-rpart.control(xval = 10,minsplit = 10,cp=0.01) #xval交叉验证迭代10次，最小分支节点数10，cp代表复杂度越小越复杂
fit<-rpart(违约~.,
             data = bankloan,method = "class",control = ct, #method=anova连续型 class离散型
             parms = list(split="gini")) #gini指标 表示一个随机选中的样本在子集中被分错的可能性
fit$cptable  #xerror交叉验证误差要小 先降再升抛物线 4 0.02185792      4 0.7595628 0.8797814 0.06084252
##优化模型
opt<-which.min(fit$cptable[,"xerror"]) #选择预测误差最小值的预测树
cp<-fit$cptable[opt,"CP"] ##返回最小xerror对应的cp
fit.prune<-prune(fit,cp=cp) #按照xerror最小的那个cp剪枝
rpart.plot(fit.prune,branch=1,branch.type=2,type=1,extra = 1,
           shadow.col = "gray",box.col="green",
           border.col="blue",split.col="red",
           split.cex=1.2,main="银行贷款信用识别决策树")
fit.prune.fig<-prune(fit,cp=0.08) ##剪成两层为了好的可视化结果
par(mfrow=c(2,3)) ##设置我们的画布，按行填充 画布为两行三列，一共可以填充6幅画
for(i in 1:6)
{
  title=paste("extra=",i)
  rpart.plot(fit.prune.fig,branch=1,branch.type=i,type = 1,extra = i,
             shadow.col = "gray",box.col="green",
             border.col="blue",split.col="red",
             split.cex=1.2,main=title)
}
#####决策树评估
bankloan.pre=predict(fit.prune,bankloan.test[,-9],type = "class") #测试集预测
table(bankloan.pre,bankloan.test$违约)  ##查看预测效果
sum(bankloan.pre==bankloan.test$违约)/nrow(bankloan.test)

##用成本矩阵来进行模型改进
(error_cost<-matrix(c(0,2,1,0),nrow=2)) ##预测正确的赋0，将1预测为0的赋予成本2，将0预测为1的情况赋予成本1 
fit.c<-rpart(违约~.,
               data=bankloan,method = "class",control = ct,
               parms = list(loss=error_cost,split="gini"))
fit.c.prune<-prune(fit.c,cp=0.017)
rpart.plot(fit.c.prune,branch=1,branch.type=2,type=1,extra = 1,
           shadow.col = "gray",box.col="green",
           border.col="blue",split.col="red",
           split.cex=1.2,main="银行贷款信用识别决策树")
bankloan.c.pre=predict(fit.c.prune,bankloan.test[,-9],type="class")
table(bankloan.c.pre,bankloan.test$违约)
sum(bankloan.c.pre==bankloan.test$违约)/nrow(bankloan.test) 
##改进在于加大了1被预测为0的成本

###################################
#决策树 基于C5.0算法的银行贷款风险识别（C50包）
setwd("D:\\KARASU\\Graduated\\数据挖掘\\R任务数据 （出版社）\\任务25决策树——银行贷款风险识别")
credit<-read.csv("bankloan.csv",stringsAsFactors = F)
credit$违约=as.factor(credit$违约)
str(credit)
table(credit$违约)
set.seed(12345)
credit_rand<-credit[order(runif(nrow(credit))),] ##产生与credit样本相同数量的随机数，并将其按照随机数大小排列，即随机排序
head(credit$年龄) #随机前
head(credit_rand$年龄) #随机后 #比较随机前后的amount变量
cutoff=floor(nrow(credit_rand)*0.7) #设置70%的分割点，并取整
credit_train<-credit_rand[1:cutoff,] #创建训练集和测试集
credit_test<-credit_rand[(cutoff+1):nrow(credit_rand),]
prop.table(table(credit_train$违约)) #检查训练和测试集中违约的比率情况
prop.table(table(credit_test$违约)) 
install.packages("C50")
library(C50)
credit_model<-C5.0(credit_train[,-9],credit_train$违约) #训练模型
summary(credit_model) ##函数查看树的详情
credit_pred<-predict(credit_model,credit_test[,-9])
install.packages("gmodels")
library(gmodels)
CrossTable(credit_test$违约,credit_pred,prop.chisq = FALSE,prop.c = FALSE,
           prop.r = FALSE,dnn = c("actual default","predicted default"))
####模型改进###Boosting算法 #重复20次 提高效率 trials
credit_boost20<-C5.0(credit_train[,-9],credit_train$违约,trials = 20)
credit_boost_pred20<-predict(credit_boost20,credit_test[,-9])
CrossTable(credit_test$违约,credit_boost_pred20,prop.chisq = FALSE,prop.c = FALSE,
           prop.r = FALSE,dnn = c("actual default","predicted default"))
####cost参数####
(error_cost<-matrix(c(0,2,1,0),nrow=2))
credit_cost<-C5.0(credit_train[,-9],credit_train$违约,costs = error_cost,trials = 20,
                  control = C5.0Control(minCases = 5)) ###每个叶节点上最小样本数为5 防止过度拟合
credit_cost_pred<-predict(credit_cost,credit_test[,-9])
CrossTable(credit_test$违约,credit_cost_pred,prop.chisq = FALSE,prop.c = FALSE,
           prop.r = FALSE,dnn = c("actual default","predicted default"))
###C5.0的可视化##
plot(credit_cost,main="Whole Tree")
plot(credit_cost,subtree = 3,main="NO.3")
plot(credit_cost,subtree = 13,main="NO.13")

########################################

##支持向量机SVM####
install.packages("e1071")
install.packages("reshape2")
library(e1071)
setwd("D:\\KARASU\\Graduated\\数据挖掘\\R任务数据 （出版社）\\任务26支持向量机——智能选股策略设计")
data=read.csv("300DATA.csv",stringsAsFactors = F)
library("reshape2") #载入数据变形包
data1="X2013Q1"    ###设置数据的三个时间
data2="X2013Q2"
data3="X2013Q3"
rawdata1=data[c('factor','stock',data1)]  ##抽取13年前3个季度数据
rawdata2=data[c('factor','stock',data2)]
rawdata3=data[c('factor','stock',data3)]
rawdata1=dcast(rawdata1,stock~factor,value.var = data1) ##将三个季度的长数据变成宽数据
rawdata2=dcast(rawdata2,stock~factor,value.var = data2)
rawdata3=dcast(rawdata3,stock~factor,value.var = data3)

##########数据的预处理############
###构造训练集#######
traindata=merge(rawdata1,rawdata2[c('stock','收盘价','涨跌幅')],by='stock') ##将2季度的收盘价和涨跌幅合并入1季度
index300.train=traindata[which(traindata$stock=="沪深300"),'涨跌幅.y']  ##读取2季度沪深300指数的涨跌幅
overrate=traindata$涨跌幅.y-index300.train #构造超额收益率
traindata=cbind(traindata,overrate) ##将超额收益率并入训练集
traindata=na.omit(traindata) ##删除缺失值 NA.OMIT
label=ifelse(traindata$overrate>0,1,0) ##超额收益率标识
traindata=cbind(traindata,label) #将标识合并入训练集
traindata$label=as.factor(traindata$label) #标识转因子型
traindata.svm=traindata[,c(21,20,19,16)] #删除不进入模型的列
####构造测试集#######
testdata=merge(rawdata2,rawdata3[c('stock','收盘价','涨跌幅')],by='stock') ##将2季度的收盘价和涨跌幅合并入1季度
index300.test=testdata[which(testdata$stock=="沪深300"),'涨跌幅.y']  ##读取2季度沪深300指数的涨跌幅
overrate=testdata$涨跌幅.y-index300.test #构造超额收益率
testdata=cbind(testdata,overrate) ##将超额收益率并入训练集
testdata=na.omit(testdata) ##删除缺失值 NA.OMIT
label=ifelse(testdata$overrate>0,1,0) ##超额收益率标识
testdata=cbind(testdata,label) #将标识合并入训练集
testdata$label=as.factor(testdata$label) #标识转因子型
testdata.svm=testdata[,c(21,20,19,16)] 

library(e1071)
svm.kernel=c("linear","polynomial","sigmoid","radial") ##设置4个核函数选项
for(i in 1:length(svm.kernel)) #对4个核函数循环
{                            ##建立SVM模型，其中class.weights表示不同分类错分的成本权重，cross表示交叉验证数
  svm.class=svm(label ~ .,data=traindata.svm[,-1],kernel=svm.kernel[i],class.weights=c('1'=1,'0'=1.2),cross=50)
  svm.predictions<-predict(svm.class,traindata.svm[,c(-1,-18)])  ##对训练集进行预测
  svm.agreement<-svm.predictions==traindata.svm$label  ##逻辑判断对于训练集的预测结果
  suss.table=table(svm.predictions,traindata.svm$label)
  suss.prop=prop.table(table(svm.agreement))
  result=c(svm.kernel[i],suss.table[2,2]/sum(suss.table[2,]),suss.prop[2])
  print(result)
}

svm.class=svm(label ~ .,data=traindata.svm[,-1],kernel=svm.kernel[i],class.weights=c('1'=1,'0'=1.2),cross=20)
svm.predictions<-predict(svm.class,testdata.svm[,c(-1,-18)])
##### not available

#######################################

######关联分析####
install.packages("arules")
library(arules)
setwd("D:\\KARASU\\Graduated\\数据挖掘\\R任务数据 （出版社）\\任务27关联分析——互联网投资标的的智能推荐")
data<-read.transactions("investment.csv",format = "single",sep = ",",cols = c("UserId","ProId"))  ##读进来数据是交易 固定形式 #single每行内容是交易号+单个商品
attributes(data)  ##查看数据
summary(data)
itemFreq<-itemFrequency(data)  #计算各个标的出现的频率
ordereditemFreq<-sort(itemFrequency(data),decreasing = T)  ###频率排序
ordereditemFreq[1:10]
itemFrequencyPlot(data,support=0.1)  ##画出最小支持度为0.1的频率图
itemFrequencyPlot(data,topN=5,horiz=T)  ##排名前5的标的频率图
rules<-apriori(data,parameter = list(minlen=3,supp=0.01,conf=0.8)) ##生成规则
rules.sorted<-sort(rules,decreasing = T,by="lift") ##将规则结果按照lift降序排序
inspect(rules.sorted)  ###查看排序后的变量
 ##规则去重
redundant=is.redundant(rules.sorted) ##识别是否是多余的规则
rules.pruned<-rules.sorted[!redundant] ##删除多余规则
inspect(rules.pruned)
##生成特定规则
#要求生成后项为“P003”“P047”的规则
rules.tar<-apriori(data,parameter = list(minlen=3,supp=0.01,conf=0.8),appearance = list(rhs=c("P003","P047"),default="lhs"))
inspect(rules.tar)
subset.rules<-subset(rules.tar,items %in% c("P047")& lift>2)
inspect(subset.rules)
#####关联规则的可视化
install.packages("arulesViz")
install.packages("RColorBrewer")
library(arulesViz)
library(RColorBrewer)
###散点图
plot(rules.sorted,control = list(jitter=2,col=rev(brewer.pal(9,"Greens")[4:9])),shading = "lift")  ####最好的规则应该位于右上角，且颜色较深
###关系图
plot(rules.pruned,measure = "confidence",method = "graph",control = list(type="items"),shading = "lift")


############################################
######BP神经网络########
library(AMORE)
setwd("D:\\KARASU\\Graduated\\数据挖掘\\R任务数据 （出版社）\\任务28BP神经网络——P2P网贷的逾期风险识别")
set.seed(1234)
samp.rate=0.7  ##设置训练集比例
p2pdata=read.csv("p2p_lending.csv")
str(p2pdata)
p2pdata$周期_月=as.numeric(p2pdata$周期_月)
p2pdata$借款描述字数=as.numeric(p2pdata$借款描述字数)
p2pdata$累积已还次数=as.numeric(p2pdata$累积已还次数)
p2pdata$累积逾期次数=as.numeric(p2pdata$累积逾期次数)
p2pdata$安全标=as.numeric(p2pdata$安全标)
samp.index=sample(1:nrow(p2pdata),size=floor(samp.rate*nrow(p2pdata)))  ##按比例抽样
train=p2pdata[samp.index,]
test=p2pdata[-samp.index,]
##设置神经网络参数
net=newff(n.neurons = c(13,10,1),learning.rate.global = 0.0001,momentum.global = 0.01,
          error.criterium = "LMS",hidden.layer = "tansig",
          output.layer = "purelin",method = "ADAPTgdwm") ##LMS最小均方误差 purelin线性函数 tansig传递函数 ADAPTgdwm 含有动量的自适应梯度下降法
##训练模型
model=train(net,train[,-14],train[,14],error.criterium="LMS",report=T,show.step=100,n.show=10)
test.predict=sim(model$net,test[,-14])


library(AMORE)
setwd("D:\\KARASU\\Graduated\\数据挖掘\\R任务数据 （出版社）\\任务28BP神经网络——P2P网贷的逾期风险识别")
set.seed(1234)
samp.rate=0.7

fun.dummy<-function(data)
{
  name.level=levels(data)
  dummy=c()
  for(i in 1:(length(name.level)-1))
  {
    temp.dummy=ifelse(data==name.level[i],1,0)
    temp.dummy=as.numeric(temp.dummy)
    dummy=cbind(dummy,temp.dummy)
  }
  colnames(dummy)=name.level[1:(length(name.level)-1)]
  dummy=as.data.frame(dummy)
  return(dummy)
}

p2pdata=read.csv("p2p_lending.csv")
str(p2pdata)
part.dummy=fun.dummy(p2pdata[,1])
for(i in 2:5)
{
  dummy=fun.dummy(p2pdata[,i])
  part.dummy=cbind(part.dummy,dummy)
}
#str(part.dummy)
p2pdata=cbind(part.dummy,p2pdata[,6:19])
p2pdata$周期_月=as.numeric(p2pdata$周期_月)
p2pdata$借款描述字数=as.numeric(p2pdata$借款描述字数)
p2pdata$累积已还次数=as.numeric(p2pdata$累积已还次数)
p2pdata$累积逾期次数=as.numeric(p2pdata$累积逾期次数)
p2pdata$安全标=as.numeric(p2pdata$安全标)

min.vec=apply(p2pdata[,16:28],2,min)
max.vec=apply(p2pdata[,16:28],2,max)
range.vec=max.vec-min.vec
std=p2pdata[,16:28]
for(i in 1:ncol(std))
{
  std[,i]=(std[,i]-min.vec[i]/range.vec[i])
}
clean.data=p2pdata
clean.data[,16:28]=std
apply(clean.data,2,max)
samp.index=sample(1:nrow(clean.data),size=floor(samp.rate*nrow(clean.data)))
train=clean.data[samp.index,]
test=clean.data[-samp.index,]

net=newff(n.neurons = c(28,10,1),learning.rate.global = 0.0001,momentum.global = 0.01,
          error.criterium = "LMS",hidden.layer = "tansig",
          output.layer = "purelin",method = "ADAPTgdwm") ##LMS最小均方误差 purelin线性函数 tansig传递函数 ADAPTgdwm 含有动量的自适应梯度下降法
model=train(net,train[,-29],train[,29],error.criterium="LMS",report=T,show.step=100,n.show=10)
test.predict=sim(model$net,test[,-29])

install.packages("ROCR")
library("ROCR")


