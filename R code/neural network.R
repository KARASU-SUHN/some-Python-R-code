library(AMORE)
setwd("D:\\Document\\学习\\2018-2019  1\\数据挖掘(R语言)\\第三次报告")
usedcar<-read.csv("数据2-3000条.csv",stringsAsFactors = F)
set.seed(1234)
samp.rate=0.7

# fun.dummy<-function(data)#将分类变量转化为哑变量
# {
#   name.level=levels(data)
#   dummy=c()
#   for(i in 1:(length(name.level)-1))
#   {
#     temp.dummy=ifelse(data==name.level[i],1,0)
#     temp.dummy=as.numeric(temp.dummy)
#     dummy=cbind(dummy,temp.dummy)
#   }
#   colnames(dummy)=name.level[1:(length(name.level)-1)]
#   dummy=as.data.frame(dummy)
#   return(dummy)
# }

str(usedcar)
# usedcar$燃料类型=as.factor(usedcar$燃料类型)
# usedcar$变速箱=as.factor(usedcar$变速箱)
# part.dummy=fun.dummy(usedcar$燃料类型)
# part.dummy1=fun.dummy(usedcar$变速箱)
# part.dummy=cbind(part.dummy,part.dummy1)
# part.dummy=cbind(usedcar$燃料类型,usedcar$变速箱)
# for(i in 10:22)
# {
#   dummy=fun.dummy(usedcar[,i])
#   part.dummy=cbind(part.dummy,dummy)
# }

# usedcar=cbind(part.dummy,usedcar[,c(1,4:9)])
usedcar$type=as.numeric(usedcar$type)
usedcar$变速箱=as.numeric(usedcar$变速箱)
usedcar$燃料类型=as.numeric(usedcar$燃料类型)
usedcar$重大事故27=as.numeric(usedcar$重大事故27)
usedcar$水泡火烧18=as.numeric(usedcar$水泡火烧18)
usedcar$调表排查11=as.numeric(usedcar$调表排查11)
usedcar$底盘悬架顶46=as.numeric(usedcar$底盘悬架顶46)
usedcar$轻微碰撞21=as.numeric(usedcar$轻微碰撞21)
usedcar$易损耗部件18=as.numeric(usedcar$易损耗部件18)
usedcar$安全系统15=as.numeric(usedcar$安全系统15)
usedcar$外部配置26=as.numeric(usedcar$外部配置26)
usedcar$内部配置20=as.numeric(usedcar$内部配置20)
usedcar$仪表台指示灯9=as.numeric(usedcar$仪表台指示灯9)
usedcar$发动机状态5=as.numeric(usedcar$发动机状态5)
usedcar$变速箱及转向5=as.numeric(usedcar$变速箱及转向5)
usedcar$总异常数=as.numeric(usedcar$总异常数)
usedcar$type=as.numeric(usedcar$type)
str(usedcar)

min.vec=apply(usedcar[,1:22],2,min)
max.vec=apply(usedcar[,1:22],2,max)
range.vec=max.vec-min.vec
std=usedcar[,1:22]
for(i in 1:ncol(std))
{
  std[,i]=(std[,i]-min.vec[i])/range.vec[i]
}
clean.data=usedcar
clean.data[,1:22]=std
apply(clean.data,2,max)

#产生与usedcar样本数量相同的随机数，并将usedcar按照随机数的大小排序。
rand<-usedcar[order(runif(nrow(usedcar))),]
#设置70%的分割点，并取整
cutoff=floor(nrow(rand)*0.7)
#创建训练和测试集
train<-rand[1:cutoff,]
test<-rand[(cutoff+1):nrow(rand),] 




# samp.index=sample(1:nrow(clean.data),size=floor(samp.rate*nrow(clean.data)))
# train=clean.data[samp.index,]
# test=clean.data[-samp.index,]
prop.table(table(train$type))
prop.table(table(test$type))
table(test$type)
#########
net=newff(n.neurons=c(21,20,1),learning.rate.global=0.0001,momentum.global=0.01,error.criterium="LMS",hidden.layer="tansig",output.layer="purelin",method="ADAPTgdwm")
model=train(net,train[,-1],train[,1],error.criterium="LMS",report=T,show.step=100,n.show=10)
test.predict=sim(model$net,test[,-1])
test.class=ifelse(test.predict<0.1,0,1)
table(test.class,test[,1])
sum(diag(table(test.class,test[,1])))/nrow(test)
table(test$type)
table(test.class)

library("ROCR")
pred<-prediction(test.predict,test[,1])
perf<-performance(pred,'tpr','fpr')
plot(perf,colorize=FALSE)
auc<-performance(pred,measure="auc")
auc@y.values[[1]]

