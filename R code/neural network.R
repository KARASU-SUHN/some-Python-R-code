library(AMORE)
setwd("D:\\Document\\ѧϰ\\2018-2019  1\\�����ھ�(R����)\\�����α���")
usedcar<-read.csv("����2-3000��.csv",stringsAsFactors = F)
set.seed(1234)
samp.rate=0.7

# fun.dummy<-function(data)#���������ת��Ϊ�Ʊ���
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
# usedcar$ȼ������=as.factor(usedcar$ȼ������)
# usedcar$������=as.factor(usedcar$������)
# part.dummy=fun.dummy(usedcar$ȼ������)
# part.dummy1=fun.dummy(usedcar$������)
# part.dummy=cbind(part.dummy,part.dummy1)
# part.dummy=cbind(usedcar$ȼ������,usedcar$������)
# for(i in 10:22)
# {
#   dummy=fun.dummy(usedcar[,i])
#   part.dummy=cbind(part.dummy,dummy)
# }

# usedcar=cbind(part.dummy,usedcar[,c(1,4:9)])
usedcar$type=as.numeric(usedcar$type)
usedcar$������=as.numeric(usedcar$������)
usedcar$ȼ������=as.numeric(usedcar$ȼ������)
usedcar$�ش��¹�27=as.numeric(usedcar$�ش��¹�27)
usedcar$ˮ�ݻ���18=as.numeric(usedcar$ˮ�ݻ���18)
usedcar$�����Ų�11=as.numeric(usedcar$�����Ų�11)
usedcar$�������ܶ�46=as.numeric(usedcar$�������ܶ�46)
usedcar$��΢��ײ21=as.numeric(usedcar$��΢��ײ21)
usedcar$����Ĳ���18=as.numeric(usedcar$����Ĳ���18)
usedcar$��ȫϵͳ15=as.numeric(usedcar$��ȫϵͳ15)
usedcar$�ⲿ����26=as.numeric(usedcar$�ⲿ����26)
usedcar$�ڲ�����20=as.numeric(usedcar$�ڲ�����20)
usedcar$�Ǳ�ָ̨ʾ��9=as.numeric(usedcar$�Ǳ�ָ̨ʾ��9)
usedcar$������״̬5=as.numeric(usedcar$������״̬5)
usedcar$�����估ת��5=as.numeric(usedcar$�����估ת��5)
usedcar$���쳣��=as.numeric(usedcar$���쳣��)
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

#������usedcar����������ͬ�������������usedcar����������Ĵ�С����
rand<-usedcar[order(runif(nrow(usedcar))),]
#����70%�ķָ�㣬��ȡ��
cutoff=floor(nrow(rand)*0.7)
#����ѵ���Ͳ��Լ�
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
