setwd("C:\\Users\\KARASU\\Desktop")
library(readr)
car_price<-read.csv("��������2.csv",stringsAsFactors = F)
###  in this problem
### car is a system
##which has input features such as enginesize,body,engine locaton,Brand of car 
##output performance are messured by  such as peak rpm,city mpg,horsepower ,highway mpg,symboling 
## we have to best establish relation between them to predict price of car!
str(car_price)
name_cars=data.frame(do.call('rbind',strsplit(as.character(car_price$firm),' ',fixed=T)))
library(plyr)
count(name_cars$do.call..rbind...strsplit.as.character.car_price.firm........)

library(VIMGUI,quietly = T,warn.conflicts = F)
carData<-kNN(car_price,variable=c("type","firm","gearbox","fuel"),dist_var=c("time","distant","new price","discount","vol"),imp_var = F)

## We will try to transform the caracter variable into the numeric variable to facilite the algorithm 
##training later. The method that I use is to reorder the levels of the variable by the mean of its price.

## make a function that transform that relevel the order of the level by the mean of the price.
f<-function(variable.number){
  list.option<-levels(carData[,variable.number])
  a<-c()
  for(i in 1:length(list.option)){
    mean.i<-mean(carData[carData[,variable.number]==list.option[i],"price"])
    a<-c(a,mean.i)
  }
  b<-order(a)
  new.order<-list.option[b]
  return(new.order)
}

## Transform variable "car" into numeric value.
car_price$vol=as.numeric(car_price$vol)
car_price$gearbox=as.numeric(car_price$gearbox)
car_price$fuel=as.numeric(car_price$fuel)
car_price$discount=as.numeric(car_price$discount)
car_price$�ش��¹�27=as.numeric(car_price$�ش��¹�27)
car_price$ˮ�ݻ���18=as.numeric(car_price$ˮ�ݻ���18)
car_price$�����Ų�11=as.numeric(car_price$�����Ų�11)
car_price$�������ܶ�46=as.numeric(car_price$�������ܶ�46)
car_price$��΢��ײ21=as.numeric(car_price$��΢��ײ21)
car_price$����Ĳ���18=as.numeric(car_price$����Ĳ���18)
car_price$��ȫϵͳ15=as.numeric(car_price$��ȫϵͳ15)
car_price$�ⲿ����26=as.numeric(car_price$�ⲿ����26)
car_price$�ڲ�����20=as.numeric(car_price$�ڲ�����20)
car_price$�Ǳ�ָ̨ʾ��9=as.numeric(car_price$�Ǳ�ָ̨ʾ��9)
car_price$������״̬5=as.numeric(car_price$������״̬5)
car_price$�����估ת��5=as.numeric(car_price$�����估ת��5)
car_price$���쳣��=as.numeric(car_price$���쳣��)

str(car_price)
summary(carData)

########## **** Exploratory Analysis **** ############
summary(carData)
par(mfrow=c(3,3))
for(i in 3:10){
  hist(carData[,i], main=names(carData)[i])
}

#library(corrplot)
#correlations<-cor(carData)
#corrplot(correlations,method="circle")
#hist(carData$year)
library(latticeExtra)
library(lattice)
library(ggplot2)
library(caret)

set.seed(5)
validationIndex<-createDataPartition(carData$price,p=0.7, list=F)
validation<-carData[-validationIndex,]
training<-carData[validationIndex,]

set.seed(5)
trainControl<-trainControl(method = "repeatedcv",number=10,repeats = 3)
trainControl
## RMSE for regression training
metric<-"RMSE"
metric
## preprocessing
preProcess<-c("center","scale","BoxCox")
preProcess


set.seed(5)
fit.glm<-train(price ~ ., data=training, method="glm", metric=metric, preProc=preProcess ,trControl=trainControl)
fit.glm