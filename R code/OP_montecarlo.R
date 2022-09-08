#op_monte

op_monte=function(s0,K,rfr,sig,T){
m=10000 
n=100
dlt=T/500
v=exp(-rfr*T)
sc=0
for (j in 1:n){
sf=0
for(i in 1:m){
e=rnorm(500)
e=sum(e)
st=s0*exp((rfr-0.5*sig^2)*T+sig*sqrt(dlt)*e)
f=max(st-K,0)
sf=sf+f
}
c0=v*(sf/m)
sc=sc+c0
}
sc=sc/n
return(sc)
}

#examine the distribution of predicted value

ff=rep(0,100)
for(i in 1:100){
ff[i]=op_monte(50,50,0.1,0.2,0.25)
}

hist(ff)


GBSOption(TypeFlag = "c", S =50, X =50, Time = 1/4, r = 0.1,b=0.1,sigma = 0.20)

 

op_monte1=function(s0,K,rfr,sig,T){
m=10000 
dlt=T/500
v=exp(-rfr*T)
sf=0
for(i in 1:m){
e=rnorm(500)
e=sum(e)
st1=s0*exp((rfr-0.5*sig^2)*T+sig*sqrt(dlt)*e)
st2=s0*exp((rfr-0.5*sig^2)*T-sig*sqrt(dlt)*e)
f=0.5*max(st1-K,0)+0.5*max(st2-K,0)
sf=sf+f
}
c0=v*(sf/m)
return(c0)
}

op_monte2=function(s0,K,rfr,sig,T){
m=10000 
dlt=T/500
v=exp(-rfr*T)
sf=0
for(i in 1:m){
e=rnorm(500)
e=sum(e)
st1=s0*exp((rfr-0.5*sig^2)*T+sig*sqrt(dlt)*e)
f=max(st1-K,0)
sf=sf+f
}
c0=v*(sf/m)
return(c0)
}


c1=rep(0,100)
for(i in 1:100){
c1[i]=op_monte1(50,50,0.1,0.2,0.25)
}
c2=rep(0,100)
for(i in 1:100){
c2[i]=op_monte2(50,50,0.1,0.2,0.25)
}


