#The present value of a bond
mpv=function(c,p,r,n){
cpv=0
for(i in 1:n){
cpv=cpv+c/(1+r)^i
}
pv=cpv+p/(1+r)^n
return(pv)
}

#reduce the search area by bracketing
#rng=c(a,b)
brk=function(c,p,n,mv,xl,xu){
#initialize the values
nn=10
dlt=(xu-xl)/nn
b=xl
rng=c(0,0)

for(i in 1:n){
a=b
b=a+dlt
fr_a=mpv(c,p,a,n)-mv
fr_b=mpv(c,p,b,n)-mv
if(sign(fr_a)!=sign(fr_b)){
rng[1]=a
rng[2]=b
}
}
return(rng)
}

#yield to maturity by approach of bracketing

ytm=function(c,p,n,mv,tol){
#initial setting
xl=0.0000001
xu=10
fr=tol+1

while(fr>tol){
rng=brk(c,p,n,mv,xl,xu)
xl=rng[1]
xu=rng[2]
fr=mpv(c,p,xl,n)-mv
}
return(xl)
}

 