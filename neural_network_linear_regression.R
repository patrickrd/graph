###Caron's model
## f = function(x,y){1-exp(-2*x * y)}
## n=5000 ##because that way the readers have been the quality of the manifold approximation
## Z = rgamma(n,1,1)
## P = matrix(nrow=n,ncol=n)
## for (i in 1:n){ 
##         P[i,] = f(Z[i],Z)
## }
## d=100
## E = eigen(P)
## keep=sort(order(abs(E$values),decreasing=TRUE)[1:d])
## X = E$vectors[,keep] %*% diag(sqrt(abs(E$values[keep])))
## ##plot(X[,1:2], cex=.2)
## A=matrix(runif(n*n)<= P,nrow=n)
## ##A=matrix(rnorm(n*n,sd=0.05), nrow=n)+P
## A = sym(A)
## res=getXQ(A, X, sum(E$values[keep]>0), sum(E$values[keep]<0))
## XhatQ = as.matrix(res$Xhat %*% t(res$Q))
## save(Z,res,file="embedding_neural_network_data.Rout")


alpha = 2; beta = 5; std = 1
y = alpha + beta*Z + rnorm(n,sd=std)

ntrain = 100

library(tensorflow)
##install_tensorflow()
##tf$constant("Hellow Tensorflow")
##install.packages("keras")
library(keras)

model <- keras_model_sequential() 
model %>% 
    layer_dense(units = 256, activation = 'relu', input_shape = c(d)) %>% 
    layer_dropout(rate = 0.4) %>% 
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 1)


# Configure a model for mean-squared error regression.
model %>% compile(
  optimizer = 'adam',
  loss = 'mse',           # mean squared error
  metrics = list('mae')   # mean absolute error
)

model %>% summary()

history <- model %>% fit(
  XhatQ[1:ntrain,], y[1:ntrain], 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
  )



##points(Z, yhat, col="green")

##(model %>% evaluate(XhatQ, y, verbose = 0))

df = as.data.frame(cbind(XhatQ[1:ntrain,],rep(1,ntrain)))
modellm <- lm (y[1:ntrain] ~ ., df)

library(glmnet)
##fit.lasso <- glmnet(cbind(XhatQ,rep(1,n)), y, family="gaussian", lambda=1)
fit.lasso <- cv.glmnet(cbind(XhatQ[1:ntrain,],rep(1,ntrain)), y[1:ntrain], family="gaussian")


##yhatlassomin = predict(fit.lasso, cbind(XhatQ,rep(1,n)), lambda=fit.lasso$lambda.min)
##predict(fit.lasso, cbind(XhatQ,rep(1,n)))
## library(glmnet)
## fit.lasso <- cv.glmnet(cbind(XhatQ[1:ntrain],rep(1,ntrain)), y[1:ntrain], family="gaussian")
## fit.lasso <- cv.glmnet(cbind(XhatQ[1:n],rep(1,n)), y[1:n], family="gaussian")
## yhatlasso = predict(fit.lasso, cbind(XhatQ,rep(1,n)), lambda=fit.lasso$lambda.1se)
## ##yhatlassomin = predict(fit.lasso, cbind(XhatQ,rep(1,n)), lambda=fit.lasso$lambda.min)

## plot(Z,yhatlasso)
## plot(Z,yhatlassomin)
## mean((yhatlassomin - y)^2)

library(randomForest)
fit.rf = randomForest(cbind(XhatQ[1:ntrain,],rep(1,ntrain)), y[1:ntrain])
yhatrf = predict(fit.rf, cbind(XhatQ,rep(1,n)))
plot(Z, yhatrf, col="red")
mean((yhatrf - y)^2)






###the comparison
test = (ntrain+1):n

yhatnn=model %>% predict(XhatQ[test,])
yhatnna=model %>% predict(XhatQ)

dft = as.data.frame(cbind(XhatQ[test,],rep(1,length(test))))
dfta = as.data.frame(cbind(XhatQ,rep(1,n)))
yhatlm = predict(modellm, dft)
yhatlma = predict(modellm, dfta)

yhatlasso = predict(fit.lasso, cbind(XhatQ[test,],rep(1,length(test))), lambda=fit.lasso$lambda.1se)
yhatlassoa = predict(fit.lasso, cbind(XhatQ,rep(1,n)), lambda=fit.lasso$lambda.1se)

yhatrf = predict(fit.rf, cbind(XhatQ[test,],rep(1,length(test))))
yhatrfa = predict(fit.rf, cbind(XhatQ,rep(1,n)))

pdf("embedding_regression2.pdf", width=6, height=5)
par(mar=c(4.5, 4.5, 0.5, 0.5))
plot(Z, y, cex=.1, pch=16, xlab="Z", ylab="Y")
o = order(Z)
lines(Z[o], yhatnna[o], cex=.1, col="red")
lines(Z[o], yhatlma[o], cex=.1, col="purple")
lines(Z[o], yhatlassoa[o], cex=.1, col="orange")
lines(Z[o], yhatrfa[o], cex=.1, col="green")
lines(sort(Z),alpha + beta*sort(Z))
legend("bottomright",legend=c(paste0("Ideal regression (", round(mean((y[test]-(alpha+beta*Z[test]))^2), digit=2),")"), paste0("Random forest (",round(mean((yhatrf-y[test])^2), digit=2),")"), paste0("Neural network (",round(mean((yhatnn-y[test])^2), digit=2),")"), paste0("Lasso (",round(mean((yhatlasso-y[test])^2), digit=2),")"), paste0("Least squares (",round(mean((yhatlm-y[test])^2), digit=2),")")), col=c("black","green", "red","orange","purple"), lty=1)
dev.off()





####



##A full dimensional regression
beta_large = rnorm(d); 
yfull = alpha + -X %*% beta_large + rnorm(n,sd=std)

history <- model %>% fit(
  XhatQ[1:ntrain,], yfull[1:ntrain], 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
  )

yhatnn=model %>% predict(XhatQ[test,])
mean((yhatnn-yfull[test])^2)

fit.rf = randomForest(cbind(XhatQ[1:ntrain,],rep(1,ntrain)), yfull[1:ntrain])
yhatrf = predict(fit.rf, cbind(XhatQ[test,],rep(1,length(test))))
mean((yhatrf-yfull[test])^2)
