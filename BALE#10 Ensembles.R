#BALE 10 Ensembles
#eBay auctions--Boosting and Bagging
library(rpart)
library(rpart.plot)
library(caret)
#load the data and pre-process
ebay.df <- read.csv("ebayAuctions.csv")
str(ebay.df)
#partition the data into training and validation sets
set.seed(1)  
train.index <- sample(c(1:dim(ebay.df)[1]), dim(ebay.df)[1]*0.6)  
valid.index <- setdiff(c(1:dim(ebay.df)[1]), train.index)  
train.df <- ebay.df[train.index, ]
valid.df <- ebay.df[valid.index, ]

#1. Run a classification tree, using the default controls of rpart().
##Looking at the validation set, what is the overall accuracy? What is the 
##lift on the first decile (first 10%)?

#fit the tree model with default settings
rt <- rpart(as.factor(Competitive) ~ ., data = train.df, method = "class")
pred <- predict(rt, valid.df, type = "class")
#confusion/classification matrix
confusionMatrix(pred, as.factor(valid.df$Competitive), positive = "1")
#Visualize the gains (lift) and find how much of the data we need to use
install.packages("gains")
library(gains)
#Build lift chart
gain <- gains(as.numeric(valid.df$Competitive), predict(rt, valid.df)[,2], groups = 10)
plot(c(0, gain$cume.pct.of.total*sum(valid.df$Competitive)) ~ c(0, gain$cume.obs), 
     xlab = "# cases", ylab = "Cumulative", type="l")
lines(c(0,sum(valid.df$Competitive))~c(0,dim(valid.df)[1]), col="gray", lty=2)
#Build decile chart
heights <- gain$mean.resp/mean(as.numeric(valid.df$Competitive))
midpoints <- barplot(heights, names.arg = gain$depth, ylim = c(0,9),
                     xlab = "Percentile", ylab = "Mean Response", 
                     main = "Decile-wise chart for class tree validation data")
text(x=midpoints, y = heights, labels = round(heights, 1), pos = 3.5, cex = 0.7,
     col = "red")
gain

#2. Run a boosted tree with the same predictors (use function boosting()
##in the adabag package). For the validation set, what is the overall accuracy?
##What is the lift on the first decile?
install.packages("adabag")
install.packages("randomForest")
#boosted tree
library(adabag)
library(randomForest)
train.df$Category <- as.factor(train.df$Category)
valid.df$Category <- as.factor(valid.df$Category)
train.df$Competitive <- as.factor(train.df$Competitive)
boost <- boosting(Competitive ~ ., data = train.df, method = "class")
predboost <- predict(boost, valid.df, type = "class")
confusionMatrix(as.factor(predboost$class), as.factor(valid.df$Competitive), positive = "1")
#lift chart
gainboost <- gains(as.numeric(valid.df$Competitive), predict(boost, valid.df)$prob[,2])
plot(c(0, gainboost$cume.pct.of.total*sum(valid.df$Competitive)) ~ c(0, gainboost$cume.obs), 
     xlab = "# cases", ylab = "Cumulative", type="l")
lines(c(0,sum(valid.df$Competitive))~c(0,dim(valid.df)[1]), col="gray", lty=2)
#decile chart
heights <- gainboost$mean.resp/mean(as.numeric(valid.df$Competitive))
midpoints <- barplot(heights, names.arg = gainboost$depth, ylim = c(0,9),
                     xlab = "Percentile", ylab = "Mean Response", 
                     main = "Decile-wise chart for boosted tree validation data")
text(x=midpoints, y = heights, labels = round(heights, 1), pos = 3.5, cex = 0.7,
     col = "red")

#3. Run a bagged tree with the same predictors (use function bagging() 
##in the adabag package). For the validation set, what is the overall accuracy?
##What is the lift on the first decile?

#bagged tree
bag <- bagging(Competitive ~ ., data = train.df, method = "class")
predbag <- predict(bag, valid.df, type = "class")
confusionMatrix(as.factor(predbag$class), as.factor(valid.df$Competitive), positive = "1")
#lift chart
gainbag <- gains(as.numeric(valid.df$Competitive), predict(bag, valid.df)$prob[,2])
plot(c(0, gainbag$cume.pct.of.total*sum(valid.df$Competitive)) ~ c(0, gainbag$cume.obs), 
     xlab = "# cases", ylab = "Cumulative", type="l")
lines(c(0,sum(valid.df$Competitive))~c(0,dim(valid.df)[1]), col="gray", lty=2)
#decile chart
heights <- gainbag$mean.resp/mean(as.numeric(valid.df$Competitive))
midpoints <- barplot(heights, names.arg = gainbag$depth, ylim = c(0,9),
                     xlab = "Percentile", ylab = "Mean Response", 
                     main = "Decile-wise chart for bagged tree validation data")
text(x=midpoints, y = heights, labels = round(heights, 1), pos = 3.5, cex = 0.7,
     col = "red")

#4. Run a random forest (use function randomForest() in package 
##randomForest with argument mtry = 4). Compare the bagged tree to the random
##forest in terms of validation accuracy and lift on first decile. How are the 
##two methods conceptually different? Explain why.

#random forest
library(randomForest)
rf <- randomForest(Competitive ~ ., data = train.df, 
                   mtry = 4, method = "class")

predrf <- predict(rf, valid.df, type="class")
confusionMatrix(as.factor(predrf), as.factor(valid.df$Competitive), positive = "1")

gainrf <- gains(as.numeric(valid.df$Competitive), as.numeric(predrf), groups=2)
heights <- gainrf$mean.resp/mean(as.numeric(valid.df$Competitive))
midpoints <- barplot(heights, names.arg = gainrf$depth, ylim = c(0,9),
                     xlab = "Percentile", ylab = "Mean Response", 
                     main = "Decile-wise chart for random forest validation data")
text(x=midpoints, y = heights, labels = round(heights, 1), pos = 3.5, cex = 0.7,
     col = "red")

