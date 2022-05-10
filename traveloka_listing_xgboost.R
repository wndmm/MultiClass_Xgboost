# XGBOOST PERFORMANCE SHOWS THAT INI SI REVIEWAMOUNT MSH PUNYA PELUANG BESAR, SESUAI REGRESI MODEL
# DARI HASIL SCRAPPING WEB dengan factor: integer (cleanaccom), integer factor (randomrate) sebagai dependensi
install.packages("mice")
install.packages("mltools")
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)
library(mltools)
library(data.table)
library(ggplot2)
library(mice)
library(caret)

# LOAD DATA

travelokadata <- read.csv(file.choose(), header = T)
travelokadata$score <- as.factor(travelokadata$score)
travelokadata$negative <- as.factor(travelokadata$negative)
travelokadata$positive <- as.factor(travelokadata$positive)
travelokadata$landmarknear <- as.factor(travelokadata$landmarknear)
str(travelokadata)

# DATA PARTITION
set.seed(1234)
ind <- sample(2, nrow(travelokadata), replace = T, prob = c(0.6, 0.4))
train <- travelokadata[ind==1,]
test <- travelokadata[ind==2,]

# CREATING MATRIX - ONE HOT ENCODING FOR FACTOR VARIABLES, CREATING DUMMY VARIABLES
str(travelokadata)

trainM <- sparse.model.matrix(cleanaccom~. -1, data = train)
head(trainM)
train_label <- train [, "cleanaccom"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainM), label = train_label)

testM <- sparse.model.matrix(cleanaccom ~. -1, data = test)
test_label <- test [, "cleanaccom"]
test_matrix <- xgb.DMatrix(data = as.matrix(testM), label = test_label)

# SET PARAMETERS
nc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss", 
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)

# EXTREME GRADIENT BOOSTING MODEL awas ini susah bosku ya hehe
set.seed(1313)
boostM <- xgb.train(params = xgb_params,
                    data = train_matrix,
                    nrounds = 300, # jumlah iterasi permodel train
                    watchlist = watchlist,  # kalo udah fit cukup sampe disini aja
                    eta = 0.01, # eta is learning rate, lebih kecil bisa juga lebih dahsyat.. 
                    max.depth = 2, # sampai dimana kedalaman learningnya
                    gamma = 0, # make sure si hanya menggunakan beta dan alpha
                    subsample = 1, 
                    colsample_bytree = 1, # belajar hanya dari 1 cases
                    missing = NA, # avoiding missing value
                    seed = 666)

# TRAINING & TEST ERROR PLOT
Eplot <- data_frame(boostM$evaluation_log)
plot(Eplot$iter, Eplot$train_mlogloss, col = "dodgerblue")
lines(Eplot$iter, Eplot$test_mlogloss, col = "red")
print(boostM$evaluation_log)
#print(boostM$params)
#print(boostM$raw)
#print(boostM$feature_names)
#print(boostM$nfeatures)

# TO KNOW THE MINIMUN VALUE OF ERROR 
min(Eplot$test_mlogloss)

# TO KNOW ITERATION OF THE LAST MODEL
Eplot[Eplot$test_mlogloss == "0.612744", ] # hasil iterasi bisa berulang masukin test mlog loss nya dan hilangkan koma

# FEATURE IMPORTANCE
impo <- xgb.importance(colnames(test_matrix), model = boostM)
print(impo)
xgb.plot.importance(xgb.importance(model = boostM), 
                    col = 10, las = 2, main = "Variable Importance")
# LETS DO PREDICTION AND CONFUSION MATRIX - TEST
pred <- predict(boostM, newdata = test_matrix)
head(pred)
str(pred)
predp <- matrix(pred, nrow = nc, ncol = length(pred)/nc) %>% 
                t() %>%
                data.frame()%>%
                mutate(label = test_label, max_prob = max.col(.,"last")-1)
table(Prediction = predp$max_prob, Actual = predp$label)

# dibagi jumlah data test which is kalo traveloka itu 511 observation, jadi di bagi dari hasil prediksi "click table aja"

travelokadata <- read.csv(file.choose())
travelokadata$landmarknear <- as.factor(travelokadata$landmarknear)
str(travelokadata)
landmark <- travelokadata$landmarknear
# landmark = as.integer(landmark)
#landmarkint = as.integer(travelokadata$landmarknear)
label = as.integer(travelokadata$cleanaccom)-1
label <- as.factor(label)
travelokadata$landmarknear= NULL
travelokadata$cleanaccom = NULL
n = nrow(travelokadata)

#travelokadata$score <- as.factor(travelokadata$score)
#travelokadata$negative <- as.factor(travelokadata$negative)
#travelokadata$positive <- as.factor((travelokadata$positive))

train.index = sample(n, floor (0.6*n))
test.index = sample(n, floor (0.4*n))
test.index = as.matrix(test.index)
test.index <- as.matrix(test.index)
train.data = as.matrix(travelokadata[train.index,])
train.label = label[train.index]
test.data = as.matrix(travelokadata[-train.index,])
test.label = label [-train.index]

# TRANSFORM INTO DATA SET XGBMATRIX
xgb.train = xgb.DMatrix(data = train.data, label = train.label)
xgb.test = xgb.DMatrix(data = test.data, label = test.label)

# DEFINE  THE PARAMETER FOR MULTINOMINAL CLASSIFICATION
num_class = length(levels(landmark))
params = list(
  booster = "gbtree",
  eta = 0.001,
  max_depth = 5,
  gamma = 3,
  subsample = 0.75,
  colsample_bytree = 1, 
  objective = "multi:softprob",
  eval_metric = "mlogloss", 
  num_class = num_class)

# TRAIN THE XGBOOST CLASSIFIER
xgb.fit = xgb.train(
  params = params,
  data = xgb.train,
  nrounds = 300,
  nthread = 1,
  early_stopping_rounds = 10,
  watchlist = list(val1 = xgb.train, val2 = xgb.test), 
  verbose = 0)

# REVIEW MODEL
xgb.fit
str(xgb.fit)
summary(travelokadata)

# CV 
train_control <- trainControl(method = "cv", number = 10, verboseIter = T, allowParallel = T)
train.index <- (travelokadata[train.index,])
XGBtune <- train(x = train.index [,-27], y = train.index [,27],
                 trControl = train_control, method = "xgbTree", verbose = T)
print(XGBtune)
XGBtune = as_data_frame(XGBtune)
plot(XGBtune, main = "Cross Validation XGB")

# PREDICTION OUTCOME WITH THE TEST DATA
#xgb.pred = predict(xgb.fit, test.data)
#xgb.pred = as.data.frame(xgb.pred)
#colnames(xgb.pred) = levels(travelokadata$randomrate)
#print(xgb.pred)

# USED THE PREDICTED LABEL WITH THE HIGHEST PROBABILITY
#xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames (xgb.pred))
#xgb.pred$label = levels(travelokadata$cleanaccom)[test.label]

#result = sum(xgb.pred$prediction == xgb.pred$label)/nrow(xgb.pred)
#print(paste("Final Accuracy = ", sprintf("%1.2f%%", 100* result)))
