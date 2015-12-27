# Practical Machine Learning Course Project
mobini83  
December 23, 2015  
## Introduction   
This project is about using data from accelerometers on the belt, forearm, arm and dumbell of 6 participants. The goal of this project is to predict the manner in which they did the exercise. This report describes how the model is built, how cross validation is used, what the expected out of sample error is, and why the presented choices are made. 

## Data   
The data for this project come from this [source](http://groupware.les.inf.puc-rio.br/har).
The training data are downloadable from [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and the test data [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). The data sets are downloaded and saved on the lcoal disk, and read into RStudio:  

```r
if (!file.exists("trainDataset.csv")) {
trainDataset <- download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "trainDataset.csv")
}
if(!file.exists("testDataset.csv")){
testDataset <- download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "testDataset.csv")
}
train <- read.csv("trainDataset.csv", na.strings = c("NA", "#DIV/0!", ""))
test <- read.csv("testDataset.csv", na.strings = c("NA", "#DIV/0!", ""))
```
An initial exloratory analysis of the data is performed to take a look at the dimention of the training data set, and a summary of the class variable which is the the manner in which participants did the exercise.   

```r
dim(train)
```

```
## [1] 19622   160
```

```r
summary(train$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
plot(train$classe, main="Classe variable of the train dataset", xlab = "Classe", ylab = "Frequency")
```

![](index_files/figure-html/unnamed-chunk-2-1.png) 

#### Loading the required libraries    

```r
library(caret)
library(parallel)
library(doParallel)
```
### Partitioning the data for cross validation    
In order to be able to do the cross validation, the training data is divided into 2 training and validation subsets. For the sake of reproducibility, the seed is set to a constant first. 

```r
set.seed(123)
partition <- createDataPartition(train$classe, p = 0.8, list = FALSE)
trainingSubset <- train[partition, ]
validationSubset <- train[-partition, ]
```












