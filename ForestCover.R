# ForestCover.R
# This is the code for the Forest Cover Type Prediction competition
# The code used in the final Kaggle workbook appears first, followed
# other code used in model fitting, cross-validation, or testing


### KAGGLE WORKBOOK CODE

library(tidyverse)
library(tidymodels)
library(vroom)
library(parsnip)
library(keras)
library(baguette)
library(bonsai)

# Read in the training and testing datasets

train <- vroom("train.csv")
test <- vroom("test.csv")

# Make Cover_Type a factor

train$Cover_Type = as.factor(train$Cover_Type)


# Plot the proportion of observations in each class
ggplot(data = train) +
  geom_bar(aes(x = Cover_Type))


# Boxplot of elevation for each cover type
ggplot(data = train) +
  geom_boxplot(aes(x = Cover_Type, y = Elevation))

# Boxplot of hillshade_noon for each cover type
ggplot(data = train) +
  geom_boxplot(aes(x = Cover_Type, y = Hillshade_Noon))


# Histogram of Hillshade_Noon
ggplot(data = train) +
  geom_histogram(aes(x = Slope))

# Histogram of Elevation
ggplot(data = train) +
  geom_histogram(aes(x = Aspect))

## RANDOM FOREST

# Set up the recipe
rf_recipe <-
  recipe(Cover_Type~., data=train) %>%
  step_zv(all_predictors()) %>% # remove zero-variance columns
  step_normalize(all_numeric_predictors()) # normalize numeric predictors

# Set the model type and the values of each tuning parameter
rf_mod <- rand_forest(min_n = 1, mtry = 15, trees = 1000) %>% # chosen via 5-fold CV
  set_engine('ranger') %>%
  set_mode('classification')

# Set up the workflow
rf_wf <-
  workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_mod)


# Fit the workflow
final_wf <-
  rf_wf %>%
  fit(data = train)

# Use the workflow to make predictions for the test data
rf_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare a .csv file for submission to Kaggle competition
rf_output <- tibble(Id = test$Id, Cover_Type = rf_preds$.pred_class)

vroom_write(rf_output, "ForestCover_RFPreds.csv", delim = ",")

## NEURAL NET

# Set up recipe
nn_recipe <- recipe(Cover_Type~., data = train) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>% # remove zero-variance columns
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

# Set up ANN model
nn_model <- mlp(hidden_units = 10,
                epochs = 100) %>%
  set_engine("keras") %>%
  set_mode("classification")

# initialize the workflow
nn_wf <- 
  workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)

# finalize the workflow
final_wf <-
  nn_wf %>%
  fit(data = train)

# predict the class for the test data
nn_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")


# prepare a .csv file to submit to kaggle

nn_output <- tibble(Id = test$Id, Cover_Type = nn_preds$.pred_class)


vroom_write(nn_output, "ForestCover_NNPreds.csv", delim = ",")

## BOOSTING

# Set up the recipe
boost_recipe <-
  recipe(Cover_Type~., data=train) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Set the model type and the values of each tuning parameter
boost_mod <- boost_tree(trees = 500, learn_rate = 0.01, tree_depth = 2) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

# Set up the workflow
boost_wf <-
  workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_mod)

# finalize the workflow

final_wf <-
  boost_wf %>%
  fit(data = train)

# predict the class for the test data
boost_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")


# prepare a .csv file to submit to kaggle

boost_output <- tibble(Id = test$Id, Cover_Type = boost_preds$.pred_class)


vroom_write(boost_output, "ForestCover_BoostPreds.csv", delim = ",")

## STACKING

library(stacks)

# Set up the 2 control settings for stacking models
untuned_model <- control_stack_grid()
tuned_model <- control_stack_resamples()

# set up folds for cross-validation

folds <- vfold_cv(train, v = 5, repeats = 1)

# set up the three base learners

# RF model
rf_mod <- rand_forest(min_n = 1, mtry = 15, trees = 1000) %>%
  set_engine('ranger') %>%
  set_mode('classification')

rf_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(rf_recipe)

rf_model <-
  fit_resamples(rf_wf,
                resamples = folds,
                metrics = metric_set(roc_auc),
                control = tuned_model)
# NN model
nn_model <- mlp(hidden_units = 10,
                epochs = 100) %>%
  set_engine("keras") %>%
  set_mode("classification")

nn_wf <- 
  workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)

nn_model <-
  fit_resamples(nn_wf,
                resamples = folds,
                metrics = metric_set(roc_auc),
                control = tuned_model)
# Boosting model
boost_mod <- boost_tree(trees = 500, learn_rate = 0.01, tree_depth = 2) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

boost_wf <- 
  workflow() %>%
  add_model(boost_mod) %>%
  add_recipe(boost_recipe)


boost_model <-
  fit_resamples(boost_wf,
                resamples = folds,
                metrics = metric_set(roc_auc),
                control = tuned_model)


# set up the stacked model

my_stack <-
  stacks() %>%
  add_candidates(rf_model) %>%
  add_candidates(nn_model) %>%
  add_candidates(boost_model)

# fit the stacked model

stack_mod <-
  my_stack %>%
  blend_predictions() %>%
  fit_members()

# use the stacked model to generate predictions
stack_preds <-
  stack_mod %>%
  predict(new_data = test, type = "class")


# prepare the output for submission to kaggle

stack_output <- tibble(Id = test$Id, Cover_Type = stack_preds$.pred_class)


vroom_write(stack_output, "ForestCover_StackPreds.csv", delim = ",")

## VIEW OUTPUT

# View a plot of the predictions by class
stack_output %>% ggplot() +
  geom_bar(aes(x = Cover_Type), col = "gray", fill = "forestgreen") +
  labs(title = "Predicted Forest Cover Type")



### END OF KAGGLE WORKBOOK CODE


### Code used for other model fitting, cross-validation, or testing

### LIBRARIES  
library(tidyverse)
library(tidymodels)
library(vroom)

#library(corrplot)


### READ IN THE TEST AND TRAINING DATA

train <- vroom("train.csv")
test <- vroom("test.csv")

#train %>% View()



### MAKE COVER_TYPE A FACTOR

train$Cover_Type = as.factor(train$Cover_Type)

### EXPLORATORY DATA ANALYSIS

ggplot(data = train) +
  geom_bar(aes(x = Cover_Type))

ggplot(data = train) +
  geom_boxplot(aes(x = Cover_Type, y = Elevation))

ggplot(data = train) +
  geom_boxplot(aes(x = Cover_Type, y = Hillshade_Noon))

ggplot(data = train) +
  geom_histogram(aes(x = Elevation), binwidth = 100)


ggplot(data = train) +
  geom_histogram(aes(x = Hillshade_Noon))

ggplot(data = train) +
  geom_histogram(aes(x = Elevation))

cor(train$Hillshade_3pm, train$Hillshade_9am)
cor(train$Hillshade_3pm, train$Hillshade_Noon)
cor(train$Hillshade_9am, train$Hillshade_Noon)


###### RANDOM FOREST

rf_recipe <-
  recipe(Cover_Type~., data=train) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step

rf_mod <- rand_forest(min_n = 1, mtry = 15, trees = 1000) %>%
  set_engine('ranger') %>%
  set_mode('classification')


rf_wf <-
  workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_mod)





## set up a tuning grid
tuning_grid <-
  grid_regular(mtry(range = c(14,19)),
               min_n(range = c(1,7)),
               levels = 5)


## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv
library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

CV_results <-
  rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  rf_wf %>%
  #finalize_workflow(best_tune) %>%
  fit(data = train)


rf_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare and export preds to csv for kaggle

stopCluster(cl)


rf_output <- tibble(Id = test$Id, Cover_Type = rf_preds$.pred_class)


vroom_write(rf_output, "ForestCover_RFPreds.csv", delim = ",")


rf_output %>% ggplot() +
  geom_bar(aes(x = Cover_Type))

####### NAIVE BAYES
library(discrim)

nb_recipe <-
  recipe(Cover_Type~., data=train) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .95)


nb_mod <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_engine('naivebayes') %>%
  set_mode('classification')


nb_wf <-
  workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_mod)


## set up a tuning grid
tuning_grid <-
  grid_regular(Laplace(),
               smoothness(),
               levels = 5)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  nb_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

nb_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare and export preds to csv for kaggle

nb_output <- tibble(Id = test$Id, Cover_Type = nb_preds$.pred_class)


vroom_write(nb_output, "ForestCover_NBPreds.csv", delim = ",")






####### KNN

knn_recipe <-
  recipe(Cover_Type~., data=train) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


knn_mod <- nearest_neighbor(neighbors = tune()) %>%
  set_engine('kknn') %>%
  set_mode('classification')


knn_wf <-
  workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_mod)


## set up a tuning grid
tuning_grid <-
  grid_regular(neighbors(),
               levels = 10)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  knn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

knn_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare and export preds to csv for kaggle

knn_output <- tibble(Id = test$Id, Cover_Type = knn_preds$.pred_class)


vroom_write(knn_output, "ForestCover_KNNPreds.csv", delim = ",")


#### multinomial regression

mn_recipe <-
  recipe(Cover_Type~., data=train) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


mn_mod <- multinom_reg(penalty = tune(),
                       mixture = tune()) %>%
  set_engine('glmnet') %>%
  set_mode('classification')


mn_wf <-
  workflow() %>%
  add_recipe(mn_recipe) %>%
  add_model(mn_mod)


## set up a tuning grid
tuning_grid <-
  grid_regular(penalty(),
               mixture(),
               levels = 5)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  mn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  mn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

mn_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare and export preds to csv for kaggle

mn_output <- tibble(Id = test$Id, Cover_Type = mn_preds$.pred_class)


vroom_write(mn_output, "ForestCover_MultinomRegPreds.csv", delim = ",")


### neural net / mlp


library(parsnip)
library(baguette)
nn_recipe <- recipe(Cover_Type~., data = train) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = 3,
                epochs = 250) %>%
  set_engine("nnet") %>%
  set_mode("classification")


nn_wf <- 
  workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)


nn_tuning_grid <- grid_regular(hidden_units(range=c(1, 10)),
                               epochs(),
                               levels = 5)



## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")



# finalize wf and get preds

final_wf <-
  nn_wf %>%
  #finalize_workflow(best_tune) %>%
  fit(data = train)


nn_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")


# prepare and export preds to csv for kaggle

nn_output <- tibble(Id = test$Id, Cover_Type = nn_preds$.pred_class)


vroom_write(nn_output, "ForestCover_NNPreds.csv", delim = ",")

nn_output %>% ggplot() +
  geom_bar(aes(x = Cover_Type))


#### Boosting
library(bonsai)
library(parsnip)

boost_recipe <- recipe(Cover_Type~., data = train) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

boost_model <- boost_tree(trees = 1000,
                tree_depth = 5,
                learn_rate = 1e-10) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")



boost_wf <- 
  workflow() %>%
  add_model(boost_model) %>%
  add_recipe(boost_recipe)


tuning_grid <- grid_regular(trees(),
                            tree_depth(),
                            learn_rate(),
                               levels = 3)



## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  boost_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")



# finalize wf and get preds

final_wf <-
  boost_wf %>%
 # finalize_workflow(best_tune) %>%
  fit(data = train)


boost_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")


# prepare and export preds to csv for kaggle

boost_output <- tibble(Id = test$Id, Cover_Type = boost_preds$.pred_class)


vroom_write(boost_output, "ForestCover_BoostPreds.csv", delim = ",")


boost_output %>% ggplot() +
  geom_bar(aes(x = Cover_Type))



### SVM



svm_recipe <- recipe(Cover_Type~., data = train) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

svm_model <- svm_rbf(rbf_sigma = 1e-10,
                     cost = .176777) %>%
  set_engine("kernlab") %>%
  set_mode("classification")



svm_wf <- 
  workflow() %>%
  add_model(svm_model) %>%
  add_recipe(svm_recipe)


tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 3)




## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")



# finalize wf and get preds

final_wf <-
  svm_wf %>%
 # finalize_workflow(best_tune) %>%
  fit(data = train)


svm_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")


# prepare and export preds to csv for kaggle

svm_output <- tibble(Id = test$Id, Cover_Type = svm_preds$.pred_class)


vroom_write(svm_output, "ForestCover_SVMPreds.csv", delim = ",")


svm_output %>% ggplot() +
  geom_bar(aes(x = Cover_Type))


### stacking


library(stacks)


## control settings for stacking models
untuned_model <- control_stack_grid()
tuned_model <- control_stack_resamples()


#rf
rf_recipe <-
  recipe(Cover_Type~., data=train) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

rf_model <- rand_forest(trees = 500, mtry = 15, min_n = 1) %>%
  set_engine("ranger") %>%
  set_mode('classification')


## set up workflow
rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_model)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)


## fit to folds
rf_model <-
  fit_resamples(rf_wf,
                resamples = folds,
                metrics = metric_set(roc_auc),
                control = tuned_model)

# neural net
library(parsnip)

library(baguette)
nn_recipe <- recipe(Cover_Type~., data = train) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = 3,
                epochs = 250) %>%
  set_engine("nnet") %>%
  set_mode("classification")


nn_wf <- 
  workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)


nn_model <-
  fit_resamples(nn_wf,
                resamples = folds,
                metrics = metric_set(roc_auc),
                control = tuned_model)


my_stack <-
  stacks() %>%
  add_candidates(rf_model) %>%
  add_candidates(nn_model)

stack_mod <-
  my_stack %>%
  blend_predictions() %>%
  fit_members()


stack_preds <-
  stack_mod %>%
  predict(new_data = test, type = "class")


# prepare and export preds to csv for kaggle

stack_output <- tibble(Id = test$Id, Cover_Type = stack_preds$.pred_class)


vroom_write(stack_output, "ForestCover_StackPreds.csv", delim = ",")

stack_output %>% ggplot() +
  geom_bar(aes(x = Cover_Type), col = "gray", fill = "forestgreen") +
  labs(title = "Predicted Forest Cover Type")



# things that can improve your score
# fit 3 models, find the best
# describe feature engineering
# go deeper into best model
