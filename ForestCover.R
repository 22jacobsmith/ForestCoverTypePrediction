# ForestCover.R


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
  step_pca(all_predictors(), threshold = .9)

rf_mod <- rand_forest(min_n = tune(), mtry = tune()) %>%
  set_engine('ranger') %>%
  set_mode('classification')


rf_wf <-
  workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_mod)




## set up a tuning grid
tuning_grid <-
  grid_regular(mtry(range = c(1,5)),
               min_n(range(1,30)),
               levels = 5)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

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
  finalize_workflow(best_tune) %>%
  fit(data = train)

rf_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare and export preds to csv for kaggle

rf_output <- tibble(Id = test$Id, Cover_Type = rf_preds$.pred_class)


vroom_write(rf_output, "ForestCover_RFPreds.csv", delim = ",")




####### NAIVE BAYES
library(discrim)

nb_recipe <-
  recipe(Cover_Type~., data=train) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .8)


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

svm_model <- svm_rbf(rbf_sigma = tune(),
                     cost = tune()) %>%
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
  finalize_workflow(best_tune) %>%
  fit(data = train)


svm_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")


# prepare and export preds to csv for kaggle

svm_output <- tibble(Id = test$Id, Cover_Type = svm_preds$.pred_class)


vroom_write(svm_output, "ForestCover_SVMPreds.csv", delim = ",")


boost_output %>% ggplot() +
  geom_bar(aes(x = Cover_Type))



