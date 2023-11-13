# ForestCoverBecker.R


library(tidyverse)
library(tidymodels)
library(vroom)


train <- vroom("train.csv")
test <- vroom("test.csv")


train$Cover_Type = as.factor(train$Cover_Type)




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


svm_output %>% ggplot() +
  geom_bar(aes(x = Cover_Type))



