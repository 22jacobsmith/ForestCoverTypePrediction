install.packages("remotes")
remotes::install_github("rstudio/tensorflow")
reticulate::install_python()
keras::install_keras()


library(parsnip)
library(baguette)
library(tensorflow)
library(keras)
nn_recipe <- recipe(Cover_Type~., data = train) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = tune()) %>%
  set_engine("keras") %>%
  set_mode("classification")


nn_wf <- 
  workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)


nn_tuning_grid <- grid_regular(hidden_units(),
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