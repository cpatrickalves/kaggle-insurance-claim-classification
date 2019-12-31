import autosklearn.classification
import autosklearn.metrics
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd

train = pd.read_csv("train_final.csv")
test = pd.read_csv("test_final.csv")

X = train.drop(['target'], axis=1)
y = train['target']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, 
														  per_run_time_limit=600, 
														  n_jobs=8,
														  ensemble_memory_limit = 4096,
														  ml_memory_limit = 6000)


automl.fit(X_train, y_train,  metric=autosklearn.metrics.log_loss)

print(automl.show_models())
print(automl.sprint_statistics())
print(automl.get_params())


y_hat = automl.predict(X_test)

print("Accuracy: ", sklearn.metrics.accuracy_score(y_test, y_hat))
print("ROC: ", sklearn.metrics.roc_auc_score(y_test, y_hat))
print("Logloss: ", sklearn.metrics.log_loss(y_test, y_hat))

# This call to fit_ensemble uses all models trained in the previous call
# to fit to build an ensemble which can be used with automl.predict()
#automl.fit_ensemble(y_train, ensemble_size=50, metric=autosklearn.metrics.log_loss)
#y_hat = automl.predict(X_test)
#print("Accuracy: ", sklearn.metrics.accuracy_score(y_test, y_hat))
#print("ROC: ", sklearn.metrics.roc_auc_score(y_test, y_hat))
#print("Logloss: ", sklearn.metrics.log_loss(y_test, y_hat))


# Test data for submission
test_x = test.drop(['ID'], axis=1)

# Realizando as previsoes
test_pred_prob = automl.predict_proba(test_x)[:,1]
submission = pd.DataFrame({'ID': test["ID"], 'PredictedProb': test_pred_prob.reshape((test_pred_prob.shape[0]))})
print(submission.head(10))
submission.to_csv('submission.csv', index=False)