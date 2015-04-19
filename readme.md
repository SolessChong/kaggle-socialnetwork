# LOG
This is log for my iterations of baseline methods. Multiple features are included throughout iterations.

## Try 1: Plain SVM
I used plain SVM, or sklearn.svm.svr, achieving really poor performance. Don't know why.

## Try 2: Linear model
I used linear model, slkearn.linear_model.LogisticRegresseion to predict. Poor result even worse than `Follower baseline`.

## Try 3: Logistic regression
Transformed features by log(1+x), gets worse than BaselineSolution.

## Try 4: Logistic regression
Use `model.predict_proba` instead of `model.predict`, yeilding 0~1 probability rather than {0,1} labels, achieving desired baseline method performance.