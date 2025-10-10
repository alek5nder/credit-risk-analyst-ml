from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np



def k_cross_val_overfitting(X_train, y_train, estimator,param_name, param_range):
    train_scores, test_scores = validation_curve(
        estimator=estimator,
        X=X_train,
        y=y_train,
        param_name=param_name,
        param_range=param_range,
        cv=10
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean,
             color="blue", marker="o",
             markersize=5, label="Training accuracy")

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15, color="blue")

    plt.plot(param_range, test_mean,
             color="green", linestyle="--",
             marker="s", markersize=5, label="Validation accuracy")

    plt.fill_between(param_range, test_mean + test_std,
                     test_mean - test_std, alpha=0.15, color="green")
    plt.grid()
    plt.xscale("linear")
    plt.legend(loc="lower right")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim([0, 1.03])
    plt.show()


