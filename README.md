# AI_ML
Predicting Wild Blueberry Harvest Yields

Initial Model Training: I started by training an XGBoost regression model on the provided training data. I used the default hyperparameters to establish a baseline performance.

Feature Importance: After the initial training, I used the feature importances provided by XGBoost to identify which features were contributing the most to the model's predictions. I then selected features based on a threshold for importance, which helped in reducing the dimensionality of the data and potentially improving model performance.

Hyperparameter Tuning: I performed grid search hyperparameter tuning to find the best combination of parameters for my XGBoost model. This included tuning n_estimators, learning_rate, max_depth, subsample, and colsample_bytree. The goal was to optimize these parameters to minimize the Mean Absolute Error (MAE).

Regularization: To further refine the model and prevent overfitting, I introduced regularization parameters (reg_alpha and reg_lambda) into the hyperparameter grid. This added L1 and L2 regularization to the model, which can help improve generalization to unseen data.

Cross-Validation: Throughout the process, I used k-fold cross-validation to evaluate the model's performance. This helped ensure that the model's performance was consistent across different subsets of the data and provided a more reliable estimate of its generalization ability.

Model Evaluation: After each round of hyperparameter tuning, I evaluated the model using the MAE metric. I compared the MAE scores to the competition's leaderboard to gauge the relative performance of my model.

Stacking Consideration: Although not implemented in the final code I shared, I considered using stacking as a potential way to improve model performance. Stacking involves training a meta-model on the predictions of multiple base models, which can lead to a more accurate ensemble model.  I have no experience with Stacking and the training materials I used did not cover this subject so I discarded this potential option of model tuning.

Performance Analysis: I analyzed the MAE scores and the standard deviation of the cross-validation results to assess the model's accuracy and consistency. I aimed to reduce the MAE to be closer to the top scores on the leaderboard.

Benchmarking and Insights: I benchmarked my model's performance against the top competitors and sought insights from the competition's discussion forums, shared kernels, and winning solutions to identify strategies that could improve my model.

Iterative Improvement: Throughout the process, I iteratively improved the model by fine-tuning hyperparameters, selecting features, and considering advanced techniques like stacking. I kept track of the changes and their effects on model performance.

My finalized code reflects a systematic approach to model development, where I started with a simple model, progressively tuned and refined it, and considered advanced ensemble techniques to improve performance. The goal was to minimize the MAE and achieve a competitive score on the Kaggle leaderboard.

The best results I achieved were: Best MAE: 351.23
Best Parameters: {'colsample_bytree': 1.0, 'learning_rate': 0.04, 'max_depth': 5, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 10, 'subsample': 0.8}
[359.4992777  348.09492472 360.87806386 345.90358266 339.4782537
 355.54931898 344.53555408 343.22119438 364.66709407 350.46097809]
MAE with Best Parameters and Selected Features: 351.23 %
Standard Deviation with Best Parameters and Selected Features: 8.04 %

Based on the scores of the Kaggle competition, this would have placed around 1400 of the 1800+ submissions.  The best MAE score was 327.  My placement overall would suggest my model has room to improve but I'm confident that with more experience and more understanding of even more advanced techniques I could have built a much better model.  This project was a good way to implement the techniques I learned in the ML A to Z course and I understand where I have room for improvement.
