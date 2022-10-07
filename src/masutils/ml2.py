class ML2:
    """
    Getting Started:
    
    from IPython.display import display, Markdown
    Markdown('<weak style="font-size:12px">{}</weak>'.format(ML2.st_theory.__doc__))
    """
    def __init__(self):
        pass

    def st_theory():
        """
        Theory:
            CRISP-DM (Cross Industry Standard Process for Data Mining)
                ● Phases:
                    ● Business Understanding
                    ● Data Understanding
                    ● Data Preparation
                    ● Modeling
                    ● Evaluation
                    ● Deployment

            Assumptions of Logistic Regression:
                ● Independence of error, whereby all sample group outcomes are separate from each other (i.e., there are no duplicate responses)
                ● Linearity in the logit for any continuous independent variables
                ● Absence of multicollinearity
                ● Lack of strongly influential outliers

            Supervised vs Unsupervised Learning:
                ● Supervised LEarnign:
                    ● Uses Known and Labelled Data as Input
                    ● Very Complex
                    ● Uses off-line analysis
                    ● Number of classes are known
                    ● Accurate results
                ● Unupervised LEarnign:
                    ● Uses UnKnown Data as Input
                    ● Less Complex
                    ● Uses Real-Time analysis
                    ● Number of classes are unknown
                    ● Moderately Accurate results

            Logistic Regression Evaluation Metrics:
                ● Deviance
                ● AIC
                ● Pseudo R2:
                    ● McFadden R2
                    ● Cox-Snell R2
                    ● Nagelkerke R2

            Logistic Regression Perfromance Measures:
                ● Confusion matrix:
                    ● Accuracy : (TP+TN)/(TP+TN+FP+FN)
                    ● Precision : (TP)/(TP+FP) (Medical and Bank Trasactions)
                    ● Recall : (TP)/(TP+FN)
                    ● False Positive Rate : (FP)/(FP+TN) (1-Specificity)
                    ● Specificity : (TN)/(TN+FP)
                    ● F1 score : 2*(Precision*Recall)/(Precision+Recall)
                    ● Kappa
                ● Cross Entropy
                ● ROC: (TPR/FPR)
                ● Youden's Index: max(TPR - FPR)

            How to Handle Imbalanced Data:
                ● Up sample minority class (SMOTE)
                ● Down sample majority class
                ● Change the performance metric
                ● Try synthetic sampling approach
                ● Use different algorithm

            Naive Bayes:
                Assumptions of Naive Bayes Classification:
                    ● The predictors are independent of each other
                    ● All the predictors have an equal effect on the outcome.
                
                Naive Bayes Procedure:
                    ● Obtain Frequency of Predictors
                    ● Compute Likelihood of predictors and obtain prior probs based on train data
                    ● Compute Posterior Probabilities for each class labels
                    ● Assign most probable class

                Naive Bayes Advantages:
                    ● Easy to implement in the case of text analytics problems
                    ● Used for multiple class prediction problems
                    ● Performs better for categorical data than numeric data
                Naive Bayes Disadvantages:
                    ● Fails to find relationship among features
                    ● May not perform when the data has more number of predictor 
                    ● The assumption of independence among features may not always hold good

            KNN:
                Proximity/Similarity Measures:
                    ● Manhattan (L1)
                    ● Euclidean (L2)
                    ● Minkowski (LN)
                    ● Chebychev (L-inf)

                KNN Procedure:
                    ● Choose a Distance measure and K
                    ● Compute Distance between point whose label is to be identified and other points
                    ● Sort distances in asc order
                    ● Choose K data points having shortest distances and note the corresponding labels.
                    ● Label which has highest freq is assigned to the point.

                KNN Advantages:
                    ● Easy to implement
                    ● No training required
                    ● New data can be added at any time
                    ● Effective if training data is large
                KNN Disadvantages:
                    ● To chose apt value for K
                    ● Computational expensive
                    ● Can not tell which features gives the best result

                KNN Applications:
                    ● Image classification
                    ● Handwriting recognition
                    ● Predict credit rating of customers
                    ● Replace missing values


            Information Theory:

                Entropy:
                    ● Shannon's Entropy:
                        ● Entropy is the measure of information for classification problem, i.e. it measures the heterogeneity of a feature
                        ● The entropy of a feature is calculated = - sigma(1 to c) [pc log2 pc]
                            where pc is prob of occurence of class.
                    ● Conditional Entropy:
                        ● The conditional entropy of one feature given other is calculated as from the contingency table of the two features.
                        ● E(T|X) = sigma(x belongs to X) [P(c)*E(c)]

                Information Gain:
                    ● Information gain is the decrease in entropy at a node:
                    ● IG(T,X) = E(T) - E(T|X)
                        where T and X are events..
                    ● IG CANNOT BE NEGATIVE.

                Entropy for Numeric Features:
                    ● Sort the data in ascending order and compute the midpoints of the successive values. These midpoints act as the thresholds
                    ● For each of these threshold values, enumerate through all possible values to compute the information gain
                    ● Select the value which has the highest information gain

                
            Decision Trees:

                Terminologies/Parts:
                    ● Branch/Subtree: (a subsection of the entire decision tree.)
                    ● Root Node: (no incoming edge and zero or more outgoing edges.)
                    ● Internal Node: (exactly one incoming edge and zero or more outgoing edges.)
                    ● Leaf/Terminal Node: (exactly one incoming edge and no outgoing edge)
                    
                Measures of Purity of a Node:
                    ● Entropy: Shannon/Conditional
                    ● Gini Index: 1 - sigma(c=1,n) [pc**2]
                        IG using Gini Index:
                            IG(T|X) = GiniIndex(T) - GiniIndex(T|X)
                    ● Classification Error:
                        1 - max(pc**2)

                Consruction of Decision Trees:

                Decision Tree Algorithms:
                    ID3:
                        ● Handles only categorical data
                        ● May not converge to optimal decision tree
                        ● May overfit
                    C4.5:
                        ● Extension to ID3 algorithm
                        ● Handles both categorical and numeric data
                        ● Handles the missing data marked by '?'
                    C5.0:
                        ● Works faster than C4.5 algorithm
                        ● More memory efficient than C4.5 algorithm
                        ● Creates smaller trees with similar efficiency

                Overfitting in Decision Trees:
                    Overfitted Trees:
                        ● may have leaf nodes that contain only one sample, ie. singleton node
                        ● are complicated and long decision chains
                    Handing using Pruning:
                        ● Pruning is a technique that removes the branches of a tree that provide little power to classify instances, thus reduces the size of the tree
                        ● Pruning reduces the complexity of the tree
                        ● Pruning can be achieved in two ways:
                            ○ Pre-Pruning: The decision tree stops growing before the tree completely grown
                            ○ Post-Pruning: The decision tree is allowed to grow completely and then prune
                        
            Ensemble Learning:
                ● Bagging:
                    Homogenous Models can be built independently and outputs are aggregated at end.
                    Ex: Random Forest
                ● Boosting:
                    Homogenous Models can be built sequentially
                    Previous Model influences features of successive models.
                    Ex: AdaBoost
                ● Stacking:
                    Outputs from several base models are the inputs for Meta Model.

            Random Forest Classfier:
                ● Procedure:
                    ● Take Sample Set of M samples, N features
                    ● Bootstrap Sampling
                    ● Make different Training sets
                    ● Get all the different Predictions
                    ● Aggregate all the predictions.

            Feature Importance:
                ● Gini Importance:
                    ● Also known as mean decrease impurity
                    ● It is the average total decrease in the node impurity weighted by the probability of reaching it 
                    ● The average is taken over all the trees of the random forest
                ● Mean Decrease in accuracy:
                    ● Measure the decrease in accuracy on the out-of-bag data
                    ● Basically, the idea is to measure the decrease in accuracy on OOB data when you randomly permute the values for that feature.
                    ● If the decrease is low, then the feature is not important, and vice-versa.

            Bagging (Bootstrap Aggregation):

                ● Designed to improve the stability (small change in dataset change the model) and accuracy of classification and regression models
                ● It reduces variance errors and helps to avoid overfitting
                ● Can be used with any type of machine learning model, mostly used with Decision Tree
                ● Uses sampling with replacement to generate multiple samples of a given size. Sample may contain repeat data points
                ● For large sample size, sample data is expected to have roughly 63.2% ( 1 - 1/e) unique data points and the rest being duplicates
                ● For classification bagging is used with voting to decide the class of an input while for regression average or median values are calculate

			Boosting Techniques:
				Advantages of Boosting:
					● Enhances the efficiency of weak classifiers 
					● Both precision and recall can be enhanced through boosting algorithms
				Disadvantages of Boosting:
					● Loss of simplicity and explanatory power
					● Increased computational complexity

			Boosting vs Bagging:
				BAGGING:
					● Base learners learn is parallel 
					● Random sampling 
					● Reduces variance 
				BOOSTING:
					● Base learners learn sequentially
					● Non-random sampling
					● Reduces bias and variance 

			Adaboost Procedure:
				● Assign weights to each of the sample
				● Build stumps with each variable. Calculate the Gini Index or Entropy for each variable 
				● AdaBoost picks the variable with the smallest Gini Index for building the stump
				● Determine the ‘Amount of Say’ the stump will have in the final prediction. 
				● Incorrectly classified records from the 1st stump have a greater chance of being passed to the next stump
				● AdaBoost does this by, increasing the weights of the wrongly classified samples and decreasing the weights of the correctly classified samples
				● Increase the sample weights of incorrectly classified samples, by the formula.
				● Decrease the sample weights of correctly classified samples, by the formula.
				● Normalise the weights so that all the new sample weights add up to 1. This can be done by dividing each of the new sample weight by the sum of new sample weights
				● Adaboost creates a new collection of samples based on the normalized updated weights
				● Start with an empty dataset. Randomly pick a number between 0 and 1
				● The algorithm will check this random number falls into which bucket and populate the corresponding record in the new dataset.
				● Add samples to the new dataset by choosing random numbers as explained in step 7, till the new dataset is the same size as the original dataset
				● Get rid of the original samples and use the new collection of samples, for the next stump
				● Go back to the beginning and find the variable to make the next stump that does the best job at classifying the new collection of sample.
				
			Gradient Boosting:
				● Start with an initial leaf which is the initial prediction for all samples
				● Compute the Residuals
				● Build a tree for the residuals
				● Calculate the output( in terms of log(Odds)) for each leaf of the tree
				● Output of all the Leaf node is tabulated 
				● Scale the output by learning rate
				● Update the prediction
				● Convert Log(Odds) to probability
				● Calculate the Residuals again
				● Build the second tree 
				● GBM repeats these steps until it has built the specified number of trees or the residuals are very small or reach a threshold

			Adaboost vs GBM:
				Similarities:
					● Decision trees are the base learners
					● Trees are built based on the errors made by the previous trees
					● The size of the tree is restricted
					● Trees are scaled 
				Differences:
					AdaBoost Gradient Boost
						● Mostly made of stumps (Tree with a root and two leaves)
						● Stumps are scaled such that each stump has different amount of say in the final prediction
						● Shortcoming of the base learner is identified by high-weight samples
					GBM:
						● Starts with a leaf. Then build trees with 8 to 32 leaves. So, it does restrict the size of the trees
						● Output from each tree is scaled by a learning rate, however all trees have an equal amount of say in the final prediction
						● Shortcoming of the base learner is identified by gradients
				
			XGBoost:
				Features:
					● Regularization
					● A unique decision tree
					● Approximate Greedy algorithm
					● Weighted Quantile Sketch
					● Sparsity-Aware split finding
					● Parallel learning
					● Cache-Aware Access
					● Blocks for Out of Core Computations

			Voting:
				This is used for classification problem statements and Multiple base model can be created using using the same dataset with different algorithms
					● Majority voting:
						Every model makes a prediction (votes) for each test instance and the final output prediction is the one that receives more than half of the votes.
					● Weighted voting:
						Unlike majority voting, where each model has the same rights, we can increase the importance of one or more models.
			Stacking:
				The basic idea is to train machine learning algorithms with training dataset and then generate a new dataset with these models.
				Features:
					● The combiner system should learn how the base systems make errors
					● Stacking is a means of estimating and rectifying the errors in the base learners
					● Therefore, the combiner should be trained on the data unused in training the base learners

        """
        pass

    def st_1_eda_steps():
        """
        EDA Steps to perform:
        
            Read CSV: 
                df_data = pd.read_csv('Admission_predict.csv')
            
            Check and Change Datatypes:
                df_data.dtypes
                df_admissions['Research'] = df_admissions['Research'].astype(object)
            
            Remove Insignificant Variables:
                df_admissions.drop('Serial No.', axis = 1)
            
            Numeric Variables Boxplots (Plotting Outliers):
                fig, ax = plt.subplots(4,3, figsize=(30,16))
                for var, subplt in zip(df.select_dtypes(include=np.number).columns, ax.flatten()):
                    p = sns.boxplot(x=df[var], orient='h', ax=subplt)
                    p.set_xlabel(var, fontsize=20)
            
            Remove Outliers (by IQR):
                q1 = df.CreditScore.quantile(.25)
                iqr=df.CreditScore.quantile(.75)-q1
                lim = q1-1.5*iqr
                df["cred_score_le_iqr"] = df.CreditScore.map(lambda x : 1 if x<=lim else 0 )
                df.cred_score_le_iqr.value_counts()
            
            Correlation Map:
                plt.figure(figsize=(15,10))
                sns.heatmap(df.corr(), annot=True, linewidth=0.6, fmt=".2f", cmap="RdBu_r")

            Categorical Variables Countplot:
                sns.countplot(df_admissions.select_dtypes(include=np.object))

            Dummy Encode/ One-Hot Encode:
                dummy_var = pd.get_dummies(data = df_cat, drop_first = True)
            
            Scale Data:
                X_scaler = StandardScaler()
                num_scaled = X_scaler.fit_transform(df_num)
                df_num_scaled = pd.DataFrame(num_scaled, columns = df_num.columns)

            Join Numerical and Dummy Encoded vars:
                X = pd.concat([df_num_scaled, dummy_var], axis = 1)
        
            Train Test Split:
                X_train, X_test, y_train, y_test = train_test_split(X, df_target, random_state = 10, test_size = 0.2)
            
            SMOTE to fix Imbalance:
                X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=2, stratify=y)
                smote=SMOTE()
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        """
        pass

    def st_2_model_build_steps():
        """
        Define Scorecard:

            score_card = pd.DataFrame(columns=["model", "Accuracy Score",  "Precision Score","Recall Score", "f1-score", "Mean Cross validation score", "AUC Score", "Kappa Score"])
            kfold = KFold(n_splits=10, random_state=2, shuffle=True)
            def update_model_scores(model, test_y , pred_y):
                global score_card
                score_card = score_card.append({
                    "model": model,
                    "Mean Cross validation score":cross_val_score(model,X,y, cv=kfold).mean(),
                    "AUC Score":metrics.roc_auc_score(y_test, pred_y),
                    "Precision Score":metrics.precision_score(y_test, pred_y),
                    "Recall Score":metrics.recall_score(y_test, pred_y),
                    "Accuracy Score":metrics.accuracy_score(y_test, pred_y),
                    "Kappa Score":metrics.cohen_kappa_score(y_test, pred_y),
                    "f1-score":metrics.f1_score(y_test, pred_y)
                }, ignore_index=True)
            def build_base_models(list_of_models, train_x, train_y, test_x, test_y):
                classification_reports = {}
                for model in list_of_models:
                    model.fit(train_x, train_y)
                    pred_y = model.predict(test_x)
                    
                    update_model_scores(model, test_y, pred_y)
                    classification_reports[model.__class__.__name__]=classification_report(test_y, pred_y)

                    print(f"Trained and updated score for {model.__class__.__name__} model.")

                return classification_reports

        Build-Base-Models:

            models = [
                LogisticRegression(),
                DecisionTreeClassifier(),
                RandomForestClassifier(),
                KNeighborsClassifier(),
                SVC(),
                xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1)
            ]
            classification_reports = build_base_models(models, X_train_smote, y_train_smote, X_test, y_test)

        Log-Reg Model:
            logreg = sm.Logit(y_train, X_train).fit()
            print(logreg.summary())

        Metrics (AIC, Odds, Cross Entropy LogLoss):

            from sklearn import metrics

            print('AIC:', logreg.aic)
            print('Odds for Variables:', logreg.params)

            cutoff = 0.5
            y_pred = [ 0 if x < cutoff else 1 for x in logreg.predict(X_test)]

            metrics.accuracy_score(y_test, y_pred)
            metrics.log_loss(y_test, test_prediction)

        Do Classification Report:

            for model_name, report in classification_reports.items():
                print(f"{model_name}:\n")
                print(report)

        

        Show Highlighted Scoreboard:

            score_card.style.highlight_max(color='lightblue', subset = ["Mean Cross validation score", "Kappa Score", "f1-score"])

        Grid Search CV with XGBoost:

            from sklearn.model_selection import GridSearchCV
            params = {
                "n_estimators":[10,20,30,40,50],
                "learning_rate": [0.1, 0.01,0.2,1],
                "max_depth":[1,2,3,5,6,7]
                #'metric': ['hamming','euclidean','manhattan','Chebyshev']  # for KNeighborsClassifier()
            }
            grid_model = GridSearchCV(estimator=xgb.XGBClassifier(),
                                    param_grid=params,
                                    cv=10,
                                    scoring='accuracy',
                                    refit=True)
            grid_model.fit(X_train_smote, y_train_smote)
            print('Best Params: ',grid_model.best_params_)
            best_estimator = grid_model.best_estimator_
            y_pred_xgb_tuned = best_estimator.predict(X_test)
            update_model_scores(best_estimator, y_test, y_pred_xgb_tuned)
            print("Tuned XGB classifier:\n")
            print(classification_report(y_test, y_pred_xgb_tuned))

        Plot Feature Importances:
            df_fi = pd.DataFrame(
                {"feature importances":best_estimator.feature_importances_},
                index = X_train_smote.columns
                ).sort_values(by="feature importances", ascending=True)

            df_fi.plot.bar()

        Plot Confusion Matrix:
            from sklearn.metrics import confusion_matrix
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])
            sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, 
                        linewidths = 0.1, annot_kws = {'size':25})
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.show()
        
        Plot ROC AUC Curve:

            y_pred_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            plt.plot(fpr, tpr)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.plot([0, 1], [0, 1],'r--')
            plt.title('ROC curve for Admission Prediction Classifier', fontsize = 15)
            plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)
            plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)
            plt.text(x = 0.02, y = 0.9, s = ('AUC Score:',round(roc_auc_score(y_test, y_pred_prob),4)))
            plt.grid(True)

        Plot Error Rate for KNeighborsClassifier:

            error_rate = []

            for i in np.arange(1,25,2):
                knn = KNeighborsClassifier(i, metric = 'euclidean')
                score = cross_val_score(knn, X_train, y_train, cv = 5)
                score = score.mean()
                error_rate.append(1 - score)

            plt.plot(range(1,25,2), error_rate)
            plt.title('Error Rate', fontsize = 15)
            plt.xlabel('K', fontsize = 15)
            plt.ylabel('Error Rate', fontsize = 15)

            plt.xticks(np.arange(1, 25, step = 2))
            plt.axvline(x = 3, color = 'red')

            plt.show()

        
        """
        pass

