class ML3:
    """
    Getting Started:
    
    from IPython.display import display, Markdown
    Markdown('<weak style="font-size:12px">{}</weak>'.format(ML3.st_theory.__doc__))
    """
    def __init__(self):
        pass

    def st_theory():
        """
        Theory:
        
            Supervised vs Unsupervised Learning:
                ● Supervised Learning:
                    ● Finding model that mapsthe target variable to input variables
                    ● Consists of inputs and labels
                ● Unsupervised Learning:
                    ● Aims to learn more about the given data
                    ● Has no Labels to map
                
            Clustering:
                
                Types of Clustering:
                    ● Density Based: Clusters formed based on density of nodes
                    ● Hierarchical Based: Formed based on distances between nodes
                    ● Graph Based: Dividing a set of graphs or dividing the nodes of graph
                    ● Partition Based: Partitioned into pre-determined number of clusters
                    ● Model Based: Assumes that data is a mixture of distributions and tries to fit a model such that each distrubution represents a cluster.
                    
                Proximity/Similarity Measures:
                    ● Manhattan (L1)
                    ● Euclidean (L2)
                    ● Minkowski (LN)
                    ● Chebychev (L-inf)
                
            K-Means Algorithm:
            
                Purpose:
                    ● Used when data is numeric
                    ● Recursive Technique
                    ● Cannot train a model
                    ● Based on Proximity measures.
                    ● Greedy Algorithm
                    ● Minimizes sq. error of points in cluster
                    ● Non-Deterministic Algorithm.
                
                K-Means Procedure:
                    ● Choose a Distnace measure and K
                    ● Randomly choose K points as cluster centroids
                    ● Assign nearest cluster centroid to each point
                    ● Compute means of clusters
                    ● Reassign cluster based on means
                    ● Repeat above 2 steps until cluster means dont change
                    
                K-Means Advantages:
                    ● Easy to Understand
                    ● Simple Implementation
                    
                K-Means Disadvantages:
                    ● Finding Optimal K is intensive to compute
                    ● Initial Centroid assignment affects output (Non-deterministic)
                    ● Ineffeicient for outliers
                
                For categorical, use K-Modes Algorithm.
                For mixed, use K-Prototypes Algorithm.
                
                Finding Optimal K:
                    ● Elbow Method:
                        ● Tries to reduce WCSS
                    ● Silhouette Score Steps:
                        ● For each point, calculate Silhouette Coefficient
                        ● Calculate mean intra-cluster distance for that point
                        ● Calculate mean inter-cluster distance for that point
                        ● Calculate Si using obtained ai and bi
                        ● Similarly, calculate silhouette coefficient for each observation
                        ● Take average of all, this is the silhouette score.
                        
                Hierarchical/Agglometarive Clustering:
                    
                    Agglomerative Clustering steps:
                        Consider each point as a unique cluster
                        Calculate pairwise distance between all clusters.
                        Combine 2 nearest clusters into a single cluster.
                        Calculate distance between newly formed cluster and remaining clusters
                        Repeat above 2 steps until a single cluster is formed.
                        
                    Linkage Methods:
                        
                        Single Linkage:
                            ● It is defined as the minimum distance between the points of two clusters.
                            ● The method can create the non-elliptical clusters 
                            ● It can produce undesirable results in the presence of outliers
                            ● It causes a chaining effect, where clusters have merged since at least one point in a cluster is closest to a point in another cluster. 
                            ● This forms a long and elongated cluster.
                            
                            Adv: 
                                ● Can create non-elliptical clusters
                            Disadv:
                                ● Sensitive to outliers
                                ● Prone to chaining effect

                        Complete Linkage:
                            ● It is defined as the maximum distance between the points of the two different clusters.
                            ● The method returns more stable clusters, with nearly equal diameter
                            ● It avoids the chaining effect
                            ● It is less sensitive to outliers
                            ● It breaks the large clusters and it is biased towards globular clusters
                            
                            Adv:
                                ● Creates more compact clusters
                            Disadv:
                                ● Biased towards globular clusters
                                ● Breaks large clusters
                                
                        Average Linkage:
                            ● It is defined as the average of all the pairwise distances between the two clusters.
                            ● This method balances between the single and complete linkage
                            ● It forms compact clusters and the method is robust to outliers
                            
                            Adv:
                                ● Balances between single and complete linkage
                                ● Robust to outliers
                            Disadv:
                                ● Biased towards globular clusters
                                
                        Centroid Linkage:
                            ● It is defined as the distance between the centroids (means) of the two clusters.
                            ● It creates similar clusters as average linkage
                            ● Problem: A smaller cluster can be more similar to the newly merged larger cluster rather than the individual clusters (inversion).
                            
                            Adv:
                                ● Cluster formation is similar to average linkage
                            Disadv:
                                ● Can cause inversion

                        Ward Linkage (ward minimum variance method):
                            ● The clusters are merged; if the new cluster minimizes the variance 
                            ● It is a computationally intensive method
                            
                            Adv:
                                ● Most effective in presence of outliers
                            Disadv:
                                ● Biased towards globular clusters

                    Dendrogram:
                        
                        Advantages:
                            ● Does not require a pre-specified number of clusters
                            ● Hierarchical relation between the clusters can be identified
                            ● Dendrogram provides a clear representation of clusters
                        Disadvantages:
                            ● Different dendrograms are produced for different linkage methods
                            ● Selecting an optimal number of clusters using dendrogram is sometimes difficult
                            ● Time complexity is high
                            
                Density Based Clustering (DBSCAN) (Density-Based Spatial Clustering of Applications with Noise):
                    
                    DBSCAN Procedure:
                        ● Decide Epsilon and Min_Samples
                        ● Choose a starting point (P) randomly and find Epsilon neighbourhood
                        ● If P is a core point, find all density-reachable points from P and forma cluster, else mark P as Noise.
                        ● Find next unvisited point and repeat.
                        ● Repeat this process till all points are visited.
                        
                    Advantages:
                        ● Does not require a pre-specified number of clusters
                        ● Useful to form clusters of any size and shape
                        ● Can be used to find outliers in the data
                    Disadvantages:
                        ● Can not efficiently work with the clusters of varying densities
                        ● Does not work well with high dimensional data

                Dimension Reduction Techniques:
                    
                    Note of Curse of Dimensionality:
                        ● If the number of features 'n' is large, the number of samples m, may be too small for accurate parameter estimation
                        ● covariance matrix has n2 parameters
                        ● For accurate estimation, sample size should be much bigger than n2,to be able to accurately represent the covariance, otherwise the model may become too complicated for the data, overfitting
                        ● If m < n2 we assume that features are uncorrelated (because we cannot represent it accurately with the 
                        given m points), even if we know this assumption is wrong. Doing so, we do not represent the covariance 
                        as parameters in our model.
                        
                    Dimension Reduction Techniques:
                        
                        PCA Procedure:
                            ● Standardize Data
                            ● Compute CoV matrix
                            ● Calculate Eigenvalues and Eignevectors
                            ● Sort Eigenvalues in Desc Order
                            ● Select Eigenvectors that explain max variance in data.
                        PCA Applications:
                            ● PCA is mainly used in image compression, facial recognition models
                            ● It is also used in the exploratory analysis to reduce the dimension of data before applying machine learning methods
                            ● Used in the field of psychology, finance to identify the patterns high dimensional data
                        
                        A Note on PCA Terminologies:
                            ● Covariance:
                                ● The covariance measures how co-dependent two variables are
                                ● Positive covariance value means that the two variables are directly proportional to each other
                                ● Negative covariance value means that the two variables are inversely proportional to each other
                                ● It is similar to variance, but the variance illustrates the variation of the single variable and covariance explains how two variables vary together
                            ● Covariance Matrix:
                                ● The covariance matrix explains the covariance between the pair of variables 
                                ● The diagonal entries represent the variance of the variable, as it is the covariance of the variable with itself
                                ● The diagonal matrix is always symmetric
                                ● The off-diagonal entries are covariance between the variables that represent the distortions (redundancy) in the data
                            ● Eigenvalue:
                                ● For any nxn matrix A, we can find n eigenvalues that satisfy the characteristic equation
                                ● A characteristic equation is defined as: |A - λI| = 0 i.e. det(A - λI) = 0
                                ● The characteristic polynomial for matrix A given as |A - λI|
                                ● The scalar value λ is known as the eigenvalue of the matrix A
                                ● Eigenvalues can be real/ complex in nature
                            ● Eigenvectors:
                                ● For each eigenvalue λ of a matrix A, there exist a non-zero vector x, which satisfy the equation: (A - λI)x = 0 i.e. Ax = λx
                                ● The vector x is known as the eigenvector corresponding to the eigenvalue λ
                                ● Eigenvectors are always orthogonal to each other
                                ● The eigenvector is a vector that does not changes its direction, after transformation by matrix A
                                
                        LDA Goal:
                            ● Maximize the distance between the means (i.e. between μ1 and μ2) of classes
                            ● Minimize the variance within each class
                            
                        LDA Procedure:
                            ● Standardize Data
                            ● Compute the within class matrix
                            ● Compute the between class matrix
                            ● Find projection vectors and transform the data.
                            
                        A Note on LDA Terminologies:
                            ● Within Class Matrix (Sw):
                                ● It captures how precisely the data is scattered within the class
                                ● Consider the data divided into two classes C1 and C2, the within class matrix is given by 
                                the summation of the covariance matrix (S1) of the class C1 and the covariance matrix (S2) of the class C2.
                                    Sw = S1 + S2
                            ● Between Class Matrix (Sb):
                                It represents how precisely the data is scattered across the classes
                                Suppose the means of classes C1 and C2 are μ1 and μ2 respectively. The formula to find SB is given as:
                                    Sb = (μ1 - μ2).(μ1 - μ2)T

                        PCA vs LDA:
                            PCA:
                                ● It is described as an unsupervised machine learning technique
                                ● It considers the independent features to find principal components that maximize the variance in the data
                            LDA:
                                ● It is described as a supervised machine learning technique
                                ● It calculates the linear discriminants to obtain the maximum separation between the classes of the dependent variable
                                
                        Kernel PCA:
                            ● Kernel PCA uses a kernel function to project dataset into a higher dimensional feature space, where it is linearly separable. 
                            ● It is similar to the idea of Support Vector Machines.
                            ● There are various kernel methods like linear, polynomial, and gaussian.

                        Limitations in Kernel PCA:
                            ● Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated 
                            variables into a set of values of linearly uncorrelated variables called principal 
                            components (from Wikipedia). 
                            ● PCA assumes that the data contains continuous values only and contains no categorical variables.
                            ● It is not possible to apply PCA techniques for dimensionality reduction when the data is composed of categorical variables.

                Recommendation Systems:
                    
                    Types:
                        ● Popularity Based
                        ● Content Based
                        ● Collaborative Filtering
                            ● User Based
                            ● Item Based
                        ● Matrix Factorization
                        ● Association Rule
                        ● Hybrid
                        
                    Popularity Based System :
                        
                        Procedure:
                            ● Read the dataset
                            ● Use group by or SQL merge concept to get the ratings for each item/movie
                            ● Take mean of all the item/movie
                            ● Sort it in descending order
                            ● Recommend top ten or five to a new user
                            
                            ● Takes only mean into consideration
                            ● User bias can be removed by deducting mean rating of a user for each item.

                    Content Based System:
                        
                        Procedure:
                            ● Read data
                            ● Preprocessing
                            ● Distance matrix calculation
                            ● Check similarity
                            ● Recommend

                        Advantages:
                            ● Do not require a lot of user's data
                            ● Only item data is sufficient
                            ● Does not suffer from cold start problem
                            ● Less expensive to build and maintain
                        Disadvantages:
                            ● Feature availability
                            ● Less dynamic
                            ● Diversity is missing
                            
                    Types of Similarities:
                        ● Euclidean Similarity
                        ● Cosing Similarity
                        ● Pearson Similarity
                        ● Jaccard Similarity
                        
                    Collaborative Filtering:
                        
                        Types:
                            Memory Based:
                                ● Similarity between users and/or items are calculated and used as weights to predict a rating
                                ● Approach collaborative filtering problem using the entire dataset
                                ● Not learning any parameter here. Non-parametric approaches.
                                ● Quality of prediction is good
                                ● Scalability is an issue
                                Eg - Item based, User based, kNN clustering
                            Model Based:
                                ● Uses statistics and machine learning methods on rating matrix
                                ● Speed and scalability issues can be addressed
                                ● Can solve problem of huge database & sparse matrix
                                ● Better at dealing with sparsity
                                ● Predicts ratings of unrated items
                                ● Inference is not traceable due to hidden factors
                                Eg-Matrix factorization ( SVD, PMF, NMF), Neural nets based
                        
                        Types: (In Memory Based):
                            ● User Based:
                                ● Similar users are considered
                                ● “Users who are similar to you also liked…”
                            ● Item Based:
                                ● Similar items are considered
                                ● “Users who liked this item also liked…”
                            
                        Challenges with Collaborative Filtering:
                            ● Cold Start problem: The model requires enough amount of other users already in the system to find a good match.
                            ● Sparsity: If the user/ratings matrix is sparse, and it is not easy to find the users that have rated the same items.
                            ● Popularity Bias: Not possible to recommend items to users with unique tastes.
                                ● Popular items are recommended more to the users as more data being available on them
                                ● This may begin a positive feedback loop not allowing the model to recommend items with less popularity to the users
                            ● Shilling attacks: Users can create fake preferences to push their own items
                            ● Scalability: Collaborative filtering models are computationally expensive

                        Steps in Collaborative Filtering:
                            ● Determine similar users
                                ● Calculate similarity matrix using similarity distance and user-item ratings. Get top similar neighbors
                            ● Estimate rating that a user would give to an item based on the ratings of similar users
                                ● Estimated rating R for a user U for an item I would be close to average rating given to I by the top n users most similar to U
                                ● Ru = (∑nu=1Ru)/n
                                ● Weighted average - multiply each rating by similarity factor
                            ● Accuracy of estimated ratings 
                                ● RMSE ( root mean squared error) 
                                ● MAE ( mean absolute error)

                    Market Basket Analysis:
                        ● Uncovers association between items.
                        ● Identifies pattern of co-occurrence
                        ● Market basket analysis may provide the retailer with information to understand the behaviour of a buyer.
                        “Customers who bought book A also bought book B”
                        
                        Terminologies in Merket Basket:
                            ● Itemset - a collection of items purchased by a customer
                                ● Ex - {Pizza, pepsi, garlic bread}
                            ● Support count (o )- Frequency of occurrence of an itemset.
                                ● Ex- o( Pizza, pepsi, garlic bread) = 2
                            ● Support - fraction of transaction that contains itemset
                                ● Ex- S( Pizza, pepsi, garlic bread) = ⅖
                            ● Frequent Itemset - An itemset whose support isgreater than or equal to a min_sup threshold

                    Apriori Algorithm Steps:
                        ● Set a min. support and confidence
                        ● Take all the subsets in transactions having higher support than min support
                        ● Take all the rules of these subsets having higher confidence than min confidence
                        ● Sort the rules by decreasing lift
                        
                    Hybrid Algorithm:
                        ● Combination of multiple algorithm
                        ● Customized algorithm
                        
                        Methods of Hybrid Algorithm:
                            ● Weighted - Each system is weighted to calculate final recommendation
                            ● Switching - System switches between different recommendation model
                            ● Mixed - Recommendations from different models are presented together.
                            
                    Evaluation Metrics of Recommendation Systems:
                        ● User Satisfaction
                            ○ Subjective metric
                            ○ Measured by user survey or online experiments
                        ● Prediction Accuracy
                            ○ Rating Prediction (MAE, RMSE)
                            ○ Top-N Recommendation (Precision, Recall)
                        ● Coverage
                            ○ Ability to recommend long tail items ( entropy, gini index)
                        ● Diversity
                            ○ Ability to cover user's different interests
                        ● Novelty - Ability of Recommendation system to recommend long tail items and new items.
                        ● Trust - Trust increases the interaction of user to recommendation system.
                            ○ Transparency, social
                        ● Robust - Ability of Recommendation system to prevent attacks.
                            ○ Shilling attack
                        ● Real Time - Generate new recommendation when user has new behaviours immediately.
        """
        pass

    def st_1_eda_steps():
        """
        Imports:
            from scipy.cluster.hierarchy import linkage
            from scipy.cluster.hierarchy import dendrogram
            from scipy.cluster.hierarchy import cophenet


        EDA Steps to perform:
        
            Read CSV: 
                df_data = pd.read_csv('Admission_predict.csv')
            
            Check and Change Datatypes:
                df_data.dtypes
                df_admissions['Research'] = df_admissions['Research'].astype(object)
            
            Remove Insignificant Variables:
                df_admissions.drop('Serial No.', axis = 1)
            
            Get pairs of variables where correlation more than 0.8:
                corr_matrix = df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
                features_corr = [col for col in upper.columns if any(upper[col] > 0.8)]
                features_corr

            Numeric Variables Boxplots (Plotting Outliers):
                fig, ax = plt.subplots(4,3, figsize=(30,16))
                for var, subplt in zip(df.select_dtypes(include=np.number).columns, ax.flatten()):
                    p = sns.boxplot(x=df[var], orient='h', ax=subplt)
                    p.set_xlabel(var, fontsize=20)

            Categorical Variables Countplot:
                sns.countplot(df_admissions.select_dtypes(include=np.object))

            Dummy Encode/ One-Hot Encode: (For Categorical Data) (ONLY FOR PCA/MCA)
                dummy_var = pd.get_dummies(data = df_cat, drop_first = True)
            
            Scale Data:
                X_scaler = StandardScaler()
                num_scaled = X_scaler.fit_transform(df_num)
                df_num_scaled = pd.DataFrame(num_scaled, columns = df_num.columns)
        """
        pass

    def st_2_model_build_steps():
        """
        K-Means Clustering:
            Plot WCSS (Elbow Plot):
                def plot_wcss():
                    wcss  = []
                    for i in range(1,21):
                        kmeans = KMeans(n_clusters = i, random_state = 10)
                        kmeans.fit(X)
                        wcss.append(kmeans.inertia_)
                            pass

                    plt.plot(range(1,21), wcss)
                    plt.title('Elbow Plot', fontsize = 15)
                    plt.xlabel('No. of clusters (K)', fontsize = 15)
                    plt.ylabel('WCSS', fontsize = 15)
                    plt.axvline(x = 5, color = 'red')
                    plt.show()

            Elbow plot using KElbowVisualizer:
                from yellowbrick.cluster import KElbowVisualizer
                from sklearn.cluster import KMeans
                model = KElbowVisualizer(KMeans(), k=15)
                model.fit(X)
                model.show()

            K-Value using Silhouette Score:
                n_clusters = [2, 3, 4, 5, 6]
                for K in n_clusters:
                    cluster = KMeans (n_clusters= K, random_state= 10)
                    predict = cluster.fit_predict(X)
                    score = silhouette_score(X, predict, random_state= 10)
                    print ("For {} clusters the silhouette score is {})".format(K, score))

            Silhouette Plot:
                from yellowbrick.cluster import SilhouetteVisualizer
                X=data_dime_input_merged_pca_applied
                n_clusters=[2,3,4,5,6,7,8,9,10]
                for K in n_clusters:
                    model=KMeans(n_clusters=K,random_state=42)
                    viz = SilhouetteVisualizer(model).fit(X)
                    viz.show()

            Dendrogram Plot:
                Simple Plot:
                    from scipy.cluster.hierarchy import linkage,dendrogram
                    plt.figure(figsize=(12,10))
                    ward_merge=linkage(data_dime_input_merged,method='ward')
                    dendrogram(ward_merge,truncate_mode='lastp',p=50)
                    plt.show()

                Compare Performance of Different Linkages:
                    from scipy.cluster.hierarchy import dendrogram,linkage
                    from sklearn.metrics.pairwise import euclidean_distances
                    linkages = ['centroid','single','complete','average','median','ward']
                    coeff = []
                    plt.rcParams['figure.figsize'] = (12,12)
                    for i in linkages:
                        linkage_ = linkage(df,method = i)
                        eucli_dist = euclidean_distances(df)
                        dist_array = eucli_dist[np.triu_indices(3609,k=1)]
                        c,cophenet_dist = cophenet(linkage_,dist_array)
                        coeff.append(c)
                        dendrogram(linkage_,truncate_mode = 'lastp',p=100)
                        plt.show()
                    v = dict(zip(linkages,coeff))
                    print(v)

            Build Clusters:
                new_clusters = KMeans(n_clusters = 5, random_state = 10)
                new_clusters.fit(X)
                df_cust['Cluster'] = new_clusters.labels_

            Visualize All Cluster:
                sns.scatterplot(x = 'Cust_Spend_Score', y = 'Yearly_Income', data = df_cust, hue = 'Cluster')
                plt.title('K-means Clustering (for K=5)', fontsize = 15)
                plt.xlabel('Spending Score', fontsize = 15)
                plt.ylabel('Annual Income', fontsize = 15)
                plt.show()

            Cluster-Wise Analysis (do for each cluster):
                def describe_cluster(cluster_id):
                    print(len(df_cust[df_cust['Cluster'] == cluster_id]))
                    print(df_cust[df_cust.Cluster==cluster_id].describe())
                    print(df_cust[df_cust.Cluster==cluster_id].describe(include = object))

        Hierarchical Clustering:

            Build Linkage Matrix:
                #instantiate linkage object with scaled data and consider 'ward' linkage method 
                link_mat = linkage(features_scaled, method = 'ward')     
                #print first 10 observations of the linkage matrix 'link_mat'
                print(link_mat[0:10])
            
            Plot Dendrogram:
                from scipy.cluster.hierarchy import linkage,dendrogram
                plt.figure(figsize=(12,10))
                ward_merge=linkage(data_dime_input_merged,method='ward')
                dendrogram(ward_merge,truncate_mode='lastp',p=50)
                plt.show()

            Calculate Cophenet Coefficient:
                from scipy.cluster.hierarchy import cophenet
                from sklearn.metrics.pairwise import euclidean_distances
                eucli_dist = euclidean_distances(features_scaled)
                dist_array = eucli_dist[np.triu_indices(5192, k = 1)]
                coeff, cophenet_dist = cophenet(link_mat, dist_array)
                print(coeff)

            K-Value using Silhouette Score:
                n_clusters = [2, 3, 4, 5, 6]
                for K in n_clusters:
                    cluster = AgglomerativeClustering (n_clusters= K, random_state= 10, linkage='ward')
                    predict = cluster.fit_predict(X)
                    score = silhouette_score(X, predict, random_state= 10)
                    print ("For {} clusters the silhouette score is {})".format(K, score))

            Build Clusters:
                clusters = AgglomerativeClustering(n_clusters=2, linkage='ward')
                clusters.fit(features_scaled)
                df_prod['Cluster'] = clusters.labels_
                df_prod['Cluster'].value_counts()

            Visualize All Cluster:
                sns.scatterplot(x = 'Cust_Spend_Score', y = 'Yearly_Income', data = df_cust, hue = 'Cluster')
                plt.title('Hierarchical Clustering (for K=5)', fontsize = 15)
                plt.xlabel('Spending Score', fontsize = 15)
                plt.ylabel('Annual Income', fontsize = 15)
                plt.show()

            Cluster-Wise Analysis (do for each cluster):
                def describe_cluster(cluster_id):
                    print(len(df_cust[df_cust['Cluster'] == cluster_id]))
                    print(df_cust[df_cust.Cluster==cluster_id].describe())
                    print(df_cust[df_cust.Cluster==cluster_id].describe(include = object))
        
        DBSCAN:
            Build Model:
                model = DBSCAN(eps = 0.8, min_samples = 15)
                model.fit(features_scaled)
                df_prod['Cluster_DBSCAN'] = model.labels_
            
            Visualize All Cluster:
                sns.scatterplot(x = 'Cust_Spend_Score', y = 'Yearly_Income', data = df_cust, hue = 'Cluster')
                plt.title('DBSCAN Clustering (for K=5)', fontsize = 15)
                plt.xlabel('Spending Score', fontsize = 15)
                plt.ylabel('Annual Income', fontsize = 15)
                plt.show()

            Cluster-Wise Analysis (do for each cluster):
                def describe_cluster(cluster_id):
                    print(len(df_cust[df_cust['Cluster'] == cluster_id]))
                    print(df_cust[df_cust.Cluster==cluster_id].describe())
                    print(df_cust[df_cust.Cluster==cluster_id].describe(include = object))

            **Interpretation**: 
                We can see that the algorithm has identified most of the technical products as the outliers.
                Here we can see that the DBSCAN algorithm has not grouped the product like hierarchical clustering. Thus we can conclude that the DBSCAN algorithm is working poorly on this dataset.

        Raw PCA:
            Normalize Data:
                X = iris.data
                X_std = StandardScaler().fit_transform(X)
            Calculate Cov Matrix:
                cov_matrix = np.cov(X_std.T)
                print('Covariance Matrix:', cov_matrix)
            PairPlot:
                sns.pairplot(X_std_df)
                plt.tight_layout()
            Calculate EigenVals and Eigne Vecs:
                eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
            Cumulative Variance Explained and choosing Number of components on PCA:
                tot = sum(eig_vals)
                var_exp = [( i /tot ) * 100 for i in sorted(eig_vals, reverse=True)]
                cum_var_exp = np.cumsum(var_exp)
                print("Cumulative Variance Explained", cum_var_exp)

        PCA:
            from sklearn.decomposition import PCA
            pca = PCA()
            pca.fit(df)
            print(np.sum(pca.explained_variance_ratio_.cumsum() <= 0.9)) # Based on this, select n_components
            pca  = PCA(n_components=102)
            X_train_2 = pca.fit_transform(X_train)
            X_test_2 = pca.transform(X_test)
            explained_variance = pca.explained_variance_ratio_

        SVD:
            from sklearn.decomposition import TruncatedSVD
            tsvd  = TruncatedSVD()
            tsvd.fit(df)
            print(np.sum(tsvd.explained_variance_ratio_.cumsum() <= 0.9)) # Based on this, select n_components
            tsvd  = TruncatedSVD(n_components=2)
            X_train_2 = tsvd.fit_transform(X_train)
            X_test_2 = tsvd.transform(X_test)
            explained_variance = tsvd.explained_variance_ratio_

        Cases:

            DT Before Applying PCA:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                from sklearn import tree
                model=tree.DecisionTreeClassifier()
                model.fit(X_train,y_train)
                y_pred_DT = model.predict(X_test)
                sns.heatmap(confusion_matrix(y_test, y_pred_DT), annot=True)
                print(classification_report(y_test,y_pred_DT))
                print(accuracy_score(y_test, y_pred_DT))

            DT After applying PCA:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                from sklearn.decomposition import PCA
                pca = PCA()
                pca.fit(df)
                print(np.sum(pca.explained_variance_ratio_.cumsum() <= 0.9)) # Based on this, select n_components
                pca = PCA(n_components=52)
                X_train_2 = pca.fit_transform(X_train)
                X_test_2 = pca.transform(X_test)
                explained_variance = pca.explained_variance_ratio_
                from sklearn import tree
                model2=tree.DecisionTreeClassifier(random_state=1)
                model2.fit(X_train_2,y_train)
                y_pred_DT_2 = model2.predict(X_test_2)
                sns.heatmap(confusion_matrix(y_test, y_pred_DT_2), annot=True)
                print(classification_report(y_test,y_pred_DT_2))
                print(accuracy_score(y_test, y_pred_DT_2))

                Explained Vairance ratio with Elbow Plot for PCA:
                    plt.figure(figsize=(10,7))
                    plt.bar(range(pca_applied.shape[1]), pca.explained_variance_ratio_)
                    plt.step(range(pca_applied.shape[1]),pca.explained_variance_ratio_.cumsum() )
                    plt.xlabel("Principal components")
                    plt.ylabel("Explained variance ratio")
                    plt.axhline(0.9)
                    plt.tight_layout()

            DT After Applying LDA:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                model = LinearDiscriminantAnalysis()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                model.fit(X_train,y_train)
                print(np.sum(model.explained_variance_ratio_.cumsum() <= 0.9)) # Based on this, select n_components
                model = LinearDiscriminantAnalysis(n_components=52)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
                print(classification_report(y_test,y_pred))
                print(accuracy_score(y_test, y_pred))

                LDA Hyperparameter Tuning:
                    from sklearn.model_selection import GridSearchCV
                    from sklearn.model_selection import RepeatedStratifiedKFold
                    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                    model = LinearDiscriminantAnalysis()
                    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                    grid = dict()
                    grid['solver'] = ['svd', 'lsqr', 'eigen']
                    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
                    results = search.fit(X_train, y_train)
                    print('Mean Accuracy: %.3f' % results.best_score_)
                    print('Config: %s' % results.best_params_)

                    (With the Best Params, repeat the above LDA steps with Best Params)

            DT After Applying Kernel PCA: ( Too Complex, Will not be asked! )

            DT After Applying MCA:
                ! pip install prince
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
                cat = []
                num = []
                for i in X.columns:
                    if X[i].nunique() < X.shape[0]/4:
                        cat.append(i)
                    else:
                        num.append(i)
                X_train_cat = X_train[cat]
                X_train_num = X_train[num]
                X_test_cat = X_test[cat]
                X_test_num = X_test[num]
                import prince
                from sklearn.metrics import make_scorer
                from sklearn.metrics import explained_variance_score
                mca = prince.MCA()
                mca_X_train = mca.fit_transform(X_train_cat)
                mca_X_test = mca.fit_transform(X_test_cat)
                main_mca_X_train = pd.concat([mca_X_train,X_train_num],axis = 1)
                main_mca_X_test = pd.concat([mca_X_test,X_test_num],axis = 1)
                explained_vars = mca.explained_variance_ratio_.cumsum()
                explained_vars
                from sklearn import tree
                model2=tree.DecisionTreeClassifier()
                model2.fit(main_mca_X_train,y_train)
                y_pred_DT_2 = model2.predict(main_mca_X_test)
                sns.heatmap(confusion_matrix(y_test, y_pred_DT_2), annot=True)
                print(classification_report(y_test,y_pred_DT_2))
                print(accuracy_score(y_test, y_pred_DT_2))



        Recommendation Systems:

            Popularity Based Recommendation:
                x=df.groupby('ItemID').agg({'Rating':'mean','ItemID':'count'})
                x.rename(columns={'ItemID':'CountofReviews'},inplace=True)
                x[x['CountofReviews']>50].sort_values(by='Rating',ascending=False).head(10)

            Content Based Recommendation:
                Preprocessing:
                    genres=data1['genres'].str.split('|',expand=True)
                    genres=genres.fillna('Others')
                    genres.columns=['genre1','genre2','genre3']
                    data1=pd.concat([data1,genres],axis=1)
                    data2=data1[movie_feat]
                    data3=pd.get_dummies(data2)
                    data3=data3.dropna()
                
                from sklearn.neighbors import NearestNeighbors
                rec_model = NearestNeighbors(metric = 'cosine')
                rec_model.fit(data3)
                query_movie_index=200
                dist, ind = rec_model.kneighbors(data3.iloc[query_movie_index, :].values.reshape(1, -1), n_neighbors = 6)
                list(data3.index[ind[0]])[1:]
                for i in range(0, len(dist[0])):
                    if i == 0:
                        print('Top 5 Recommendations for the user who watched the movie :',data3.index[query_movie_index])
                    else:
                        print(i, data3.index[ind[0][i]])


            Collaborative Recommendation:
                from surprise import Dataset,Reader
                from surprise.model_selection import train_test_split,cross_validate
                from surprise import KNNWithMeans,SVDpp
                from surprise import accuracy
                reader=Reader(rating_scale=(1,5))
                rating_data=Dataset.load_from_df(data_recom[['UserID','ItemID','Rating']],reader)
                [trainset,testset]=train_test_split(rating_data,test_size=0.15,shuffle=True)
                trainsetfull=rating_data.build_full_trainset()
                print("Number of users :",trainsetfull.n_users)
                print("Number of items :",trainsetfull.n_items)
                my_k = 15
                my_min_k = 5
                my_sim_option = {'name':'pearson', 'user_based':False}
                algo = KNNWithMeans(k = my_k, min_k = my_min_k, sim_options = my_sim_option, verbose = True)
                results = cross_validate(
                    algo = algo, data = rating_data, measures=['RMSE'], 
                    cv=5, return_train_measures=True
                    )
                print(results['test_rmse'].mean())
                algo.fit(trainsetfull)
                algo.predict(uid = 50, iid =2)
                len(ratings['movieId'].unique())
                item_id_unique=ratings['movieId'].unique()
                item_id10=ratings.loc[ratings['userId']==10,'movieId']
                item_id_pred=np.setdiff1d(item_id_unique,item_id10)
                testset=[[50,iid,5] for iid in item_id_pred]
                pred=algo.test(testset)
                pred
                pred_ratings=np.array([pred1.est for pred1 in pred])
                i_max=pred_ratings.argmax()
                iid=item_id_pred[i_max]
                print("Top item for user 10 has iid {0} with predicted rating {1}".format(iid,pred_ratings[i_max]))

                Compare Different Algorithms Scoreboard:

                    from surprise import NormalPredictor
                    from surprise import KNNBasic
                    from surprise import KNNWithMeans
                    from surprise import KNNWithZScore
                    from surprise import KNNBaseline
                    from surprise import SVD
                    from surprise import BaselineOnly
                    from surprise import SVDpp
                    from surprise import NMF
                    from surprise import SlopeOne
                    from surprise import CoClustering
                    benchmark = []
                    # Iterate over all algorithms
                    algorithms = [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]
                    print ("Attempting: ", str(algorithms))
                    for algorithm in algorithms:
                        print("Starting: " ,str(algorithm))
                        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
                        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
                        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
                        benchmark.append(tmp)
                        print("Done: " ,str(algorithm))
                    print ('DONE')
                    surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
                    surprise_results

            Apriori Algorithm:
                from mlxtend.frequent_patterns import apriori
                from mlxtend.frequent_patterns import association_rules

                unique_row_items = []
                for index, row in df.iterrows():
                    items_series = str(row[0]).split(',')
                    for item in items_series:
                        if item not in unique_row_items:
                            unique_row_items.append(item)
                df_apriori = pd.DataFrame(columns=unique_row_items)
                for index, row in df.iterrows():
                    items = str(row[0]).split(',')
                    one_hot_encoding = np.zeros(len(unique_row_items),dtype=int)
                    for it in items:
                        for i,column in enumerate(df_apriori.columns):
                            #print(i,column,it)
                            if it == column:
                                one_hot_encoding[i] = 1
                    df_apriori.at[index] = one_hot_encoding
                df_apriori=df_apriori.astype('int')
                freq_items = apriori(df_apriori, min_support = 0.2, use_colnames = True, verbose = 1)
                df_association_rules = association_rules(freq_items, metric = "confidence", min_threshold = 0.2)
                df_association_rules.sort_values("confidence",ascending=False)
                cols = ['antecedents','consequents']
                df_association_rules[cols] = df_association_rules[cols].applymap(lambda x: tuple(x))#.apply(lambda x: str(x))
                print (df_association_rules)
                df_association_rules = (df_association_rules.explode('antecedents')
                    .reset_index(drop=True)
                    .explode('consequents')
                    .reset_index(drop=True))
                df_association_rules["product_group"] = df_association_rules["antecedents"].apply(lambda x: str(x)) + "," + df_association_rules["consequents"].apply(lambda x: str(x))
                df1 = df_association_rules.loc[:,["product_group","confidence","lift"]].sort_values("confidence",ascending=False)
            
                Plot:
                    sns.barplot(x="product_group",y="confidence",data=df1)
                    sns.barplot(x="product_group",y="confidence",hue="lift",data=df1);
                    df1.plot.bar()

                Example Conclusion:
                    * 80% of customers who buy MAGGI (Instant soup) buy it in tea.
                    * TEA and MAGGI products increase their sales by 2.17 times mutually.
                    * 66% of customers who buy SUGAR buy it in bread.
                    * 42% of customers who buy COFFEE buy sugar and CORNFLAKES. At the same time, 33% of these sales are in bread.
                
            Hybrid Recommendation System: ( Too Complicated, wont be asked. )

        """

