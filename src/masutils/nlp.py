class NLP:
    def __init__(self):
        """
        Theory:
            

        """
        pass
    

    def st_1_eda_steps():
        """
            Imports:
                import nltk
                from six import string_types
                from nltk.corpus import reuters
                from string import punctuation
                from nltk.corpus import stopwords
                from nltk import word_tokenize
                import numpy as np
                from gensim.parsing.preprocessing import PorterStemmer, remove_stopwords

                nltk.download('stopwords') # Downloading stopwords
                nltk.download('punkt') # Downloading tokenizer

            Read Data:
                review_text = pd.read_csv('NLP_ExamData.csv')
                review_text.head()

            Regexp Tokenizer:
                from nltk.tokenize import RegexpTokenizer
                data_list = list()
                for comp in review:
                    data_list.append(RegexpTokenizer('\w+').tokenize(comp))  # this is same as [a-zA-Z0-9_]
                ##print Example
                print(data_list[:1])

            Convert to Lowercase:
                low=[]
                for line in data_list:
                    lines = list(map(lambda x : x.lower(),line))
                    low.append(lines)
                print(low[0])

            Remove Punctuations:
                puncList = [";",":","!","?","/","\\",",","#","@","$","&",")","(","\""]
                Punc_filtered_sentence = [] 
                for lines in low:
                    punc = []
                    for w in lines: 
                        if w not in puncList: 
                            punc.append(w) 
                    Punc_filtered_sentence.append(punc)
                print(len(low[0])) 
                print(len(Punc_filtered_sentence[0])) 

            Remove Stopwords:
                stop_words = set(stopwords.words('english')) 
                filtered_sentence = [] 
                for lines in Punc_filtered_sentence:
                    word = []
                    for w in lines: 
                        if w not in stop_words: 
                            word.append(w) 
                    filtered_sentence.append(word)
                print(len(Punc_filtered_sentence[0])) 
                print(len(filtered_sentence[0])) 

            Stemming:
                from gensim.parsing.preprocessing import PorterStemmer, remove_stopwords
                stemmer = PorterStemmer()
                document_text = stemmer.stem_documents(document_text)

            Lemmatization:
                from nltk.stem.wordnet import WordNetLemmatizer
                # nltk.download('wordnet') # if required..
                lmtzr = WordNetLemmatizer()
                # nltk.download('omw-1.4')
                lemmatized=[]
                for line in filtered_sentence:
                    lines = list(map(lambda x : lmtzr.lemmatize(x),line))
                    lemmatized.append(lines) 
                print(len(lemmatized[0])) 
                            
            Print Cleaned Corpus:
                sentence = []
                for row in lemmatized:
                    sequ = ''
                    for word in row:
                        sequ = sequ + ' ' + word
                    sentence.append(sequ)
                corpus = sentence
                print(corpus[1])
        """
        pass

    def st_2_model_build_steps():
        """
        Models:

        #document_text/df is the cleaned DataFrame with document in each row.

        Bag Of Words:
            document_tokens = document_text.str.split(' ')
            tokens_all = []
            for tokens in document_tokens:
                tokens_all.extend(tokens)
            print('No. of tokens in entire corpus:', len(tokens_all))
            tokens_freq = pd.Series(tokens_all).value_counts()
            tokens_freq = pd.Series(tokens_all).value_counts().drop([''])
            tokens_freq

            Bar Chart of Bag of Words:
                df_tokens = pd.DataFrame(tokens_freq).reset_index().rename(columns={'index': 'token', 0: 'frequency'})
                df_tokens
                #Remove Stopwords:
                common_stopwords = nltk.corpus.stopwords.words('english')
                custom_stopwords = ['amp', 'rt']
                all_stopwords = np.hstack([common_stopwords, custom_stopwords])
                len(all_stopwords)
                df_tokens = df_tokens[~df_tokens['token'].isin(all_stopwords)]
                df_tokens
                #Plot:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(24,5))
                df_tokens.set_index('token')['frequency'].head(25).plot.bar()
                plt.xticks(fontsize = 20)
        
        Word Clouds:
            from wordcloud import WordCloud
            docs_strings = ' '.join(document_text)
            len(docs_strings)
            wc = WordCloud( collocations=False, background_color='white', stopwords=all_stopwords).generate(docs_strings)
            plt.figure(figsize=(20,5))
            plt.imshow(wc)
            plt.axis('off');

        Compute TF and IDF:
            data = []
            for i in range(len(filtered_sentence)):
                data.extend(filtered_sentence[i])
            vocabulary = set(data)
            word_index = {w: index for index, w in enumerate(vocabulary)}
            VOCABULARY_SIZE = len(vocabulary)
            DOCUMENTS_COUNT = len(data)
            def word_tf(word, token): 
                return float(token.count(word)) / len(token)
            def doc_idf():
                word_doc_count = np.zeros(VOCABULARY_SIZE)
                for word in data : 
                    indexes = [word_index[word]] 
                    word_doc_count[indexes] += 1.0
                print(word_doc_count) 
                return np.log(DOCUMENTS_COUNT / (1 + word_doc_count).astype(float))
            def tf_idf(word, token): 
                if word not in word_index:
                    return .0
                return word_tf(word, token) * doc_idf()[word_index[word]]
            ##Word TF:
            print( word_tf('excellent',data))
            print (word_tf('order',data))
            ##Word IDF:
            word_doc_count = np.zeros(VOCABULARY_SIZE)
            for word in data : 
                indexes = [word_index[word]] 
                word_doc_count[indexes] += 1.0
            print(word_doc_count) 
            word_idf = np.log(DOCUMENTS_COUNT / (1 + word_doc_count).astype(float))

        Count Vectorizer Model:
            #Do Preprocessing:
                #Tokenize
                #Remove Stopwords
                #Stemming/Lemmatization
            from sklearn.model_selection import train_test_split
            from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
            train_docs, test_docs = train_test_split(df['text'], test_size=0.2, random_state=1)
            vectorizer = CountVectorizer(min_df=10).fit(train_docs)
            vocab = vectorizer.get_feature_names_out()
            cv_train_dtm = vectorizer.transform(train_docs)
            cv_test_dtm = vectorizer.transform(test_docs)
            cv_train_y = df.loc[train_docs.index, 'category']
            cv_test_y = df.loc[test_docs.index, 'category']
            cv_train_y.shape
            cv_train_dtm
            ##Visualize the vectorized independent train data
            df_train_dtm = pd.DataFrame(cv_train_dtm.toarray(), index=train_docs.index, columns=vocab)
            df_train_dtm

            Check for Sparsity in Data:
                total_values = 1780 * 5051 #(Rows,Columns in Train_dtm)
                non_zero = 309154 #(Count of stored elements in cv_train_dtm)
                zero_values = total_values - non_zero
                sparsity = zero_values / total_values * 100
                sparsity

            Model Building with SVM:
                from sklearn.svm import SVC
                from sklearn.naive_bayes import MultinomialNB
                cv_naive_bayes_model = MultinomialNB().fit(cv_train_dtm, cv_train_y)
                test_y_pred = cv_naive_bayes_model.predict(cv_test_dtm)
                from sklearn.metrics import accuracy_score, f1_score
                print('Accuracy score: ', accuracy_score(cv_test_y, test_y_pred))
                print('F1 score: ', f1_score(cv_test_y, test_y_pred,average='weighted'))

        TF-IDF Model:
            #Do Preprocessing:
                #Tokenize
                #Remove Stopwords
                #Stemming/Lemmatization
            train_docs, test_docs = train_test_split(pd.Series(df['text']), test_size=0.2, random_state=1)
            vectorizer = TfidfVectorizer(min_df=10).fit(train_docs)
            vocab = vectorizer.get_feature_names_out()
            tf_train_dtm = vectorizer.transform(train_docs)
            tf_test_dtm = vectorizer.transform(test_docs)
            train_y = df.loc[train_docs.index, 'category']
            test_y = df.loc[test_docs.index, 'category']

            Model Building with MultinomialNB:
                from sklearn.svm import SVC
                from sklearn.naive_bayes import MultinomialNB
                tf_naive_bayes_model = MultinomialNB().fit(tf_train_dtm, train_y)
                test_y_pred = tf_naive_bayes_model.predict(tf_test_dtm)
                from sklearn.metrics import accuracy_score, f1_score
                print('Accuracy score: ', accuracy_score(test_y, test_y_pred))
                print('F1 score: ', f1_score(test_y, test_y_pred,average='weighted'))

        Compute Performances of Count Vectorizer and TfidfVectorizer:

            #Change test_y, test_y_pred for both methods and perform these:

            from sklearn.metrics import classification_report
            print(classification_report(test_y, test_y_pred))
            from sklearn.metrics import ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            # Multinomial NB:
            plt.figure(figsize=(15,6))
            ConfusionMatrixDisplay.from_estimator(estimator=cv_naive_bayes_model,X=cv_test_dtm, y=test_y)
            plt.show()
            #Finally,Compare the F1 Scores.


        Sentiment Analysis with TextBlob Library:

            Subjectivity Response:
                from textblob import TextBlob
                def getTextSubjectivity(txt):
                    return TextBlob(txt).sentiment.subjectivity
                review_df['Subjectivity'] = review_df['Review'].transform(lambda x: getTextSubjectivity(str(x)))
                review_df['Subjectivity'] 

            Polarity Response:
                def getTextPolarity(txt):
                    return TextBlob(txt).sentiment.polarity
                review_df['Polarity'] = review_df['Review'].transform(lambda x: getTextPolarity(str(x)))
                review_df['Polarity'] 

            Assigning sentiment to the content using polarity:
                def getTextAnalysis(a):
                    if a < 0:
                        return "Negative"
                    elif a == 0:
                        return "Neutral"
                    else:
                        return "Positive" 
                review_df['Sentiment'] = review_df['Polarity'].apply(getTextAnalysis)
                positive = review_df[review_df['Sentiment'] == 'Positive']
                print(str(positive.shape[0]/(review_df.shape[0])*100) + " % of positive Review")
            
            Plot Freqeuncy Distribution of the Sentiments:
                plt.figure(figsize = (10,5))
                labels = review_df.groupby('Sentiment').count().index.values
                values = review_df.groupby('Sentiment').size().values
                plt.bar(labels, values)

            Make a Sentiment Score Table:
                df = pd.DataFrame([list(df_news['news_category']), sentiment_scores_tb, sentiment_category_tb]).T
                df.columns = ['news_category', 'sentiment_score', 'sentiment_category']
                df['sentiment_score'] = df.sentiment_score.astype('float')
                df.groupby(by=['news_category']).describe()
            
            Catplot for sentiments of different categories.
                fc = sns.catplot(x="news_category", hue="sentiment_category", 
                    data=df, kind="count", 
                    palette={"negative": "#FE2020", 
                             "positive": "#BADD07", 
                             "neutral": "#68BFF5"})

            Heatmap of Sentiments:
                plt.figure(figsize = (5,5))
                conf = pd.DataFrame(confusion_matrix(true_labels, predicted_labels),
                            index = ['negative', 'neutral', 'positive'],
                                columns = ['negative', 'neutral', 'positive'])
                sns.heatmap(conf, annot=True,cmap='viridis')

            Getting Most Positive and Negatie Sentiments:
                pos_idx = df[(df.news_category=='world') & (df.sentiment_score == df[(df.news_category=='world')].sentiment_score.max())].index[0]
                neg_idx = df[(df.news_category=='world') & (df.sentiment_score == df[(df.news_category=='world')].sentiment_score.min())].index[0]
                print('Most Negative world-news Article:', df_news.iloc[neg_idx][['news_article']][0])
                print('Most Positive world-news Article:', df_news.iloc[pos_idx][['news_article']][0])

        Text Generation with RNN:
            #text_data will be a list of all documents.

            Preprocessing:
                tokenizer = Tokenizer()
                tokenizer.fit_on_texts(text_data)
            
            Get Features and Labels:
                features = []
                labels = []
                training_length = 50
                for seq in sequences:
                    # Create multiple training examples from each sequence
                    for i in range(training_length, training_length+300):
                        # Extract the features and label
                        extract = seq[i - training_length: i - training_length + 20]

                        # Set the features and label
                        features.append(extract[:-1])
                        labels.append(extract[-1])
            
            Train and Test Data:
                from sklearn.utils import shuffle
                import numpy as np
                features, labels = shuffle(features, labels, random_state=1)
                train_end = int(0.75 * len(labels))
                train_features = np.array(features[:train_end])
                valid_features = np.array(features[train_end:])
                train_labels = labels[:train_end]
                valid_labels = labels[train_end:]
                X_train, X_valid = np.array(train_features), np.array(valid_features)
                y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
                y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)
                for example_index, word_index in enumerate(train_labels):
                    y_train[example_index, word_index] = 1
                for example_index, word_index in enumerate(valid_labels):
                    y_valid[example_index, word_index] = 1

                Check the Feature and Labels once:
                    for i, sequence in enumerate(X_train[:1]):
                    text = []
                    for idx in sequence:
                        text.append(idx_word[idx])
                    print('Features: ' + ' '.join(text)+'\n')
                    print('Label: ' + idx_word[np.argmax(y_train[i])] + '\n')

            Define ModeL:
                model = Sequential()
                model.add(
                    Embedding(
                        input_dim=num_words,
                        output_dim=100,
                        weights=None,
                        trainable=True))
                model.add(
                    LSTM(
                        64, return_sequences=False, dropout=0.1,
                        recurrent_dropout=0.1))
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(num_words, activation='softmax'))
                model.compile(
                    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                model.summary()
            
            Train Model:
                h = model.fit(X_train, y_train, epochs = 1, batch_size = 100, verbose = 1)

            Evaluate Model:
                print(model.evaluate(X_train, y_train, batch_size = 20))
                print('\nModel Performance: Log Loss and Accuracy on validation data')
                print(model.evaluate(X_valid, y_valid, batch_size = 20))

            Generate Text:
            
                seed_length=50
                new_words=50
                diversity=1
                n_gen=1
                import random

                seq = random.choice(sequences)
                seed_idx = random.randint(0, len(seq) - seed_length - 10)
                end_idx = seed_idx + seed_length

                gen_list = []

                for n in range(n_gen):
                    seed = seq[seed_idx:end_idx]
                    original_sequence = [idx_word[i] for i in seed]
                    generated = seed[:] + ['#']

                    actual = generated[:] + seq[end_idx:end_idx + new_words]

                    for i in range(new_words):
                        preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(np.float64)
                        preds = preds / sum(preds)
                        probas = np.random.multinomial(1, preds, 1)[0]
                        next_idx = np.argmax(probas)
                        seed += [next_idx]
                        generated.append(next_idx)

                    n = []
                    for i in generated:
                        n.append(idx_word.get(i, '< --- >'))
                    gen_list.append(n)

                a = []
                for i in actual:
                    a.append(idx_word.get(i, '< --- >'))
                a = a[seed_length:]
                gen_list = [gen[seed_length:seed_length + len(a)] for gen in gen_list]
                print('Original Sequence: \n'+' '.join(original_sequence))
                print("\n")
                # print(gen_list)
                print('Generated Sequence: \n'+' '.join(gen_list[0][1:]))
                # print(a)

        """
        pass

