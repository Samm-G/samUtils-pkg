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
                df = pd.read_csv('bbc-text.csv')

            Regexp Tokenizer:
                reports = df.iloc[0]['text']
                tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
                tokenizer.tokenize(reports)

            Convert to Lowercase:
                low=[]
                for line in data_list:
                    lines = list(map(lambda x : x.lower(),line))
                    low.append(lines) 

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

            Lemmatization:
                import nltk
                from nltk.stem.wordnet import WordNetLemmatizer

                # nltk.download('wordnet')
                lmtzr = WordNetLemmatizer()
                # nltk.download('omw-1.4')
                lemmitized=[]
                for line in filtered_sentence:
                    lines = list(map(lambda x : lmtzr.lemmatize(x),line))
                    lemmitized.append(lines) 
                print(len(lemmitized[0])) 
            
            Print Cleaned Corpus:
                sentence = []
                for row in lemmitized:
                    sequ = ''
                    for word in row:
                        sequ = sequ + ' ' + word
                    sentence.append(sequ)

                corpus = sentence
                print(corpus[1])

            Assign Sentiment using Polarity:
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
        """
        pass

    def st_2_model_build_steps():
        """
        TextBlob Library:

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

        """
        pass

