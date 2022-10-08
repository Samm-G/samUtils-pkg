class NLP:
    def __init__(self):
        """
        Theory:
            ● Pre-Processing in NLP:
                ● Tokenization
                ● Lemmatization and Stemming
            
            ● Feature Extraction:
                ● POS
                ● TF-IDF
                
            ● NER
            
            ● Applications of NLP:
                ● Check Credit worthiness
                ● Language Translation
                ● Sentiment Analysis
                ● Customer Support
                ● Work Routing
                ● Identify Similar Legal cases

            Terminologies:
                ● Corpus: A body of text samples
                ● Document: A text sample
                ● Vocabulary: A list of words used in the corpus
                ● Language model: How the words are supposed to be organized
                
            Challenges in NLP:
                ● Large vocabulary
                ● Multiple meanings
                ● Many word forms
                ● Synonyms
                ● Sarcasm, jokes, idioms, figures of speech
                ● Fluid style and usage
                
            Text Pre-Processing Steps:
            
                ● Tokenization:
                    ● Chopping up text into pieces called tokens
                        ● Split up at all non-alphanumeric characters
                
                ● Stopwords Removal:
                    ● Stopwords:
                        ● Words that are common
                        ● Non-selective (excluding negation)
                        ● Need not be used to classify text

                    ● High frequency words i.e present in most documents
                    ● Can not be used to distinguish between documents
                    ● Hence can be removed as features
                
                ● Normalization:
                    ● Normalization is counting equivalent forms as one term.
                    ● Words appear in many forms:
                        ● School, school, schools
                
                ● Stemming and Lemmatization:
                    ● Stemming:
                        ● chopping off the end of words
                            ● Nannies becomes nanni (Rule: .ies 🡪 .i)
                        ● Converts inflections to root or word stem
                        ● Used for dimensionality reduction
                        ● Word stem may not be present in dictionary
                        ● Popular algorithms include Potter Stemmer, Lovins Stemmer etc
                    ● Lemmatization:
                        ● Finding the lemma of a word is the more exact task
                            ● Nannies should become nanny
                        ● Very similar to Stemming
                        ● Converts inflections to root word or Lemma
                        ● Word stem may not be present in dictionary
                            
                ● Word Vectors: (Convert words to Numbers):
                    
                    ● Bag of Words:
                        ● Chart of Document vs Vocabulary.
                        ● Counts number of words in each document.
                        
                    ● TF-IDF:
                        ● TF: Count of term t in document d.
                            TF = (Freq of word in a doc) / (Total words in the Doc)
                            TF(t,d) = f(t,d) / sigma[t E-> D](f(t',d))
                        ● IDF: 
                            IDF Penalizes terms that occur often in all documents.
                            IDF = log(Num of docs / Num of Docs the word is present)
                            IDF(t,D) = Log( |D| / 1+|{d E-> D, t E-> d}|)
                            
                        ● TF-IDF = tf(t,D)*idf(t,D)
                        
            ● Applications of NLP:
                ● POS Tagging
                    ● Assign grammatical properties (e.g. noun, verb, adverb, adjective etc.) to words. 
                    ● Allows understanding of language structure and syntax.
                    ● These properties can used to extract information by using language rules.
                    ● Multiple NLP libraries support POS tagging e.g. NLTK, spaCy
                                
                    ● Challenges of POS Tagging:
                        ● Ambiguity that needs context
                            ● It is a quick read (NN)
                            ● I like to read (VB)
                        ● Differences in numbers of tags
                            ● Brown has 87 tags
                            ● British National Corpus has 61 tags

                    ● Approaches of POS Tagging:
                        ● Learn from corpora
                        ● Use regular expressions
                        ● Words ending with ‘ed’ or ‘ing’ are likely to be of a certain kind
                        ● Use context
                        ● POS of preceding words and grammar structure
                        ● For example, n-gram approaches
                        ● Map untagged words using an embedding
                        ● Use recurrent neural networks
                
                ● Named Entity Recognition (NER):
                    ● Classifies text into predefined categories or real world entities.
                    ● Used for information extraction, improve search algorithms, content recommendations.
                    
                    ● Challenges of NER:
                        ● Different entities sharing the same name
                        ● Manish Jindal 🡪 Person
                        ● Jindal Steel 🡪 Thing (company)
                        ● Common words that are also names
                        ● Do you want it with curry or dry
                        ● Tyler Curry
                        ● Ambiguity in the order, abbreviation, style
                        ● Jindal, Manish
                        ● Dept. of Electrical Engineering
                        ● De Marzo, DeMarzo
            
                    ● Approaches to NER:
                        ● Match to an NE in a tagged corpus
                            ● Fast, but cannot deal with ambiguities
                        ● Rule based
                            ● E.g. capitalization of first letter
                            ● Does not always work, especially between different types of proper nouns
                        ● Recurrent neural network based
                            ● Learn from a NE tagged corpus

                ● Sentence Parsing:
                    ● Parsing implies finding structure in an input
                        ● That is, there is an order in PoS tags
                        ● We cannot have “Dog cat beautiful rat”
                        ● But, “A dog is more beautiful than a rat” is fine
                    ● We expect inputs to follow some local and some global rules
                    ● Parsing implies finding structure in an input
                        ● That is, there is an order in PoS tags
                        ● We cannot have “Dog cat beautiful rat”
                        ● But, “A dog is more beautiful than a rat” is fine
                    ● These rules set the context for PoS tags
                        ● E.g. “I am walking” vs. “Walking is good”
                        
                ● Dependency Parsing:
                    ● Shows how words in a sentence relate to each other. 
                    ● Allows further understanding of language structure and syntax.
                        
                ● Chunking and Chinking:
                    ● We can use regular expressions to parse sentences with NLTK into chunks and chinks
                    ● Regular expression is a template for searching parts of a sentence
                        ● A sentence has a noun phrase followed by a verb phrase
                    ● So, if we have PoS tags correct, we can parse the sentences
                    
                ● Chunking vs Chinking:
                
                    ● Chunking is used to find chunks
                        ● Noun phrase complete set of rules (for example):
                            ● An optional determinant (article)
                            ● An optional adverb
                            ● Followed by an optional gerund verb (ending in ‘ing’)
                            ● Followed by a mandatory noun or a pronoun
                        ● E.g., “… is good for health”
                        
                    ● Chinking is used to code exceptions to chunking rules that should not be chunked
                        ● E.g., verb phrase
                        ● Look at the rules
                    ● Except, when the a gerund verb (actually, a noun, e.g. “walking”) is 
                        followed by a regular verb.
                    ● E.g., “… is good for health”
                    
                    Rules of Chunking and Chinking:
                        ● Define a noun phrase
                            ● Which starts with an optional determinant
                            ● Has an optional adverb
                            ● Has an optional verb gerund
                            ● Ends with a mandatory noun/pronoun
                        ● Define a verb phrase
                            ● Starts with a mandatory verb
                            ● Can have any PoS else trailing it
                            ● Except when it starts with a gerund verb
                    
                ● Sentiment analysis
                    ● Is a given product review positive or negative?
                    ● Which are the most significant reviews?
                ● Text generation
                    ● Question answering, e.g. chatbots
                    ● Language translation, e.g. English to Kannada

                    
            ● Building Blocks of NLP:
                ● BOW:
                    ● Frequency Table/Bar Plot
                    ● Summary Statistics
                ● Numerical Variable:
                    ● Histogram, Violin and Boxplot
                    ● Summary Statistics
                
            ● Bag Of Words:
                ● A simple feature extraction approach in NLP
                ● Ignores grammar / structure
                ● Represents each document by measuring presence of vocabulary words

                ● One Hot Encoding:
                    ● Table of Vocabulary vs Vocabulary (A Singular Matrix)
                    ● Assign index for each word in vocabulary.
                    
                    ● Disadvantages of One-Hot Encoding:
                        ● In one-hot-encoding, the order of the words in a sentence does not get considered.
                        ● The context of a sentence gets missed out.
                ● Document as Matrix:
                    ● Table of (New Document in order of sentences) vs Vocabulary
                    
                ● Document Similarity:
                    ● Features of Document Similarity:
                        ● Histograms considers four main aspects of data viz., shape, center, spread, and outliers
                        ● Shape can be symmetric, skewed, or have multiple peaks
                        ● Center refers to the mean or median
                        ● Spread refers to the variability of the data

                    ● Struge's Rule to find optimum Bin Size:
                        ● Sturge’s rule is one of the methods to choose the optimum bin size for a histogram. 
                        ● This method is useful if the dataset is symmetric
                        ● The rule is given as:
                            K = 1 + 3.322*log(N)
                                Where,
                                K = Number of bins
                                N = Number of observations in the dataset
                                
            ● Context Bag of Words: (CBOW):
                ● Lets consider an example : 
                    ● It was a noisy Deer into the woods.
                ● Now focus on the word `Deer`. 
                    ● It was a noisy Deer into the woods.
                ● In continuous bag-of-words(CBOW) , we try to predict a word given its surrounding context (e.g location ± 2)
                ● (A 🡪 Deer),(noisy,Deer),(into,Deer),(the,Deer).
            
            ● Skip Gram Model:
                ● We try to model the contextual words (e.g location ± 2) given a particular word.
                ● Considering the previous example , It was a noisy Deer into the woods. ■ Now focus on the word `Deer`. It was a noisy Deer into the woods. 
                ● Considering the previous example , 
                    ● It was a noisy Deer into the woods.
                ● Now focus on the word `Deer`. 
                    ● It was a noisy Deer into the woods.
                ● (Deer 🡪 a) , (Deer 🡪 noisy) , (Deer 🡪 into) , (Deer 🡪 the) 
                
            ● Building Skip-Gram Neural Network:
                ● Steps:
                    ● Create Input Vector: 
                        ● Size same as Vocabulary Size (10000)
                    ● Create Hidden Layer: 
                        ● Maybe 50 neurons
                    ● Create Hidden Layer Outputs: 
                        ● 50, same as number of hidden layers.
                    ● Create Output Layer: 
                        ● Same as Vocabulary Size (10000)
                        
                    ● There are too Many Predictions:
                        ● We Handle it with Negative Sampling.
                    ● After Training:
                        ● Output Layer is Discarded
                        ● For each Word in vocabulary, we get... 50 numbers 
                        ● 50 is the Embedding Size
                        
                ● Negative Sampling:
                    ● Only a few weights are updated
                    ● Weights corresponding to Positive outputs (Window Size)
                    ● Very small number of weights for Negative output:
                        ● 5-20 for small datasets
                        ● 2-5 for Large dataset
                        
            ● Word2Vec:
                ● Training Steps:
                    ● The objective is to maximize the probability of actual skip-gram, while minimizing the probability of no-existent skip-gram.
                    ● We try to find the probability of a presence of a particular word with a particular contextual word with a corpus, 
                        arg maxθ π(w, c ϵ D) p(D = 1 | w, c ; θ) 
                            ● Where , w is a particular word
                            ● C is a contextual word
                            ● θ is the model parameter
                    ● We try to find the probability of a absence of a particular word with a particular contextual word with a corpus, 
                        arg maxθ π(w′, c′ ϵ D′ ) p(D= 0 | w′, c′ ; θ) …. ( 2 )
                            ● Where , w′ is a particular word
                            ● c′ is a contextual word
                            ● θ is the model parameter 

                ● Build Word2Vec Model Steps:
                    ● Load Movie Reviews Data
                    ● Convert Text to Numbers (Keras Tokenizer)
                    ● Make all reviews of equal size
                    ● Build an array with Embeddings from pre-trained Word2Vec Models.
                    ● Build Model using Embedding Layers
                    ● Train Model.
                    
                ● Keras Embedding Layer 
                    ● Input_dim → Possible Input values (vocabulary length) 
                    ● Output_dim → How many numbers for each Input value 
                    ● Input_length → How many input numbers in each Example ( the length of the sentence we pass) 
                    ● Weights → Pre-trained Embeddings, if any.
                    
            ● Global Vectors (GloVe): 
                ● GloVe captures wird-word co-occurances in the entire corpus.
                ● Glove Models is: F((wi-wj)T wk) = Pik / jk
                
                ● Cost Function: J = Sigma(f(Xij)(wiTw^j - log(Xij)))^2
                    For words i,j cooccurence probability is Xij
                ● A weighted function F suppresses rare cooccurences.
                
                
            ● Web Scraping:
            
                ● Use cases of web scraping
                    ● Price monitoring
                    ● Price intelligence
                    ● News monitoring 
                    ● Lead generation and 
                    ● Market research.
                
                ● Python Packages for Web Scraping:
                    ● Beautiful Soup -
                        ● For Stock market price – gets update regularly
                    ● Scrapy –
                        ● Less for Web Scrapping - more for building web spider for web crawler
                    ● XTML
                    ● Specific API packages
                        ● newsapi
                        ● tweepy

                ● Steps in Web Scraping:
                    ● Define Task
                    ● Inspect elements
                    ● Look at Element structure
                    ● Approach:
                        ● Step 1 : Extract the web page HTML content & convert it to text to view the Elements of the HTML page
                        ● Step 2 : Convert the HTML content into XML object
                        ● Step 3 : `prettify` helps to have a look proper intendent look of the XML page
                        ● Step 4 : `title` helps to extract the title of the web page 
                        ● Step 5 : `string` helps to convert the tags into string
                        ● Step 6 : We can fetch content from first mention of any tags by having the tag extension with the XML content
                        ● Step 7 : `find_all` helps to fetch content from all the mentioned of any tags by having the tag extension with the XML content
                        ● Step 8 : We can also fetch information for a tag with a specific class type
                        ● Step 9 : Fetch and arrange the information following the above steps.
                
                ● Retrieving Data Through API:
                    ● Step 1 : `Install and Import the appropriate API package
                        Here we are using NewsApi
                    ● Step 2 : Initialize the API by providing access to the confidential API key
                        We would be creating an account in NewsApi. (https://newsapi.org/ )
                    ● Step 3 : We can fetch all the Headline in two ways
                        ● Step 3.1 : We can fetch all the Headline using a particular URL 
                        ● Step 3.2 : We can fetch all the Headline using a particular Headline parameters Approach
                    ● Step 4 : We can fetch the entire Body of an article or `everything` about an article in two ways
                        ● Step 4.1 : We can fetch the entire Body of an article or `everything` about an article using a particular URL 
                        ● Step 4.2 : We can the entire Body of an article or `everything` about an article using a particular Headline parameters
                    ● Step 5 : We can fetch the source of an article in two ways.
                        ● Step 5.1 : We can fetch the source of an article using a particular URL 
                        ● Step 5.2 : We can fetch the source of an article using a particular Headline parameters

                ● Sentiment Analysis of Web Scraped Data:
                    ● Step 1 : Convert the available information in a dataset.
                    ● Step 2 : Clean the data
                    ● Step 3 : Assign the subjectivity 
                        Return the subjectivity score as a float within the range [0.0, 1.0] where 0.0 is very objective 
                        and 1.0 is very subjective.
                    ● Step 4 : Assign the Polarity
                        Return the polarity score as a float within the range [-1.0, 1.0]
                    ● Step 5 : Visualize the distribution of the sentiment over the entire content
                    
            ● Issues using ANN for sequence problems:
                ● No Fixed size for neurons in a layer
                ● Too much computation
                ● No parameter sharing
                
            ● NLP Using RNN:
                
                ● Sequential Data:
                    ● One-dimensional discrete index
                        ● Example: time instances, character position
                    ● Each data point can be a scalar, vector, or a symbol from an alphabet
                    Ex: Speech, Text (NLP), Music, Protein and DNA sequences, etc
                    
                ● Traditional DL vs RNN:
                    ● Trasitional DL:
                        ● Cannot take past data in Need for past data or context
                        ● Work with a fixed window
                    ● RNN:
                        ● A memory state is computed in addition to an output, which is sent to the next time instance
                        ● The order of the data is accounted for.
                        ● Types of analysis possible on sequential data using “recurrence”:
                            ● One to one:
                                Ex: POS Tagging, Stock Trading
                            ● One to many:
                                Ex: Generate text given topic
                                Ex: Generate caption based on an image
                            ● Many to one:
                                Ex: Sentiment Analysis
                            ● Many to many:
                                Ex: Language translation
                                
            ● LSTM RNN (Long Short Term Memory) :
                ● Introducing a forget gate to control the gradient
                ● Adding input and output gates
                
                ● LSTM Few Words:
                    ● CEC: With the forget gate, influence of the state forward can be modulated such that it can be remembered for a long time, until the state or the input changes to make LSTM forget it. 
                    ● This ability or the path to pass the past-state unaltered to the future-state (and the gradient backward) is called constant error carrousel (CEC). 
                    ● It gives LSTM the ability to remember long term (hence, long short term memory)
                    ● Blocks: Since there are just too many weights to be learnt for a single state bit, several state bits can be combined into a single block such that the state bits in a block share gates
                    ● Peepholes: The state itself can be an input for the gate using peephole connections
                    ● GRU: In a variant of LSTM called gated recurrent unit (GRU), input gate can simply be one-minus-forget-gate. 
                    ● hat is, if the state is being forgotten, then replace it by input, and if it is being remembered, then block the input

            ● Applications of LSTMs:
                ● Pre-processing for NLP:
                    ● Alternative to converting words into an embedding using Word2Vec or GloVe.
                    ● One-hot-bit input vector can also be too long and sparse, and require lots on input weights.
                ● Sentiment analysis:
                    ● Very common for customer review or new article analysis
                    ● Output before the end can be discarded (not used for backpropagation)
                    ● This is a many-to-one task
                ● Sentence generation:
                    ● Very common for image captioning
                    ● Input is given only in the beginning
                    ● This is a one-to-many task
                ● Pre-training LSTMs:
                    ● Learning to predict the next word can imprint powerful language models in LSTMs
                    ● This captures the grammar and syntax
                ● Machine translation:
                    ● A naïve model would be to use a many-to- many network and directly train it
                    
            ● Advanced LSTM Structures:
                ● Multi-layer LSTM:
                    ● More than one hidden layer can be used
                ● Bi-directional LSTM:
                    ● Many problems require a reverse flow of information as well
                    ● For example, POS tagging may require context from future words
                ● LSTM with Attention Mechanism:
                    ● LSTM gets confused between the words and sometimes can predict the wrong word.
                    ● The encoder step needs to search for the most relevant information, this idea is called 'Attention'.
                
            ● Problems with LSTM:
                ● Inappropriate model
                    ● Identify the problem: One-to-many, many-to-one etc.
                    ● Loss only for outputs that matter
                    ● Separate LSTMs for separate languages
                ● High training loss
                    ● Model not expressive
                    ● Too few hidden nodes
                    ● Only one hidden layer
                ● Overfitting
                    ● Model has too much freedom
                    ● Too many hidden nodes
                    ● Too many blocks
                    ● Too many layers
                    ● Not bi-directional


            ● Advanced Language Models:
                ● Attenstion between Encoder and Decoder
                ● No vs Global vs Local Attention
                ● Transformer Networks
                ● Attention in Transformer Networks
                ● BERT
                ● XLNet
                ● DistilBERT (Current Gen).
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

