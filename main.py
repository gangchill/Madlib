
from file_reader import FileReader
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
matplotlib.use('Agg')


class FakeReviewsDetection(object):
    def __init__(self, genuine_file, fake_file):
        # Get True Reviews
            
        self.true_filereader = FileReader(genuine_file, "True") # new object
        self.true_filereader.parse_file()
        self.genuine_reviews = self.true_filereader.get_review_list()
        # print self.genuine_reviews

        # Get Fake Reviews
        self.fake_file_reader = FileReader(fake_file, "Fake") # new object
        self.fake_file_reader.parse_file()
        self.fake_reviews = self.fake_file_reader.get_review_list()

        # Merge both the Reviews
        self.combined_reviews = []
        self.combined_reviews.extend(self.genuine_reviews)
        self.combined_reviews.extend(self.fake_reviews)
        
        '''all_reviews_docs = []
        for f_review in self.combined_reviews:
            all_reviews_docs.append(f_review["review"])
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_reviews_docs)'''
        
        self.Train_Classifier()

    def Train_Classifier(self):
        features = []
        correct_labels = []
        
        # getting features
        
        TruthfullAll = pd.read_csv('TruthfulALLNew.csv')
        Truthful_labels = (np.full((1,800),1)).tolist()
     
        FakeAll = pd.read_csv('FakeALLNew.csv')
        Fake_labels = (np.full((1,800),0)).tolist()
    
        data = TruthfullAll.copy()
        data = data.append(FakeAll,ignore_index = True)
        data = data.astype(float)
        dataMatrix = data.as_matrix()
        
        labels = Truthful_labels[0]
        labels.extend(Fake_labels[0])
        
        corpus = self.get_corpus(self.combined_reviews)
        
        length = 0
        for line in corpus:
            token = line.split()
            length +=len(token)
        
        print "number of terms in corpus : ",length
        

        
        print "creating n-grams....."
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 1, stop_words = 'english')
        
        tfidf_matrix =  tf.fit_transform(corpus)
        print tfidf_matrix.shape[1]
        print "The matrix size before appling SVD:", tfidf_matrix.shape

        #Feature reduction using SVD
        svd = TruncatedSVD(n_components=100, random_state=50)
        U = svd.fit_transform(tfidf_matrix)
        print "The matrix size after appling SVD:", U.shape
         
        print "creating features......"
        count = 0
        
        for review_object in self.combined_reviews:
            # Get review tex
            #print "feat # ", count
            feat = []
            feat.extend(dataMatrix[count, :])
            feat.extend(U[count,:])
            count+=1
            features.append(feat)
        
        
        X = np.matrix(features)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minmax = min_max_scaler.fit_transform(X)
        
        y = np.asarray(labels)
        
        # Selecting important features
        print "Original Features : ", X_minmax.shape
        
        columns = data.columns
        
        self.importantFeatures(X_minmax, y, columns)
        
        X_Train, X_Test, y_Train, y_Test = train_test_split(X_minmax, y, test_size=0.2, random_state=0)
        
        skf = cross_validation.StratifiedKFold(y_Train, 4)
        nb_accuracy = []
        lg_accuracy = []
        svm_accuracy = []
        rd_forest = []
        count = 0
        
        for train,test in skf:
            count += 1
            print "\n\n\nIteration # ", count
            
            X_train, X_test, y_train, y_test = X_Train[train], X_Train[test], y_Train[train], y_Train[test]
            
            
            print "\nRandom Forest Classifier..."
            forest = RandomForestClassifier(n_estimators = 100)
            forest = forest.fit(X_train, y_train)

            y_pred = forest.predict(X_test)

            rd_forest.append(metrics.accuracy_score(y_test, y_pred))
            print("accuracy:", metrics.accuracy_score(y_test, y_pred))
            
            
            print "\nTraining Naive Bayes..."
            clf = MultinomialNB().fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            nb_accuracy.append(metrics.accuracy_score(y_test, y_pred))
            print("accuracy:", metrics.accuracy_score(y_test, y_pred))
            
            print "\nTraining Logistic Regression..."
            log_reg_classifier = LogisticRegression().fit(X_train, y_train)
            y_pred = log_reg_classifier.predict(X_test)
            
            lg_accuracy.append(metrics.accuracy_score(y_test, y_pred))
            print("accuracy:", metrics.accuracy_score(y_test, y_pred))
            
            print "\nTrainig SVM..."
            svm_classifier = SVC().fit(X_train, y_train)
            y_pred = svm_classifier.predict(X_test)
            
            svm_accuracy.append(metrics.accuracy_score(y_test, y_pred))
            print("accuracy:", metrics.accuracy_score(y_test, y_pred))
            
        print "\n\n#####Random Forest#####"
        print "Average Training Accuracy : ",np.array(rd_forest).mean()
        y_pred = forest.predict(X_Test)
        rf_a = metrics.accuracy_score(y_Test, y_pred)
        print "Testing Accuracy : ",rf_a
        print(metrics.classification_report(y_Test, y_pred,
                                    target_names=['Genuine', 'Fake']))
        
        print "\n\n#####Naive Bayes#####"
        print "Average Training Accuracy : ",np.array(nb_accuracy).mean()
        y_pred = clf.predict(X_Test)
        nb_a = metrics.accuracy_score(y_Test, y_pred)
        print "Testing Accuracy : ",nb_a
        print(metrics.classification_report(y_Test, y_pred,
                                    target_names=['Genuine', 'Fake']))
        
        print "\n\n#####Logistic Regression#####"
        print "Average Training Accuracy : ",np.array(lg_accuracy).mean()
        y_pred = log_reg_classifier.predict(X_Test)
        lr_a = metrics.accuracy_score(y_Test, y_pred)
        print "Testing Accuracy : ",lr_a
        print(metrics.classification_report(y_Test, y_pred,
                                    target_names=['Genuine', 'Fake']))
        print '\nConfussion matrix:\n',confusion_matrix(y_Test, y_pred)
        
        print "\n\n#####Support Vector Machine#####"
        print "Average Training Accuracy : ",np.array(svm_accuracy).mean()
        y_pred = svm_classifier.predict(X_Test)
        svm_a = metrics.accuracy_score(y_Test, y_pred)
        print "Testing Accuracy : ",svm_a
        print(metrics.classification_report(y_Test, y_pred,
                                        target_names=['Genuine', 'Fake']))
        
        
        accuracy = []
        accuracy.append(nb_a)
        accuracy.append(svm_a)
        accuracy.append(lr_a)
        accuracy.append(rf_a)
    
    def get_corpus(self, reviews):
        words = []
        corpus = []
        for review in reviews:
            words.append(review["review"])
        
        return words
    
    def importantFeatures(self,X,Y,names):
        #from sklearn.feature_selection import RFE, f_regression
        from sklearn.feature_selection import f_regression
        import operator
            
        f, pval  = f_regression(X, Y, center=True)
        ranks = rank_to_dict(f, names)
        
        print "printing top features according to their importance (rank)...."
        #sorted_ranks = sorted(ranks.items(), key=operator.itemgetter(1), reverse=False)
        sorted_ranks = sorted(ranks.items(), key=operator.itemgetter(1))
        #print "\n\n"
        print sorted_ranks


def rank_to_dict(ranks, names, order=1):
    minmax = preprocessing.MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))

a = FakeReviewsDetection("TruthfulNew.txt","FakeNew.txt")