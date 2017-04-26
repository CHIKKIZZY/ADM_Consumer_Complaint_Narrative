import numpy as np
import string
import pandas as pd
import math
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn_pandas import DataFrameMapper, cross_val_score
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import csr_matrix

def lsa(data, pipeline=None, dimensions=200):
    vectorized =[]
    stemmer = SnowballStemmer("english")#Remove stemming words
    table = str.maketrans({key: None for key in string.punctuation})#removing punctuations
    stemmed = data['Consumer complaint narrative']
    stemmed = stemmed.map(lambda x: ' '.join([stemmer.stem(y.lower().strip('xx')) for y in x.translate(table).split(' ')]))
    if not pipeline:
        vectorizer = TfidfVectorizer(min_df=0.0025, max_df=0.1, stop_words='english', ngram_range=(1,2), sublinear_tf=True, use_idf=True)
        svd = TruncatedSVD(dimensions)
        normalizer = Normalizer(copy=False)
        pipeline = make_pipeline(vectorizer, svd, normalizer).fit(stemmed)
    vectorized= pipeline.transform(stemmed)
    df=pd.DataFrame()
    df=df.append({'features':vectorized},ignore_index=True)
    data = data.reset_index()
    data = pd.concat([data, df], axis=1)
    data= data.dropna()
    return (pipeline, data)


def train_test_split(data):
    print("test")
    test = set(range(len(data))[::10])
    train = sorted(set(range(len(data))) - test)
    test = sorted(test)
    print("test out")
    return data.iloc[train], data.iloc[test]

def cosine_sim(a, b):
    
    x = (a.data * a.data)
    X = math.sqrt(x.sum())    
    y = (b.data * b.data)
    Y = math.sqrt(y.sum())	
    dotProduct = (a).dot(b.transpose())

    Sum_dot = dotProduct.sum()
    if (X != 0) | (Y !=0) :
        Cos_Sim = Sum_dot / ( X * Y )
    else :
        Cos_Sim = 0

    return(Cos_Sim)
    
   
	
def make_predictions(data, data_train, data_test):

    class1 = 0
    class2 = 0
    class3 = 0   
    class4 = 0
    class5 = 0
    class6 = 0   
    class7 = 0   
    class8 = 0
   
    result = [] 
    wrong_pred = 0
    
    for index, row in data_test.iterrows():
        Complain_ID = row['Complaint ID']
        product = row['Product']
        issue = row['Issue']      
        responce_Topredict = data[(data['Complaint ID'] == Complain_ID )]
        a =  responce_Topredict['features'].values[0]
        #print(a.data)
        trains_data = data_train[ (data_train['Complaint ID'] != Complain_ID) & ((data_train.Product == product) | (data_train.Issue == issue))]
        update_cos_sim = 0
        higest_cos_match = 0
        
        for index1, row1 in trains_data.iterrows():          
                data_Hist = data[(data['Complaint ID'] == row1['Complaint ID'])]
                b =  data_Hist['features'].values[0]
                print(data_Hist)
                cos_sim = cosine_sim(csr_matrix(a),csr_matrix(b))
                if (cos_sim > 0) & (update_cos_sim < cos_sim ):
                        update_cos_sim = cos_sim   
                        higest_cos_match = row1['Complaint ID']
                        
        
        #print(len(row['Company response to consumer'])	,len(row1['Company response to consumer']))
        result.append((higest_cos_match,update_cos_sim))
        
        
        for index2, row2 in trains_data.iterrows():
            responce = ''
            if higest_cos_match == row2['Complaint ID'] :
                responce =  row2['Company response to consumer']

                    
                if row['Company response to consumer'] != responce :
                    wrong_pred = wrong_pred + 1
                
                if row['Company response to consumer'] == responce :
                    if   'Closed with explanation'.lower()   in row['Company response to consumer'].lower() :
                        class1 = class1 + 1;
                    if  'Closed with non-monetary relief'.lower()   in row['Company response to consumer'].lower() :
                        class2 = class2 + 1;
                    if  'Closed with monetary relief'.lower()   in row['Company response to consumer'].lower() :
                        class3 = class3 + 1;
                    if  'Untimely response'.lower()  in row['Company response to consumer'].lower() :
                        class4 = class4 + 1;
                    if  'Closed without relief'.lower()   in row['Company response to consumer'].lower() :
                        class5 = class5 + 1; 
                    if  'Closed with relief'.lower()   in row['Company response to consumer'].lower() :
                        class6 = class6 + 1;
                    if  'In progress'.lower()   in row['Company response to consumer'].lower() :
                        class7 = class7 + 1;
                    if  'Closed'.lower() == row['Company response to consumer'].lower() :
                        class8 = class8 + 1;

    
    print("*** Test Data Classification ***")                
    print('Closed with explanation = ',class1)
    print('Closed with non-monetary relief = ',class2)
    print('Closed with monetary relief = ',class3)
    print('Untimely response = ',class4)
    print('Closed without relief = ',class5)
    print('Closed with relief = ',class6)
    print('In progress = ',class7)  
    print('Closed = ',class8)
    
    class12 = 0
    class22 = 0
    class32 = 0   
    class42 = 0
    class52 = 0
    class62 = 0   
    class72 = 0   
    class82 = 0
    
    
    print("*** Test Data Classification Accuracy ***")   
    for index3, row3 in data_test.iterrows():
        if   'Closed with explanation'.lower()   in row3['Company response to consumer'].lower() :
             class12 = class12 + 1;
        if  'Closed with non-monetary relief'.lower()   in row3['Company response to consumer'].lower() :
             class22 = class22 + 1;
        if  'Closed with monetary relief'.lower()   in row3['Company response to consumer'].lower() :
             class32 = class32 + 1;
        if  'Untimely response'.lower()  in row3['Company response to consumer'].lower() :
             class42 = class42 + 1;
        if  'Closed without relief'.lower()   in row3['Company response to consumer'].lower() :
             class52 = class52 + 1; 
        if  'Closed with relief'.lower()   in row3['Company response to consumer'].lower() :
             class62 = class62 + 1;
        if  'In progress'.lower()   in row3['Company response to consumer'].lower() :
             class72 = class72 + 1;
        if  'Closed'.lower() == row3['Company response to consumer'].lower() :
             class82 = class82 + 1;
     
    print('Wrong Predictions -Closed with explanation = ',class12-class1)
   
    if class12 != 0 :
        print('Error % Closed with explanation = ',((class12-class1)/class12)*100 )
    else :
        print('Error % Closed with explanation = 0')

    print('Wrong Predictions -Closed with non-monetary relief = ',class22-class2)
        
    if class22 != 0 :
        print('Error % Closed with non-monetary relief = ',((class22-class2)/class22)*100 )
    else :
        print('Error % Closed with monetary relief = 0')

    print('Wrong Predictions -Closed with monetary relief = ',class32-class3)
        
    if class32 != 0 :
        print('Error % Closed with monetary relief = ',((class32-class3)/class32)*100 )
    else :
        print('Error % Closed with monetary relief = 0')

    print('Wrong Predictions -Untimely response = ',class42-class4)
       
    if class42 != 0 :
        print('Error % Untimely response = ',((class42-class4)/class42)*100 )
    else :
        print('Error % Closed with monetary relief = 0')

    print('Wrong Predictions- Closed without relief = ',class52-class5)
      
    if class52 != 0 :
        print('Error % Closed without relief = ',((class52-class5)/class52)*100 )
    else :
        print('Error % Closed with monetary relief = 0')

    print('Wrong Predictions -Closed with relief = ',class62-class6)
        
    if class62 != 0 :    
        print('Error % Closed with relief = ',((class62-class6)/class62)*100 )
    else :
        print('Error % Closed with monetary relief = 0')

    print('Wrong Predictions -In progress = ',class72-class7)  
       
    if class72 != 0 :        
        print('Error % In progress = ',((class72-class7) /class72)*100 ) 
    else :
        print('Error % Closed with monetary relief = 0')

    print('Wrong Predictions -Closed = ',class82-class8)
       
    if class82 != 0 :
        print('Error % Closed = ',((class82-class8)/class82)*100 )    
    else :
        print('Error % Closed with monetary relief = 0')
    
    
    return(result,wrong_pred)    
    pass


def mean_error(wrong_pred, data_test):
    error = 0.0
    error = (wrong_pred / len(data_test) ) * 100
    return error
	
def main():

    # Read the input
    d = pd.read_csv("D:\Subjects\Adv Data Mining - CS522\Project\Consumer_Complaints_new.csv") # the consumer dataset is now a Pandas DataFrame
    #d = d[1:500]
    d= d[d["Consumer complaint narrative"]!='null']
    pipeline, training = lsa(d)
    data_train, data_test = train_test_split(training)
    print(training.columns.values)
    predictions,wrong_pred = make_predictions(training, data_train, data_test)
    
	
if __name__ == '__main__':
    main()
