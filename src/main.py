# import pandas as pd

# from src.preprocessing import clean_text

 

 

# def run_pipeline():

#     df = pd.read_csv(".data/spam.csv", encoding="latin-1")

#     df = df[["v1", "v2"]]

#     df.columns = ["label", "text"]

#     ## Encode the labels

#     df["label"] = df["label"].map({"ham": 0, "spam": 1})

#     # Preprocess the text to get the cleaned version with no URLs, special characters, or stopwords

#     df["clean_text"] = df["text"].apply(clean_text)

import pandas as pd
from src.preprocessing import clean_text
from src.vectorization import tfidf_vectorize
from src.analysis import  get_top_words
from src.word2vec_demo import train_word2vec,demo_similarity

def run_pipeline():
    print("loading dataset....")
    df=pd.read_csv(r"D:\AI_ML\AI PROJECTS\Deep Learning\NLP spam datection\data\spam.csv",encoding="latin-1")
    df=df[["v1","v2"]]
    df.columns=['label','text']

    #Encode labels
    df['label_num']=df['label'].map({'ham':0,'spam':1})

    print("Cleaning text....")
    df['clean_text']=df['text'].apply(clean_text)

    #TF-IDF 
    print("Permorfing TF-IDF...")
    X,vectorizer=tfidf_vectorize(df['clean_text'])

    #Anlysis
    print("\n top Spam Words:")
    spam_words=get_top_words(vectorizer,X,df['label_num'],1)

    for word,score in spam_words[:15]:
        print(f"{word}->{score:.4f}")

    # #Word2vec Demo
    print("\n training Word2vec.....")
    w2v_model=train_word2vec()

    print("\n Similarity Demo:")
    demo_similarity(w2v_model)

    print("Pipeline completed!")


if __name__=="__main__" :
    run_pipeline()




