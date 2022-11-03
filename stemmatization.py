from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def stemmatize_sentence(sentence):
    stemmer = PorterStemmer()
    word_list = word_tokenize(sentence)
    stemmatized_ouput = ' '.join([stemmer.stem(w) for w in word_list])
    return stemmatized_ouput

def stemmatize(train_texts, test_texts=None):
    ### Stemmatize Sentences
    stemmatized_texts_train = []
    stemmatized_texts_test  = []
    for text in train_texts:
        stemmatized_texts_train.append(stemmatize_sentence(text))
    if test_texts is not None:
        for text in test_texts:
            stemmatized_texts_test.append(stemmatize_sentence(text))

    return stemmatized_texts_train, stemmatized_texts_test