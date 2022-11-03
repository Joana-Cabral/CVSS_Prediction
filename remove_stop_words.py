from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stop_words_from_sentence(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_ouput = ' '.join([w for w in word_tokens if not w in stop_words])
    return filtered_ouput

def remove_stop_words(train_texts, test_texts=None):
    ### Remove stop words from sentences
    filtered_texts_train = []
    filtered_texts_test  = []
    for text in train_texts:
        filtered_texts_train.append(remove_stop_words_from_sentence(text))
    if test_texts is not None:
        for text in test_texts:
            filtered_texts_test.append(remove_stop_words_from_sentence(text))

    return filtered_texts_train, filtered_texts_test