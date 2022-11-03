from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_sentence(sentence):
    word_list = word_tokenize(sentence)
    # lemmatized_output = ' '.join([lemmatize_word(w) for w in word_list]) # ALL LEMMATIZATION
    lemmatized_output = ' '.join([lemmatize_noun(w) for w in word_list]) # NOUN LEMMATIZATION (OLD)

    return lemmatized_output

def lemmatize(train_texts, test_texts=None):
    ### Lemmatize Sentences
    lemmatized_texts_train = []
    lemmatized_texts_test  = []
    for text in train_texts:
        lemmatized_texts_train.append(lemmatize_sentence(text))
    if test_texts is not None:
        for text in test_texts:
            lemmatized_texts_test.append(lemmatize_sentence(text))

    return lemmatized_texts_train, lemmatized_texts_test

def lemmatize_word(word):
    lemmatizer = WordNetLemmatizer()
    pos_tag = get_wordnet_pos(word)
    word_lemmatized = lemmatizer.lemmatize(word, pos_tag)

    if pos_tag == "r" or pos_tag == "R":
        try:
            lemmas = wordnet.synset(word+'.r.1').lemmas()
            pertainyms = lemmas[0].pertainyms()
            name = pertainyms[0].name()
            return name
        except Exception:
            return word_lemmatized
    else:
        return word_lemmatized

def lemmatize_noun(word):
    lemmatizer = WordNetLemmatizer()
    word_lemmatized = lemmatizer.lemmatize(word)

    return word_lemmatized

