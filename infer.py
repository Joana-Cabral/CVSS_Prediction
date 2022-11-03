from pathlib import Path
from nltk import tokenize
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from CVSSDataset import CVSSDataset, read_cvss_csv, read_cvss_txt
import numpy as np
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score
from lemmatization import lemmatize, lemmatize_noun
from remove_stop_words import remove_stop_words
from stemmatization import stemmatize

# -------------------------------------- MODEL -------------------------------------

def load_model(model_path, model):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    return model

def select_tokenizer_model(model_name, extra_tokens, token_file, model_path, config_path):
    global lemmatization
    
    if model_name == 'distilbert':
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertConfig
        config = DistilBertConfig.from_pretrained(config_path)
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
        model = DistilBertForSequenceClassification(config)
    
    elif model_name == 'bert':
        from transformers import BertTokenizerFast, BertForSequenceClassification, BertConfig
        config = BertConfig.from_pretrained(config_path)
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification(config)

    elif model_name == 'deberta':
        from transformers import DebertaConfig, DebertaTokenizerFast, DebertaForSequenceClassification
        config = DebertaConfig.from_pretrained(config_path)
        tokenizer = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base')
        model = DebertaForSequenceClassification(config)

    elif model_name == 'albert':
        from transformers import AlbertConfig, AlbertTokenizerFast, AlbertForSequenceClassification
        config = AlbertConfig.from_pretrained(config_path)
        tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v1')
        model = AlbertForSequenceClassification(config)

    elif model_name == 'roberta':
        from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForSequenceClassification
        config = RobertaConfig.from_pretrained(config_path)
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification(config)
        
    ### Add Tokens
    if extra_tokens:
        add_tokens_from_file(token_file, tokenizer, lemmatization)
    number_tokens = len(tokenizer)

    print("### Number of tokens in Tokenizer")
    print(number_tokens)

    model.resize_token_embeddings(number_tokens) 

    return tokenizer, model

def add_tokens_from_file(token_file, tokenizer, lemmatize=False):
    print("### Adding Tokens")
    
    file_      = open(token_file, 'r')
    token_list = []
    
    for line in file_:
        if lemmatize:
            token_list.append(lemmatize_noun(line.rstrip("\n")))
        else:
            token_list.append(line.rstrip("\n"))
    file_.close()
    tokenizer.add_tokens(token_list)

# -------------------------------------- METRICS -----------------------------------

def get_pred_accuracy(target, output):
    output = output.argmax(axis=1) # -> multi label

    tot_right = np.sum(target == output)
    tot = target.size

    return (tot_right/tot) * 100

def get_accuracy_score(target, output):
    return accuracy_score(target, output)

def get_f1_score(target, output):
    return f1_score(target, output, average='weighted')

def get_precision_score(target, output):
    return precision_score(target, output, average='weighted')

def get_recall_score(target, output):
    return recall_score(target, output, average='weighted')

def get_mean_accuracy(target, output):
    eps = 1e-20
    output = output.argmax(axis=1)

    # TP + FN
    gt_pos = np.sum((target == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((target == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((target == 1) * (output == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((target == 0) * (output == 0), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    
    # mean accuracy
    return (label_pos_recall + label_neg_recall) / 2

def get_balanced_accuracy(target, output):
    return balanced_accuracy_score(target, output)

# -------------------------------------- MAIN -----------------------------------

def main():
    global lemmatization

    parser = argparse.ArgumentParser()
    parser.add_argument('--classes_names', type=str, required=True, help='Names used to distinguish class values')
    parser.add_argument('--label_position', type=int, required=True, help='The label position in CSV file')
    parser.add_argument('--root_dir', type=str, required=True, help='Path to model and config files')
    parser.add_argument('--model', type=str, help='The name of the model to use')
    parser.add_argument('--test_batch', type=int, help='Batch size for test')
    parser.add_argument('--extra_tokens', type=int, help='Extra tokens')
    parser.add_argument('--lemmatization', type=int, help='Lemmatization')
    parser.add_argument('--stemming', type=int, help='Stemming')
    parser.add_argument('--rem_stop_words', type=int, help='Remove Stop Words')
    parser.add_argument('--token_file', type=str, help='Tokens file')
    args = parser.parse_args()

    model_name  = args.model if args.model else 'distilbert'
    extra_tokens = bool(args.extra_tokens) if args.extra_tokens else False
    token_file   = args.token_file
    lemmatization  = bool(args.lemmatization) if args.lemmatization else False
    stemming  = bool(args.stemming) if args.stemming else False
    rem_stop_words = bool(args.rem_stop_words) if args.rem_stop_words else False

    root_dir    = args.root_dir
    model_path  = root_dir + 'pytorch_model.bin'
    config_path = root_dir + 'config.json'
    
    batch_size     = args.test_batch if args.test_batch else 2
    list_classes   = args.classes_names.rsplit(",")
    label_position = args.label_position

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("### Device: ",device)

    ### Select Model
    tokenizer, model = select_tokenizer_model(model_name, extra_tokens, token_file, model_path, config_path)

    ### Load Dataset
    print("### Loading Dataset")
    
    test_texts, test_labels = read_cvss_csv('data/test.csv', label_position, list_classes)


    ### Lemmatize Sentences
    if lemmatization:
        print("### Lemmatizing Sentences")
        lemmatized_test, _ = lemmatize(test_texts)

    if stemming:
        print("### Stemmatize Sentences")
        stemmatized_test, _ = stemmatize(test_texts)

    if rem_stop_words:
        print("### Remove Stop Words from Sentences")
        filtered_test, _ = remove_stop_words(test_texts)


    ### Tokenize Sentences
    print("### Tokenizing Sentences")

    if lemmatization:
        test_encodings = tokenizer(lemmatized_test, truncation=True, padding=True)
    elif stemming:
        test_encodings = tokenizer(stemmatized_test, truncation=True, padding=True)
    elif rem_stop_words:
        test_encodings = tokenizer(filtered_test, truncation=True, padding=True)
    else:
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    ### Dataset Encodings
    test_dataset = CVSSDataset(test_encodings, test_labels)

    print("### Dataset Encodings")

    model = load_model(model_path, model)
    model.to(device)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    pred_probs = []
    gt_list = []

    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        soft = torch.nn.Softmax(dim=1)
        output_logits = soft(outputs.logits)
        
        gt_list.append(labels.cpu().detach().numpy())
        pred_probs.append(output_logits.cpu().detach().numpy())

    gt_list = np.concatenate(gt_list, axis=0)
    pred_probs = np.concatenate(pred_probs, axis=0)
    pred_probs = pred_probs.argmax(axis=1)


    print("Accuracy = {:.6f}   F1-score = {:.6f}   Precision = {:.6f}   Recall = {:.6f}   mean Accuracy = {:.6f}".format(get_accuracy_score(gt_list, pred_probs), get_f1_score(gt_list, pred_probs), get_precision_score(gt_list, pred_probs), get_recall_score(gt_list, pred_probs), balanced_accuracy_score(gt_list, pred_probs)))

if __name__ == '__main__':
    main()
