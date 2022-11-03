from transformers import Trainer, TrainingArguments, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from CVSSDataset import CVSSDataset, read_cvss_csv
from lemmatization import lemmatize, lemmatize_word, lemmatize_noun
from remove_stop_words import remove_stop_words
from stemmatization import stemmatize
import numpy as np
import argparse
import os

def select_tokenizer_model(model_name, extra_tokens, token_file, num_labels):
    global lemmatization

    print("### Selecting Model and Tokenizer")

    if model_name == 'distilbert':
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertConfig
        config = DistilBertConfig.from_pretrained('distilbert-base-cased')
        config.num_labels = num_labels
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
        model = DistilBertForSequenceClassification(config)
    
    elif model_name == 'bert':
        from transformers import BertTokenizerFast, BertForSequenceClassification, BertConfig
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_labels = num_labels
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification(config)

    elif model_name == 'deberta':
        from transformers import DebertaConfig, DebertaTokenizerFast, DebertaForSequenceClassification
        config = DebertaConfig.from_pretrained('microsoft/deberta-base')
        config.num_labels = num_labels
        tokenizer = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base')
        model = DebertaForSequenceClassification(config)

    elif model_name == 'albert':
        from transformers import AlbertConfig, AlbertTokenizerFast, AlbertForSequenceClassification
        config = AlbertConfig.from_pretrained('albert-base-v1')
        config.num_labels = num_labels
        tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v1')
        model = AlbertForSequenceClassification(config)

    elif model_name == 'roberta':
        from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForSequenceClassification
        config = RobertaConfig.from_pretrained('roberta-base')
        config.num_labels = num_labels
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification(config)

    ### Add Tokens
    if extra_tokens:
        add_tokens_from_file(token_file, tokenizer, lemmatization)
    number_tokens = len(tokenizer)

    print("### Number of tokens in Tokenizer")
    print(number_tokens)

    # print("### Configuration")
    # print(model.config)

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

def get_pred_accuracy(target, output):
    output = output.argmax(axis=1) # -> multi label

    tot_right = np.sum(target == output)
    tot = target.size

    return (tot_right/tot) * 100

def get_binary_mean_accuracy(target, output):
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

def get_evaluation_metrics(target, output, num_labels):
    accuracy      = get_pred_accuracy(target, output, num_labels)
    precision     = get_precision(target, output)
    recall        = get_recall(target, output)
    f1_score      = get_f1_score(target, output)

    return accuracy, precision, recall, f1_score

def infer(trainer, test_loader, num_labels):
    predicts   = trainer.predict(test_loader)
    soft       = torch.nn.Softmax(dim=1)
    pred_probs = torch.from_numpy(predicts.predictions)
    pred_probs = soft(pred_probs).numpy()
    gt_list    = predicts.label_ids

    return get_pred_accuracy(gt_list, pred_probs)

def main():
    global lemmatization


    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labels', type=int, required=True, default=2, help='Number of classes in 1 label')
    parser.add_argument('--classes_names', type=str, required=True, help='Names used to distinguish class values')
    parser.add_argument('--label_position', type=int, required=True, help='The label position in CSV file')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model', type=str, help='The name of the model to use')
    parser.add_argument('--train_batch', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Epochs for training')
    parser.add_argument('--lr', type=float, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for training')
    parser.add_argument('--warmup_steps', type=int, help='Warmup steps for training')
    parser.add_argument('--warmup_ratio', type=float, help='Warmup ratio for training')
    parser.add_argument('--extra_tokens', type=int, help='Extra tokens')
    parser.add_argument('--lemmatization', type=int, help='Lemmatization')
    parser.add_argument('--stemming', type=int, help='Stemming')
    parser.add_argument('--rem_stop_words', type=int, help='Remove Stop Words')
    parser.add_argument('--token_file', type=str, help='Tokens file')
    args = parser.parse_args()

    extra_tokens = bool(args.extra_tokens) if args.extra_tokens else False
    token_file   = args.token_file
    lemmatization  = bool(args.lemmatization) if args.lemmatization else False
    stemming  = bool(args.stemming) if args.stemming else False
    rem_stop_words = bool(args.rem_stop_words) if args.rem_stop_words else False

    # Automatic
    list_classes   = args.classes_names.rsplit(",")
    label_position = args.label_position
    output_dir     = args.output_dir
    model_name     = args.model if args.model else 'distilbert'
    num_labels     = args.num_labels

    train_batch_size = args.train_batch if args.train_batch else 8
    test_batch_size  = 4

    epochs        = args.epochs if args.epochs else 3 
    learning_rate = args.lr if args.lr else 5e-5
    weight_decay  = args.weight_decay if args.weight_decay else 0
    warmup_steps  = args.warmup_steps if args.warmup_steps else 0
    warmup_ratio  = args.warmup_ratio if args.warmup_ratio else 0

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("### Device: ",device)

    os.makedirs(output_dir, exist_ok=True)

    ### Select Model
    tokenizer, model = select_tokenizer_model(model_name, extra_tokens, token_file, num_labels)

    ### Split Dataset
    print("### Splitting Dataset")

    train_texts, train_labels = read_cvss_csv('data/train.csv', label_position, list_classes)
    test_texts, test_labels = read_cvss_csv('data/test.csv', label_position, list_classes)


    ### Remove Stop Words from Sentences
    if rem_stop_words:
        print("### Remove Stop Words from Sentences")
        filtered_train, filtered_test = remove_stop_words(train_texts, test_texts)

    
    ### Lemmatize Sentences
    if lemmatization:
        print("### Lemmatizing Sentences")
        if rem_stop_words:
            lemmatized_train, lemmatized_test = lemmatize(filtered_train, filtered_test)
        else:
            lemmatized_train, lemmatized_test = lemmatize(train_texts, test_texts)

    
    ### Stemmatize Sentences
    if stemming:
        print("### Stemmatize Sentences")
        stemmatized_train, stemmatized_test = stemmatize(train_texts, test_texts)
        


    ### Tokenize Sentences
    print("### Tokenizing Sentences")

    if lemmatization:
        train_encodings = tokenizer(lemmatized_train, truncation=True, padding=True) # truncate to the model max length and pad all sentences to the same size
        test_encodings = tokenizer(lemmatized_test, truncation=True, padding=True)
    elif rem_stop_words:
        train_encodings = tokenizer(filtered_train, truncation=True, padding=True) # truncate to the model max length and pad all sentences to the same size
        test_encodings = tokenizer(filtered_test, truncation=True, padding=True)
    elif stemming:
        train_encodings = tokenizer(stemmatized_train, truncation=True, padding=True) # truncate to the model max length and pad all sentences to the same size
        test_encodings = tokenizer(stemmatized_test, truncation=True, padding=True)
    else:
        train_encodings = tokenizer(train_texts, truncation=True, padding=True) # truncate to the model max length and pad all sentences to the same size
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    
    ### Dataset Encodings
    print("### Encoding Dataset")

    train_dataset = CVSSDataset(train_encodings, train_labels)
    test_dataset = CVSSDataset(test_encodings, test_labels)
    

    print("### Training")
    
    training_args = TrainingArguments(
        output_dir=output_dir,                          # output directory
        num_train_epochs=epochs,                        # total # of training epochs
        per_device_train_batch_size=train_batch_size,   # batch size per device during training
        per_device_eval_batch_size=test_batch_size,     # batch size for evaluation
        learning_rate=learning_rate,                    # learning rate
        save_strategy="epoch",
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,           # evaluation dataset
    #    optimizers=(optimizer, scheduler),   # optimizer and scheduler 
    )

    trainer.train()
    trainer.save_model()
    acc = infer(trainer, test_dataset, num_labels)
    print("Accuracy = {:.6f}".format(acc))
    

if __name__ == '__main__':
    main()
