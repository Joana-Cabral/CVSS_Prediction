# Possible values for variable MODELO: bert / deberta / albert  / roberta / distilbert
ROOTDIR=models/DistilBERT/tok_remstopwords/
MODELO=distilbert
LEMMATIZATION=0
EXTRAVOCAB=0
TOKENFILE=vocab/CVSS_25k.vocab
REMOVESTOPWORDS=0
STEM=0

python infer.py --classes_names NETWORK,LOCAL,PHYSICAL,ADJACENT_NETWORK --label_position 1 --root_dir ${ROOTDIR}AV/_91.35/ --model ${MODELO} --lemmatization ${LEMMATIZATION} --rem_stop_words ${REMOVESTOPWORDS} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python infer.py --classes_names LOW,HIGH --label_position 2 --root_dir ${ROOTDIR}AC/92.68/ --model ${MODELO} --lemmatization ${LEMMATIZATION} --rem_stop_words ${REMOVESTOPWORDS} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python infer.py --classes_names NONE,LOW,HIGH --label_position 3 --root_dir ${ROOTDIR}PR/66.78_lr3e-5_ep5/ --model ${MODELO} --lemmatization ${LEMMATIZATION} --rem_stop_words ${REMOVESTOPWORDS} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python infer.py --classes_names NONE,REQUIRED --label_position 4 --root_dir ${ROOTDIR}UI/_93.26/ --model ${MODELO} --lemmatization ${LEMMATIZATION} --rem_stop_words ${REMOVESTOPWORDS} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python infer.py --classes_names UNCHANGED,CHANGED --label_position 5 --root_dir ${ROOTDIR}S/96.23/ --model ${MODELO} --lemmatization ${LEMMATIZATION} --rem_stop_words ${REMOVESTOPWORDS} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python infer.py --classes_names NONE,LOW,HIGH --label_position 6 --root_dir ${ROOTDIR}C/86.18_lr3e-5_ep5/ --model ${MODELO} --lemmatization ${LEMMATIZATION} --rem_stop_words ${REMOVESTOPWORDS} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python infer.py --classes_names NONE,LOW,HIGH --label_position 7 --root_dir ${ROOTDIR}I/87.34/ --model ${MODELO} --lemmatization ${LEMMATIZATION} --rem_stop_words ${REMOVESTOPWORDS} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python infer.py --classes_names NONE,LOW,HIGH --label_position 8 --root_dir ${ROOTDIR}A/88.98/ --model ${MODELO} --lemmatization ${LEMMATIZATION} --rem_stop_words ${REMOVESTOPWORDS} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
