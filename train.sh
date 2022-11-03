LEMMA=0
EXTRAVOCAB=0
TOKENFILE=vocab/CVSS_25k.vocab
REMOVESTOPWORDS=0
STEM=1

python train.py --num_labels 4 --classes_names NETWORK,LOCAL,PHYSICAL,ADJACENT_NETWORK --label_position 1 --output_dir output/attackVector --rem_stop_words ${REMOVESTOPWORDS} --lemmatization ${LEMMA} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python train.py --num_labels 2 --classes_names LOW,HIGH --label_position 2 --output_dir output/attackComplexity --rem_stop_words ${REMOVESTOPWORDS} --lemmatization ${LEMMA} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python train.py --num_labels 3 --classes_names NONE,LOW,HIGH --label_position 3 --output_dir output/privilegeReq --rem_stop_words ${REMOVESTOPWORDS} --lemmatization ${LEMMA} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python train.py --num_labels 2 --classes_names NONE,REQUIRED --label_position 4 --output_dir output/userInteraction --rem_stop_words ${REMOVESTOPWORDS} --lemmatization ${LEMMA} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python train.py --num_labels 2 --classes_names UNCHANGED,CHANGED --label_position 5 --output_dir output/scope --rem_stop_words ${REMOVESTOPWORDS} --lemmatization ${LEMMA} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python train.py --num_labels 3 --classes_names NONE,LOW,HIGH --label_position 6 --output_dir output/confidentiality --rem_stop_words ${REMOVESTOPWORDS} --lemmatization ${LEMMA} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python train.py --num_labels 3 --classes_names NONE,LOW,HIGH --label_position 7 --output_dir output/integrity --rem_stop_words ${REMOVESTOPWORDS} --lemmatization ${LEMMA} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}
python train.py --num_labels 3 --classes_names NONE,LOW,HIGH --label_position 8 --output_dir output/availability --rem_stop_words ${REMOVESTOPWORDS} --lemmatization ${LEMMA} --stemming ${STEM} --extra_tokens ${EXTRAVOCAB} --token_file ${TOKENFILE}