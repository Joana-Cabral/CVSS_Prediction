# Possible values for variable MODELO: bert / deberta / albert / roberta / distilbert
MODELO=bert
BATCH=4
EPOCHS=3
LR=5e-5
WD=0

python train.py --num_labels 4 --classes_names NETWORK,LOCAL,PHYSICAL,ADJACENT_NETWORK --label_position 1 --output_dir output/attackVector --model ${MODELO} --train_batch ${BATCH} --epochs ${EPOCHS} --lr ${LR} 
python train.py --num_labels 2 --classes_names LOW,HIGH --label_position 2 --output_dir output/attackComplexity --model ${MODELO} --train_batch ${BATCH} --epochs ${EPOCHS} --lr ${LR} --weight_decay ${WD}
python train.py --num_labels 3 --classes_names NONE,LOW,HIGH --label_position 3 --output_dir output/privilegeReq --model ${MODELO} --train_batch ${BATCH} --epochs ${EPOCHS} --lr ${LR} --weight_decay ${WD}
python train.py --num_labels 2 --classes_names NONE,REQUIRED --label_position 4 --output_dir output/userInteraction --model ${MODELO} --train_batch ${BATCH} --epochs ${EPOCHS} --lr ${LR} --weight_decay ${WD}
python train.py --num_labels 2 --classes_names UNCHANGED,CHANGED --label_position 5 --output_dir output/scope --model ${MODELO} --train_batch ${BATCH} --epochs ${EPOCHS} --lr ${LR} --weight_decay ${WD}
python train.py --num_labels 3 --classes_names NONE,LOW,HIGH --label_position 6 --output_dir output/confidentiality --model ${MODELO} --train_batch ${BATCH} --epochs ${EPOCHS} --lr ${LR} --weight_decay ${WD}
python train.py --num_labels 3 --classes_names NONE,LOW,HIGH --label_position 7 --output_dir output/integrity --model ${MODELO} --train_batch ${BATCH} --epochs ${EPOCHS} --lr ${LR} --weight_decay ${WD}
python train.py --num_labels 3 --classes_names NONE,LOW,HIGH --label_position 8 --output_dir output/availability --model ${MODELO} --train_batch ${BATCH} --epochs ${EPOCHS} --lr ${LR} --weight_decay ${WD}
