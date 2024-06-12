#生成数据,需要提前把paper-xml文件夹放在data目录下
python gen_data_2.py

#模型1
python train_1.py  --bert_dir sci_bert --model_name model_1.pt
# #模型2
python train_1.py --seed 2024 --bert_dir sci_bert --model_name model_2.pt

#预测 在relust文件夹生成两个结果文件
python step_2_predict.py --bert_dir sci_bert --model_name model_1.pt --test_file sub_v1.json
python step_2_predict2.py  --bert_dir sci_bert --model_name model_2.pt --test_file sub_v2.json

python gen_data.py
python test_train.py --seed 42 --train_batch_size 128 --lr 3e-5 --model_name model_3.pt
python test_infer.py --test_file sub_v3.json --model_name model_3.pt

#融合
python merge.py
