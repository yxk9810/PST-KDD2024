#将微调好的权重(model_1.pt、model_2.pt、model_3.pt)放入model_out文件加中，执行下列脚本
python step_2_predict.py --bert_dir sci_bert --model_name model_1.pt --test_file sub_v1.json
python step_2_predict2.py  --bert_dir sci_bert --model_name model_2.pt --test_file sub_v2.json
python test_infer.py --test_file sub_v3.json --model_name model_3.pt

#融合
python merge.py