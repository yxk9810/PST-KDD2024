### 依赖：
-python3.8 
-transformers 

### 数据处理
下载对应的数据到data目录下
模型 下载sci-bert(https://huggingface.co/allenai/scibert_scivocab_uncased) 到sci-bert下

### 训练和预测
从训练到最终结果，一键运行：sh run.sh
运行之后 merge_submit.json 文件为最终提交文件
result文件夹里面是三个融合前的原始文件，分别有三个模型预测所得

### 方案：
bert 二分类

### 模型权重下载[路径](https://drive.google.com/file/d/1R3F4KokHusrL-HKnZ88aALyqHXWBP9Pg/view?usp=drive_link)
