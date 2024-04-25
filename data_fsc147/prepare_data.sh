#!/bin/bash

# ori_root should contain the following files:
#     {ori_root}/
#     ├── annotation_FSC147_384.json
#     ├── ImageClasses_FSC147.txt
#     ├── images_384_VarV2/  # (unzip from `FSC147_384_V2.zip`)
#     |       ├── 2.jpg
#     |       ├── 3.jpg
#     |       .
#     |       .
#     |       .
#     |       └── 7607.jpg
#     └── Train_Test_Val_FSC_147.json
#
# it would be better to set the path as absolute path
ori_root=/home/wlin38/fixedpoint_prompt_counting/data_fsc147/origional_fsc147
new_root=/home/wlin38/fixedpoint_prompt_counting/data_fsc147/train_fsc147

mkdir $new_root
ln -s $ori_root/Train_Test_Val_FSC_147.json $new_root/

# generate base information of fsc147 for training
python gendata_512x768.py $ori_root $new_root

# download pre-trained llama-adapter
llama7b_url=https://huggingface.co/nyanko7/LLaMA-7B/resolve/main
mkdir llama_model_weights
mkdir llama_model_weights/7B

for reqf in checklist.chk params.json consolidated.00.pth
do
    wget $llama7b_url/$reqf -O llama_model_weights/7B/$reqf
done
wget $llama7b_url/tokenizer.model llama_model_weights/tokenizer.model
wget https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.1.0.0/llama_adapter_len10_layer30_release.pth -O llama_model_weights/llama_adapter_len10_layer30_release.pth

# generate nouns dict using llama
python nouns_extration.py $new_root

# generate clip mask
python llama2clip.py $new_root llama2