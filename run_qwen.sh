
set -x

model_path=/root/data/zhangxiao/work_xiao/aicas2024/Qwen/output_qwen/checkpoint-2100

pushd quantization
python autoclip.py \
    --model_path $model_path \
    --calib_dataset pile \
    --quant_type int --w_bit 4 --q_group_size -1 \
    --run_clip --dump_clip ./clip_cache/qwen_1_8B/int4-g2048.pt

popd

pushd data/generation

# vllm
python generate_vllm.py \
    --base_model $model_path \
    --dataset_name wikitext \
    --out_path ./datasets/qwen_1_8B/ \
    --max_sample 3000

python generate_vllm.py \
    --base_model $model_path \
    --dataset_name alpaca \
    --out_path ./datasets/qwen_1_8B/ \
    --max_sample 5000

# change to path in .py
python mix_data.py

popd


# Specify the pre-trained model path
# Specify the num_gpus and batch_size according to your GPU devices
# Specify the clipping cache path to the --clip

pushd train

bash train.sh  \
    ../data/generation/datasets/qwen_1_8B/mix_wiki_alpaca_8000.json \
    ./ckpts/qwen_1_8B/int4-g2048-0328/ \
    ./logs/qwen_1_8B/int4-g2048/ 4


popd

pushd test/general
CUDA_VISIBLE_DEVICES=0 python llm_eval.py \
    --model ../../train/ckpts/qwen_1_8B/int4-g2048/checkpoint-600/ \
    --eval_tasks arc_challenge,winogrande,hellaswag,piqa \
    --test_set \
    --bits 4 --group_size -1 \
    --quant_type int --num_fewshot 0 

