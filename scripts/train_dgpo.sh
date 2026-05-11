#!/bin/bash

WAND_PROJECT=DGPO

# Default values
DEFAULT_DATA_NAME="nq_hotpotqa_train"
DEFAULT_CUDA_DEVICES="0,1,2,3,4,5,6,7"
DEFAULT_STUDENT_MODEL="omron-sinicx/Qwen2.5-0.5B-Instruct-kd"
DEFAULT_TEACHER_MODEL="omron-sinicx/SearchR1-ppo-qwen2.5-3b-instruct"
DEFAULT_EXPERIMENT_NAME="dgpo-qwen2.5-0.5b"

# Initialize variables with defaults
DATA_NAME="$DEFAULT_DATA_NAME"
CUDA_DEVICES="$DEFAULT_CUDA_DEVICES"
STUDENT_MODEL="$DEFAULT_STUDENT_MODEL"
TEACHER_MODEL="$DEFAULT_TEACHER_MODEL"
EXPERIMENT_NAME="$DEFAULT_EXPERIMENT_NAME"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-name)
            DATA_NAME="$2"
            shift 2
            ;;
        --gpu-ids)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --student-model)
            STUDENT_MODEL="$2"
            shift 2
            ;;
        --teacher-model)
            TEACHER_MODEL="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --data-name <name>              Data name (default: $DEFAULT_DATA_NAME)"
            echo "  --gpu-ids <devices>             CUDA visible devices (default: $DEFAULT_CUDA_DEVICES)"
            echo "  --student-model <model>         Student model path (default: $DEFAULT_STUDENT_MODEL)"
            echo "  --teacher-model <model>         Teacher model path (default: $DEFAULT_TEACHER_MODEL)"
            echo "  --experiment-name <name>        Experiment name (default: $DEFAULT_EXPERIMENT_NAME)"
            echo "  --help, -h                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Export environment variables
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export STUDENT_MODEL="$STUDENT_MODEL"
export TEACHER_MODEL="$TEACHER_MODEL"
export EXPERIMENT_NAME="$EXPERIMENT_NAME"
export DATA_DIR="${DATA_ROOT}/${DATA_NAME}"
export VLLM_ATTENTION_BACKEND=XFORMERS

# Create necessary directories
mkdir -p ${EXP_ROOT}/logs
mkdir -p ${EXP_ROOT}/verl_checkpoints/$EXPERIMENT_NAME


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \ #学習データを何件使うか。null=全件
    data.val_data_num=null \
    data.train_batch_size=512 \ #1 training step で使う train batch size
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \ #モデルに入力する prompt の最大 token 長
    data.max_response_length=500 \ #モデルが1回の generation で出せる最大 response token 数
    data.max_start_length=2048 \ #multi-turn rollout 開始時に保持する初期 prompt の最大長
    data.max_obs_length=500 \ #検索結果 observation の最大 token 長
    data.shuffle_train_dataloader=True \ #学習データを shuffle するか
    algorithm.adv_estimator=gae \ #advantage 推定方法
    actor_rollout_ref.model.path=$STUDENT_MODEL \
    actor_rollout_ref.model.ref_path=$TEACHER_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \ #一部を保存せず、backward 時に再計算
    actor_rollout_ref.model.use_remove_padding=True \ #padding token を除去
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \ 
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=128 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \ #actor の parameter を CPU に offload
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \ #actor optimizer state を CPU に offload
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \ #モデルを複数GPUに分割しない
    actor_rollout_ref.rollout.name=vllm \ #rollout generationにvllmを使用
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \ #1つの prompt に対し1つのrollout
    actor_rollout_ref.rollout.temperature=1 \ #生成時の sampling temperature
    actor_rollout_ref.rollout.top_p=1.0 \ #確率が高い順から合計がpに達するまでのトークンを候補とする。1.0 は全候補を対象。
    actor_rollout_ref.actor.state_masking=true \ #state / information を actor loss から mask
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_type=kl \ #KL loss の種類
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$STUDENT_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=64 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_penalty=kl \
    algorithm.no_think_rl=false \ #think を使わない RL にするか
    trainer.critic_warmup=0 \ #critic だけを先に warmup する step 数
    trainer.logger=['wandb'] \
    +trainer.val_only=false \ #validation だけでなく training を実行
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \ #1 node あたりの GPU 数
    trainer.nnodes=1 \ #使用node数
    trainer.save_freq=50 \ #checkpoint 保存頻度
    trainer.test_freq=200 \ #validation / test の頻度
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1000 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${EXP_ROOT}/verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee ${EXP_ROOT}/logs/$EXPERIMENT_NAME.log
