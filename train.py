from model import TransformerModel, TransformerModelArgs
from logger import Logger,LogPlotter
from dataset import PretrainDataset
from pretrainer import LLMPreTrainConfig, LLMPreTrainer
from evaluator import TrainLossEvaluator
import os,datetime
 
wandb_log=False
wandb_project_name="my_llama"

model_name="my_llama_v1_64M_fp8"
pretrain_dataset_path='./data/tokenized_train_data/'
eval_dataset_path='./data/tokenized_train_data/eval'

args=TransformerModelArgs(
    vocab_size=6400,
    dim=512,
    n_layers=20,
    n_heads=8,
    n_kv_heads=2,
    max_position_embeddings=2048,
    embedding_weight_tying=True,
    dropout=0.1,
)
cf=LLMPreTrainConfig(
    tokens_per_iter=524288,
    max_seq_len=1024,
    batch_size=16, # 16GB
    # batch_size=48, # 40GB
    optimizer_type='AdamW_ortho',
    loss_fn='cross_entropy',
    stable_lr=5e-4,
    min_lr=5e-5,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_norm_clip=1.0,
    warmup_iters=1000,
    stable_iters=99000,
    decay_iters=1000,
    finetuning_iters=1000,
    checkpoint_dir='./data/checkpoints/'+model_name,
    log_interval=1,
    eval_interval=10,
    checkpoint_interval=25,
    backup_checkpoint_interval=200,
    iter_per_shard=1000,
    max_opened_shards=16,
    device='cuda',
    dtype='bfloat16',
    fp8_training=True,
    compile=True,
    num_workers=4,
    use_gradient_checkpoint=False,
)




if __name__=="__main__":

    model=TransformerModel(args)
    print("total parameters:", format(sum(p.numel() for p in model.parameters()), "_"))
    dataset=PretrainDataset(
        dataset_dir=pretrain_dataset_path,
        seq_len=cf.max_seq_len+1,
        iter_per_shard=cf.iter_per_shard,
        max_opened_shards=cf.max_opened_shards,
    )


    logger=Logger(
        folder=cf.checkpoint_dir,
        config={
            "model":model_name, 
            "model_args":args,
            "training_config":cf
        },
        use_wandb=wandb_log,
        wandb_project_name=wandb_project_name,
        wandb_run_name=model_name

    )
    logPlotter=LogPlotter(logger)

    evaluator=TrainLossEvaluator(
        eval_dataset_dir=eval_dataset_path,
        model=model,
        logger=logger,
        max_seq_len=cf.max_seq_len,
        n_eval_samples=5,
    )

    trainer=LLMPreTrainer(
        model=model,
        dataset=dataset,
        config=cf,
        logger=logger,
    )
    trainer.add_callback(evaluator, interval=cf.eval_interval)
    trainer.add_callback(logPlotter, interval=10, post_step=True)

    trainer.train(resume=True)