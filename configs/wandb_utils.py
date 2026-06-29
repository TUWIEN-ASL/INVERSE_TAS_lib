import wandb


def init_wandb(model_name, dataset, split, f_type, extra_config=None):
    config = dict(model=model_name, dataset=dataset, split=split, f_type=f_type)
    if extra_config:
        config.update(extra_config)
    return wandb.init(
        project="inverse_tas",
        name=f"{model_name}_{dataset}_split{split}_{f_type}",
        config=config,
    )
