import transformers
from dataloader import get_cpt_data

def get_config():
    config = transformers.HfArgumentParser(transformers.TrainingArguments)
    config = config.parse_args_into_dataclasses()[0]
    config.report_to = "none"
    config.per_device_train_batch_size = 1
    config.gradient_accumulation_steps = 32
    config.bf16 = True
    return config

def train():
    # training config
    config = get_config()
    # loading model
    model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Base")
    # loading dataset
    data_module = get_cpt_data()
    # setting up trainer
    trainer = transformers.Trainer(
        model=model, args=config, **data_module)
    trainer.train()

if __name__ == "__main__":
    train()