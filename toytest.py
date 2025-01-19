import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_scheduler
)
from datasets import load_dataset
from accelerate import Accelerator
from evaluate import load as load_metric


def preprocess_function(examples, tokenizer, max_input_length=128, max_target_length=128):
    # 'translation' field has "en" and "et" keys in this dataset
    # Adjust if your version of the dataset differs.
    inputs = examples["translation"]["en"]
    targets = examples["translation"]["et"]

    model_inputs = tokenizer(
        text=inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    # Setup the tokenizer for targets
   # with tokenizer.as_target_tokenizer():
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def do_eval(model, dataloader, tokenizer):
    model.eval()
    frc_bos = tokenizer.convert_tokens_to_ids("est_Latn")
    hyps = []
    refs = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            generated_ids = model.generate(**batch, forced_bos_token_id=frc_bos)
        decoded_hyps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        hyps.extend(decoded_hyps)

        decoded_refs = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        refs.extend(decoded_refs)

    metric_bleu = load_metric("sacrebleu")
    result = metric_bleu.compute(predictions=hyps, references=refs)

    print(f"Validation BLEU: {result['score']:.2f}")
    print(f"1st hyp: {hyps[0]}")
    print(f"1st ref: {refs[0]}")
    print(f"2nd hyp: {hyps[1]}")
    print(f"2nd ref: {refs[1]}")



def do_eval_loss(model, dataloader, print=False):
    model.eval()

    losses = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(loss.item())

    val_loss = sum(losses) / len(losses)

    if print:
        print(f"Validation Loss: {val_loss:.4f}")
    else:
        return val_loss


def freeze_mdl(mdl):
    for p in mdl.parameters():
        p.requires_grad = False


def chain(*iterables):
    for it in iterables:
        yield from it


def main():
    #
    # 1. Load dataset
    #
    # The 'Helsinki-NLP/europarl' dataset with configuration "en-et"
    # may require manual validation/test splitting, but let's assume
    # 'train'/'validation' are provided (or you can split yourself).
    #
    raw_datasets = load_dataset("Helsinki-NLP/europarl", "en-et")
    # If needed:
    # raw_datasets = raw_datasets["train"].train_test_split(test_size=0.01)
    # Then rename keys to mimic a train/val structure if your dataset lacks it

    raw_data = raw_datasets["train"].select(range(11000))

    #
    # 2. Initialize tokenizer
    #
    # For NLLB, you must specify the correct source/target languages as special tokens.
    #
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600m",
        src_lang="eng_Latn",
        tgt_lang="est_Latn",
    )

    #
    # 3. Preprocessing / tokenization
    #
    max_input_length = 128
    max_target_length = 128

    batch_size = 16

    prepr_func = lambda ex: preprocess_function(ex, tokenizer, max_input_length, max_target_length)

    tok_data = raw_data.map(prepr_func).shuffle(seed=42)
    clean_tok_data = tok_data.remove_columns(["translation"])

    train_dataset = clean_tok_data.select(range(10000))
    valid_dataset = clean_tok_data.select(range(10101,10200))

    #
    # 4. Create DataLoaders
    #
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size  # adjust as needed
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size  # adjust as needed
    )

    #
    # 5. Initialize model
    #
    xmodel = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600m")

    xanc_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600m")
    freeze_mdl(xanc_model)

    #
    # 6. Setup optimizer and (optional) scheduler
    #
    optimizer = torch.optim.AdamW(chain(xmodel.parameters(), xanc_model.parameters()), lr=1e-4)

    # Example linear scheduler; adjust warmup_steps, etc. as needed
    num_epochs = 1
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps
    )

    #
    # 7. Use Accelerate to prepare
    #
    accelerator = Accelerator()

    models = [None, None]

    (
        models[0],
        models[1],
        optimizer,
        train_dataloader,
        valid_dataloader,
        lr_scheduler
    ) = accelerator.prepare(
        xmodel, xanc_model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    do_eval(models[0], valid_dataloader, tokenizer)

    indices = [(0, 0), (0, 1), (1, 0)]

    #
    # 8. Training loop
    #
    for epoch in range(num_epochs):
        models[0].train()

        for step, batch in enumerate(train_dataloader):
            if step < 51:
                these_indices = indices[step % 3]

                src_model = models[these_indices[0]]
                tgt_model = models[these_indices[1]]

                #outputs = model(**batch)
                enc_vecs = src_model.model.encoder(attention_mask=batch['attention_mask'], input_ids=batch['input_ids'])
                outputs = tgt_model(attention_mask=batch['attention_mask'], labels=batch['labels'], encoder_outputs=enc_vecs)

                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                try:
                    #mem = torch.cuda.memory_allocated(0) / 1024**2
                    mem = torch.mps.current_allocated_memory() / 1024 ** 2
                except:
                    mem = -1

                if step % 10 == 0:
                    print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f} | Device {models[0].device} | Mem {mem:.2f} Mb")

        #
        # 9. Validation loop (simple loss computation)
        #
        do_eval(models[0], valid_dataloader, tokenizer)

    #
    # 10. Save final model
    #
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(models[0])
    unwrapped_model.save_pretrained("nllb_en-et_finetuned", save_function=accelerator.save)
    tokenizer.save_pretrained("nllb_en-et_finetuned")


if __name__ == "__main__":
    main()
