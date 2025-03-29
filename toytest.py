from time import sleep

import torch.optim
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from modelops import report_devices
from translate import hf_tok

def main(mdl_id = "meta-llama/Llama-3.2-1B"):
    acc = Accelerator()

    report_devices("Initial state:", accelerator=acc)
    sleep(3)

    m = AutoModelForCausalLM.from_pretrained(mdl_id, token=hf_tok, torch_dtype=torch.bfloat16)
    t = AutoTokenizer.from_pretrained(mdl_id)

    report_devices("Model and tokenizer loaded:", accelerator=acc)
    sleep(3)


    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    lrs = get_scheduler("linear", optimizer=opt, num_warmup_steps=100, num_training_steps=1000)
    opt, lrs, m = acc.prepare(opt, lrs, m)

    report_devices("Accelerator initialized:", accelerator=acc)
    sleep(3)

    m.train()

    sleep(3)
    report_devices("model.train() called:", accelerator=acc)
    sleep(3)

    txt = "Kui ma laulan, kui me leelon, laulan lained laksuma, siis jääb küla kuulamaie"
    raw_inp = [txt] * 64
    t.pad_token = '<|reserved_special_token_0|>'
    inp = t(raw_inp, return_tensors="pt", max_length=512, truncation=True, add_special_tokens=True, padding=True, padding_side='left')
    inp['labels'] = inp['input_ids']
    inp.to(m.device)
    sleep(3)

    report_devices("batch loaded:", accelerator=acc)
    sleep(3)

    state = AcceleratorState()
    print(f"Num proc: {acc.num_processes}, proc ID: {acc.process_index}, num proc per node {state.num_processes_per_node}, node rank {state.node_rank}, num nodes {state.num_nodes}")

    for _ in range(10):
        outputs = m(**inp)
        loss = outputs.loss
        acc.backward(loss)
        report_devices("training cycle:", accelerator=acc)


if __name__ == "__main__":
    main()

