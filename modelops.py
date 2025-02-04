

def mdl_param_count(model):
    result = 0
    embedding_size = -1

    for n, p in model.named_parameters():
        this_count = 1

        for s in p.shape:
            this_count *= s

        result += this_count

        # if n == "model.shared.weight":

        if "shared.weight" in n:
            embedding_size = this_count

    return result, embedding_size
