def print_shapes(w):
    for i in range(len(w)):
        print(f"Layer {i}: {w[i].shape}")

# separate weigts and biases
def separate_weights(w):
    weights = []
    biases = []
    for i in range(len(w)):
        if(len(w[i].shape) == 2):
            weights.append(w[i])
        elif(len(w[i].shape) == 1):
            biases.append(w[i])
    return weights, biases


def load_weigths(df, round):
    w_name = df["weights"][round]
    w_file = glob.glob(f"{fed_avg}/../{w_name}", recursive=True)
    return torch.load(w_file[0])

def total_l2_norm(w0, w1):
    """Returns total L2 norm over all parameters."""
    diffs = [(torch.tensor(a) - torch.tensor(b)).flatten() for a, b in zip(w0, w1)]
    total_diff = torch.cat(diffs)
    return torch.norm(total_diff).item()

def l2norm_perlayer(w0, w1):
    """Returns L2 norm per layer."""
    diffs = [(torch.tensor(a) - torch.tensor(b)).flatten() for a, b in zip(w0, w1)]
    total_diff = torch.cat(diffs)
    return [torch.norm(diff).item() for diff in diffs]

# for each round, compute the l2 norm of the weights between current and previous
def compute_l2_norm(round):
    if round == 0:
        return 0
    else:
        return total_l2_norm(load_weigths(round-1), load_weigths(round))


