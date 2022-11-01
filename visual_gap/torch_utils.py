import torch

is_cuda_available = torch.cuda.is_available()

if is_cuda_available: 
    print("Using CUDA...\n")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor
    BoolTensor = torch.cuda.BoolTensor
else:
    LongTensor = torch.LongTensor
    FloatTensor = torch.FloatTensor
    BoolTensor = torch.BoolTensor

def get_model_class(hyper_params):
    from pytorch_models import MLP, RNN, Transformer, GapFormer

    return {
		# Baselines
        "copy": MLP.MLP,
        "linear": MLP.MLP,

		# Non-sequential
        "MLP": MLP.MLP,
        
		# Sequential
        "RNN": RNN.RNN,
        "Transformer": Transformer.Transformer,
        "GapFormer": GapFormer.GapFormer,
    }[hyper_params['model_type']]

def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    for name, param in model.named_parameters():
        try: torch.nn.init.xavier_uniform_(param.data)
        except: pass # just ignore those failed init layers
