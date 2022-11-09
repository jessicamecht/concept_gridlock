import os
import gc
import sys
import time
import humanize
import datetime as dt
from tqdm import tqdm

from utils import file_write, log_end_epoch, get_common_path, valid_hyper_params, INF

def train(model, criterion, optimizer, reader, hyper_params):
    import torch

    model.train()
    
    # Initializing metrics since we will calculate MSE on the train set on the fly
    metrics = {
        'RMSE': 0.0
    }
    
    # Train for one epoch, batch-by-batch
    loop = tqdm(reader)
    for data, y in loop:
        # Empty the gradients
        model.zero_grad()
        optimizer.zero_grad()
    
        # Forward pass
        # print (type(data))
        # print (y.size())
        output = model(*data)
        # print (output.size())
        # sys.exit()
        # Compute per-interaction loss
        loss = criterion(output, y, return_mean = False)

        # loop.set_description("Loss: {}".format(round(float(loss), 4)))

        # Backward pass
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        metrics['RMSE'] += float(loss)

    metrics['RMSE'] = float(metrics['RMSE']) / len(reader)
    if hyper_params['horizon'] == True: metrics['RMSE'] /= hyper_params['horizon_K']
    metrics['RMSE'] = round(metrics['RMSE'] ** 0.5, 4)

    return metrics

def train_complete(hyper_params, train_reader, val_reader, model, model_class):
    import torch

    from loss import CustomLoss
    from eval import evaluate
    from torch_utils import is_cuda_available

    if hyper_params['model_type'] in { 'copy', 'linear' }: 
        print("\nNo training required for this baseline.\n")
        return model

    criterion = CustomLoss(hyper_params)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyper_params['lr'], betas=(0.9, 0.98),
        weight_decay=hyper_params['weight_decay']
    )

    file_write(hyper_params['log_file'], str(model))
    file_write(hyper_params['log_file'], "\nModel Built!\nStarting Training...\n")

    try:
        best_MSE = float(INF)
        decreasing_streak = 0

        for epoch in range(1, hyper_params['epochs'] + 1):
            epoch_start_time = time.time()
            
            # Training for one epoch
            train_metrics = train(model, criterion, optimizer, train_reader, hyper_params)
            # print("GPU usage while training: {}".format(humanize.naturalsize(torch.cuda.max_memory_reserved())))

            # Calulating the metrics on the validation set
            metrics = {}
            if epoch % hyper_params['validate_every'] == 0:
                metrics = evaluate(model, criterion, val_reader, hyper_params)
                decreasing_streak += 1

                # Save best model on validation set
                if metrics['RMSE_mean'] < best_MSE:
                    print("Saving model...")
                    torch.save(model.state_dict(), hyper_params['model_path'])
                    decreasing_streak, best_MSE = 0, metrics['RMSE_mean']
            
            metrics['Train RMSE'] = train_metrics['RMSE']
            log_end_epoch(hyper_params, metrics, epoch, time.time() - epoch_start_time, metrics_on = '(VAL)')
            
            if hyper_params['model_type'] == "GapFormer" and hyper_params['gapformer_fusion'] == 'global_softmax_attention':
                rnn_weight, transformer_weight = map(lambda x: round(x, 4), model.softmax(model.weights).cpu().detach().numpy().tolist())
                file_write(hyper_params['log_file'], "RNN weight: {} ; Transformer weight: {}".format(rnn_weight, transformer_weight))
            if hyper_params['model_type'] == "GapFormer" and hyper_params['gapformer_fusion'] == 'global_raw_attention':
                rnn_weight, transformer_weight = map(lambda x: round(x, 4), model.weights.cpu().detach().numpy().tolist())
                file_write(hyper_params['log_file'], "RNN weight: {} ; Transformer weight: {}".format(rnn_weight, transformer_weight))

            # Check if need to early-stop
            if 'early_stop' in hyper_params and decreasing_streak >= hyper_params['early_stop']:
                file_write(hyper_params['log_file'], "Early stopping..")
                break
            
    except KeyboardInterrupt: print('Exiting from training early')

    # Load best model and return it for evaluation on test-set
    if os.path.exists(hyper_params['model_path']):
        model = model_class(hyper_params)
        if is_cuda_available: model = model.cuda()
        model.load_state_dict(torch.load(hyper_params['model_path']))
    
    model.eval()

    return model

def main_pytorch(hyper_params, just_eval, eval_full = True):
    from data import load_data
    from eval import evaluate
    
    from torch_utils import is_cuda_available, xavier_init, get_model_class
    from loss import CustomLoss

    import torch

    # Load the data readers
    train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params)
    hyper_params = train_reader.hyper_params
    file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
    file_write(hyper_params['log_file'], "Data reading complete!")
    file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(len(train_reader)))
    file_write(hyper_params['log_file'], "Number of validation batches: {:4d}".format(len(val_reader)))
    file_write(hyper_params['log_file'], "Number of test batches: {:4d}".format(len(test_reader)))

    # Initialize & train the model
    start_time = time.time()

    model = get_model_class(hyper_params)(hyper_params)
    if is_cuda_available: model = model.cuda()

    if not just_eval:
        xavier_init(model)
        model = train_complete(
            hyper_params, train_reader, test_reader, model, get_model_class(hyper_params)
        )
    else:
        model.load_state_dict(torch.load(hyper_params['model_path']))
        model.eval()

    metrics = {}
    if eval_full:
        # Calculating MSE on test-set
        criterion = CustomLoss(hyper_params)
        metrics = evaluate(model, criterion, test_reader, hyper_params, test = True)
        log_end_epoch(hyper_params, metrics, 'final', time.time() - start_time, metrics_on = '(TEST)')

    # We have no space left for storing the models
    del model, train_reader, test_reader, val_reader
    return metrics

def main(hyper_params, gpu_id = None, just_eval = False): 
    if not valid_hyper_params(hyper_params): 
        print("Invalid task combination specified, exiting.")
        return

    # Setting GPU ID for running entire code ## Very Very Imp.
    if gpu_id is not None: 
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # torch.cuda.set_device(int(gpu_id))

    hyper_params['log_file'] = "../results/{}/logs/{}.txt".format(hyper_params['dataset'], get_common_path(hyper_params))
    # hyper_params['log_file'] = "../results/{}/logs/test.txt".format(hyper_params['dataset'])
    hyper_params['model_path'] = "../results/{}/models/{}.pt".format(hyper_params['dataset'], get_common_path(hyper_params))

    import torch    
    torch.cuda.empty_cache() ; gc.collect()
    main_pytorch(hyper_params, just_eval)
    torch.cuda.empty_cache() ; gc.collect()

if __name__ == '__main__':
    from hyper_params import hyper_params
    main(hyper_params)
