hyper_params = {

    ## Dataset
    # [ 'toyota', 'openacc', 'once' ]
    'dataset':              'once',

    ## Models 
    # Baselines: 		[ 'copy', 'linear' ] 
	# Non-sequential: 	[ 'MLP' ] 
	# Sequential: 		[ 'RNN', 'Transformer', 'GapFormer' ]
    'model_type':           'RNN',
	'horizon': 				True,  # NOTE: not applicable for baselines
	'staggered_training': 	False, # NOTE: not included in the paper, check slides for staggered training
    'image_feature':        True,

	## Horizon-conifgurations (only if horizon == True)
    'horizon_K':              10,
    'horizon_decay':         'None', # [ 'None', 'linear', 'inverse' ]

	## Staggered-training configurations
	'seq_step_size':        100,

    ## All methods
    'latent_size':          512, 
    'dropout':              0.0,
    'weight_decay':         float(1e-6),
    'lr':                   0.0005,

	# ALL Sequential Methods
    'max_seq_len':          1_000,
    'pad_with':             0.0,

    ## Transformer / GapFormer
    'num_heads':            1,
    'num_blocks':           1,
	'linear_attention': 	True,
	'transformer_seq_len':  50, # Window-size in Transformer/GapFormer's linear self-attention, if enabled

    ## GapFormer
    # [ 'concat', 'add', 'global_softmax_attention', 'global_raw_attention' ]
    'gapformer_fusion':     'global_softmax_attention',
    
    ## General stuff
	'eval_K':               100, # Timesteps: if sampling freq is 10Hz === 100/10 seconds
    'batch_size':           2, # ~64-128 for MLP and ~2-4 for sequential models
    'epochs':               20,
    'validate_every':       1,
    'early_stop':           50, # Stop if perf. doesn't increase for `early_stop` X `validate_every` epochs
}
