import os
import re
import numpy as np

def get_best(all_logs, model, time_horizon, VAL_METRIC = "RMSE_mean"):
    search_key = ".*" + {
        "copy": "copy.txt",
        "linear": "linear.txt",

        "MLP": "MLP(?!.*horizon.*)",
        'MLP_Horizon': "MLP.*horizon",
        
        "RNN": "RNN(?!.*horizon.*)",
        "RNN_Horizon": "RNN.*horizon",
        
        "Transformer": "Transformer(?!.*horizon.*)",
        "Transformer_Horizon": "Transformer.*horizon",

        "GapFormer": "GapFormer(?!.*horizon.*)",
        "GapFormer_Horizon": "GapFormer.*horizon",
    }[model] + ".*"
    
    relevant_logs = []
    for log in all_logs:
        if re.search(search_key, log): 
            f = open(log, "r") ; lines = f.readlines() ; f.close()
            relevant_logs.append([ log, lines ])
    
    NEEDED_METRICS = { "RMSE_{}".format(t) for t in range(1, time_horizon + 1) }

    final = [ { VAL_METRIC: float(1e6) } ] 
    best_log, best_overall = None, float(1e6)
    for log_name, lines in relevant_logs:

        best, all_metrics = float(1e6), {}       
        for line in lines:
            line = line.strip()
            
            if line.endswith("(TEST)"):
                this_metrics = {}
                for m in line[:-7].split(" | "): # removing " (TEST)"
                    if "=" not in m: continue
                    key, val = m.split(" = ")
                    if key in [ 'time', 'CPU_RAM', 'GPU', 'GPU_RAM' ]: continue
                    
                    this_metrics[key] = float(val)
                
                if any(map(lambda x: x not in this_metrics, NEEDED_METRICS)): continue

				# RMSE_mean could be different while logging and while computing this function
                this_metrics['RMSE_mean'] = round(np.mean([ this_metrics[i] for i in NEEDED_METRICS ]), 4)

                if this_metrics[VAL_METRIC] < best:
                    best, all_metrics = float(this_metrics[VAL_METRIC]), this_metrics
        
        if any(map(lambda x: x not in all_metrics, NEEDED_METRICS)): continue
        all_metrics['RMSE_mean'] = round(np.mean([ all_metrics[i] for i in NEEDED_METRICS ]), 4)

        if best < best_overall: best_overall, best_log = best, log_name

        if len(all_metrics) > 0: 
            final.append(all_metrics)

    if len(final) > 1: final = final[1:]

    return list(sorted(final, key = lambda x: float(x[VAL_METRIC])))[:10], best_log

if __name__ == "__main__":
	for dataset, sampling_freq in [ 
		[ "toyota", 10.0 ], 
		[ "openacc", 10.0 ],
		[ "once", 2.0 ]
	]:
		
		BASE_PATH = "../results/{}/logs/".format(dataset)
		all_logs = os.listdir(BASE_PATH)
		all_logs = list(map(lambda x: BASE_PATH + x, all_logs))
		print("="*30, "Dataset =", dataset, "="*30)

		for kind in [ 
			'copy',
			'linear',
			'MLP',
			'MLP_Horizon',
			'RNN',
			'RNN_Horizon',
			'Transformer',
			'Transformer_Horizon',
			'GapFormer',
			'GapFormer_Horizon',
		]:
			results, best_log = get_best(all_logs, kind, int(10.0 * sampling_freq)) # 10 seconds in the future

			def pretty(all_metrics):
				return_metrics = [ 'RMSE_mean', 'RMSE_1', 'RMSE_25', 'RMSE_50', 'RMSE_75', 'RMSE_100' ]
				if dataset == "once": return_metrics = [ 'RMSE_mean', 'RMSE_1', 'RMSE_5', 'RMSE_10', 'RMSE_15', 'RMSE_20' ]
				return { k:all_metrics[k] for k in return_metrics if k in all_metrics }

			print("Method = {}\nBest results:".format(kind))
			for i in range(min(5, len(results))):
				print("{}".format(pretty(results[i])))
			print(best_log)
			print()
