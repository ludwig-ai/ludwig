## Using a queue for holding `gpu_ids`
from mulitprocessing import Pool, Queue

gpus = "0,1"
gpu_fraction = 0.25
gpu_ids = gpus.split(",")
process_per_gpu = int(1 /gpu_fraction)

queue = Queue()

for gpu_id in range(gpu_ids):
    for _ in range(process_per_gpu):
        queue.put(gpu_id)
				
				
def train_and_eval_model(self, hyperopt_dict):
		gpu_id = queue.get()
		try:			
			parameters = hyperopt_dict['parameters']
			train_stats, eval_stats = train_and_eval_on_split(**hyperopt_dict, gpus=gpu_id)
			metric_score = get_metric_score(eval_stats)
		finally:
			queue.put(gpu_id)
			return {
					'parameters': parameters,
					'metric_score': metric_score,
					'training_stats': train_stats,
					'eval_stats': eval_stats
			}
		

pool = multiprocessing.Pool(num_workers)

hyperopt_results = pool.map(train_and_eval_model, hyperopt_parameters)
@kaushikb11
 