class NewLudwigModel:

    def __init__(self,
                 model_definition,
                 use_horovod=False,
                 gpus=None,
                 gpu_memory_limit=None,
                 allow_parallel_threads=True,
                 random_seed=default_random_seed):
        self._horovod = None
        if should_use_horovod(use_horovod):
            import horovod.tensorflow
            self._horovod = horovod.tensorflow
            self._horovod.init()

        initialize_tensorflow(gpus, gpu_memory_limit, allow_parallel_threads,
                              self._horovod)
        tf.random.set_seed(random_seed)

        self.model = ECD(model_definition)

    def train(self, data, training_params):
        preproc_data = preprocess_data(data)
        trainer = Trainer(training_aprams)
        training_stats = trainer.train(self.model, preproc_data)
        return training_stats

    def predict(self, data):
        preproc_data = preprocess_data(data)
        preds = self.model.batch_predict(preproc_data)
        postproc_preds = postprocess_data(preds)
        return postproc_preds

    def evaluate(self, data, return_preds=False):
        preproc_data = preprocess_data(data)
        if return_preds:
            eval_stats, preds = self.model.batch_evaluate(
                preproc_data, return_preds=return_preds
            )
            postproc_preds = postprocess_data(preds)
            return eval_stats, postproc_preds
        else:
            eval_stats = self.model.batch_evaluate(
                preproc_data, return_preds=return_preds
            )
            return eval_stats
