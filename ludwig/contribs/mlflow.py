from ludwig.callbacks import Callback

try:
    import mlflow
except:
    mlflow = None


class MlflowCallback(Callback):
    def on_batch_start(self, trainer, progress_tracker, save_path):
        pass

    def on_batch_end(self, trainer, progress_tracker, save_path):
        pass

    def on_epoch_start(self, trainer, progress_tracker, save_path):
        pass

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        pass

    def on_validation_start(self, trainer, progress_tracker, save_path):
        pass

    def on_validation_end(self, trainer, progress_tracker, save_path):
        pass

    def on_test_start(self, trainer, progress_tracker, save_path):
        pass

    def on_test_end(self, trainer, progress_tracker, save_path):
        pass
