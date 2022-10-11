import os

from ludwig.utils.data_utils import load_json, load_yaml


class BenchmarkingArtifact:
    def __init__(self, benchmarking_config: dict, experiment_idx: int):
        self.benchmarking_config = benchmarking_config
        self.experiment_config = benchmarking_config["experiments"][experiment_idx]
        self.ludwig_config = load_yaml(self.experiment_config["config_path"])
        self.process_config_file = None
        if self.experiment_config["process_config_file_path"]:
            with open(self.experiment_config["process_config_file_path"]) as f:
                self.process_config_file = "".join(f.readlines())
        self.experiment_run_path = os.path.join(self.experiment_config["experiment_name"], "experiment_run")
        self.description = load_json(os.path.join(self.experiment_run_path, "description.json"))
        self.test_statistics = load_json(os.path.join(self.experiment_run_path, "test_statistics.json"))
        self.training_statistics = load_json(os.path.join(self.experiment_run_path, "training_statistics.json"))
        self.model_hyperparameters = load_json(
            os.path.join(self.experiment_run_path, "model", "model_hyperparameters.json")
        )
        self.training_progress = load_json(os.path.join(self.experiment_run_path, "model", "training_progress.json"))
        self.training_set_metadata = load_json(
            os.path.join(self.experiment_run_path, "model", "training_set_metadata.json")
        )
