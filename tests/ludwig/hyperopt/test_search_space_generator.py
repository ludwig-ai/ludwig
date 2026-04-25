"""Tests for search space auto-generation."""

from ludwig.hyperopt.search_space_generator import generate_trainer_search_space


class TestGenerateTrainerSearchSpace:
    def test_default_fields(self):
        space = generate_trainer_search_space()
        assert "trainer.learning_rate" in space
        assert space["trainer.learning_rate"]["space"] == "loguniform"

    def test_custom_fields(self):
        space = generate_trainer_search_space(tunable_fields=["learning_rate", "num_layers"])
        assert "trainer.learning_rate" in space
        assert "combiner.num_layers" in space
        assert "trainer.batch_size" not in space

    def test_batch_size_is_int(self):
        space = generate_trainer_search_space(tunable_fields=["batch_size"])
        assert space["trainer.batch_size"]["space"] == "int"
