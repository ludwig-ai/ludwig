from tests.training_success.test_training_success import (
    combiner_config_generator,
    defaults_config_generator,
    ecd_trainer_config_generator,
)


def full_config_generator(generator_fn, *args):
    return len(list(generator_fn(*args)))


def test_config_sampling():
    total_count = 0

    total_count += full_config_generator(defaults_config_generator, "number", "preprocessing")
    total_count += full_config_generator(defaults_config_generator, "number", "encoder")
    total_count += full_config_generator(defaults_config_generator, "number", "decoder")
    total_count += full_config_generator(defaults_config_generator, "number", "loss")

    total_count += full_config_generator(defaults_config_generator, "category", "preprocessing")
    total_count += full_config_generator(defaults_config_generator, "category", "encoder")
    total_count += full_config_generator(defaults_config_generator, "category", "decoder")
    total_count += full_config_generator(defaults_config_generator, "category", "loss")

    total_count += full_config_generator(defaults_config_generator, "binary", "preprocessing")
    total_count += full_config_generator(defaults_config_generator, "binary", "encoder")
    total_count += full_config_generator(defaults_config_generator, "binary", "decoder")
    total_count += full_config_generator(defaults_config_generator, "binary", "loss")

    total_count += full_config_generator(ecd_trainer_config_generator)

    total_count += full_config_generator(combiner_config_generator, "sequence_concat")
    total_count += full_config_generator(combiner_config_generator, "sequence")
    total_count += full_config_generator(combiner_config_generator, "comparator")
    total_count += full_config_generator(combiner_config_generator, "concat")
    total_count += full_config_generator(combiner_config_generator, "project_aggregate")
    total_count += full_config_generator(combiner_config_generator, "tabnet")
    total_count += full_config_generator(combiner_config_generator, "tabtransformer")
    total_count += full_config_generator(combiner_config_generator, "transformer")

    # expecting 139
    assert total_count == 139
