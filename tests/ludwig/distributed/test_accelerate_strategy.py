"""Tests for AccelerateStrategy."""

import torch


class TestAccelerateStrategy:
    def test_import_and_instantiate(self):
        from ludwig.distributed.accelerate import AccelerateStrategy

        strategy = AccelerateStrategy()
        assert strategy.size() >= 1
        assert strategy.rank() >= 0
        assert strategy.local_rank() >= 0

    def test_is_available(self):
        from ludwig.distributed.accelerate import AccelerateStrategy

        assert AccelerateStrategy.is_available()

    def test_barrier_no_hang(self):
        from ludwig.distributed.accelerate import AccelerateStrategy

        strategy = AccelerateStrategy()
        strategy.barrier()  # Should not hang in single-process mode

    def test_broadcast_object(self):
        from ludwig.distributed.accelerate import AccelerateStrategy

        strategy = AccelerateStrategy()
        result = strategy.broadcast_object({"key": "value"})
        assert result == {"key": "value"}

    def test_allreduce(self):
        from ludwig.distributed.accelerate import AccelerateStrategy

        strategy = AccelerateStrategy()
        t = torch.tensor([1.0, 2.0, 3.0])
        result = strategy.allreduce(t)
        assert torch.allclose(result, t)  # Single process: identity

    def test_registered_in_strategies(self):
        from ludwig.distributed import STRATEGIES

        assert "accelerate" in STRATEGIES

    def test_init_dist_strategy(self):
        from ludwig.distributed import init_dist_strategy

        strategy = init_dist_strategy("accelerate")
        assert strategy.size() >= 1

    def test_context_managers(self):
        from ludwig.distributed.accelerate import AccelerateStrategy

        strategy = AccelerateStrategy()
        model = torch.nn.Linear(10, 10)

        with strategy.prepare_model_update(model, should_step=True):
            pass

        with strategy.prepare_optimizer_update(torch.optim.SGD(model.parameters(), lr=0.01)):
            pass
