"""Tests for model soup utilities."""

import torch

from ludwig.utils.model_soup import uniform_soup


class TestUniformSoup:
    def test_two_state_dicts(self):
        sd1 = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.0])}
        sd2 = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([2.0])}
        avg = uniform_soup([sd1, sd2])
        assert torch.allclose(avg["w"], torch.tensor([2.0, 3.0]))
        assert torch.allclose(avg["b"], torch.tensor([1.0]))

    def test_three_state_dicts(self):
        sds = [
            {"w": torch.tensor([1.0, 1.0])},
            {"w": torch.tensor([2.0, 2.0])},
            {"w": torch.tensor([3.0, 3.0])},
        ]
        avg = uniform_soup(sds)
        assert torch.allclose(avg["w"], torch.tensor([2.0, 2.0]))

    def test_single_state_dict(self):
        sd = {"w": torch.tensor([5.0])}
        result = uniform_soup([sd])
        assert torch.equal(result["w"], sd["w"])

    def test_preserves_dtype(self):
        sd1 = {"w": torch.tensor([1.0, 2.0], dtype=torch.float16)}
        sd2 = {"w": torch.tensor([3.0, 4.0], dtype=torch.float16)}
        avg = uniform_soup([sd1, sd2])
        assert avg["w"].dtype == torch.float16

    def test_multiple_keys(self):
        sd1 = {"a": torch.ones(3), "b": torch.zeros(2), "c": torch.tensor([10.0])}
        sd2 = {"a": torch.ones(3) * 3, "b": torch.ones(2) * 2, "c": torch.tensor([20.0])}
        avg = uniform_soup([sd1, sd2])
        assert torch.allclose(avg["a"], torch.ones(3) * 2)
        assert torch.allclose(avg["b"], torch.ones(2))
        assert torch.allclose(avg["c"], torch.tensor([15.0]))
