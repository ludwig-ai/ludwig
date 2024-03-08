from ludwig.encoders.image.torchvision import TVSwinTransformerEncoder
from ludwig.modules.gradual_unfreezer import GradualUnfreezer, GradualUnfreezerConfig
from ludwig.utils.misc_utils import set_random_seed


def test_gradual_unfreezer():
    set_random_seed(13)

    config = GradualUnfreezerConfig(thaw_epochs=[1, 2], layers_to_thaw=[["features.0", "features.1"]])

    model = TVSwinTransformerEncoder(
        model_variant="t",
        use_pretrained=False,
        saved_weights_in_checkpoint=True,
        trainable=False,
    )
    print(model)

    unfreezer = GradualUnfreezer(config=config, model=model)

    for epoch in range(10):
        unfreezer.thaw(epoch)

    for name, p in model.named_parameters():
        if config.layers_to_thaw[0][0] in name:
            assert p.requires_grad
        elif config.layers_to_thaw[0][1] in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad
