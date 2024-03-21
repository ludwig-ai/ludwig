from ludwig.schema.gradual_unfreezer import GradualUnfreezerConfig


class GradualUnfreezer:
    def __init__(self, config: GradualUnfreezerConfig, model):
        self.config = config
        self.model = model
        self.thaw_epochs = self.config.thaw_epochs
        self.layers_to_thaw = self.config.layers_to_thaw

        if len(self.thaw_epochs) != len(self.layers_to_thaw):
            raise ValueError("The length of thaw_epochs and layers_to_thaw must be equal.")
        self.layers = dict(zip(self.thaw_epochs, self.layers_to_thaw))

    def thaw(self, current_epoch: int) -> None:
        if current_epoch in self.layers:
            current_layers = self.layers[current_epoch]
            for layer in current_layers:
                self.thawParameter(layer)

    # thaw individual layer
    def thawParameter(self, layer):
        # is there a better way to do this instead of iterating through all parameters?
        for name, p in self.model.named_parameters():
            if layer in str(name):
                p.requires_grad_(True)
            else:
                raise ValueError("Layer type doesn't exist within model architecture")
