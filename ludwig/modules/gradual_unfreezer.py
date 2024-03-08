from ludwig.schema.gradual_unfreezer import GradualUnfreezerConfig


class GradualUnfreezer:
    def __init__(self, config: GradualUnfreezerConfig, model):
        self.config = config
        self.model = model
        self.thaw_epochs = self.config.thaw_epochs
        self.layers_to_thaw = self.config.layers_to_thaw
        self.freeze_before_training()

        self.layers = dict(zip(self.thaw_epochs, self.layers_to_thaw))

    def freeze_before_training(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def thaw(self, current_epoch):
        if current_epoch in self.layers:
            current_layers = self.layers[current_epoch]
            for layer in current_layers:
                self.thawParameter(layer)

    def thawParameter(self, layer):
        for name, p in self.model.named_parameters():
            if layer in str(name):
                p.requires_grad_(True)
