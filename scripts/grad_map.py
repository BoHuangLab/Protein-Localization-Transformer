import torch


class ActivationsAndGradients:
    """Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):

        if len(output) == 2:
            output = output["attn"]

        if len(self.activations) == 0:
            self.min_val = torch.min(output)

        attn = torch.nn.functional.pad(
            output,
            (0, 1512 - (output.shape[-1]), 0, 0, 0, 0, 0, 0),
            "constant",
            self.min_val,
        )

        self.activations.append(attn.cpu().detach())

    def save_gradient(self, module, input, output):
        # if not hasattr(output, "requires_grad") or not output.requires_grad:
        #     # You can only register hooks on tensor requires grad.
        #     return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        # output.register_hook(_store_grad(output))

    def __call__(self, x, filter_thres=0.5):
        self.gradients = []
        self.activations = []
        sequence, nucleus, target = x
        return self.model.generate_images(
            text=sequence,
            condition=nucleus,
            return_logits=True,
            progress=True,
            use_cache=True,
            filter_thres=filter_thres,
        )

    def release(self):
        for handle in self.handles:
            handle.remove()