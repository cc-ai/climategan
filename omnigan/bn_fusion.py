import torch
from copy import deepcopy


class FlattableModel(object):
    def __init__(self, model):
        self.model = deepcopy(model)
        self._original_model = model
        self._flat_model = None
        self._attr_names = self.get_attributes_name()

    def flatten_model(self):
        if self._flat_model is None:
            self._flat_model = self._flatten_model(self.model)
        return self._flat_model

    @staticmethod
    def _selection_method(module):
        return not (
            isinstance(module, torch.nn.Sequential)
            or isinstance(module, torch.nn.ModuleList)
        ) and not hasattr(module, "_restricted")

    @staticmethod
    def _flatten_model(module):
        modules = []
        child = False
        for (name, c) in module.named_children():
            child = True
            flattened_c = FlattableModel._flatten_model(c)
            modules += flattened_c
        if not child and FlattableModel._selection_method(module):
            modules = [module]
        return modules

    def get_layer_io(self, layer, nb_samples, data_loader):
        ios = []
        hook = layer.register_forward_hook(
            lambda m, i, o: ios.append((i[0].data.cpu(), o.data.cpu()))
        )

        nbatch = 1
        for batch_idx, (xs, ys) in enumerate(data_loader):
            # -1 takes all of them
            if nb_samples != -1 and nbatch > nb_samples:
                break
            _ = self.model(xs.cuda())
            nbatch += 1

        hook.remove()
        return ios

    def get_attributes_name(self):
        def _real_get_attributes_name(module):
            modules = []
            child = False
            for (name, c) in module.named_children():
                child = True
                flattened_c = _real_get_attributes_name(c)
                modules += map(lambda e: [name] + e, flattened_c)
            if not child and FlattableModel._selection_method(module):
                modules = [[]]
            return modules

        return _real_get_attributes_name(self.model)

    def update_model(self, flat_model):
        """
        Take a list representing the flatten model and rebuild its internals.
        :type flat_model: List[nn.Module]
        """

        def _apply_changes_on_layer(block, idxs, layer):
            assert len(idxs) > 0
            if len(idxs) == 1:
                setattr(block, idxs[0], layer)
            else:
                _apply_changes_on_layer(getattr(block, idxs[0]), idxs[1:], layer)

        def _apply_changes_model(model_list):
            for i in range(len(model_list)):
                _apply_changes_on_layer(self.model, self._attr_names[i], model_list[i])

        _apply_changes_model(flat_model)
        self._attr_names = self.get_attributes_name()
        self._flat_model = None

    def cuda(self):
        self.model = self.model.cuda()
        return self

    def cpu(self):
        self.model = self.model.cpu()
        return self


def bn_fuse(model):
    model = model.cpu()
    flattable = FlattableModel(model)
    fmodel = flattable.flatten_model()

    for index, item in enumerate(fmodel):
        if (
            isinstance(item, torch.nn.Conv2d)
            and index + 1 < len(fmodel)
            and isinstance(fmodel[index + 1], torch.nn.BatchNorm2d)
        ):
            alpha, beta = _calculate_alpha_beta(fmodel[index + 1])
            if item.weight.shape[0] != alpha.shape[0]:
                # this case happens if there was actually something else
                # between the conv and the
                # bn layer which is not picked up in flat model logic. (see densenet)
                continue
            item.weight.data = item.weight.data * alpha.view(-1, 1, 1, 1)
            item.bias = torch.nn.Parameter(beta)
            fmodel[index + 1] = _IdentityLayer()
    flattable.update_model(fmodel)
    return flattable.model


def _calculate_alpha_beta(batchnorm_layer):
    alpha = batchnorm_layer.weight.data / (
        torch.sqrt(batchnorm_layer.running_var + batchnorm_layer.eps)
    )
    beta = (
        -(batchnorm_layer.weight.data * batchnorm_layer.running_mean)
        / (torch.sqrt(batchnorm_layer.running_var + batchnorm_layer.eps))
        + batchnorm_layer.bias.data
    )
    alpha = alpha.cpu()
    beta = beta.cpu()
    return alpha, beta


class _IdentityLayer(torch.nn.Module):
    def forward(self, input):
        return input
