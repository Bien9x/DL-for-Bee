def freeze(module):
    """
    Freezes module's parameters.
    """

    for parameter in module.parameters():
        parameter.requires_grad = False


def get_frozen_parameters(module):
    """
    Returns names of frozen parameters of the given module.
    """

    frozen_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            frozen_parameters.append(name)

    return frozen_parameters
