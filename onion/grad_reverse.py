# https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
class GradReverse(Function):
    "Implementation of GRL from DANN (Domain Adaptation Neural Network) paper"
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    """
    GRL must be placed between the feature extractor and the domain classifier
    """
    return GradReverse.apply(x)

