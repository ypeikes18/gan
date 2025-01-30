import torch as t
import torch.nn as nn

def discriminator_guidance(D, x, num_steps=100):
    """
    :param D: The discriminator
    :param x: The output of the generator
    :param num_steps: The number of gradient descent steps to take
    """
    x = x.detach()
    x = nn.Parameter(x)
    optimizer = t.optim.Adam([x], lr=1e-3)

    for i in range(num_steps):
        optimizer.zero_grad()
        prediction = D(x.view(x.size(0), 1, 28, 28))  # reshape back to (b, c, h, w)
        loss = nn.BCEWithLogitsLoss()(prediction, t.ones_like(prediction))
        print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    # Return the updated tensor (reshaped to image dimension if needed)
    return x.view(x.size(0), 1, 28, 28)