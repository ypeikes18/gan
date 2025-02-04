import torch as t
import torch.nn as nn

def discriminator_guidance(discriminator, x, num_steps=100, lr=1e-3):
    """
    :param discriminator: The discriminator
    :param x: The output of the generator
    :param num_steps: The number of gradient descent steps to take
    """
    x = nn.Parameter(x.detach())
    optimizer = t.optim.Adam([x], lr=1e-3)

    for i in range(num_steps):
        optimizer.zero_grad()
        prediction = discriminator(x)
        loss = nn.BCEWithLogitsLoss()(prediction, t.ones_like(prediction))
        print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    return x