import torch as t
from torch import nn
from torch.utils.data import DataLoader
from utils import create_nd_conv, create_nd_conv_transpose
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class Generator(nn.Module):
    def __init__(self, *, latent_dim: int, conv_dims: int, output_shape: tuple[int, ...]):
        """
        :param latent_dim: The dimension of the latent space
        :param conv_dims: The number of dimensions of the convolutional layers (e.g. 2 for MNIST)
        :param output_shape: The shape of the input to the generator (e.g. (1, 28, 28) for MNIST)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 7 * 7),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (latent_dim, 7, 7)),
            create_nd_conv_transpose(conv_dims, in_channels=latent_dim, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            create_nd_conv_transpose(conv_dims, in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            create_nd_conv(conv_dims, in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            create_nd_conv(conv_dims, in_channels=16, out_channels=output_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
        
    def grad_free_forward(self, x):
        with t.no_grad():
            return self.model(x)
        
    def sample(self, batch_size=1):
        with t.no_grad():
            return self.model(t.randn(batch_size, self.latent_dim))

class Discriminator(nn.Module):
    def __init__(self, *, input_shape: tuple[int, ...], conv_dims: int):
        super().__init__()
        self.model = nn.Sequential(
            create_nd_conv(conv_dims, in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            create_nd_conv(conv_dims, in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(28*28, 1),
        )

    def forward(self, x):
        return self.model(x)
    
    def frozen_forward(self, x: t.Tensor) -> t.Tensor:
        """
        :param x: Tensor of shape (b, c, h, w)
        :return: The output of the discriminator
        """
        for param in self.model.parameters():
            param.requires_grad = False
        res = self.forward(x)  
        for param in self.model.parameters():
            param.requires_grad = True
        return res


def train(G, D, data, epochs=1, batch_size=32, lr=1e-3, k=1, batches=float('inf')):
    """
    :param G: The generator
    :param D: The discriminator
    :param data: The dataset
    :param epochs: The number of epochs to train for
    :param batch_size: The batch size
    :param lr: The learning rate for both the generator and discriminator optimizers
    :param k: The number of times to train the discriminator before training the generator
    :param batches: The number of batches to train for
    """
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    generator_optimizer = t.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    discriminator_optimizer = t.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    labels = t.concat([t.ones(batch_size,1) , t.zeros(batch_size,1) * 0.1], dim=0)
    for epoch in range(epochs):
        for i ,(batch, classes) in enumerate(data):
            if i >= batches:
                break
            print(f"Epoch {epoch}, Batch {i}")
              
            real_fake_data = t.concat(
                [batch, G.grad_free_forward(t.randn(batch_size, G.latent_dim))], 
                dim=0
            )
            # for shuffling the batch and labels
            indices = t.randperm(real_fake_data.shape[0])

            pred = D(real_fake_data[indices])
            discriminator_optimizer.zero_grad()
            loss = t.nn.BCEWithLogitsLoss()(pred, labels[indices])
            writer.add_scalar("Discriminator loss", loss, i)
            print(f"Discriminator loss: {loss}")
            loss.backward()
            discriminator_optimizer.step()
            
            if i % k == 0:  
                pred = D.frozen_forward(G(t.randn(batch_size,G.latent_dim)))
                generator_optimizer.zero_grad()
                loss = loss = t.nn.BCEWithLogitsLoss()(pred, t.ones_like(pred))
                writer.add_scalar("Generator loss", loss, i)
                print(f"Generator loss: {loss}")
                loss.backward()
                generator_optimizer.step()
    writer.close()
