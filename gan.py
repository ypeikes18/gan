import torch as t
from torch import nn
from torch.utils.data import DataLoader
from nn_utils import create_nd_conv, create_nd_conv_transpose
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from utils import upsample_sizes

writer = SummaryWriter()


class Generator(nn.Module):
    def __init__(self, *, latent_size: tuple[int, ...], conv_dims: int, output_shape: tuple[int, ...]):
        """
        :param latent_dim: The dimension of the latent space
        :param conv_dims: The number of dimensions of the convolutional layers (e.g. 2 for MNIST)
        :param output_shape: The shape of the input to the generator (e.g. (1, 28, 28) for MNIST)
        """
        super().__init__()
        # TODO unhardcode channels
        self.conv_dims = conv_dims
        self.latent_size = latent_size
        self.output_shape = output_shape
        out_shape_prod = t.prod(t.tensor(output_shape))
        self.to_pixel = nn.Sequential(
            nn.Linear(t.prod(t.tensor(latent_size)), out_shape_prod),
            nn.LeakyReLU(0.2),
            nn.Linear(out_shape_prod, out_shape_prod),
            nn.LeakyReLU(0.2),
            nn.Linear(out_shape_prod, out_shape_prod),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, output_shape)
        )
        self.activation = nn.LeakyReLU(0.2)
        self.convs = nn.Sequential(
            create_nd_conv(conv_dims=conv_dims, stride=1, padding=1, kernel_size=3, in_channels=output_shape[-3], out_channels=64),
            nn.LeakyReLU(0.2),
            create_nd_conv(conv_dims=conv_dims, stride=1, padding=1, kernel_size=3, in_channels=64, out_channels=128),
            nn.LeakyReLU(0.2),
            create_nd_conv(conv_dims=conv_dims, stride=1, padding=1, kernel_size=3, in_channels=128, out_channels=64),
            nn.LeakyReLU(0.2),
            create_nd_conv(conv_dims=conv_dims, stride=1, padding=1, kernel_size=3, in_channels=64, out_channels=output_shape[-3]),
            nn.LeakyReLU(0.2),
        )
        self.out_activation = nn.Sigmoid()


    def get_seed(self, batch_size: int) -> t.Tensor:
        return t.randn(batch_size, self.latent_size, requires_grad=False)

    def forward(self, x):
        x = self.to_pixel(x)
        x = self.convs(x)
        return self.out_activation(x)
        
    def grad_free_forward(self, x):
        with t.no_grad():
            return self.forward(x)
        
    def sample(self, batch_size=1):
        with t.no_grad():
            breakpoint()
            return self.forward(self.get_seed(batch_size))


class Discriminator(nn.Module):
    def __init__(self, *, input_shape: tuple[int, ...], conv_dims: int):
        super().__init__()
        self.model = nn.Sequential(
            create_nd_conv(conv_dims, in_channels=input_shape[0], out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            create_nd_conv(conv_dims, in_channels=128, out_channels=input_shape[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(t.prod(t.tensor(input_shape)), 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1)
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


def train(
G: Generator, 
D: Discriminator, 
data, 
epochs=1, 
batch_size=32, 
lr=1e-3, 
k_discriminator=1, 
batches=float('inf'),
k_generator=1):
    """
    :param G: The generator
    :param D: The discriminator
    :param data: The dataset
    :param epochs: The number of epochs to train for
    :param batch_size: The batch size
    :param lr: The learning rate for both the generator and discriminator optimizers
    :param k_discriminator: The number of times to train the discriminator before training the generator
    :param k_generator: The number of times to train the generator before training the discriminator
    :param batches: The number of batches to train for
    """
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    generator_optimizer = t.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    discriminator_optimizer = t.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    labels = t.concat([t.ones(batch_size,1) , t.zeros(batch_size,1)], dim=0)
    for epoch in range(epochs):
        for i ,(batch, classes) in enumerate(data):
            if i >= batches:
                break
            print(f"Epoch {epoch}, Batch {i}")
            z = G.get_seed(batch.shape[0])
            real_fake_data = t.concat(
                [batch, G.grad_free_forward(z)], 
                dim=0
            )
            # for shuffling the batch and labels
            indices = t.randperm(real_fake_data.shape[0])

            pred = D(real_fake_data[indices])
            if i % 100 == 0:
                print(pred)
            discriminator_optimizer.zero_grad()
            loss = t.nn.BCEWithLogitsLoss()(pred, labels[indices])
            writer.add_scalar("Discriminator loss", loss, i)
            print(f"Discriminator loss: {loss}")
            loss.backward()
            discriminator_optimizer.step()
            
            if i % k_discriminator == 0:
                for _ in range(k_generator):
                    z = G.get_seed(batch.shape[0])
                    pred = D.frozen_forward(G(z))
                    generator_optimizer.zero_grad()
                    loss = loss = t.nn.BCEWithLogitsLoss()(pred, t.ones_like(pred))
                    writer.add_scalar("Generator loss", loss, i)
                    print(f"Generator loss: {loss}")
                    loss.backward()
                    generator_optimizer.step()
    
    writer.close()
