import torch as t
import matplotlib.pyplot as plt
from gan import Generator, Discriminator, train
import torchvision
if __name__ == "__main__":

    # The following line of code normalizes the data to the range [0, 1]
    data = torchvision.datasets.MNIST(
        root="mnist/", train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
    )
    x = next(iter(data))
    G = Generator()
    D = Discriminator()
    train(G, D, data, epochs=20, batch_size=64, lr=0.0002, k=1, batches=200)
    generated_sample = G.sample()
    plt.imshow(generated_sample[0].squeeze().detach().numpy(), cmap='gray')
    plt.savefig("images/gan_sample.png")
    plt.clf()