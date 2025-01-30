import torch as t
import matplotlib.pyplot as plt
from gan import Generator, Discriminator, train
import torchvision
if __name__ == "__main__":

    # The following line of code normalizes the data to the range [0, 1]
    data = torchvision.datasets.MNIST(
        root="mnist/", train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()        
        ])
    )
    x = next(iter(data))
    G = Generator(latent_dim=250, conv_dims=2, output_shape=(1, 28, 28))
    D = Discriminator(conv_dims=2, input_shape=(1, 28, 28))

    train(G, D, data, epochs=5, batch_size=64, lr=1e-4, k=1)
    t.save(G.state_dict(), "weights/generator_weights_2.pth")
    t.save(D.state_dict(), "weights/discriminator_weights_2.pth")
    
    generated_sample = G.sample(10)
    breakpoint()
    num_samples = 10
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(10):
        axes[i].imshow(generated_sample[i].squeeze().detach().numpy(), cmap='gray')
        axes[i].axis('off')
    plt.savefig("images/gan_sample_2.png")

