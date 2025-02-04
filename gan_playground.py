import torch as t
import matplotlib.pyplot as plt
from gan import Generator, Discriminator, train
from discriminator_guidance import discriminator_guidance
import torchvision

if __name__ == "__main__":

    # The following line of code normalizes the data to the range [0, 1]
    data = torchvision.datasets.MNIST(
        root="mnist/", train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()        
        ])
    )
    x = next(iter(data))
    G = Generator(latent_size=100, output_shape=(1, 28, 28), conv_dims=2)
    D = Discriminator(conv_dims=2, input_shape=(1, 28, 28))

    train(G, D, data, epochs=1, batch_size=64, lr=5e-4, k_discriminator=1, k_generator=1)

    t.save(G.state_dict(), f"weights/generator_weights_020325.pth")
    t.save(D.state_dict(), f"weights/discriminator_weights_020325.pth")
    
    generated_sample = G.sample(10)
    generated_sample = discriminator_guidance(D, generated_sample, num_steps=0)
    num_samples = 10
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(10):
        axes[i].imshow(generated_sample[i].squeeze().detach().numpy(), cmap='gray')
        axes[i].axis('off')
    plt.savefig(f"images/gan_sample_020325_guided{0}.png")

    print("Done")

