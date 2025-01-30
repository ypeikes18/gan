import torch as t
import matplotlib.pyplot as plt
from gan import Generator, Discriminator, train
from guidance import discriminator_guidance
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


    i = 6
    G.load_state_dict(t.load(f"weights/generator_weights_{i}.pth"))
    D.load_state_dict(t.load(f"weights/discriminator_weights_{i}.pth"))

    # train(G, D, data, epochs=1, batch_size=64, lr=4e-4, k=1)

    # t.save(G.state_dict(), f"weights/generator_weights_{i+1}.pth")
    # t.save(D.state_dict(), f"weights/discriminator_weights_{i+1}.pth")
    
    generated_sample = G.sample(10)
    generated_sample = discriminator_guidance(D, generated_sample, num_steps=50)
    num_samples = 10
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(10):
        axes[i].imshow(generated_sample[i].squeeze().detach().numpy(), cmap='gray')
        axes[i].axis('off')
    plt.savefig(f"images/gan_sample_{6}_guided{50}.png")

    print("Done")

