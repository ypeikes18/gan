import torch as t
from torch import nn
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
        
    def grad_free_forward(self, x):
        with t.no_grad():
            return self.model(x)
        
    def sample(self):
        with t.no_grad():
            return self.model(t.randn(1, self.latent_dim))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(28*28, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
    
    def frozen_forward(self, x):
        for param in self.model.parameters():
            param.requires_grad = False
        output = self.model(x)
        for param in self.model.parameters():
            param.requires_grad = True
        return output


    
    
def train(G, D, data, epochs=1, batch_size=32, lr=1e-3, k=2, batches=200):
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    generator_optimizer = t.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    discriminator_optimizer = t.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    labels = t.concat([t.ones(batch_size,1), t.zeros(batch_size,1)], dim=0)
    for epoch in range(epochs):
        for i ,(batch, classes) in enumerate(data):
            if i >= batches:
                break
            print(f"Epoch {epoch}, Batch {i}")
              
            # TODO unhardcode
            generated = G.grad_free_forward(t.randn(batch_size, G.latent_dim))
            # TODO maybe shuffle this along with the labels
            pred = D(t.concat([batch, generated], dim=0))
            discriminator_optimizer.zero_grad()
            loss = t.nn.BCELoss()(pred, labels)
            print(f"Discriminator loss: {loss}")
            loss.backward()
            discriminator_optimizer.step()
            
            if i % k == 0:  
                pred = D.frozen_forward(G(t.randn(batch_size,G.latent_dim)))
                generator_optimizer.zero_grad()
                loss = t.nn.BCELoss()(pred, t.ones(batch_size,1))
                print(f"Generator loss: {loss}")
                loss.backward()
                generator_optimizer.step()
