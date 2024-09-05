import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)

#------------------------------------------- TRAINING VARIABLES -------------------------------------------#    
CRITERION = nn.BCEWithLogitsLoss()
N_EPOCH = 100
Z_DIM = 64
DISPLAY_STEP = 469
BATCH_SIZE = 128
LEARNING_RATE = 0.00001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#------------------------------------------- DATASET -------------------------------------------#

MNIST_dataset = MNIST(
            "MNIST",
            download = False,
            transform = transforms.ToTensor()
        )


dataloader = DataLoader(
    MNIST_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)


def show_tensor_images(image_tensor, num_images = 25, size = (1, 28, 28)):
    '''
    Shows images from the dataset
    '''
    
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


#------------------------------------------- GENERATOR -------------------------------------------#    
def get_generator_block(input_dims: int, output_dims: int):
    """
    Given a functions input and output dimensions, will generate a neural network "block" with the following layers:
        Linear
        BatchNormalization
        ReLU
    """

    return nn.Sequential(
        nn.Linear(input_dims, output_dims),
        nn.BatchNorm1d(output_dims),
        nn.ReLU(inplace = True)
    )

# Child of the nn.module class
class Generator(nn.Module):
    """
    Generator Class:

        Dimensions:
            z_dim - Dimension of the noise vector
            im_dim - Dimension of the image vector
                (since MNIST images are 784, im_dim should be 784)
            hidden_dim - Dimension of the hidden layers
    """

    def __init__(self, z_dim:int = 10, im_dim:int = 784, hidden_dim:int = 128):
        super(Generator, self).__init__() # Is there a point to having parameters filled out?
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()            
        )

    def forward(self, noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen
    

def get_noise(n_samples:int, z_dims:int, device = "cpu"):
    '''
    Returns a (n_samples x z_dims) vector of random numbers selected from a 
    normal distribution
    '''
    return torch.randn(n_samples, z_dims, device = device)

#------------------------------------------- DISCRIMINATOR -------------------------------------------#

def get_discriminator_block(input_dims:int, output_dims:int):
    '''
    Returns a block of a Discriminator NN
    Consists of a linear layer followed by a leaky ReLu layer w/ -0.2 slope
    '''
    return nn.Sequential(
        nn.Linear(input_dims, output_dims),
        nn.LeakyReLU(0.2, inplace = True)
    )

class Discriminator(nn.Module):
    '''
    Discriminator Neural Network:
        Takes in a generated image and outputs the likelyhood of it being real or generated
    '''

    def __init__(self, im_dim:int = 784, hidden_dim:int = 128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1) # Note that no sigmoid function is neccesary due to it being within the loss function
        )

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc
    
#------------------------------------------- TRAINING -------------------------------------------#

gen = Generator(Z_DIM).to(device = device)
gen_opt = torch.optim.Adam(gen.parameters(), lr = LEARNING_RATE)
disc = Discriminator().to(device = device)
disc_opt = torch.optim.Adam(disc.parameters(), lr = LEARNING_RATE)

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''

    # Creating fake images
    noise = get_noise(num_images, z_dim, device)
    generated_images = gen(noise)

    # Making predictions for fake images
    fake_pred = disc(generated_images.detach())
    fake_ground_truth = torch.zeros_like(fake_pred)

    fake_loss = criterion(fake_pred, fake_ground_truth)

    # Making predictions for real images
    real_pred = disc(real)
    real_ground_truth = torch.ones_like(real_pred)

    real_loss = criterion(real_pred, real_ground_truth)
    
    disc_loss = (real_loss + fake_loss)/2

    return disc_loss

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    
    # Creating fake images
    noise = get_noise(num_images, z_dim, device)
    generated_images = gen(noise)

    # Getting predictions on fake images
    pred = disc(generated_images)
    
    # Basing loss off how similar fake images are classified as real
    real = torch.ones_like(pred)
    gen_loss = criterion(pred, real)

    return gen_loss

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True
gen_loss = False
error = False

# For each epoch
for epoch in range(N_EPOCH):

    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device) # Reshapes the input pictures to be (batch_size x 784)

        ### Updating the discriminator ###

        # Zeroing out the discrimator gradiants before backpropagation
        disc_opt.zero_grad() 

        # Getting the discriminator loss
        disc_loss = get_disc_loss(gen, disc, CRITERION, real, cur_batch_size, Z_DIM, device = device)

        # Update Gradiants
        disc_loss.backward(retain_graph = True)

        # Backpropagate
        disc_opt.step()

        ### Updating the generator ###

        # Zeroing out the generator gradiants before backpropagation
        gen_opt.zero_grad()
        
        # Getting the generator loss
        gen_loss = get_gen_loss(gen, disc, CRITERION, cur_batch_size, Z_DIM, device = device)

        # Updating gradiants
        gen_loss.backward(retain_graph = True)

        # Backpropagate
        gen_opt.step()

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / DISPLAY_STEP

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / DISPLAY_STEP

        ### Visualization code ###
        if cur_step % DISPLAY_STEP == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, Z_DIM, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1        


