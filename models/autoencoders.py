import torch.nn as nn
import torch as tc

class AutoencoderConvolutional(nn.Module):

    def __init__(self) -> None:
        
        super(AutoencoderConvolutional, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), # 8, 64, 64
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # 16, 64, 64
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 32, 64, 64 
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    
class Model_1(nn.Module):

    def __init__(self) -> None:
        
        super(Model_1, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    

class Model_2(nn.Module):

    def __init__(self) -> None:
        
        super(Model_2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),    
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    


class Model_with_Max_pool(nn.Module):

    def __init__(self) -> None:
        
        super(Model_with_Max_pool, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        
        decoded = self.decoder(encoded)

        return decoded
    
class Model_with_Max_pool_2(nn.Module):

    def __init__(self) -> None:
        
        super(Model_with_Max_pool_2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        
        decoded = self.decoder(encoded)

        return decoded


class Model_with_Max_pool_3(nn.Module):

    def __init__(self) -> None:
        
        super(Model_with_Max_pool_3, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.SiLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        
        decoded = self.decoder(encoded)

        return decoded

class Small_Model_with_Max_pool(nn.Module):

    def __init__(self) -> None:
        
        super(Small_Model_with_Max_pool, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),  
            
        )

        self.decoder = nn.Sequential(
            
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        
        decoded = self.decoder(encoded)

        return decoded   

class Small_Model_Fully_Conv(nn.Module):

    def __init__(self) -> None:
        
        super(Small_Model_Fully_Conv, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),  
            
        )

        self.decoder = nn.Sequential(
            
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        
        decoded = self.decoder(encoded)

        return decoded 

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(8, 16, 
							kernel_size=3, 
							stride=2, 
							padding=1, 
							output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(16, 1, 
							kernel_size=3, 
							stride=2, 
							padding=1, 
							output_padding=1),
			nn.Sigmoid()
		)
		
	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

class Model_Linear(nn.Module):

    def __init__(self) -> None:
        
        super(Model_Linear, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(60*60, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 60*60),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    