import torch
import torch.nn.functional as F



class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1, dropout_p=0.2):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Dropout(p=dropout_p),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Dropout(p=dropout_p),
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))
        
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity
        
    def __init__(self, layers=[32,64,128], n_input_channels=3, n_output_channels=6,
        dropout_p=0.2):
        super().__init__()
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2, bias=False),
             torch.nn.BatchNorm2d(32),
             torch.nn.ReLU(),
             torch.nn.Dropout(p=dropout_p),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ]
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2, dropout_p=dropout_p))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)
    
    def forward(self, x):
        return self.classifier(self.network(x).mean(dim=[2, 3]))


# class CNNClassifier(torch.nn.Module):
#     def __init__(self, layers=[16, 32, 64, 128], n_input_channels=3, n_output_channels=6, kernel_size=5):
#         super().__init__()

#         L = []
#         c = n_input_channels
#         for l in layers:
#             L.append(torch.nn.Conv2d(c, l, kernel_size, stride=2, padding=kernel_size//2))
#             L.append(torch.nn.ReLU())
#             c = l
#         self.network = torch.nn.Sequential(*L)
#         self.classifier = torch.nn.Linear(c, n_output_channels)

#     def forward(self, x):
#         return self.classifier(self.network(x).mean(dim=[2, 3]))


# class CNNClassifier(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         """
#         Your code here
#         Hint: Base this on yours or HW2 master solution if you'd like.
#         Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
#         """
#         raise NotImplementedError('CNNClassifier.__init__')

#     def forward(self, x):
#         """
#         Your code here
#         @x: torch.Tensor((B,3,64,64))
#         @return: torch.Tensor((B,6))
#         Hint: Apply input normalization inside the network, to make sure it is applied in the grader
#         """
#         raise NotImplementedError('CNNClassifier.forward')




class FCN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1, dropout_p=0.2):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              # torch.nn.Dropout(p=dropout_p),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              # torch.nn.Dropout(p=dropout_p),
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))
        
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity
        
    def __init__(self, layers=[32,64], n_input_channels=3, n_output_channels=5, # <- 5 dense labels 
        dropout_p=0.2, input_width=64):
        super().__init__()
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2, bias=False),
             torch.nn.BatchNorm2d(32),
             torch.nn.ReLU(),
             torch.nn.Dropout(p=dropout_p),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ]
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2, dropout_p=dropout_p))
            c = l

        L.append(torch.nn.Conv2d(c, n_output_channels, kernel_size=1, bias=False))
        
        


        self.network = torch.nn.Sequential(*L)
        

        self.classifier = torch.nn.Linear(c, n_output_channels)
    
    def forward(self, x):

        z = self.network(x)

        U = []
        for l in range(x.shape[-1]-4):
            U.append(torch.nn.ConvTranspose2d(5, 5, kernel_size=4, padding=1, stride=1, bias=False))
        self.up = torch.nn.Sequential(*U)

        z = self.up(z)
        return z
        #return self.classifier(self.network(x).mean(dim=[2, 3]))


    # def __init__(self):
    #     super().__init__()
    #     """
    #     Your code here.
    #     Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
    #     Hint: Use up-convolutions
    #     Hint: Use skip connections
    #     Hint: Use residual connections
    #     Hint: Always pad by kernel_size / 2, use an odd kernel_size
    #     """
    #     raise NotImplementedError('FCN.__init__')

    # def forward(self, x):
    #     """
    #     Your code here
    #     @x: torch.Tensor((B,3,H,W))
    #     @return: torch.Tensor((B,6,H,W))
    #     Hint: Apply input normalization inside the network, to make sure it is applied in the grader
    #     Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
    #           if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
    #           convolution
    #     """
    #     raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
