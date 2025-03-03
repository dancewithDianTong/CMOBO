import torch

class Design:
    def __init__(self):
        self.n_objectives = 2
        self.n_variables = 4
        self.n_constraints = 0
        self.n_original_constraints = 4
        
        # Define bounds
        self.bounds = torch.tensor([[55.0, 75.0, 1000.0, 11.0],[80.0, 110.0, 3000.0, 20.0]])

    def evaluate(self, x):
        # x is an n x 4 tensor
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First objective function (n x 1)
        f1 = 4.9e-5 * (x2**2 - x1**2) * (x4 - 1.0)
        
        # Second objective function (n x 1)
        f2 = (9.82e6 * (x2**2 - x1**2)) / (x3 * x4 * (x2**3 - x1**3))
        
        # Stack the objectives (n x 2)
        f = torch.stack([f1, f2], dim=1)
        
        # Original constraints (n x 4)
        g1 = (x2 - x1) - 20.0
        g2 = 0.4 - (x3 / (3.14 * (x2**2 - x1**2)))
        g3 = 1.0 - (2.22e-3 * x3 * (x2**3 - x1**3)) / torch.pow((x2**2 - x1**2), 2)
        g4 = (2.66e-2 * x3 * x4 * (x2**3 - x1**3)) / (x2**2 - x1**2) - 900.0

        # Reformulate the constraints (n x 4) with constraint violation
        g = torch.stack([g1, g2, g3, g4], dim=1)


        return -f, g
