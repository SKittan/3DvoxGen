import torch

class CAclouds3D():
    """ Cloud simulation by Cellular Automaton for creating nice clouds.
    Source:
    Application of cellular automata approach for cloud simulation and rendering
    Chaos 24, 013125 (2014); https://doi.org/10.1063/1.4866854
    W. Christopher Immanuel and S. Paul Mary Deborrah and R. Samuel Selvaraj
    """
    def __init__(self, width, depth, height, device='cpu', *args, **kwargs):
        """ Initialsation

        Arguments:
            width {int} -- Grid size in x-direction
            depth {int} -- Grid size in y-direction
            height {int} -- Grid size in z-direction

        Keyword Arguments:
            device {str} -- Device on which the calculation is done ('cpu', 'cuda', 'cuda:0', ...) (default: {'cpu'})

        Returns:
            CAcloud3D -- Object for 3D cloud creation/simulation
        """
        self.width = width
        self.depth = depth
        self.height = height
        self.dev = torch.device(device)
        # one-time init for all coordinate combinations
        self.x, self.y, self.z = torch.meshgrid(torch.arange(0, self.width, device=self.dev),
                                                torch.arange(0, self.depth, device=self.dev),
                                                torch.arange(0, self.height, device=self.dev)
                                               )

        # init CA grids
        self.hum = torch.zeros((width, depth, height), device=self.dev, dtype=torch.uint8) # humidity/vapor
        self.act = torch.zeros((width, depth, height), device=self.dev, dtype=torch.uint8) # activation/phase transition factor
        self.cld = torch.zeros((width, depth, height), device=self.dev, dtype=torch.uint8) # clouds
        self.f_act = torch.zeros_like(self.act) # activation factor -> helper variable for calculation
        self.hum_temp = torch.zeros_like(self.hum) # temporary tensor for humidity, since it is also used for act calculation
        # Variables for formation and extinction process
        # reserve memory for random number creation
        self.rnd_hum = torch.zeros_like(self.hum, dtype=torch.int16)
        self.rnd_act = torch.zeros_like(self.act, dtype=torch.int16)
        self.rnd_ext = torch.zeros_like(self.cld, dtype=torch.int16)
        # probability areas for random variable changes
        self.P_hum = torch.zeros_like(self.hum, dtype=torch.int16)
        self.P_act = torch.zeros_like(self.act, dtype=torch.int16)
        self.P_ext = torch.zeros_like(self.cld, dtype=torch.int16)

        return super().__init__(*args, **kwargs)

    def init_elliptic_probabilities(self, c_x, c_y, c_z, f_x, f_y, f_z,
                                    P_hum0, P_act0, P_ext0,
                                    radius=1., overlap=1.):
        """ Init elliptic propability distribution for formation and extinction process
        Clouds are created in defined volume and extinct in the outside volume
        In the overlapping volume formation and extinction occurs
        Humidity will be created in the complete volume, for better cloud distribution / growth

        Elliptic volume: (x-c_x)**2/f_x + (y-c_y)**2/f_y + (z-c_z)**2/f_z <= radius + overlap

        Arguments:
            c_x {int} -- Center position of elliptical volume (x-direction) [0...width-1]
            c_y {int} -- Center position of elliptical volume (y-direction) [0...depth-1]
            c_z {int} -- Center position of elliptical volume (z-direction) [0...height-1]
            f_x {int} -- Stretch factor for elliptical volume (x-direction) [>0]
            f_y {int} -- Stretch factor for elliptical volume (y-direction) [>0]
            f_z {int} -- Stretch factor for elliptical volume (z-direction) [>0]
            P_hum0 {int16} -- Probability for a cell getting humiditiy status = 1
                              (0 is equal to 0.00% and 10000 is equal to 100.00% Probability)
            P_act0 {int16} -- Probability for a cell getting activation factor = 1
                              (0 is equal to 0.00% and 10000 is equal to 100.00% Probability)
            P_ext0 {int16} -- Probability for a cell getting cloud status = 0
                              (0 is equal to 0.00% and 10000 is equal to 100.00% Probability)
        Keyword Arguments:
            radius {float} -- Default distance for ellipsoid calculation (default: 1.)
            overlap {float} -- Overlapping of formation and extinction volume (default: 1.)
        """
        if overlap < 0.:
            overlap = 0.

        if c_x < 0.:
            print("c_x has to be in range of 0 and %i, hence it was set to 0." % (self.width-1))
            c_x = 0
        elif c_x >= self.width:
            print("c_x has to be in range of 0 and %i, hence it was set to %i." % (self.width-1, self.width-1))
            c_x = self.width - 1

        if c_y < 0.:
            print("c_y has to be in range of 0 and %i, hence it was set to 0." % (self.depth-1))
            c_y = 0
        elif c_y >= self.width:
            print("c_y has to be in range of 0 and %i, hence it was set to %i." % (self.depth-1, self.depth-1))
            c_y = self.depth - 1

        if c_z < 0.:
            print("c_z has to be in range of 0 and %i, hence it was set to 0." % (self.height-1))
            c_z = 0
        elif c_z >= self.width:
            print("c_z has to be in range of 0 and %i, hence it was set to %i." % (self.height-1, self.height-1))
            c_z = self.height - 1

        if f_x <= 0:
            print("f_x has to be greater than 0, hence it is ignored.")
            f_x = 1.

        if f_y <= 0:
            print("f_y has to be greater than 0, hence it is ignored.")
            f_y = 1.

        if f_z <= 0:
            print("f_z has to be greater than 0, hence it is ignored.")
            f_z = 1.

        if P_hum0 < 0:
            print("P_hum0 has to be greater or equal to 0, hence it is set to 0.")
            P_hum0 = 0
        elif P_hum0 > 10000:
            print("P_hum0 has to be smaller or equal to 10000, hence it is set to 10000.")
            P_hum0 = 10000

        if P_act0 < 0:
            print("P_act0 has to be greater or equal to 0, hence it is set to 0.")
            P_act0 = 0
        elif P_act0 > 10000:
            print("P_act0 has to be smaller or equal to 10000, hence it is set to 10000.")
            P_act0 = 10000

        if P_ext0 < 0:
            print("P_ext0 has to be greater or equal to 0, hence it is set to 0.")
            P_ext0 = 0
        elif P_ext0 > 10000:
            print("P_ext0 has to be smaller or equal to 10000, hence it is set to 10000.")
            P_ext0 = 10000

        # create selection masks
        distance = (self.x-c_x)**2/f_x + (self.y-c_y)**2/f_y + (self.z-c_z)**2/f_z
        # inner
        sel_inner = distance <= radius + overlap
        # outer
        sel_outer = distance > radius - overlap

        self.P_hum = P_hum0 # humidity for complete volume
        self.P_act[sel_inner] = P_act0
        self.P_ext[sel_outer] = P_ext0

    def __cloud_growth__(self):
        """ Application of CA rules to hum, act, cld grids
        """
        # calculate new hum
        self.hum_temp = self.hum & ~self.act
        # calculate new cld
        self.cld = self.cld | self.act
        # calculate new activation factor
        self.f_act = (
                torch.cat((self.act[-2:,:,:], self.act[:-2,:,:]), dim=0) | torch.cat((self.act[-1:,:,:], self.act[:-1,:,:]), dim=0) |
                torch.cat((self.act[1:,:,:], self.act[:1,:,:]), dim=0) | torch.cat((self.act[2:,:,:], self.act[:2,:,:]), dim=0) |
                torch.cat((self.act[:,-2:,:], self.act[:,:-2,:]), dim=1) | torch.cat((self.act[:,-1:,:], self.act[:,:-1,:]), dim=1) |
                torch.cat((self.act[:,1:,:], self.act[:,:1,:]), dim=1) |
                torch.cat((self.act[:,:,-2:], self.act[:,:,:-2]), dim=2) | torch.cat((self.act[:,:,-1:], self.act[:,:,:-1]), dim=2) |
                torch.cat((self.act[:,:,1:], self.act[:,:,:1]), dim=2) | torch.cat((self.act[:,:,2:], self.act[:,:,:2]), dim=2)
                )
        # calculate new act
        self.act = ~self.act & self.hum & self.f_act
        self.act2 = self.act
        # copy hum_temp to hum
        self.hum = self.hum_temp

    def __cloud_FormationExtinction__(self):
        """ Apllication of formation and extinction rules
        """
        # update random values
        self.rnd_hum.random_(0, 10001)
        self.rnd_act.random_(0, 10001)
        self.rnd_ext.random_(0, 10001)
        # update cell states
        self.hum = self.hum | (self.rnd_hum < self.P_hum)
        self.act = self.act | (self.rnd_act < self.P_act)
        self.cld = self.cld & (self.rnd_ext > self.P_ext)

    def step(self):
        """ Execute one simulation step (for external simulation loops)
        (application of growth and formation/extinction rules)
        """
        self.__cloud_growth__()
        self.__cloud_FormationExtinction__()

    def simulate(self, n_iterations):
        """ Execute simulation steps
        (application of growth and formation/extinction rules)

        Arguments:
            n_iterations {int} -- Number of steps to be executed
        """
        for i in range(n_iterations):
            self.__cloud_growth__()
            self.__cloud_FormationExtinction__()

    def get_cloud_positions(self):
        """Calculate a 2D Vector of x/y/z-Positions for all Cells with cld state equal to 1.

        Returns:
            [tensor] -- [[x0, y0, z0], [x1, y1, z1], ...] Positions of cld=1 cells
        """
        # create selection mask
        selection = self.cld.reshape(1, -1).squeeze() == 1
        # flatten coordinate arrays
        x = self.x.reshape(1, -1).squeeze()[selection]
        y = self.y.reshape(1, -1).squeeze()[selection]
        z = self.z.reshape(1, -1).squeeze()[selection]

        return torch.cat((x, y, z), -1).view(3, -1).transpose(0,1)