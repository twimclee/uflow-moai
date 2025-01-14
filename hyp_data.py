class MHyp:

    def __init__(self):
        self.input_size = None
        self.flow_step = None
        self.backbone = None
        self.epochs = None
        self.batch_train = None
        self.batch_val = None
        self.learning_rate = None
        self.weight_decay = None
        self.log_every_n_epochs = None
        self.save_ckpt_every = None
        self.save_debug_images_every = None
        self.log_predefined_debug_images = None
        self.log_n_images = None
        self.patience = None

        self.brightness = None
        self.contrast = None
        self.saturation = None
        self.hue = None
        self.hflip = None
        self.vflip = None
        self.rotation = None

    def print_data(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

class MData:

    def __init__(self):
        self.names = None

    def print_data(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

