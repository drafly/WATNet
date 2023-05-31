from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='train', help='train, test')
        self.parser.add_argument('--mode', type=str, default='train', help='train, val, test')
        self.parser.add_argument('--images_path', type=str, default="/data/dataset/wqq/mitUPE")
        self.parser.add_argument('--images_val_path', type=str, default="/data/dataset/wqq/mitUPE")