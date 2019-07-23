import os
from sigver.datasets.base import IterableDataset
from skimage.io import imread
from skimage import img_as_ubyte

MAX_X = 3000
MAX_Y = 3000

class PersianDataset(IterableDataset):
    """ Helper class to load the brazilian PUC-PR dataset
    """
    def __init__(self, path, file_extension='PNG'):
        self.path = path
        self.users = list(range(1, 115))

        self.file_extension = file_extension

    @property
    def genuine_per_user(self):
        return 27

    @property
    def skilled_per_user(self):
        return 6

    @property
    def simple_per_user(self):
        return 35

    @property
    def maxsize(self):
        return MAX_X, MAX_Y

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        genuine_files = ['C{:03d}G{:02d}.{}'.format(user, img, self.file_extension)
                         for img in range(1, self.genuine_per_user+1)]
        for f in genuine_files:
            full_path = os.path.join(self.path, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""
        forgery_files = ['C{:03d}F{:02d}.{}'.format(user, img, self.file_extension)
                         for img in range(4, self.skilled_per_user+4)]
        for f in forgery_files:
            full_path = os.path.join(self.path, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        """ Iterate over simple forgeries for the given user"""
        forgery_files = ['C{:03d}F{:02d}.{}'.format(user, img, self.file_extension)
                         for img in range(11, self.simple_per_user+11)]
        for f in forgery_files:
            full_path = os.path.join(self.path, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f