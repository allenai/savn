import h5py


class Glove:
    def __init__(self, glove_file):
        self.glove_embeddings = h5py.File(glove_file, "r")
