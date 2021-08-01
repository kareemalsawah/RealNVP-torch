class CelebADataset():
    def __init__(self,images):
        self.images = images
    
    def __len__(self):
        return self.images.shape[0]
    
    def __idx__(self,idx):
        return self.images[idx]