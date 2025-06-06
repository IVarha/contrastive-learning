import random
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class FragmentDataset(Dataset):
    def __init__(self, root,limit=None,image_size=64,
                 patch_size=16, fragments_per_image=16, augment=False,augmenter = None):

        self.images = []#[Image.open(x).convert("RGB") for x in Path(root).glob("*.png")]
        i = 0
        for x in tqdm(list(Path(root).glob("*.png"))):
            self.images.append(Image.open(x).convert("RGB"))
            i += 1
            if limit is not None and i >= limit:
                break
        self.patch_size = patch_size
        self.image_size = image_size
        self.fragments_per_image = fragments_per_image
        self.augment = augment
        self.augmenter = augmenter
        #self.files = list(Path(root).glob("*.png"))  # or '*.jpg'
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor()
        ])

    def get_full_image(self, idx):
        img = self.images[idx]
        try:
            img = self.transform(img)
        except Exception as e:
            print(f"Error transforming image {idx}: {e}")
            print(f"Image shape: {img.size}")
        return img


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(idx, list): # If idx is a list, return a list of items
            return [self.__getitem__(i) for i in idx]
        img = self.images[idx]
        try:
            img = self.transform(img)
        except Exception as e:
            print(f"Error transforming image {idx}: {e}")
            print(f"Image shape: {img.size}")

        fragments = []

        # Compute number of non-overlapping positions along H and W
        positions = []
        # For example, if IMAGE_SIZE=64 and PATCH_SIZE=4, we have 16 positions along each dimension
        # and 16 * 16 = 256 total positions
        for i in range(0, self.image_size - self.patch_size + 1, self.patch_size):
            for j in range(0, self.image_size - self.patch_size + 1, self.patch_size):
                positions.append((i, j))

        # Randomly sample without replacement
        #print(f"Positions: {positions}")
        selected_positions = random.sample(positions, self.fragments_per_image)

        for (i, j) in selected_positions:
            fragment = img[:, i:i+self.patch_size, j:j+self.patch_size]
            # Apply augmentation if specified
            if self.augment:
                if self.augmenter is not None:
                    try:
                        fragment = self.augmenter(fragment)
                    except Exception as e:
                        print(f"Error applying augmenter to fragment {idx}: {e}")
                        print(f"Fragment shape: {fragment.shape}")
            fragments.append(fragment)
        try:
            fragments = torch.stack(fragments)
        except Exception as e:
            print(f"Error stacking fragments for image {idx}: {e}")
            print(f"Fragment shapes: {[f.shape for f in fragments]}")

            # If stacking fails, return empty tensor
        return fragments, idx  # synthetic label

    def __getitems__(self, indices):
        return [self.__getitem__(i) for i in indices]