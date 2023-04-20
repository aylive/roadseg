import os
import random
from PIL import Image

import pandas as pd

# Set the input directory where the images are located
input_dir = '/home/aylive/data/peng/data/20cities'
output_dir = '/home/aylive/workspace/peng_pku/samples'

# Set the size of the tiles
tile_size = 352

# Set the propotion of train, val, test size (%)
train_size, val_size = 80, 5    # test_size = 100 - train_size - val_size

if __name__ == "__main__":

    # Set the output directory where the cropped tiles will be saved
    output_dir = output_dir + f'_tiles_{tile_size}'
    for sub_dir in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, sub_dir))

    #TODO: image labeling
    # (1) extract image label from OSM
    # (2) set the road buffer as 5m

    # image tiling
    # (1) rolling tile 
    # (2) upsample to specified size
    # (3) inplace split the tiles into train, val & test
    # (4) save tiles & meta-data
    #TODO: spatial correlation
    #TODO: imbalance adjusting
    metadata = []

    # For 20cities dataset
    for img_name in os.listdir(input_dir):
        # Only process relevant images
        if img_name.endswith('sat.png'):
            # Extract the reigon info
            region = img_name.split('_')[1]

            # Find corresponding ground truth image
            mask_name = 'gt'.join(img_name.split('sat'))

            # Open image & corresponding mask
            img = Image.open(os.path.join(input_dir, img_name))
            mask = Image.open(os.path.join(input_dir, mask_name))

            if img.size != mask.size:
                raise ValueError('Incompatible size between image and mask.')

            width, height = img.size
            width_step, height_step = tile_size, tile_size

            # Loop through each tile
            for w in range(tile_size, width, width_step):
                for h in range(tile_size, height, height_step):
                    # Locate the tile relative to the original image
                    left = w - tile_size
                    upper = h - tile_size
                    right = w
                    lower = h

                    # Crop the tile of specified size
                    tile_img = img.crop((left, upper, right, lower))
                    tile_mask = mask.crop((left, upper, right, lower))

                    # In-place train-test-val splitting
                    split_num = random.randint(0, 100)

                    if split_num <= train_size:
                        split = 'train'
                    elif split_num <= (train_size + val_size):
                        split = 'valid'
                    else:
                        split = 'test'

                    # Generate the path to save tile
                    pth = os.path.join(output_dir, split)
                    tile_img_pth = os.path.join(pth, f'region_{region}_{right}_{lower}_sat.png')
                    tile_mask_pth = os.path.join(pth, f'region_{region}_{right}_{lower}_mask.png')
                    
                    # Save tiles 
                    tile_img.save(tile_img_pth)
                    tile_mask.save(tile_mask_pth)

                    # Save the tile info as metadata
                    metadata.append(
                        pd.DataFrame({
                            'region': f'region_{region}',
                            'split': split,
                            'sat_img_pth': os.path.join(split, tile_img_pth),
                            'mask_pth': os.path.join(split, tile_mask_pth),
                        }, index=[len(metadata)])
                    )

    # Concatnate & Save the metadata of tiles
    metadata = pd.concat(metadata, axis=0)
    metadata.to_csv(
        os.path.join(output_dir, 'metadata.csv'), index=False,
    )

    for split in ['train', 'valid', 'test']:
        sub_metadata = metadata[metadata['split'] == split]
        sub_metadata.to_csv(
            os.path.join(output_dir, f'metadata_{split}.csv'), index=False,
        )
