# Global setting
directory = 'Satellite Images Semantic Segmentation/data/raw/'

patch_size = 256
num_channels = 3
num_classes = 6
batch_size = 32
smooth=1e-8
epochs = 150
verbose = 1
seed = 42

checkpoint_path = 'Unet.weights.h5'
model_path = 'Unet_model.h5'

class_colors_hex = {
    'Building': '#3C1098',
    'Land': '#8429F6',
    'Road': '#6EC1E4',
    'Vegetation': '#FEDD3A',
    'Water': '#E2A929',
    'Unlabeled': '#9B9B9B'
}

data_gen_args = dict(
    rotation_range=45.0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

