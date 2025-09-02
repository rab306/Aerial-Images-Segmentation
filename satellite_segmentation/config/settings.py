class Config:
    def __init__(self):
        # Data settings
        self.directory = 'Satellite Images Semantic Segmentation/data/raw/'
        self.patch_size = 256
        self.num_channels = 3
        self.num_classes = 6
        
        # Training settings
        self.batch_size = 32
        self.epochs = 500
        self.verbose = 1
        self.seed = 42
        self.smooth = 1e-8
        
        # Loss function parameters 
        self.focal_alpha = 0.5
        self.focal_gamma = 1.0
        
        # File paths
        self.checkpoint_path = 'Unet.weights.h5'
        self.model_path = 'Unet_model.h5'

        # Metrics settings
        self.primary_monitor_metric = 'val_jaccard_coeff'
        self.secondary_monitor_metric = 'val_mean_iou'  
        self.patience = 100
        
        # Class information
        self.class_colors_hex = {
            'Building': '#3C1098',
            'Land': '#8429F6',
            'Road': '#6EC1E4',
            'Vegetation': '#FEDD3A',
            'Water': '#E2A929',
            'Unlabeled': '#9B9B9B'
        }