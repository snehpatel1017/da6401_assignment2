# Custom transform to handle image orientation
class RotateIfNeeded:
    def __call__(self, img):
        width, height = img.size
        if height > width:
            return img.transpose(2)  # Rotate 90 degrees if portrait
        return img

def center_crop_image(img, target_size=(400, 400)):
    """Crop the center of the image to the target size."""
    width, height = img.size
    target_width, target_height = target_size
    left = max(0, (width - target_width) // 2)
    top = max(0, (height - target_height) // 2)
    right = left + target_width
    bottom = top + target_height
    return img.crop((left, top, right, bottom))

def count_parameters(model):
    """Count the number of trainable and non-trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable

def freeze_all_layers_except_last(model):
    """Freeze all layers except the last fully connected layer"""
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last layer (classifier)
    for param in model.classifier[1].parameters():
        param.requires_grad = True
    
    return model

def freeze_first_k_layers(model, k):
    """Freeze first K layers, train the rest"""
    # First, unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Get list of all layers in features
    features = list(model.features)
    
    # Freeze the first k layers
    for i in range(min(k, len(features))):
        for param in features[i].parameters():
            param.requires_grad = False
    
    return model

def no_freezing(model):
    """Train all layers (no freezing)"""
    for param in model.parameters():
        param.requires_grad = True
    return model