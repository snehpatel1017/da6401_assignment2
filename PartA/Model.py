import torch
import torch.nn as nn
class FlexibleCNN(nn.Module):
  
    def __init__(self, in_channels=3, num_filters=32, filter_size=3,
                 activation_fn=nn.ReLU, filter_organization='same',
                 use_batchnorm=False, dropout_rate=0.0,
                 dense_neurons=128, num_classes=10, input_size=(224, 224)):
        """
        Initialize the CNN model
        
        Args:
            in_channels: Number of input channels (3 for RGB images)
            num_filters: Base number of filters in convolutional layers
            filter_size: Size of convolutional filters
            activation_fn: Activation function to use
            filter_organization: Strategy for organizing filter counts across layers
                                ('same', 'double', or 'half')
            use_batchnorm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            dense_neurons: Number of neurons in the dense layer
            num_classes: Number of output classes
            input_size: Size of input images
        """
        super(FlexibleCNN, self).__init__()

        # Determine filter counts based on organization strategy
        if filter_organization == 'same':
            filters = [num_filters] * 5
        elif filter_organization == 'double':
            filters = [num_filters * (2**i) for i in range(5)]
        elif filter_organization == 'half':
            filters = [num_filters // (2**(i)) for i in range(5)]
            filters = [max(16, f) for f in filters]  # Ensure minimum filter count

        # Build layers list
        layers = []

        # First conv block
        layers.append(nn.Conv2d(in_channels, filters[0], kernel_size=filter_size, padding='same'))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(filters[0]))
        layers.append(activation_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Remaining conv blocks
        for i in range(1, 5):
            layers.append(nn.Conv2d(filters[i-1], filters[i], kernel_size=filter_size, padding='same'))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(filters[i]))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)

        # Calculate output dimensions dynamically
        with torch.no_grad():
            x = torch.randn(1, in_channels, *input_size)
            x = self.features(x)
            self.flattened_size = x.numel() // x.size(0)

        # Classifier
        classifier_layers = [
            nn.Flatten(),
            nn.Linear(self.flattened_size, dense_neurons),
            activation_fn()
        ]

        if dropout_rate > 0:
            classifier_layers.append(nn.Dropout(dropout_rate))

        classifier_layers.append(nn.Linear(dense_neurons, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        """Forward pass through the network"""
        x = self.features(x)
        x = self.classifier(x)
        return x
