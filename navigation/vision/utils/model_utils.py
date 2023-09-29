import torch

def get_output_shape(model, input_shape):
        # Compute the shape of the output tensor
        # `input_shape` should be the shape of the input tensor (batch_size, 3, C, H, W)
        with torch.no_grad():
            dummy_input = torch.randn(input_shape)
            output = model.forward(dummy_input)
        return output.shape[1:]  # Exclude batch dimension

def get_predicted_class(class_logits):
        """Returns the index that contains the highest class energy.
        """
        _, predicted = torch.max(class_logits, 1)
        return predicted