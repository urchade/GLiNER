import torch

def pad_2d_tensor(key_data):
    """
    Pad a list of 2D tensors to have the same size along both dimensions.
    
    :param key_data: List of 2D tensors to pad.
    :return: Tensor of padded tensors stacked along a new batch dimension.
    """
    if not key_data:
        raise ValueError("The input list 'key_data' should not be empty.")

    # Determine the maximum size along both dimensions
    max_rows = max(tensor.shape[0] for tensor in key_data)
    max_cols = max(tensor.shape[1] for tensor in key_data)
    
    tensors = []

    for tensor in key_data:
        rows, cols = tensor.shape
        row_padding = max_rows - rows
        col_padding = max_cols - cols

        # Pad the tensor along both dimensions
        padded_tensor = torch.nn.functional.pad(tensor, (0, col_padding, 0, row_padding),
                                                                 mode='constant', value=0)
        tensors.append(padded_tensor)

    # Stack the tensors into a single tensor along a new batch dimension
    padded_tensors = torch.stack(tensors)

    return padded_tensors