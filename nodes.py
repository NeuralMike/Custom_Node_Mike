import torch
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from torch.utils.tensor import tensor
from torch.utils.tensor import is_tensor

def pil2tensor(pil_image, dtype=torch.float32, normalize=False):
    """
    Converts a PIL Image object to a PyTorch tensor.

    Args:
    - pil_image (PIL.Image.Image): The PIL Image object to convert.
    - dtype (torch.dtype, optional): The desired data type of the returned tensor. Defaults to torch.float32.
    - normalize (bool, optional): If True, the returned tensor is divided by 255.0. Defaults to False.

    Returns:
    torch.Tensor: The converted PyTorch tensor.
    """
    if is_tensor(pil_image):
        return pil_image

    if pil_image.mode == 'L':
        mode = 'grayscale'
    elif pil_image.mode == 'RGB':
        mode = 'RGB'
    elif pil_image.mode == 'RGBA':
        mode = 'RGBA'

    image = Image.fromarray(np.asarray(pil_image), mode=pil_image.mode)
    image = image.convert('RGB')

    if normalize:
        return tensor(np.array(image) / 255.0, dtype=dtype)
    else:
        return tensor(np.asarray(image), dtype=dtype)

class ConcatenateSentences(nn.Module):
    """
    A custom PyTorch module for concatenating two input sentences.

    Attributes:
    - input_1 (str): The first input sentence.
    - input_2 (str): The second input sentence.
    - output (str): The concatenated output sentence.
    """
    def __init__(self):
        super(ConcatenateSentences, self).__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"input_1": ("STRING", {}), "input_2": ("STRING", {})}}

    @classmethod
    def RETURN_TYPES(cls):
        return {"output": ("STRING", {})}

    def forward(self, input_1, input_2):
        self.input_1 = input_1
        self.input_2 = input_2

        # Concatenate the input sentences
        self.output = f"{self.input_1} {self.input_2}"

        return self.output
