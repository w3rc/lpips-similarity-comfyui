import torch
import lpips
import numpy as np

class LPIPSSimilarityNode:
    """
    Computes LPIPS perceptual similarity between
    one input image and three reference images.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = lpips.LPIPS(net="alex").to(self.device)
        self.model.eval()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("score_ref1", "score_ref2", "score_ref3", "best_match_index")
    FUNCTION = "compute"
    CATEGORY = "Similarity"

    def _to_tensor(self, image):
        # IMAGE may be numpy or torch tensor depending on pipeline
        if isinstance(image, torch.Tensor):
            img = image
        else:
            img = torch.from_numpy(image)
    
        # ComfyUI IMAGE format: [B, H, W, C], range 0..1
        if img.ndim == 4:
            img = img.permute(0, 3, 1, 2)
        elif img.ndim == 3:
            img = img.permute(2, 0, 1).unsqueeze(0)
    
        img = img * 2.0 - 1.0  # normalize to [-1, 1]
        return img.to(self.device).float()


    def compute(self, input_image, reference_image_1, reference_image_2, reference_image_3):
        with torch.no_grad():
            inp = self._to_tensor(input_image)
            ref1 = self._to_tensor(reference_image_1)
            ref2 = self._to_tensor(reference_image_2)
            ref3 = self._to_tensor(reference_image_3)

            s1 = self.model(inp, ref1).mean().item()
            s2 = self.model(inp, ref2).mean().item()
            s3 = self.model(inp, ref3).mean().item()

            scores = [s1, s2, s3]
            best_idx = int(np.argmin(scores))  # lower = more similar

        return (s1, s2, s3, best_idx)


NODE_CLASS_MAPPINGS = {
    "LPIPSSimilarity": LPIPSSimilarityNode
}
