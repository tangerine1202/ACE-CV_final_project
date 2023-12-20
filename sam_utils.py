from tqdm.auto import tqdm
import numpy as np
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

cfg = {
    'l0': 'assets/checkpoints/sam/l0.pt',
    'l1': 'assets/checkpoints/sam/l1.pt',
    'l2': 'assets/checkpoints/sam/l2.pt',
}


class SAM:
    def __init__(self, model_name, device='cpu'):
        efficientvit_sam = create_sam_model(
            name=model_name, weight_url=cfg[model_name], pretrained=True).to(device).eval()
        efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
        self.sam = efficientvit_sam
        self.predictor = efficientvit_sam_predictor

    def point_prompt(self, img, point_coords, point_labels):
        '''
        point_coords (np.ndarray or None): 
            A Nx2 array of point prompts to the model. 
            Each point is in (X,Y) in pixels.
        point_labels (np.ndarray or None): 
            A length N array of labels for the point prompts. 
            1 indicates a foreground point and 0 indicates a background point.
        '''
        assert img.shape[2] == 3, 'image shape should be (H, W, 3)'
        self.predictor.set_image(img)
        masks, _, _ = self.predictor.predict(
            point_coords,
            point_labels,
            multimask_output=False,
        )
        return masks
