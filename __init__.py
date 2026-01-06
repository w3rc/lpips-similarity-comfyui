from dotenv import load_dotenv
from similarity import LPIPSSimilarityNode

NODE_CLASS_MAPPINGS = {
    "GetSimilarity": LPIPSSimilarityNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LPIPSSimilarity": "LPIPS Image Similarity"
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]