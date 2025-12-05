"""
Weights Manager to download and manage weights from Hugging Face
"""

import os
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightsManager:
    def __init__(self, repo_id: str = "2uanDM/complex-surveillance-ai-system", cache_dir: str = "weights"):
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.weights_info = {}
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_available_weights(self) -> Dict[str, List[str]]:
        """Get list of available weights from the Hugging Face repo"""
        try:
            files = list_repo_files(self.repo_id)
            
            # Filter for classifier weights
            classifier_files = [f for f in files if f.endswith("_classifier.pth")]
            
            # Group by model architecture
            weights_by_model = {
                "mobilenet": [],
                "efficientnet_b0": [],
                "efficientnet_b4": []
            }
            
            for file in classifier_files:
                if file.startswith("mobilenet_"):
                    task = file.replace("mobilenet_", "").replace("_classifier.pth", "")
                    weights_by_model["mobilenet"].append(task)
                elif file.startswith("efficientnet_b0_"):
                    task = file.replace("efficientnet_b0_", "").replace("_classifier.pth", "")
                    weights_by_model["efficientnet_b0"].append(task)
                elif file.startswith("efficientnet_b4_"):
                    task = file.replace("efficientnet_b4_", "").replace("_classifier.pth", "")
                    weights_by_model["efficientnet_b4"].append(task)
            
            self.weights_info = weights_by_model
            return weights_by_model
            
        except Exception as e:
            logger.error(f"Error getting available weights: {e}")
            return {}
    
    def download_weight(self, model_type: str, task: str, force_download: bool = False) -> Optional[str]:
        """Download specific weight file"""
        try:
            filename = f"{model_type}_{task}_classifier.pth"
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                cache_dir=self.cache_dir,
                force_download=force_download
            )
            
            logger.info(f"✅ Downloaded {filename} to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"❌ Error downloading {model_type}_{task}_classifier.pth: {e}")
            return None
    
    def download_all_weights(self, force_download: bool = False) -> Dict[str, Dict[str, str]]:
        """Download all weights"""
        weights_info = self.get_available_weights()
        downloaded_weights = {}
        
        for model_type, tasks in weights_info.items():
            downloaded_weights[model_type] = {}
            for task in tasks:
                local_path = self.download_weight(model_type, task, force_download)
                if local_path:
                    downloaded_weights[model_type][task] = local_path
        
        return downloaded_weights
    
    def load_model_state_dict(self, model_type: str, task: str) -> Optional[Dict]:
        """Load state dict from weight file"""
        try:
            # Download if not available
            weight_path = self.download_weight(model_type, task)
            if not weight_path:
                return None
            
            # Load state dict
            state_dict = torch.load(weight_path, map_location="cpu")
            logger.info(f"✅ Loaded state dict for {model_type}_{task}")
            return state_dict
            
        except Exception as e:
            logger.error(f"❌ Error loading state dict for {model_type}_{task}: {e}")
            return None
    
    def get_weight_info(self) -> Dict[str, Dict[str, Dict]]:
        """Get detailed information about weights"""
        weights_info = {}
        
        try:
            files = list_repo_files(self.repo_id)
            classifier_files = [f for f in files if f.endswith("_classifier.pth")]
            
            for file in classifier_files:
                # Parse filename
                if "_" in file:
                    parts = file.replace("_classifier.pth", "").split("_")
                    if len(parts) >= 2:
                        model_type = "_".join(parts[:-1])  # efficientnet_b0, efficientnet_b4, mobilenet
                        task = parts[-1]  # gender, glasses, etc.
                        
                        if model_type not in weights_info:
                            weights_info[model_type] = {}
                        
                        weights_info[model_type][task] = {
                            "filename": file,
                            "repo_id": self.repo_id,
                            "cache_dir": self.cache_dir
                        }
            
        except Exception as e:
            logger.error(f"Error getting weight info: {e}")
        
        return weights_info

def main():
    """Test weights manager"""
    manager = WeightsManager()
    
    print("🔍 Getting available weights...")
    weights_info = manager.get_available_weights()
    
    print("📊 Available weights:")
    for model_type, tasks in weights_info.items():
        print(f"  {model_type}: {tasks}")
    
    print("\n📥 Downloading sample weight (mobilenet_gender)...")
    local_path = manager.download_weight("mobilenet", "gender")
    if local_path:
        print(f"✅ Downloaded to: {local_path}")
        
        # Test loading state dict
        state_dict = manager.load_model_state_dict("mobilenet", "gender")
        if state_dict:
            print(f"✅ Loaded state dict with {len(state_dict)} keys")

if __name__ == "__main__":
    main()
