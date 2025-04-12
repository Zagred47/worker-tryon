import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# Scarica i checkpoint (se non già scaricati)
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")

# Inizializzazione dei modelli
#mask_predictor = AutoMasker(densepose_path="./ckpts/densepose", schp_path="./ckpts/schp")
#densepose_predictor = DensePosePredictor(config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml", 
#                                         weights_path="./ckpts/densepose/model_final_162be9.pkl")
#parsing = Parsing(atr_path="./ckpts/humanparsing/parsing_atr.onnx", 
#                  lip_path="./ckpts/humanparsing/parsing_lip.onnx")
#openpose = OpenPose(body_model_path="./ckpts/openpose/body_pose_model.pth")

#vt_model_hd = LeffaModel(pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting", 
#                          pretrained_model="./ckpts/virtual_tryon.pth", dtype="float16")
#vt_inference_hd = LeffaInference(model=vt_model_hd)

#vt_model_dc = LeffaModel(pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting", 
#                          pretrained_model="./ckpts/virtual_tryon_dc.pth", dtype="float16")
#vt_inference_dc = LeffaInference(model=vt_model_dc)