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

def leffa_predict(src_image_path, ref_image_path, step=50, scale=2.5, seed=42, vt_model_type="viton_hd", 
                  vt_garment_type="upper_body", vt_repaint=False, preprocess_garment=False):
    src_image = Image.open(src_image_path)
    src_image = resize_and_center(src_image, 768, 1024)
    
    if preprocess_garment and ref_image_path.lower().endswith('.png'):
        ref_image = preprocess_garment_image(ref_image_path)
    else:
        ref_image = Image.open(ref_image_path)
    ref_image = resize_and_center(ref_image, 768, 1024)
    
    src_image_array = np.array(src_image)
    
    model_parse, _ = parsing(src_image.resize((384, 512)))
    keypoints = openpose(src_image.resize((384, 512)))
    
    if vt_model_type == "viton_hd":
        mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
    else:
        mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
    mask = mask.resize((768, 1024))
    
    if vt_model_type == "viton_hd":
        src_image_seg_array = densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
    else:
        src_image_iuv_array = densepose_predictor.predict_iuv(src_image_array)
        src_image_seg_array = src_image_iuv_array[:, :, 0:1]
        src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
    
    densepose = Image.fromarray(src_image_seg_array)
    
    transform = LeffaTransform()
    data = {"src_image": [src_image], "ref_image": [ref_image], "mask": [mask], "densepose": [densepose]}
    data = transform(data)
    
    inference = vt_inference_hd if vt_model_type == "viton_hd" else vt_inference_dc
    output = inference(data, num_inference_steps=step, guidance_scale=scale, seed=seed, repaint=vt_repaint)
    
    return np.array(output["generated_image"][0]), np.array(mask), np.array(densepose)