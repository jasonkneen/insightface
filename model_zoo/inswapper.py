import time
import numpy as np
import onnxruntime
import cv2
import onnx
from onnx import numpy_helper
from ..utils import face_align




class INSwapper():
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = None #session
        model = onnx.load(self.model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.input_mean = 0.0
        self.input_std = 255.0
        #print('input mean and std:', model_file, self.input_mean, self.input_std)
        if self.session is None:
            # self.session = onnxruntime.InferenceSession(self.model_file, None)
            self.session = onnxruntime.InferenceSession(self.model_file, 
                                                        providers=[
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider"
            ])
        inputs = self.session.get_inputs()
        self.input_names = []
        for inp in inputs:
            self.input_names.append(inp.name)
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.output_names = output_names
        assert len(self.output_names)==1
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        print('inswapper-shape:', self.input_shape)
        self.input_size = tuple(input_shape[2:4][::-1])

    def forward(self, img, latent):
        img = (img - self.input_mean) / self.input_std
        pred = self.session.run(self.output_names, {self.input_names[0]: img, self.input_names[1]: latent})[0]
        return pred

    # def get(self, img, target_face, source_face, paste_back=True):
    #     aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
    #     blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
    #                                   (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
    #     latent = source_face.normed_embedding.reshape((1,-1))
    #     latent = np.dot(latent, self.emap)
    #     latent /= np.linalg.norm(latent)
    #     pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
    #     #print(latent.shape, latent.dtype, pred.shape)
    #     img_fake = pred.transpose((0,2,3,1))[0]
    #     bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
    #     if not paste_back:
    #         return bgr_fake, M
    #     else:
    #         target_img = img
    #         fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
    #         fake_diff = np.abs(fake_diff).mean(axis=2)
    #         fake_diff[:2,:] = 0
    #         fake_diff[-2:,:] = 0
    #         fake_diff[:,:2] = 0
    #         fake_diff[:,-2:] = 0
    #         IM = cv2.invertAffineTransform(M)
    #         img_white = np.full((aimg.shape[0],aimg.shape[1]), 255, dtype=np.float32)
    #         bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    #         img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    #         fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    #         img_white[img_white>20] = 255
    #         fthresh = 10
    #         fake_diff[fake_diff<fthresh] = 0
    #         fake_diff[fake_diff>=fthresh] = 255
    #         img_mask = img_white
    #         mask_h_inds, mask_w_inds = np.where(img_mask==255)
    #         mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    #         mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    #         mask_size = int(np.sqrt(mask_h*mask_w))
    #         k = max(mask_size//10, 10)
    #         #k = max(mask_size//20, 6)
    #         #k = 6
    #         kernel = np.ones((k,k),np.uint8)
    #         img_mask = cv2.erode(img_mask,kernel,iterations = 1)
    #         kernel = np.ones((2,2),np.uint8)
    #         fake_diff = cv2.dilate(fake_diff,kernel,iterations = 1)
    #         k = max(mask_size//30, 5)
    #         #k = 3
    #         #k = 3
    #         kernel_size = (k, k)
    #         blur_size = tuple(2*i+1 for i in kernel_size)
    #         img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    #         k = 5
    #         kernel_size = (k, k)
    #         blur_size = tuple(2*i+1 for i in kernel_size)
    #         fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
    #         img_mask /= 255
    #         fake_diff /= 255
    #         #img_mask = fake_diff
    #         img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
    #         fake_merged = img_mask * bgr_fake + (1-img_mask) * target_img.astype(np.float32)
    #         fake_merged = fake_merged.astype(np.uint8)
    #         return fake_merged

    def get(self, img, target_face, source_face, paste_back=True):
        """High resolution face swapping with enhanced detail preservation"""
        # Calculate optimal size based on input face size
        face_size = max(target_face.bbox[2] - target_face.bbox[0], 
                        target_face.bbox[3] - target_face.bbox[1])
        scale_factor = max(1, face_size / self.input_size[0])
        
        # Get aligned face with increased resolution
        aimg, M = face_align.norm_crop2(img, target_face.kps, 
                                       int(self.input_size[0] * scale_factor))
        
        # Preserve high-res version for later
        high_res_aligned = aimg.copy()
        
        # Resize to model input size for inference
        aimg_input = cv2.resize(aimg, self.input_size, 
                               interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to blob with minimal preprocessing
        blob = cv2.dnn.blobFromImage(aimg_input, 1.0 / self.input_std, self.input_size,
                                  (self.input_mean, self.input_mean, self.input_mean), 
                                  swapRB=True)
        
        # Get and normalize latent vector
        latent = source_face.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        
        # Run inference
        pred = self.session.run(self.output_names, 
                               {self.input_names[0]: blob, 
                                self.input_names[1]: latent})[0]
        
        # Convert prediction back to image
        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        
        # Upscale the swapped face to match original resolution
        bgr_fake = cv2.resize(bgr_fake, (high_res_aligned.shape[1], high_res_aligned.shape[0]), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        if not paste_back:
            return bgr_fake, M
        
        # Enhanced paste-back process
        target_img = img
        
        # Calculate difference mask with color-aware sensitivity
        fake_diff = bgr_fake.astype(np.float32) - high_res_aligned.astype(np.float32)
        fake_diff_lab = cv2.cvtColor(fake_diff.astype(np.uint8), cv2.COLOR_BGR2Lab)
        fake_diff = np.mean(np.abs(fake_diff_lab), axis=2)
        
        # Minimal border clearing to preserve detail
        fake_diff[:1,:] = 0
        fake_diff[-1:,:] = 0
        fake_diff[:,:1] = 0
        fake_diff[:,-1:] = 0
        
        # Get inverse transform
        IM = cv2.invertAffineTransform(M)
        
        # Create base mask
        img_white = np.full((high_res_aligned.shape[0], high_res_aligned.shape[1]), 
                           255, dtype=np.float32)
        
        # Warp results back with high quality interpolation
        bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), 
                                 borderValue=0.0, flags=cv2.INTER_LANCZOS4)
        img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), 
                                  borderValue=0.0)
        fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), 
                                  borderValue=0.0)
        
        # Fine-tune edge detection
        img_white[img_white > 10] = 255
        fthresh = 3  # More sensitive threshold
        fake_diff[fake_diff < fthresh] = 0
        fake_diff[fake_diff >= fthresh] = 255
        
        # Create refined mask
        img_mask = img_white
        mask_h_inds, mask_w_inds = np.where(img_mask==255)
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h*mask_w))
        
        # Minimal erosion to preserve edge detail
        k = max(mask_size//20, 5)  # Smaller kernel for better detail
        kernel = np.ones((k,k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        
        # Enhanced edge preservation
        kernel = np.ones((2,2), np.uint8)
        fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
        
        # Adaptive blending
        k = max(mask_size//30, 3)  # Reduced blur for sharper results
        kernel_size = (k, k)
        blur_size = tuple(2*i+1 for i in kernel_size)
        
        # Minimal blur while preventing artifacts
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        fake_diff = cv2.GaussianBlur(fake_diff, (3, 3), 0)
        
        # Color correction
        target_face_region = target_img[int(target_face.bbox[1]):int(target_face.bbox[3]),
                                      int(target_face.bbox[0]):int(target_face.bbox[2])]
        if target_face_region.size > 0:
            src_lab = cv2.cvtColor(bgr_fake, cv2.COLOR_BGR2LAB)
            tgt_lab = cv2.cvtColor(target_face_region, cv2.COLOR_BGR2LAB)
            
            src_mean = np.mean(src_lab, axis=(0,1))
            tgt_mean = np.mean(tgt_lab, axis=(0,1))
            src_std = np.std(src_lab, axis=(0,1))
            tgt_std = np.std(tgt_lab, axis=(0,1))
            
            # Adjust color statistics
            adjusted_lab = ((src_lab - src_mean) * (tgt_std / (src_std + 1e-7))) + tgt_mean
            bgr_fake = cv2.cvtColor(adjusted_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Normalize masks
        img_mask = img_mask.astype(np.float32) / 255
        fake_diff = fake_diff.astype(np.float32) / 255
        
        # Prepare final mask
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
        
        # Final blend with high precision
        fake_merged = img_mask * bgr_fake + (1-img_mask) * target_img.astype(np.float32)
        fake_merged = np.clip(fake_merged, 0, 255).astype(np.uint8)
        
        return fake_merged
