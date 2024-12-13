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
        """
        Enhanced version of get() with improved resolution and blending
        """
        # Calculate a larger aligned face region for higher quality
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        
        # Convert to blob while maintaining maximum quality
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                    (self.input_mean, self.input_mean, self.input_mean), 
                                    swapRB=True)
        
        # Process latent vector
        latent = source_face.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        
        # Run inference
        pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
        
        # Convert prediction to image with maximum quality preservation
        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        
        if not paste_back:
            return bgr_fake, M
        
        # Enhanced paste-back process
        target_img = img
        
        # Calculate face bounds from landmarks for precise blending
        face_bounds = target_face.bbox.astype(np.int32)
        face_width = face_bounds[2] - face_bounds[0]
        face_height = face_bounds[3] - face_bounds[1]
        
        # Get inverse transform
        IM = cv2.invertAffineTransform(M)
        
        # Create detailed feature mask using landmarks
        feature_mask = np.zeros((aimg.shape[0], aimg.shape[1]), dtype=np.float32)
        if hasattr(target_face, 'landmark_2d_106') and target_face.landmark_2d_106 is not None:
            landmarks = target_face.landmark_2d_106
            # Create convex hull from landmarks
            hull = cv2.convexHull(landmarks.astype(np.int32))
            cv2.fillConvexPoly(feature_mask, hull, 1.0)
        else:
            # Fallback to elliptical mask if landmarks aren't available
            center = (aimg.shape[1] // 2, aimg.shape[0] // 2)
            axes = (aimg.shape[1] // 2 - 1, aimg.shape[0] // 2 - 1)
            cv2.ellipse(feature_mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Warp results back to original image space with high quality interpolation
        bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]),
                                 flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        feature_mask = cv2.warpAffine(feature_mask, IM, (target_img.shape[1], target_img.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Create refined blending mask
        mask_h_inds, mask_w_inds = np.where(feature_mask > 0)
        if len(mask_h_inds) == 0 or len(mask_w_inds) == 0:
            return target_img
            
        # Calculate optimal blur size based on face size
        mask_size = int(np.sqrt((np.max(mask_h_inds) - np.min(mask_h_inds)) * 
                               (np.max(mask_w_inds) - np.min(mask_w_inds))))
        blur_size = max(mask_size // 32, 3) # Smaller blur for sharper edges
        blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size
        
        # Apply minimal feathering
        feature_mask = cv2.GaussianBlur(feature_mask, (blur_size, blur_size), 0)
        
        # Color correction
        source_region = bgr_fake[feature_mask > 0]
        target_region = target_img[feature_mask > 0]
        if len(source_region) > 0 and len(target_region) > 0:
            source_mean = np.mean(source_region, axis=0)
            target_mean = np.mean(target_region, axis=0)
            source_std = np.std(source_region, axis=0)
            target_std = np.std(target_region, axis=0)
            
            # Adjust color statistics
            for c in range(3):
                if source_std[c] > 0:
                    bgr_fake[:,:,c] = ((bgr_fake[:,:,c] - source_mean[c]) * 
                                     (target_std[c] / source_std[c]) + target_mean[c])
        
        # Expand mask to 3 channels
        feature_mask = np.expand_dims(feature_mask, axis=-1)
        
        # Final high-quality blend
        blended = target_img.copy()
        blended_region = (bgr_fake * feature_mask + target_img * (1 - feature_mask))
        blended[feature_mask[:,:,0] > 0] = blended_region[feature_mask[:,:,0] > 0]
        
        return blended
