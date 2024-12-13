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
        # Initial face alignment
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        
        # Convert to higher precision for better quality
        aimg = aimg.astype(np.float32)
        
        # Improve blob creation with better precision
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                    (self.input_mean, self.input_mean, self.input_mean), 
                                    swapRB=True)
        
        # Process embedding
        latent = source_face.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        
        # Get prediction
        pred = self.session.run(self.output_names, 
                               {self.input_names[0]: blob, 
                                self.input_names[1]: latent})[0]
        
        # Convert prediction to image
        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        
        if not paste_back:
            return bgr_fake, M
        
        # Enhanced blending process
        target_img = img.astype(np.float32)
        
        # Calculate difference mask with higher precision
        fake_diff = bgr_fake.astype(np.float32) - aimg
        fake_diff = np.abs(fake_diff).mean(axis=2)
        
        # Improve border handling
        border_width = 3  # Increased border width for smoother transition
        fake_diff[:border_width,:] = 0
        fake_diff[-border_width:,:] = 0
        fake_diff[:,:border_width] = 0
        fake_diff[:,-border_width:] = 0
        
        # Transform coordinates
        IM = cv2.invertAffineTransform(M)
        
        # Create high resolution mask
        img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
        
        # Use better interpolation method for warping
        bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), 
                                 borderValue=0.0, flags=cv2.INTER_LANCZOS4)
        img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), 
                                  borderValue=0.0, flags=cv2.INTER_LANCZOS4)
        fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), 
                                  borderValue=0.0, flags=cv2.INTER_LANCZOS4)
        
        # Enhance mask processing
        img_white[img_white > 20] = 255
        
        # Adjust threshold for better edge detection
        fthresh = 5  # Reduced threshold for more detailed edges
        fake_diff[fake_diff < fthresh] = 0
        fake_diff[fake_diff >= fthresh] = 255
        
        # Calculate dynamic kernel sizes based on face size
        mask_h_inds, mask_w_inds = np.where(img_white == 255)
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        
        # Refined kernel sizes for better edge quality
        erode_size = max(mask_size // 15, 8)  # Smaller erosion kernel
        kernel = np.ones((erode_size, erode_size), np.uint8)
        img_mask = cv2.erode(img_white, kernel, iterations=1)
        
        # Enhanced edge processing
        edge_kernel = np.ones((2, 2), np.uint8)
        fake_diff = cv2.dilate(fake_diff, edge_kernel, iterations=1)
        
        # Multi-step blurring for better transitions
        blur_sizes = [
            max(mask_size // 40, 3),  # Small blur for detail
            max(mask_size // 30, 5),  # Medium blur for transition
            max(mask_size // 20, 7)   # Large blur for overall smoothing
        ]
        
        img_mask_temp = img_mask.copy()
        for blur_size in blur_sizes:
            kernel_size = (blur_size, blur_size)
            blur_size = tuple(2 * i + 1 for i in kernel_size)
            img_mask_temp = cv2.GaussianBlur(img_mask_temp, blur_size, 0)
        img_mask = img_mask_temp
        
        # Fine detail preservation in fake_diff
        fake_diff = cv2.GaussianBlur(fake_diff, (5, 5), 0)
        
        # Normalize masks
        img_mask = img_mask / 255.0
        fake_diff = fake_diff / 255.0
        
        # Apply mask with high precision blending
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
        fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img
        
        # Ensure output maintains high quality
        fake_merged = np.clip(fake_merged, 0, 255)
        fake_merged = fake_merged.astype(np.uint8)
        
        return fake_merged
