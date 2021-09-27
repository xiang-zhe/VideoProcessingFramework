import torch
import numpy as np
import PyNvCodec as nvc

def converter2torch(rawFrame, nvUpl, surface_tensor, w, gpuID):
    #h, w, c = rawFrame.shape
    #nvUpl = nvc.PyFrameUploader(int(w), int(h), nvc.PixelFormat.RGB, gpuID)
    #surface_tensor = torch.zeros(h, w, 3, dtype=torch.uint8, device=torch.device(f'cuda:{gpuID}'))
    rawSurface = nvUpl.UploadSingleFrame(rawFrame)  #rawSurface.Format() == nvc.PixelFormat.RGB
    rawSurface.PlanePtr().Export(surface_tensor.data_ptr(), w * 3, gpuID)  #surface_tensor.dtype == (torch.tensor, device = "cuda")
    return surface_tensor

def converter2numpy(surface_tensor, nvDwn, rawFrame, gpuID):
    #nvDwn = nvc.PySurfaceDownloader(hwidth, hheight, sureface_tensor.Format(), gpuID)
    #rawFrame = np.ndarray(shape=(), dtype=np.uint8)
    success = nvDwn.DownloadSingleSurface(surface_tensor, rawFrame)
    if success:
        return rawFrame
    else:
        print('Failed to download surface')
    
    
