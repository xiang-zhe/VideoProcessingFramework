import torch
import numpy as np
import PyNvCodec as nvc

def numpy2surface2torch(rawFrame, nvUpl, surface_tensor, w, gpuID):
    #h, w, c = rawFrame.shape
    #nvUpl = nvc.PyFrameUploader(int(w), int(h), nvc.PixelFormat.RGB, gpuID)
    #surface_tensor = torch.zeros(h, w, 3, dtype=torch.uint8, device=torch.device(f'cuda:{gpuID}'))
    rawSurface = nvUpl.UploadSingleFrame(rawFrame)  #rawSurface.Format() == nvc.PixelFormat.RGB
    rawSurface.PlanePtr().Export(surface_tensor.data_ptr(), w * 3, gpuID)  #surface_tensor.dtype == (torch.tensor, device = "cuda")
    return surface_tensor

def torch2surface2numpy(surface_tensor, nvDwn, rawFrame, w gpuID):
    #h, w, c = surface_tensor.shape
    #surface_rgb = nvc.Surface.Make(nvc.PixelFormat.RGB, w, h, gpuID)
    #surface_rgb.PlanePtr().Import(surface_tensor.data_ptr(), w * 3, gpuID)
    #nvDwn = nvc.PySurfaceDownloader(hwidth, hheight, nvc.PixelFormat.RGB, gpuID)
    #rawFrame = np.ndarray(shape=(), dtype=np.uint8)
    success = nvDwn.DownloadSingleSurface(surface_tensor, rawFrame)
    if success:
        return rawFrame
    else:
        print('Failed to download surface')
    
    
