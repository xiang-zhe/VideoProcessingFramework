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
    surface_rgb = nvc.Surface.Make(nvc.PixelFormat.RGB, w, h, gpuID)
    surface_rgb.PlanePtr().Import(surface_tensor.data_ptr(), w * 3, gpuID)
    #nvDwn = nvc.PySurfaceDownloader(hwidth, hheight, nvc.PixelFormat.RGB, gpuID)
    #rawFrame = np.ndarray(shape=(surface_tensor.HostSize()), dtype=np.uint8)
    success = nvDwn.DownloadSingleSurface(surface_rgb, rawFrame)
    if success:
        return rawFrame
    else:
        print('Failed to download surface')
    
    
class VPF():
    def __init__(self, width, height, gpuID):
        #self.init_vpf()
        self.w = width
        self.h = height
        self.gpuID = gpuID 
        self.surface_tensor = torch_zeros(self.h, self.w, 3, dtype=torch_uint8, device=torch_device(f'cuda:{self.gpuID}'))
        self.surface_rgb = nvc.Surface.Make(nvc.PixelFormat.RGB, self.w, self.h, self.gpuID)
        self.frame = numpy_ndarray(shape=(self.surface_rgb.HostSize()), dtype=numpy_uint8)
        self.nvUpl = nvc.PyFrameUploader(self.w, self.h, nvc.PixelFormat.RGB, self.gpuID)
        self.nvDwn = nvc.PySurfaceDownloader(self.w, self.h, nvc.PixelFormat.RGB, self.gpuID)

    def numpy2tensor(self, rawFrame):
        rawSurface = self.nvUpl.UploadSingleFrame(rawFrame)  #rawSurface.Format() == nvc.PixelFormat.RGB
        rawSurface.PlanePtr().Export(self.surface_tensor.data_ptr(), self.w * 3, self.gpuID)  #surface_tensor.dtype == (torch.tensor, device = "cuda")
        return self.surface_tensor

    def tensor2numpy(self, rawTensor):
        self.surface_rgb.PlanePtr().Import(rawTensor.data_ptr(), self.w * 3, self.gpuID)
        success = self.nvDwn.DownloadSingleSurface(self.surface_rgb, self.frame)
        if success:
            return self.frame
        else:
            print('Failed to download surface')
