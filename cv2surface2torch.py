import torch
import PyNvCodec as nvc

def converter(im, nvUpl, surface_tensor, w, gpuID):
    #h, w, c = im.shape
    #nvUpl = nvc.PyFrameUploader(int(w), int(h), nvc.PixelFormat.RGB, gpuID)
    #surface_tensor = torch.zeros(h, w, 3, dtype=torch.uint8, device=torch.device(f'cuda:{gpuID}'))
    rawSurface = nvUpl.UploadSingleFrame(im)  #rawSurface.Format() == nvc.PixelFormat.RGB
    rawSurface.PlanePtr().Export(surface_tensor.data_ptr(), w * 3, gpuID)  #surface_tensor.dtype == (torch.tensor, device = "cuda")
    return surface_tensor
