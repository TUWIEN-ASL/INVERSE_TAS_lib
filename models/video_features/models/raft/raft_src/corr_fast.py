import torch
import torch.nn.functional as F
from models.video_features.models.raft.raft_src.utils.utils import bilinear_sampler

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlockFast:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # Pre-compute constants
        self.delta = self._create_delta(radius).to(fmap1.device)
        
        # Optimize correlation computation
        corr = self._efficient_corr(fmap1, fmap2)
        
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        
        # Build pyramid more efficiently
        self.corr_pyramid = self._build_pyramid(corr, num_levels)

    def _efficient_corr(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        
        # Use torch.baddbmm for more efficient matrix multiplication
        # Initialize output tensor for in-place operation
        corr = torch.empty(batch, ht * wd, ht * wd, 
                          device=fmap1.device, dtype=fmap1.dtype)
        
        # Compute correlation with scaling factor
        scaling = 1.0 / torch.sqrt(torch.tensor(dim).float())
        torch.bmm(fmap1.transpose(1, 2), fmap2, out=corr)
        corr.mul_(scaling)
        
        return corr.view(batch, ht, wd, 1, ht, wd)

    @staticmethod
    def _create_delta(radius):
        """Pre-compute delta grid for all levels"""
        dx = torch.linspace(-radius, radius, 2 * radius + 1)
        dy = torch.linspace(-radius, radius, 2 * radius + 1)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), dim=-1)
        return delta.view(1, 2 * radius + 1, 2 * radius + 1, 2)

    def _build_pyramid(self, corr, num_levels):
        """Efficiently build correlation pyramid"""
        pyramid = [corr]
        
        # Pre-allocate memory for pyramid levels
        for i in range(num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            pyramid.append(corr)
            
        return pyramid

    def __call__(self, coords):
        batch, h1, w1, _ = coords.permute(0, 2, 3, 1).shape
        out_pyramid = []
        
        # Pre-compute reshape dimensions
        view_dims = (batch * h1 * w1, 1, 1, 2)
        final_dims = (batch, h1, w1, -1)
        
        for i in range(self.num_levels):
            # Efficient coordinate computation
            centroid_lvl = coords.permute(0, 2, 3, 1).reshape(view_dims) / (2 ** i)
            coords_lvl = centroid_lvl + self.delta
            
            # Sample correlations
            corr = bilinear_sampler(self.corr_pyramid[i], coords_lvl)
            corr = corr.view(final_dims)
            out_pyramid.append(corr)

        # Single concatenation operation
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

# def bilinear_sampler(img, coords):
#     """Optimized bilinear sampler implementation"""
#     H, W = img.shape[-2:]
    
#     # Normalize coordinates
#     coords_x, coords_y = coords.split(1, dim=-1)
#     coords_x = coords_x.view(-1, H, W)
#     coords_y = coords_y.view(-1, H, W)

#     x0 = torch.floor(coords_x).long()
#     x1 = x0 + 1
#     y0 = torch.floor(coords_y).long()
#     y1 = y0 + 1

#     # Clip coordinates to image boundaries
#     x0 = torch.clamp(x0, 0, W-1)
#     x1 = torch.clamp(x1, 0, W-1)
#     y0 = torch.clamp(y0, 0, H-1)
#     y1 = torch.clamp(y1, 0, H-1)
    
#     # Compute interpolation weights
#     wa = ((x1.type_as(coords_x) - coords_x) * 
#           (y1.type_as(coords_y) - coords_y))
#     wb = ((x1.type_as(coords_x) - coords_x) * 
#           (coords_y - y0.type_as(coords_y)))
#     wc = ((coords_x - x0.type_as(coords_x)) * 
#           (y1.type_as(coords_y) - coords_y))
#     wd = ((coords_x - x0.type_as(coords_x)) * 
#           (coords_y - y0.type_as(coords_y)))

#     # Gather pixel values and apply weights
#     img_a = img[..., y0, x0]
#     img_b = img[..., y1, x0]
#     img_c = img[..., y0, x1]
#     img_d = img[..., y1, x1]

#     return (wa * img_a + wb * img_b + wc * img_c + wd * img_d)