import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """Converts image into flattened patch embeddings."""
    def __init__(self, img_size=(256,128), patch_size=(16,16), in_chans=1, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = self.grid_size[0] * self.grid_size[1]
        # use a conv to both patchify and embed
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = num_patches

    def forward(self, x):
        # Handle both 2D and 3D inputs
        if x.dim() == 3:  # (B, H, W) - add channel dimension
            x = x.unsqueeze(1)  # (B, 1, H, W)
        # x: (B, C, 256, 128)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class ViTEncoder(nn.Module):
    def __init__(self,
                 img_size=(256,128),
                 patch_size=(16,16),
                 in_chans=1,
                 embed_dim=512,
                 depth=6,
                 nhead=8,
                 mlp_ratio=4.0,
                 latent_dim=1024):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # learnable cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # projection to latent space
        self.to_latent = nn.Linear(embed_dim, latent_dim)

        # init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                          # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)    # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)            # (B, N+1, D)
        x = x + self.pos_embed                           # add pos emb
        # With batch_first=True, no need to transpose
        x = self.transformer(x)                          # (B, S, D)
        cls_out = x[:, 0]                                # (B, D)
        latent = self.to_latent(cls_out)                 # (B, latent_dim)
        return latent

class ViTDecoder(nn.Module):
    def __init__(self,
                 img_size=(256,128),
                 patch_size=(16,16),
                 out_chans=1,
                 embed_dim=512,
                 depth=6,
                 nhead=8,
                 mlp_ratio=4.0,
                 latent_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = self.grid_size[0] * self.grid_size[1]

        # from latent back to token embeddings
        self.from_latent = nn.Linear(latent_dim, embed_dim)

        # learnable dummy tokens for image patches
        self.patch_tokens = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer decoder (we’ll use Encoder layers as a simple decoder)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)

        # project back to image patches via a conv transpose
        self.deproj = nn.ConvTranspose2d(embed_dim,
                                         out_chans,
                                         kernel_size=patch_size,
                                         stride=patch_size)

        # init
        nn.init.trunc_normal_(self.patch_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z):
        B = z.size(0)
        # project latent to same dim as tokens
        z = self.from_latent(z)                          # (B, D)
        # broadcast to each patch position
        tokens = self.patch_tokens.expand(B, -1, -1)     # (B, N, D)
        # add a “summary” via the latent as a bias to each token
        tokens = tokens + z.unsqueeze(1)                 # (B, N, D)
        tokens = tokens + self.pos_embed                 # (B, N, D)
        # With batch_first=True, no need to transpose
        x = self.transformer(tokens)                     # (B, N, D)
        # reshape to (B, D, H/ps, W/ps)
        H, W = self.grid_size
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        # de-patch to image
        x = self.deproj(x)                               # (B, 1, 256, 128)
        # Remove channel dimension to match input format
        x = x.squeeze(1)                                 # (B, 256, 128)
        return x

class AutoViT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Set default parameters for brain data
        default_kwargs = {
            'img_size': (256, 128),
            'patch_size': (16, 16),
            'embed_dim': 512,
            'depth': 6,
            'nhead': 8,
            'mlp_ratio': 4.0,
            'latent_dim': 1024
        }
        default_kwargs.update(kwargs)
        
        self.encoder = ViTEncoder(**default_kwargs)
        self.decoder = ViTDecoder(**default_kwargs)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        # Apply sigmoid activation to match brain data range [0, 1]
        recon = torch.sigmoid(recon)
        return recon

# # Example usage:
# if __name__ == "__main__":
#     model = ViTAutoencoder(
#         img_size=(256,128),
#         patch_size=(16,16),
#         in_chans=3,
#         embed_dim=512,
#         depth=4,
#         nhead=8,
#         mlp_ratio=4.0,
#         latent_dim=1024
#     )
#     imgs = torch.randn(2, 3, 256, 128)
#     recon_imgs, latents = model(imgs)
#     print("Reconstructed:", recon_imgs.shape)   # (2,3,256,128)
#     print("Latent vector:", latents.shape)      # (2,1024)
