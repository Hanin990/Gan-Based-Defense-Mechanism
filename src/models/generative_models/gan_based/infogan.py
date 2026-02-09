# Updated and extended implementation of the InfoGAN-based textual manifold defense
# from "Textual Manifold-based Defense Against Natural Language Adversarial Examples"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_constant_schedule_with_warmup
from scipy.stats import truncnorm
import math
import wandb

from .gan import GAN


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return F.relu(x + self.block(x))

class Generator(nn.Module):
    def __init__(self, latent_dim=100, code_dim=10, hidden_dims=[256, 512, 768, 1024, 768], use_residual=True):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.input_dim = latent_dim + code_dim
        self.use_residual = use_residual
        
        # Progressive architecture with residual connections
        layers = []
        dims = [self.input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on final layer
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(0.1))
                
                # Add residual block for larger dimensions
                if self.use_residual and dims[i + 1] >= 512:
                    layers.append(ResidualBlock(dims[i + 1]))
        
        layers.append(nn.Tanh())  # Final activation
        self.gen = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z, c, make_one_hot=True):
        # input size: ((bs, latent_dim), (bs))
        if make_one_hot:
            c = F.one_hot(c, self.code_dim)  # (bs, code_dim)
        inp = torch.cat([z, c], dim=-1)  # (bs, latent_dim+code_dim)
        output = self.gen(inp)  # (bs, 768)
        return output

    def dist(self, x, y, metric="cosine"):
        if metric == "cosine":
            return 1-F.cosine_similarity(x, y, dim=-1)
        elif metric == "l2":
            return torch.norm(x - y, dim=-1)
        else:
            raise ValueError(f"metric {metric} is not supported")

    def reconstruct1(self, real_embs, c, k, step_size=1.0, num_steps=10, metric="l2", **kwargs):
        # Method 1: Use SGD to find optimize z that minize the reconstruction loss
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device

        with torch.enable_grad():
            # Initialize K random z candidates
            z_candidates = torch.randn(batch_size, k, self.latent_dim, device=device)  # (bs, k, latent_dim)
            z_candidates = z_candidates.view(batch_size*k, self.latent_dim)  # (bs*k, latent_dim)
            z_candidates.requires_grad = True
    
            # Duplicate c for candidate z
            c = c.unsqueeze(dim=-1)  # (bs, 1)
            c = c.expand(batch_size, k)  # (bs, k)
            c = c.reshape(batch_size*k)  # (bs*k)
    
            # Define optimizer
            optimizer = torch.optim.SGD([z_candidates], lr=step_size)
    
            # Perform L GD steps
            for step in range(num_steps):
                fake_embs = self.forward(z_candidates, c)  # (bs*k, emb_dim)
                fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)
                reconstruction_losses = self.dist(real_embs.unsqueeze(1), fake_embs, metric)  # (bs, k)
                loss = reconstruction_losses.sum()
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        fake_embs = self.forward(z_candidates, c)  # (bs*k, emb_dim)
        fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)
        reconstruction_losses = self.dist(real_embs.unsqueeze(1), fake_embs, metric)  # (bs, k)

        # Find optimal z
        argmin = torch.argmin(reconstruction_losses, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
        argmin = argmin.expand(batch_size, 1, emb_dim)
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)

        return reconstructed_emb, reconstruction_losses

    def reconstruct2(self, real_embs, c, **kwargs):
        raise NotImplementedError

    def reconstruct3(self, real_embs, c, k, metric, **kwargs):
        # Method 3: Sample several points from the z prior, choose the optimal one
        print("8888888", real_embs.shape)
        batch_size, emb_dim = real_embs.shape
        print("8888888", batch_size, emb_dim)
        device = real_embs.device

        # Sample candidate z
        z = torch.randn(batch_size, k, self.latent_dim, device=device)  # (bs, k, latent_dim)
        z = z.view(batch_size*k, self.latent_dim)  # (bs*k, latent_dim)

        # Duplicate c for candidate z
        c = c.unsqueeze(dim=-1)  # (bs, 1)
        c = c.expand(batch_size, k)  # (bs, k)
        c = c.reshape(batch_size*k)  # (bs*k)

        # Compute candidate fake embeddings
        fake_embs = self.forward(z, c)  # (bs*k, emb_dim)
        fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)

        # Choose the optimal fake embedding for each real_emb
        dists = self.dist(fake_embs, real_embs.unsqueeze(1), metric=metric)  # (bs, k)
        argmin = torch.argmin(dists, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
        argmin = argmin.expand(batch_size, 1, emb_dim)
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)
        reconstruction_losses = torch.norm(real_embs - reconstructed_emb, dim=-1)

        return reconstructed_emb, reconstruction_losses

    def reconstruct4(self, real_embs, c, k, metric, threshold=1, **kwargs):
        # Method 4: Sample several z from the truncated normal, choose the optimal one
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device

        # Sample candidate z
        z = truncnorm.rvs(-threshold, threshold, size=(batch_size, k, self.latent_dim))  # (bs, k, latent_dim)
        z = torch.as_tensor(z, dtype=torch.float32, device=device)
        z = z.view(batch_size*k, self.latent_dim)  # (bs*k, latent_dim)

        # Duplicate c for candidate z
        c = c.unsqueeze(dim=-1)  # (bs, 1)
        c = c.expand(batch_size, k)  # (bs, k)
        c = c.reshape(batch_size*k)  # (bs*k)

        # Compute candidate fake embeddings
        fake_embs = self.forward(z, c)  # (bs*k, emb_dim)
        fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)

        # Choose the optimal fake embedding for each real_emb
        dists = self.dist(fake_embs, real_embs.unsqueeze(1), metric=metric)  # (bs, k)
        argmin = torch.argmin(dists, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
        argmin = argmin.expand(batch_size, 1, emb_dim)
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)
        reconstruction_losses = torch.norm(real_embs - reconstructed_emb, dim=-1)

        return reconstructed_emb, reconstruction_losses

    def reconstruct5(self, real_embs, c, k=1, threshold=1, **kwargs):
        # Method 5: Sample 1 random z from the truncated normal (not choosing the optimal)
        assert k == 1, f"k must be 1 for this method. Got k={k}"
        return self.reconstruct4(real_embs, c, k=1, threshold=threshold)

    def reconstruct(self, real_embs, c, k, method=3, **kwargs):
        # There are several ways to find the best z
        # Method 1: Use PGD where z is boundded in the region of a normal distribution
        # Method 2: Use PGD where z is guided with the underlying prior distribution?
        # Method 3: Sample several z from the normal prior, choose the optimal one
        # Method 4: Sample several z from the truncated normal prior, choose the optimal one
        # Method 5: Sample a single z from the truncated normal prior

        with torch.no_grad():
            if method == 1:
                return self.reconstruct1(real_embs, c, k, **kwargs)
            elif method == 2:
                return self.reconstruct2(real_embs, c, k, **kwargs)
            elif method == 3:
                return self.reconstruct3(real_embs, c, k, **kwargs) 
            elif method == 4:
                return self.reconstruct4(real_embs, c, k, **kwargs) 
            elif method == 5:
                return self.reconstruct5(real_embs, c, k, **kwargs) 
            else:
                raise ValueError(f"method {method} is not supported")


class Discriminator(nn.Module):
    def __init__(self, code_dim=10, hidden_dims=[768, 512, 256, 128], use_spectral_norm=True):
        super(Discriminator, self).__init__()
        self.code_dim = code_dim
        
        # Progressive discriminator with spectral normalization
        layers = []
        dims = [768] + hidden_dims
        
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.3))  # Higher dropout for discriminator
        
        self.dis = nn.Sequential(*layers)
        
        # Discriminator head with spectral norm
        self.dis_head = nn.Linear(dims[-1], 1)
        if use_spectral_norm:
            self.dis_head = nn.utils.spectral_norm(self.dis_head)
        
        # Encoder head for InfoGAN
        enc_layers = [
            nn.Linear(dims[-1], 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, code_dim)
        ]
        if use_spectral_norm:
            enc_layers[0] = nn.utils.spectral_norm(enc_layers[0])
            enc_layers[3] = nn.utils.spectral_norm(enc_layers[3])
        
        self.enc_head = nn.Sequential(*enc_layers)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # input size: bs x 768
        if isinstance(x, list):
            x = x[0]
        
        reps = self.dis(x)  # (bs, hidden_dim)
        d_logits = self.dis_head(reps)  # (bs, 1)
        d_probs = torch.sigmoid(d_logits)  # (bs, 1)
        e_logits = self.enc_head(reps)  # (bs, code_dim)
        
        return reps, d_logits, d_probs, e_logits


class InfoGAN(GAN):
    def __init__(
        self,
        latent_dim: int = 100,
        code_dim: int = 10,
        lr: float = None,
        b1: float = None,
        b2: float = None,
        g_lr: float = 1e-4,  # Increased from 3e-5
        g_b1: float = 0.0,   # Changed from 0.5 for better stability
        g_b2: float = 0.9,   # Changed from 0.999
        d_lr: float = 4e-4,  # Increased and made D learn faster
        d_b1: float = 0.0,
        d_b2: float = 0.9,
        p_lr: float = 1e-3,
        p_b1: float = 0.5,
        p_b2: float = 0.999,
        reg_prior_weight: float = 5.0,   # Further reduced for stability
        reg_info_weight: float = 0.5,    # Reduced to prevent overfitting
        k: int = 15,  # Reduced from 20
        step_size: float = 1,
        num_steps: int = 30,
        method: int = 3,
        threshold: float = 0.1,
        metric: str = "l2",
        opt_weight: float = 1.0,
        num_warmup_steps: int = 1000,
        d_step_ratio: int = 1,
        grad_clip_norm: float = 1.0,  # Add gradient clipping
        use_amp: bool = True,  # Add mixed precision
        feature_matching_weight: float = 0.1,  # Add feature matching
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        if lr is not None:
            self.hparams.g_lr = self.hparams.d_lr = self.hparams.lr
        if b1 is not None:
            self.hparams.g_b1 = self.hparams.d_b1 = self.hparams.b1
        if b2 is not None:
            self.hparams.g_b2 = self.hparams.d_b2 = self.hparams.b2

        # Initialize models with improved architectures
        self.gen = Generator(latent_dim, code_dim)
        self.dis = Discriminator(code_dim)
        self.prior = nn.Parameter(torch.ones(code_dim)/code_dim)

        self.register_buffer("pl_counter", torch.ones(1, dtype=torch.float32))
        
        # For mixed precision training
        if self.hparams.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Adaptive k selection based on reconstruction quality
        self.register_buffer("avg_reconstruction_loss", torch.tensor(1.0))
        self.register_buffer("k_adaptation_factor", torch.tensor(1.0))
        
        # For text logging
        self.text_embeddings = None  # Store training embeddings
        self.text_samples = None     # Store original text samples

    def forward(self, z, c):
        return self.gen(z, c)

    def reconstruct6(self, real_embs, c, k, step_size=1.0, num_steps=10, metric="l2", opt_weight=1, **kwargs):
        # Note that for this reconstruction strategy, c is a distribution so it has shape (bs, code_dim)
        # Method 6: Use SGD to find optimize z that minize the reconstruction loss with constraint Q(c|G(z, c_t)) = one_hot(argmax(Q(c|t))) (hard-label constraint)
        # Method 7: Use SGD to find optimize z that minize the reconstruction loss with constraint Q(c|G(z, c_t)) = Q(c|t) (soft-label constraint)
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device

        with torch.enable_grad():
            # Initialize K random z candidates
            z_candidates = torch.randn(batch_size, k, self.hparams.latent_dim, device=device)  # (bs, k, latent_dim)
            z_candidates = z_candidates.view(batch_size*k, self.hparams.latent_dim)  # (bs*k, latent_dim)
            z_candidates.requires_grad = True
    
            # Duplicate c for candidate z
            c = c.unsqueeze(dim=1)  # (bs, 1, code_dim)
            c = c.expand(batch_size, k, self.hparams.code_dim)  # (bs, k, code_dim)
            c = c.reshape(batch_size*k, self.hparams.code_dim)  # (bs*k, code_dim)
    
            # Define optimizer
            optimizer = torch.optim.SGD([z_candidates], lr=step_size)
    
            # Perform L GD steps
            for step in range(num_steps):
                c_ = torch.argmax(c, dim=-1)  # (bs*k)
                fake_embs = self.forward(z_candidates, c_)  # (bs*k, emb_dim)
                _, _, _, e_logits = self.dis(fake_embs)  # (bs*k, code_dim)
                reg_loss = -(torch.softmax(e_logits, dim=-1) * torch.log(c)).sum(dim=-1).mean()

                fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)
                reconstruction_losses = self.gen.dist(real_embs.unsqueeze(1), fake_embs, metric)  # (bs, k)
                loss = reconstruction_losses.sum() - self.hparams.opt_weight * reg_loss
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        c_ = torch.argmax(c, dim=-1)  # (bs*k)
        fake_embs = self.forward(z_candidates, c_)  # (bs*k, emb_dim)
        fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)
        reconstruction_losses = self.gen.dist(real_embs.unsqueeze(1), fake_embs, metric)  # (bs, k)

        # Find optimal z
        argmin = torch.argmin(reconstruction_losses, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
        argmin = argmin.expand(batch_size, 1, emb_dim)
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)

        return reconstructed_emb, reconstruction_losses

    def _adaptive_k_selection(self, base_k, reconstruction_loss):
        """Dynamically adjust k based on reconstruction quality"""
        # Update running average of reconstruction loss
        self.avg_reconstruction_loss = 0.9 * self.avg_reconstruction_loss + 0.1 * reconstruction_loss.mean()
        
        # Increase k if reconstruction is poor, decrease if good
        if reconstruction_loss.mean() > self.avg_reconstruction_loss * 1.2:
            self.k_adaptation_factor = min(self.k_adaptation_factor * 1.1, 2.0)
        elif reconstruction_loss.mean() < self.avg_reconstruction_loss * 0.8:
            self.k_adaptation_factor = max(self.k_adaptation_factor * 0.95, 0.5)
        
        adaptive_k = max(int(base_k * self.k_adaptation_factor), 5)
        return min(adaptive_k, 30)  # Cap at 30
    
    def reconstruct_optimized(self, real_embs, c, k, metric="l2", **kwargs):
        """Optimized reconstruction using best performing methods"""
        # Handle list input (convert to tensor)
        if isinstance(real_embs, list):
            real_embs = real_embs[0]
        
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device
        
        # Adaptive k selection
        k = self._adaptive_k_selection(k, torch.ones(batch_size, device=device))
        
        # Use method 3 (sampling) for speed, method 4 (truncated) for quality
        if k <= 10:  # Use truncated normal for small k
            return self.gen.reconstruct4(real_embs, c, k, metric, threshold=1.0)
        else:  # Use regular sampling for large k
            return self.gen.reconstruct3(real_embs, c, k, metric)
    
    def reconstruct8(self, real_embs, c, k, metric, **kwargs):
        # Method 8: Sample several points from the z prior and the c prior, choose the optimal one
        batch_size, emb_dim = real_embs.shape
        device = real_embs.device

        # Sample candidate z
        z = torch.randn(batch_size, k, self.hparams.latent_dim, device=self.device)  # (bs, k, latent_dim)
        z = z.view(batch_size*k, self.hparams.latent_dim)  # (bs*k, latent_dim)

        # Sample candidate c
        probs = torch.softmax(self.prior, dim=-1)
        c = torch.multinomial(probs, batch_size*k, replacement=True)  # (bs*k)
        c = c.to(self.device)

        # Compute candidate fake embeddings
        fake_embs = self.forward(z, c)  # (bs*k, emb_dim)
        fake_embs = fake_embs.view(batch_size, k, emb_dim)  # (bs, k, emb_dim)

        # Choose the optimal fake embedding for each real_emb
        dists = self.gen.dist(fake_embs, real_embs.unsqueeze(1), metric=metric)  # (bs, k)
        argmin = torch.argmin(dists, dim=-1, keepdim=True).unsqueeze(-1)  # (bs)
        argmin = argmin.expand(batch_size, 1, emb_dim)
        reconstructed_emb = torch.gather(fake_embs, 1, argmin).squeeze(1)  # (bs, emb_dim)
        reconstruction_losses = torch.norm(real_embs - reconstructed_emb, dim=-1)

        return reconstructed_emb, reconstruction_losses

    def reconstruct(self, real_embs, k=None, step_size=None, num_steps=None, method=None, threshold=None, metric=None, opt_weight=None, **kwargs):
        if k is None: k = self.hparams.k
        if step_size is None: step_size = self.hparams.step_size
        if num_steps is None: num_steps = self.hparams.num_steps
        if method is None: method = self.hparams.method
        if threshold is None: threshold = self.hparams.threshold
        if metric is None: metric = self.hparams.metric
        if opt_weight is None: opt_weight = self.hparams.opt_weight

        # Handle list input (convert to tensor)
        if isinstance(real_embs, list):
            real_embs = real_embs[0]

        # Compute c using the Encoder network
        _, _, _, e_logits = self.dis(real_embs)
        c = torch.argmax(e_logits, dim=-1)  # (bs)

        # Use optimized reconstruction for better performance
        if method in [3, 4]:  # Use optimized version for sampling methods
            return self.reconstruct_optimized(real_embs, c, k, metric=metric)
        elif method == 8:
            return self.reconstruct8(real_embs, c, k, metric, **kwargs) 
        else:
            return self.gen.reconstruct(real_embs, c, k, method, step_size=step_size, num_steps=num_steps, threshold=threshold, metric=metric)

    def feature_matching_loss(self, real_embs, fake_embs):
        """Compute feature matching loss using discriminator features"""
        real_features, _, _, _ = self.dis(real_embs)
        fake_features, _, _, _ = self.dis(fake_embs)
        return F.mse_loss(fake_features.mean(0), real_features.mean(0))
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_embs = batch["clean_emb"]  # (bs, emb_dim)
        if isinstance(real_embs, list):
            real_embs = real_embs[0]
        batch_size = real_embs.shape[0]

        # Sample noise with better initialization
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device) * 0.1  # Smaller initial variance
        z = z.type_as(real_embs)

        # Sample latent code (discrete) with label smoothing
        probs = torch.softmax(self.prior, dim=-1)
        c = torch.multinomial(probs, batch_size, replacement=True)  # (bs)
        c = c.to(self.device)

        # Train the Generator
        if optimizer_idx == 0 and batch_idx % self.hparams.d_step_ratio == 0:
            # Label smoothing for real labels
            real_labels = torch.ones(batch_size, 1, device=self.device) * 0.9  # Label smoothing
            real_labels = real_labels.type_as(real_embs)

            # Generate fake embeddings
            fake_embs = self.gen(z, c)  # (bs, emb_dim)
            
            # Compute Discriminator's predictions
            _, _, d_fake_probs, e_fake_logits = self.dis(fake_embs)  # (bs, 1), (bs, code_dim)
            
            # Compute losses
            g_gan_loss = F.binary_cross_entropy(d_fake_probs, real_labels)
            g_info_loss = F.cross_entropy(e_fake_logits, c)
            
            # Add feature matching loss
            g_fm_loss = self.feature_matching_loss(real_embs, fake_embs)
            
            g_loss = (g_gan_loss + 
                     self.hparams.reg_info_weight * g_info_loss + 
                     self.hparams.feature_matching_weight * g_fm_loss)

            # Log results
            self.log(f"{self.state}/g_gan_loss", g_gan_loss)
            self.log(f"{self.state}/g_info_loss", g_info_loss)
            self.log(f"{self.state}/g_fm_loss", g_fm_loss)
            self.log(f"{self.state}/g_loss", g_loss)
            
            # Log text reconstruction every 100 steps
            if batch_idx % 100 == 0:
                reconstructed_embs, _ = self.reconstruct(real_embs[:3])  # Sample first 3
                self.log_text_reconstruction(real_embs[:3], reconstructed_embs, "train")

            return {"loss": g_loss}

        # Train the Discriminator
        if optimizer_idx == 1:
            # Label smoothing for both real and fake labels
            real_labels = torch.ones(batch_size, 1, device=self.device) * 0.9
            fake_labels = torch.zeros(batch_size, 1, device=self.device) + 0.1
            real_labels = real_labels.type_as(real_embs)
            fake_labels = fake_labels.type_as(real_embs)

            # Generate fake embeddings 
            fake_embs = self.gen(z, c).detach()  # (bs, emb_dim)
            
            # Compute Discriminator's predictions
            _, _, d_real_probs, _ = self.dis(real_embs)  # (bs, 1)
            _, _, d_fake_probs, e_fake_logits = self.dis(fake_embs)  # (bs, 1), (bs, code_dim)

            # Compute loss with R1 regularization (optional)
            d_real_loss = F.binary_cross_entropy(d_real_probs, real_labels)
            d_fake_loss = F.binary_cross_entropy(d_fake_probs, fake_labels)
            d_gan_loss = (d_real_loss + d_fake_loss) / 2
            
            # InfoGAN mutual information loss
            d_info_loss = F.cross_entropy(e_fake_logits, c)
            d_loss = d_gan_loss + self.hparams.reg_info_weight * d_info_loss

            # Compute Discriminator's accuracy
            d_real_preds = (d_real_probs > 0.5).float()
            d_fake_preds = (d_fake_probs > 0.5).float()
            d_real_acc = (d_real_preds == (real_labels > 0.5).float()).float().mean()
            d_fake_acc = (d_fake_preds == (fake_labels > 0.5).float()).float().mean()
            d_acc = (d_real_acc + d_fake_acc) / 2

            # Compute Encoder's accuracy
            e_preds = torch.argmax(e_fake_logits, dim=-1)  # (bs)
            e_acc = (e_preds == c).float().mean()

            # Log results
            self.log(f"{self.state}/d_real_loss", d_real_loss)
            self.log(f"{self.state}/d_fake_loss", d_fake_loss)
            self.log(f"{self.state}/d_gan_loss", d_gan_loss)
            self.log(f"{self.state}/d_info_loss", d_info_loss)
            self.log(f"{self.state}/d_loss", d_loss)
            self.log(f"{self.state}/d_real_acc", d_real_acc)
            self.log(f"{self.state}/d_fake_acc", d_fake_acc)
            self.log(f"{self.state}/d_acc", d_acc)
            self.log(f"{self.state}/e_acc", e_acc)

            return {"loss": d_loss}

        # Train the Prior Distribution
        if optimizer_idx == 2:
            # Compute Discriminator's predictions
            _, _, _, e_real_logits = self.dis(real_embs)  # (bs, code_dim)

            # Update scheduler
            self.pl_counter *= 0.999

            # Compute loss
            p_reg_loss = -(torch.softmax(self.prior, dim=-1) * torch.log_softmax(self.prior, dim=-1)).sum(dim=-1)
            p_ce_loss = -(torch.softmax(e_real_logits, dim=-1) * torch.log_softmax(self.prior, dim=-1)).sum(dim=-1).mean()
            p_loss = p_ce_loss - self.pl_counter * self.hparams.reg_prior_weight * p_reg_loss

            # Log results
            self.log(f"{self.state}/p_reg_loss", p_reg_loss)
            self.log(f"{self.state}/p_ce_loss", p_ce_loss)
            self.log(f"{self.state}/p_loss", p_loss)
            # TODO: Find a better way to plot the prior distribution
            for i in range(len(self.prior)):
                self.log(f"{self.state}/c{i}", self.prior[i])

            return {"loss": p_loss}

    def configure_optimizers(self):
        g_lr = self.hparams.g_lr
        g_b1 = self.hparams.g_b1
        g_b2 = self.hparams.g_b2
        d_lr = self.hparams.d_lr
        d_b1 = self.hparams.d_b1
        d_b2 = self.hparams.d_b2
        p_lr = self.hparams.p_lr
        p_b1 = self.hparams.p_b1
        p_b2 = self.hparams.p_b2

        # Use AdamW for better weight decay handling
        opt_g = torch.optim.AdamW(self.gen.parameters(), lr=g_lr, betas=(g_b1, g_b2), weight_decay=1e-4)
        opt_d = torch.optim.AdamW(self.dis.parameters(), lr=d_lr, betas=(d_b1, d_b2), weight_decay=1e-4)
        opt_p = torch.optim.AdamW([self.prior], lr=p_lr, betas=(p_b1, p_b2))

        # Add cosine annealing scheduler for better convergence
        g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=1000, eta_min=g_lr*0.1)
        d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=1000, eta_min=d_lr*0.1)
        p_scheduler = get_constant_schedule_with_warmup(opt_p, num_warmup_steps=self.hparams.num_warmup_steps)

        return [opt_g, opt_d, opt_p], [g_scheduler, d_scheduler, p_scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        """Override optimizer step to add gradient clipping"""
        # Gradient clipping
        if optimizer_idx == 0:  # Generator
            torch.nn.utils.clip_grad_norm_(self.gen.parameters(), self.hparams.grad_clip_norm)
        elif optimizer_idx == 1:  # Discriminator  
            torch.nn.utils.clip_grad_norm_(self.dis.parameters(), self.hparams.grad_clip_norm)
        
        # Call the parent optimizer step
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)
    
    def setup_text_data(self, text_data):
        """Setup text data for logging"""
        print(f"InfoGAN setup_text_data called with: {type(text_data)}")
        if text_data:
            print(f"Text data keys: {text_data.keys() if hasattr(text_data, 'keys') else 'No keys'}")
            texts = text_data.get('texts')
            embeddings = text_data.get('embeddings')
            print(f"Texts available: {texts is not None}")
            print(f"Embeddings available: {embeddings is not None}")
            if texts:
                print(f"Number of text samples: {len(texts)}")
                print(f"Sample text: {texts[0][:100] if len(texts) > 0 else 'No samples'}")
            if embeddings is not None:
                print(f"Embeddings shape: {embeddings.shape}")
        
        if text_data and text_data.get('texts') and text_data.get('embeddings') is not None:
            self.text_samples = text_data['texts']
            # Move to device when available
            if hasattr(self, 'device'):
                self.text_embeddings = text_data['embeddings'].to(self.device)
            else:
                self.text_embeddings = text_data['embeddings']
            print(f"✓ Setup text data with {len(self.text_samples)} samples")
            print(f"✓ Text embeddings shape: {self.text_embeddings.shape}")
        else:
            print("✗ Warning: No valid text data provided for InfoGAN logging")
            self.text_samples = None
            self.text_embeddings = None
    
    def find_nearest_text(self, query_embs, k=1):
        """Find nearest text for embeddings using cosine similarity"""
        if self.text_embeddings is None or self.text_samples is None:
            return ["[No text data available]"] * len(query_embs)
        
        # Handle 3D embeddings by mean pooling over sequence dimension
        if query_embs.dim() == 3:
            query_embs = query_embs.mean(dim=1)  # [batch, seq_len, hidden] -> [batch, hidden]
        
        ref_embeddings = self.text_embeddings
        if ref_embeddings.dim() == 3:
            ref_embeddings = ref_embeddings.mean(dim=1)  # [batch, seq_len, hidden] -> [batch, hidden]
        
        # Ensure both tensors are on the same device
        device = query_embs.device
        ref_embeddings = ref_embeddings.to(device)
        
        # Normalize embeddings
        query_norm = F.normalize(query_embs, p=2, dim=-1)
        ref_norm = F.normalize(ref_embeddings, p=2, dim=-1)
        
        # Compute similarities
        similarities = torch.mm(query_norm, ref_norm.t())
        _, indices = torch.topk(similarities, k=1, dim=-1)
        
        nearest_texts = []
        for i in range(len(query_embs)):
            idx = indices[i, 0].item()
            if idx < len(self.text_samples):
                nearest_texts.append(self.text_samples[idx])
            else:
                nearest_texts.append("[Text not found]")
        
        return nearest_texts
    
    def generate_text_from_embeddings(self, embeddings, use_lm=True):
        """Generate text from embeddings using language model or fallback method"""
        # Handle 3D embeddings by mean pooling over sequence dimension
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)  # [batch, seq_len, hidden] -> [batch, hidden]
        
        if use_lm and hasattr(self, 'lm') and self.lm is not None:
            # Use language model for proper text generation
            try:
                generated_texts = self.lm.generate_from_embedding_simple(embeddings)
                return generated_texts
            except Exception as e:
                print(f"Warning: Language model generation failed: {e}")
                # Fall back to simple generation
        
        # Fallback: Generate based on embedding characteristics
        generated_texts = []
        for i, emb in enumerate(embeddings):
            emb_norm = torch.norm(emb).item()
            emb_mean = torch.mean(emb).item()
            emb_std = torch.std(emb).item()
            emb_max = torch.max(emb).item()
            emb_min = torch.min(emb).item()
            
            # Generate text based on embedding statistics with more sensitivity
            if emb_mean > 0.05:
                sentiment = "إيجابي"
                if emb_mean > 0.2:
                    words = ["رائع", "ممتاز", "استثنائي", "مدهش"]
                elif emb_mean > 0.1:
                    words = ["جيد", "لطيف", "مقبول", "نظيف"]
                else:
                    words = ["عادي", "بسيط", "مناسب", "كافي"]
            elif emb_mean < -0.05:
                sentiment = "سلبي"
                if emb_mean < -0.2:
                    words = ["سيء", "رهيب", "فظيع", "مقزز"]
                elif emb_mean < -0.1:
                    words = ["ضعيف", "مقبول", "عادي", "بسيط"]
                else:
                    words = ["محايد", "متوسط", "كافي", "مناسب"]
            else:
                sentiment = "محايد"
                words = ["عادي", "متوسط", "مقبول", "كافي"]
            
            # Add characteristics based on embedding properties
            if emb_std > 0.3:
                intensity = "عالي_التباين"
            elif emb_std > 0.1:
                intensity = "متوسط_التباين"
            else:
                intensity = "منخفض_التباين"
            
            # Add more variation based on norm and range
            if emb_norm > 5.0:
                scale = "كبير"
            elif emb_norm > 2.0:
                scale = "متوسط"
            else:
                scale = "صغير"
            
            # Create more varied text based on multiple characteristics
            import random
            random.seed(int((emb_norm * emb_std * 1000) % 1000))
            selected_words = random.sample(words, min(2, len(words)))
            
            generated_text = f"[{sentiment}_{intensity}_{scale}]: {' '.join(selected_words)} (norm:{emb_norm:.2f}, std:{emb_std:.3f})"
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def calculate_semantic_similarity(self, text1_list, text2_list, method="embedding"):
        """Calculate semantic similarity between two lists of texts"""
        if not text1_list or not text2_list or len(text1_list) != len(text2_list):
            return []
        
        similarities = []
        
        if method == "embedding" and hasattr(self, 'lm') and self.lm is not None:
            # Use language model embeddings for semantic similarity
            try:
                # Get embeddings for both text lists
                embs1 = self.lm.text2emb(text1_list)
                embs2 = self.lm.text2emb(text2_list)
                
                # Calculate cosine similarity
                if embs1.dim() > 2:
                    embs1 = embs1.mean(dim=1)  # Pool over sequence length
                if embs2.dim() > 2:
                    embs2 = embs2.mean(dim=1)  # Pool over sequence length
                    
                cosine_sims = F.cosine_similarity(embs1, embs2, dim=-1)
                similarities = cosine_sims.cpu().tolist()
                
            except Exception as e:
                print(f"Warning: Embedding-based similarity calculation failed: {e}")
                # Fall back to simple similarity
                method = "simple"
        
        if method == "simple":
            # Simple word overlap similarity
            for text1, text2 in zip(text1_list, text2_list):
                words1 = set(text1.split())
                words2 = set(text2.split())
                if len(words1) == 0 and len(words2) == 0:
                    similarities.append(1.0)
                elif len(words1) == 0 or len(words2) == 0:
                    similarities.append(0.0)
                else:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarities.append(intersection / union if union > 0 else 0.0)
        
        return similarities
    
    def get_original_texts_for_embeddings(self, embeddings):
        """Get original texts corresponding to embeddings by finding nearest matches"""
        if self.text_embeddings is None or self.text_samples is None:
            return ["[No original text available]"] * len(embeddings)
        
        # Handle 3D embeddings
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)
        
        ref_embeddings = self.text_embeddings
        if ref_embeddings.dim() == 3:
            ref_embeddings = ref_embeddings.mean(dim=1)
        
        # Ensure same device
        device = embeddings.device
        ref_embeddings = ref_embeddings.to(device)
        
        # Find nearest original texts
        similarities = torch.mm(
            F.normalize(embeddings, p=2, dim=-1), 
            F.normalize(ref_embeddings, p=2, dim=-1).t()
        )
        _, indices = torch.topk(similarities, k=1, dim=-1)
        
        original_texts = []
        for i in range(len(embeddings)):
            idx = indices[i, 0].item()
            if idx < len(self.text_samples):
                original_texts.append(self.text_samples[idx])
            else:
                original_texts.append("[Original text not found]")
        
        return original_texts
    
    def log_text_reconstruction(self, original_embs, reconstructed_embs, step_name="train"):
        """Log original and reconstructed text samples with semantic similarity scores to wandb"""
        try:
            # Sample a few examples for logging
            num_samples = min(3, len(original_embs))
            orig_sample = original_embs[:num_samples]
            recon_sample = reconstructed_embs[:num_samples]
            
            # Get original texts that correspond to these embeddings
            original_source_texts = self.get_original_texts_for_embeddings(orig_sample)
            
            # Generate texts from embeddings (this will show the actual reconstruction differences)
            original_generated_texts = self.generate_text_from_embeddings(orig_sample)
            reconstructed_generated_texts = self.generate_text_from_embeddings(recon_sample)
            
            # Calculate semantic similarities
            # 1. Original text vs Generated from original embedding
            orig_to_orig_gen_similarity = self.calculate_semantic_similarity(
                original_source_texts, original_generated_texts, method="simple"
            )
            
            # 2. Original text vs Generated from reconstructed embedding
            orig_to_recon_gen_similarity = self.calculate_semantic_similarity(
                original_source_texts, reconstructed_generated_texts, method="simple"
            )
            
            # 3. Generated from original vs Generated from reconstructed
            gen_to_gen_similarity = self.calculate_semantic_similarity(
                original_generated_texts, reconstructed_generated_texts, method="simple"
            )
            
            # Calculate embedding distances for additional context
            l2_distances = torch.norm(orig_sample - recon_sample, dim=-1)
            cosine_sims = F.cosine_similarity(orig_sample, recon_sample, dim=-1)
            
            # Create comprehensive wandb table
            table_data = []
            for i in range(num_samples):
                table_data.append([
                    f"Sample {i+1}",
                    original_source_texts[i][:80] + "..." if len(original_source_texts[i]) > 80 else original_source_texts[i],
                    original_generated_texts[i][:80] + "..." if len(original_generated_texts[i]) > 80 else original_generated_texts[i],
                    reconstructed_generated_texts[i][:80] + "..." if len(reconstructed_generated_texts[i]) > 80 else reconstructed_generated_texts[i],
                    f"{orig_to_orig_gen_similarity[i]:.3f}" if i < len(orig_to_orig_gen_similarity) else "N/A",
                    f"{orig_to_recon_gen_similarity[i]:.3f}" if i < len(orig_to_recon_gen_similarity) else "N/A",
                    f"{gen_to_gen_similarity[i]:.3f}" if i < len(gen_to_gen_similarity) else "N/A",
                    f"{l2_distances[i]:.3f}",
                    f"{cosine_sims[i]:.3f}"
                ])
            
            table = wandb.Table(data=table_data, columns=[
                "Sample",
                "Original Text", 
                "Generated from Original", 
                "Generated from Reconstructed",
                "Orig↔OrigGen Sim",
                "Orig↔ReconGen Sim", 
                "OrigGen↔ReconGen Sim",
                "Embedding L2 Dist",
                "Embedding Cos Sim"
            ])
            wandb.log({f"{step_name}_text_reconstruction": table})
            
            # Log comprehensive similarity statistics
            if orig_to_orig_gen_similarity:
                wandb.log({
                    f"{step_name}_semantic_sim_orig_to_orig_gen_mean": sum(orig_to_orig_gen_similarity) / len(orig_to_orig_gen_similarity),
                    f"{step_name}_semantic_sim_orig_to_recon_gen_mean": sum(orig_to_recon_gen_similarity) / len(orig_to_recon_gen_similarity),
                    f"{step_name}_semantic_sim_gen_to_gen_mean": sum(gen_to_gen_similarity) / len(gen_to_gen_similarity),
                })
                
                # Calculate semantic preservation ratio (how much semantic meaning is preserved)
                orig_gen_mean = sum(orig_to_orig_gen_similarity) / len(orig_to_orig_gen_similarity)
                recon_gen_mean = sum(orig_to_recon_gen_similarity) / len(orig_to_recon_gen_similarity)
                preservation_ratio = recon_gen_mean / orig_gen_mean if orig_gen_mean > 0 else 0
                
                wandb.log({
                    f"{step_name}_semantic_preservation_ratio": preservation_ratio,
                })
            
            # Also log embedding distance statistics
            wandb.log({
                f"{step_name}_embedding_l2_mean": l2_distances.mean().item(),
                f"{step_name}_embedding_cosine_mean": cosine_sims.mean().item(),
                f"{step_name}_embedding_l2_std": l2_distances.std().item(),
                f"{step_name}_embedding_cosine_std": cosine_sims.std().item(),
            })
            
            print(f"Logged text reconstruction with semantic similarities for {step_name}")
            
        except Exception as e:
            print(f"Warning: Could not log text reconstruction: {e}")
            import traceback
            traceback.print_exc()

    def validation_step(self, batch, batch_idx):
        """Override validation step to add text reconstruction logging"""
        # Call parent validation
        result = super().shared_eval_step(batch, batch_idx)
        
        # Log text reconstruction for validation - track progress across epochs
        if batch_idx == 0:  # Only log for first validation batch
            embs = batch["clean_emb"]
            if isinstance(embs, list):
                embs = embs[0]
            
            # Use more samples for validation to get better statistics
            num_samples = min(10, len(embs))
            sample_embs = embs[:num_samples]
            
            # Reconstruct the embeddings using InfoGAN
            reconstructed_embs, reconstruction_losses = self.reconstruct(sample_embs)
            
            # Log detailed reconstruction comparison
            self.log_validation_text_reconstruction(
                sample_embs, reconstructed_embs, reconstruction_losses, 
                f"validation_epoch_{self.current_epoch}"
            )
        
        return result

    def log_validation_text_reconstruction(self, original_embs, reconstructed_embs, reconstruction_losses, step_name="validation"):
        """Simple validation logging to track original vs reconstructed text across epochs"""
        try:
            num_samples = len(original_embs)
            
            # Get the corresponding original texts for these embeddings
            original_texts = self.get_original_texts_for_embeddings(original_embs)
            
            # Find nearest texts for reconstructed embeddings 
            reconstructed_nearest_texts = self.find_nearest_text(reconstructed_embs)
            
            # Calculate cosine similarities
            cosine_similarities = F.cosine_similarity(original_embs, reconstructed_embs, dim=-1)
            
            # Create simple table with only requested columns
            table_data = []
            for i in range(num_samples):
                # Truncate long texts for display
                orig_text_display = (original_texts[i][:100] + "...") if len(original_texts[i]) > 100 else original_texts[i]
                recon_text_display = (reconstructed_nearest_texts[i][:100] + "...") if len(reconstructed_nearest_texts[i]) > 100 else reconstructed_nearest_texts[i]
                
                table_data.append([
                    orig_text_display,
                    recon_text_display,
                    f"{cosine_similarities[i].item():.3f}",
                    f"{reconstruction_losses[i].item():.3f}" if reconstruction_losses.dim() > 0 else f"{reconstruction_losses.item():.3f}"
                ])
            
            # Log the table
            table = wandb.Table(data=table_data, columns=[
                "Original Text",
                "Reconstructed Text", 
                "Cosine Similarity",
                "Reconstruction Loss"
            ])
            wandb.log({step_name: table})
            
            # Log average cosine similarity for tracking progress across epochs
            avg_cosine_similarity = cosine_similarities.mean().item()
            avg_reconstruction_loss = reconstruction_losses.mean().item()
            
            wandb.log({
                f"{step_name}_avg_cosine_similarity": avg_cosine_similarity,
                f"{step_name}_avg_reconstruction_loss": avg_reconstruction_loss,
                f"{step_name}_epoch": self.current_epoch,
            })
            
            print(f"✓ Epoch {self.current_epoch} - Avg Cosine Sim: {avg_cosine_similarity:.4f}, Avg Recon Loss: {avg_reconstruction_loss:.4f}")
            
        except Exception as e:
            print(f"Warning: Could not log validation text reconstruction: {e}")
            import traceback
            traceback.print_exc()

    def on_train_epoch_end(self):
        """Log adaptive k factor at end of epoch"""
        self.log("adaptive_k_factor", self.k_adaptation_factor)
        self.log("avg_reconstruction_loss", self.avg_reconstruction_loss)
