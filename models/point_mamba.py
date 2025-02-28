from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message  
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

#from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pytorch3d.loss import chamfer_distance
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

#from knn_cuda import KNN
from .block import Block
from .build import MODELS

from pytorch3d.ops import knn_points, sample_farthest_points    


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        #self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        ##center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # import ipdb; ipdb.set_trace()
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        center = sample_farthest_points(points = xyz, K = self.num_group)     
        center = center[0]      
        # knn to get the neighborhood
        ###_, idx = self.knn(xyz, center)  # B G M  
        idx = knn_points(center.cuda(), xyz.cuda(), K=self.group_size, return_sorted=False)     
        idx = idx.idx
        idx = idx.long()
        #########_, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood_org = neighborhood
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, neighborhood_org


class Group_Affine(nn.Module):  # FPS + KNN 
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        #self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, affine_alpha, affine_beta):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        ########center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # import ipdb; ipdb.set_trace()
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        center = sample_farthest_points(points = xyz, K = self.num_group)     
        center = center[0]      
        # knn to get the neighborhood
        #_, idx = self.knn(xyz, center)  # B G M  
        idx = knn_points(center.cuda(), xyz.cuda(), K=self.group_size, return_sorted=False)     
        idx = idx.idx
        idx = idx.long()
        #########_, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood_org = neighborhood
        #neighborhood = neighborhood - center.unsqueeze(2)     
        mean = torch.mean(neighborhood, dim=2, keepdim=True)
        std = torch.std((neighborhood - mean).reshape(batch_size, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        neighborhood = (neighborhood - mean) / (std + 1e-5)
        neighborhood = affine_alpha * neighborhood + affine_beta
        return neighborhood, center, neighborhood_org   


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


@MODELS.register_module()
class PointMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointMamba, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)


        self.pos_embed = nn.Sequential(   
            nn.Linear(3, 128),     
            nn.GELU(),
            nn.Linear(128, self.trans_dim)  
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,    
                                 drop_path=self.drop_path)

        self.norm = nn.LayerNorm(self.trans_dim) 

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )


        self.build_loss_func()

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

        self.method = config.method
        self.reverse = config.reverse
        self.k_top_eigenvectors = config.k_top_eigenvectors
        self.smallest = config.smallest 
        self.knn_graph = config.knn_graph 
        self.level = config.level
        self.steps = config.steps
        self.sign = config.sign
        self.affine_geometric = config.affine_geometric
        self.alpha = config.alpha
        self.all_distance = config.all_distance  
        self.symmetric = config.symmetric
        self.self_loop = config.self_loop 


        if (self.method == "k_top_eigenvectors_wighted_traversing" or self.method == "k_top_eigenvectors_wighted_traversing_2"):
            self.learnable_tokens = nn.Parameter(torch.zeros(1, 1, self.trans_dim)).cuda() 
            trunc_normal_(self.learnable_tokens, std=.02)  


        if (self.method == "k_top_eigenvectors_wighted_traversing" or self.method == "k_top_eigenvectors_wighted_traversing_2"): 
            self.cls_head_learnable = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )    

        if (self.affine_geometric):
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, 3]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, 3]))
            self.group_divider_affine = Group_Affine(num_group=self.num_group, group_size=self.group_size)

    def build_loss_func(self): 
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def create_graph_from_feature_space_gpu(self, points, k=5, symmetric= False, self_loop= False):
        """
        Create a graph from point cloud data in a feature space using k-nearest neighbors, optimized for GPU.
        
        Parameters:
        - points: Tensor of shape (B, N, F) where B is the batch size,
                N is the number of points in the point cloud, and F is the feature dimensions.
                This tensor should already be on the GPU.
        - k: The number of nearest neighbors to consider for graph edges.
        
        Returns:
        - adjacency_matrix: Tensor of shape (B, N, N) representing the adjacency matrix of the graph.
        """
        # Ensure the input tensor is on the GPU 
        points = points.to('cuda')
        
        B, N, _ = points.shape 
        
        # Compute pairwise distances, shape (B, N, N), on GPU
        dist_matrix = torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1)
        
        # Find the k-nearest neighbors for each point, excluding itself
        _, indices = torch.topk(-dist_matrix, k=k+1, largest=True, dim=-1)
        if (self_loop):
            indices = indices[:, :, :]  # Remove self-loops 
        else:
            indices = indices[:, :, 1:]  # Remove self-loops 
        
        # Create adjacency matrix on GPU
        adjacency_matrix = torch.zeros(B, N, N, device='cuda')
        b_idx = torch.arange(B, device='cuda')[:, None, None]
        adjacency_matrix[b_idx, torch.arange(N, device='cuda')[:, None], indices] = 1
        if (symmetric):
            adjacency_matrix[b_idx, indices, torch.arange(N, device='cuda')[:, None]] = 1  # Ensure symmetry  
        
        return adjacency_matrix


    def create_graph_from_feature_space_gpu_weighted_adjacency(self, points, k=5, alpha=1, symmetric = False, self_loop = False):
        """
        Create a graph from point cloud data in a feature space using k-nearest neighbors with Euclidean distances as weights, optimized for GPU.
        
        Parameters:
        - points: Tensor of shape (B, N, F) where B is the batch size,
                N is the number of points in the point cloud, and F is the feature dimensions.
                This tensor should already be on the GPU.
        - k: The number of nearest neighbors to consider for graph edges.
        
        Returns:
        - adjacency_matrix: Tensor of shape (B, N, N) representing the weighted adjacency matrix of the graph,
                            where weights are the Euclidean distances between points.
        """
        points = points.to('cuda')
        B, N, _ = points.shape
        
        # Compute pairwise Euclidean distances, shape (B, N, N), on GPU
        dist_matrix = torch.sqrt(torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1))
        
        # Find the k-nearest neighbors for each point, including itself
        distances, indices = torch.topk(-dist_matrix, k=k+1, largest=True, dim=-1)
        #distances, indices = -distances[:, :, 1:], indices[:, :, 1:]  # Remove self-loops
        distances, indices = -distances[:, :, :], indices[:, :, :]  

        if (self_loop):
            indices = indices[:, :, :]  # Remove self-loops    
            distances = distances
        else:
            indices = indices[:, :, 1:]  # Remove self-loops 
            distances = distances[..., 1:]

        # Create a weighted adjacency matrix on GPU 
        adjacency_matrix = torch.zeros(B, N, N, device='cuda') 
        b_idx = torch.arange(B, device='cuda')[:, None, None]
        n_idx = torch.arange(N, device='cuda')[:, None]
        
        # Use gathered distances as weights
        distances_new = torch.exp((-1) * alpha * (distances)**2)
        adjacency_matrix[b_idx, n_idx, indices] = distances_new
        if (symmetric):
            adjacency_matrix[b_idx, indices, n_idx] = distances_new  # Ensure symmetry    

        dist_matrix_new = torch.exp((-1) * alpha * (dist_matrix)**2)  
        
        return adjacency_matrix, dist_matrix_new 


    def calc_top_k_eigenvalues_eigenvectors(self, adj_matrices, k, smallest):  
        """
        Calculate the top k eigenvalues and eigenvectors of a batch of adjacency matrices 
        using Random-walk normalization.

        Parameters:
        - adj_matrices: A tensor of shape (B, 128, 128) representing a batch of adjacency matrices.
        - k: The number of top eigenvalues (and corresponding eigenvectors) to return.

        Returns:
        - top_k_eigenvalues: A tensor of shape (B, k) containing the top k eigenvalues for each matrix.
        - top_k_eigenvectors: A tensor of shape (B, 128, k) containing the eigenvectors corresponding to the top k eigenvalues for each matrix.
        """
        B, N, _ = adj_matrices.shape
        top_k_eigenvalues = torch.zeros((B, k+1)).cuda()
        top_k_eigenvectors = torch.zeros((B, N, k+1)).cuda()   
        #top_k_eigenvalues = torch.zeros((B, k)).cuda()
        #top_k_eigenvectors = torch.zeros((B, N, k)).cuda()  
        eigenvalues_l = torch.zeros((B, N)).cuda()
        eigenvectors_l = torch.zeros((B, N, N)).cuda()

        for i in range(B):
            # Extract the i-th adjacency matrix
            A = adj_matrices[i]

            # Ensure A is symmetric
            #A = (A + A.t()) / 2

            # Compute the degree matrix D
            D = torch.diag(torch.sum(A, dim=1))

            # Compute D^-1
            D_inv = torch.diag(1.0 / torch.diag(D))

            # Perform Random-walk normalization: D^-1 * A
            ####normalized_A = torch.matmul(D_inv, A)
            I = torch.eye(N).cuda()    
            normalized_A = I - torch.matmul(D_inv, A)       
 
            # Compute eigenvalues and eigenvectors
            #eigenvalues, eigenvectors = torch.linalg.eigh(normalized_A)
            eigenvalues, eigenvectors = torch.linalg.eig(normalized_A)   
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real

            # Select the top k eigenvalues and corresponding eigenvectors
            if (smallest == False):
                top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=True, sorted=True) 
                #top_vals, top_indices = torch.topk(eigenvalues, k, largest=True, sorted=True)  
                top_vecs = eigenvectors[:, top_indices]
            else:
                top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=False, sorted=True) 
                #top_vals, top_indices = torch.topk(eigenvalues, k, largest=False, sorted=True) 
                top_vecs = eigenvectors[:, top_indices]   

            # Store the results
            top_k_eigenvalues[i] = top_vals
            top_k_eigenvectors[i, :, :] = top_vecs                 

            eigenvalues_l[i] = eigenvalues
            eigenvectors_l[i, :, :] = eigenvectors    

        return top_k_eigenvalues[:, 1:], top_k_eigenvectors[:, :, 1:], eigenvalues_l, eigenvectors_l
        #return top_k_eigenvalues, top_k_eigenvectors, eigenvalues_l, eigenvectors_l   
    


    def calc_top_k_eigenvalues_eigenvectors__(self, adj_matrices, k, smallest):  
        """
        Calculate the top k eigenvalues and eigenvectors of a batch of adjacency matrices 
        using Random-walk normalization.

        Parameters:
        - adj_matrices: A tensor of shape (B, 128, 128) representing a batch of adjacency matrices.
        - k: The number of top eigenvalues (and corresponding eigenvectors) to return.

        Returns:
        - top_k_eigenvalues: A tensor of shape (B, k) containing the top k eigenvalues for each matrix.
        - top_k_eigenvectors: A tensor of shape (B, 128, k) containing the eigenvectors corresponding to the top k eigenvalues for each matrix.
        """
        B, N, _ = adj_matrices.shape

        # Ensure A is symmetric
        adj_matrices = (adj_matrices + adj_matrices.transpose(1, 2)) / 2

        # Compute the degree matrices D and D^-1
        degrees = torch.sum(adj_matrices, dim=2)
        D_inv = torch.zeros_like(adj_matrices)
        for i in range(B):
            D_inv[i] = torch.diag(1.0 / degrees[i])

        # Perform Random-walk normalization: D^-1 * A
        I = torch.eye(N, device=adj_matrices.device)
        normalized_A = I.unsqueeze(0).repeat(B, 1, 1) - torch.matmul(D_inv, adj_matrices)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(normalized_A)   
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        # Select the top k eigenvalues and corresponding eigenvectors
        if smallest:
            top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=False, sorted=True)
        else:
            top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=True, sorted=True)
        
        # Gather the top eigenvectors
        top_vecs = torch.gather(eigenvectors, 2, top_indices.unsqueeze(1).expand(-1, N, -1))

        # Slice off the smallest eigenvalue and corresponding eigenvector
        return top_vals[:, 1:], top_vecs[:, :, 1:], eigenvalues, eigenvectors




    def calc_top_k_eigenvalues_eigenvectors_symmetric(self, adj_matrices, k, smallest=True):
        B, N, _ = adj_matrices.shape
        top_k_eigenvalues = torch.zeros((B, k), device='cuda')
        top_k_eigenvectors = torch.zeros((B, N, k), device='cuda')
        eigenvalues_l = torch.zeros((B, N), device='cuda')
        eigenvectors_l = torch.zeros((B, N, N), device='cuda')

        for i in range(B):
            # Extract the i-th adjacency matrix
            A = adj_matrices[i].to('cuda')

            # Ensure A is symmetric
            A = (A + A.t()) / 2

            # Compute the degree matrix D
            D = torch.diag_embed(torch.sum(A, dim=1))

            # Compute D^(-1/2)
            D_inv_sqrt = torch.diag_embed(torch.pow(torch.diag(D), -0.5))

            # Compute the symmetric normalized Laplacian
            I = torch.eye(N, device='cuda')
            L_sym = I - torch.matmul(D_inv_sqrt, torch.matmul(A, D_inv_sqrt))

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)

            # Select the top k eigenvalues and corresponding eigenvectors
            if not smallest:
                top_vals, top_indices = torch.topk(eigenvalues, k, largest=True, sorted=True)
                top_vecs = eigenvectors[:, top_indices]
            else:
                top_vals, top_indices = torch.topk(eigenvalues, k, largest=False, sorted=True)
                top_vecs = eigenvectors[:, top_indices]

            # Store the results
            top_k_eigenvalues[i] = top_vals
            top_k_eigenvectors[i, :, :] = top_vecs

            eigenvalues_l[i] = eigenvalues
            eigenvectors_l[i, :, :] = eigenvectors

        return top_k_eigenvalues, top_k_eigenvectors, eigenvalues_l, eigenvectors_l


    def sort_points_by_fiedler(self, points, fiedler_vector):
        """
        Sorts points based on the Fiedler vector, optimized for GPU execution.

        Parameters:
        - points: Tensor of shape (B, N, 3) where B is the batch size, 
                N is the number of points in the point cloud, and 3 are the xyz coordinates.
        - fiedler_vector: The Fiedler vector used for sorting, of shape (B, N).

        Returns:
        - sorted_points: Tensor of sorted points.
        """
        B, N, _ = points.shape
        # Generate indices from Fiedler vector for sorting
        fiedler_vector = fiedler_vector.real         
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)
        
        # Expand the indices to work for x, y, z coordinates
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 384)      
        
        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, expanded_indices)
        
        return sorted_points 
    
    def sort_points_by_fiedler_for_center(self, points, fiedler_vector):
        """
        Sorts points based on the Fiedler vector, optimized for GPU execution.

        Parameters:
        - points: Tensor of shape (B, N, 3) where B is the batch size, 
                N is the number of points in the point cloud, and 3 are the xyz coordinates.
        - fiedler_vector: The Fiedler vector used for sorting, of shape (B, N).

        Returns:
        - sorted_points: Tensor of sorted points.
        """
        B, N, _ = points.shape
        # Generate indices from Fiedler vector for sorting
        fiedler_vector = fiedler_vector.real         
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)
        
        # Expand the indices to work for x, y, z coordinates
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)      
         
        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, expanded_indices)
        
        return sorted_points


    def sort_points_by_fiedler_for_position(self, points, fiedler_vector):
        """
        Sorts points based on the Fiedler vector, optimized for GPU execution.

        Parameters:
        - points: Tensor of shape (B, N, 3) where B is the batch size, 
                N is the number of points in the point cloud, and 3 are the xyz coordinates.
        - fiedler_vector: The Fiedler vector used for sorting, of shape (B, N).

        Returns:
        - sorted_points: Tensor of sorted points.
        """
        B, N = points.shape
        # Generate indices from Fiedler vector for sorting
        fiedler_vector = fiedler_vector.real         
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)
        
        # Expand the indices to work for x, y, z coordinates
        #expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)      
         
        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, sorted_indices)
        
        return sorted_points 
    
    def sort_points_by_fiedler_for_neighberhood(self, points, fiedler_vector):
        """
        Sorts points based on the Fiedler vector, optimized for GPU execution.

        Parameters:
        - points: Tensor of shape (B, N, 3) where B is the batch size, 
                N is the number of points in the point cloud, and 3 are the xyz coordinates.
        - fiedler_vector: The Fiedler vector used for sorting, of shape (B, N).

        Returns:
        - sorted_points: Tensor of sorted points.
        """
        B, N, _, _ = points.shape
        # Generate indices from Fiedler vector for sorting
        fiedler_vector = fiedler_vector.real         
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)
        
        # Expand the indices to work for x, y, z coordinates
        expanded_indices = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3)      
        
        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, expanded_indices)
        
        return sorted_points 
    

    def multilevel_travers(self, eigen_vectors, level):
        # Calculate mean for each batch element
        means = eigen_vectors.mean(dim=1, keepdim=True)
        # Compute binaries for each batch element
        binaries = eigen_vectors >= means 
        # Take the first 'level' rows and transpose
        binaries = binaries[:, :, :level]
        # Calculate integers
        num_bits = level
        powers_of_2 = 2 ** torch.arange(start=num_bits-1, end=-1, step=-1, device=eigen_vectors.device)
        integers = torch.sum(binaries * powers_of_2[None, None, :], dim=-1, keepdim=True)
        return integers.squeeze()
    

    def multilevel_travers_2(self, eigen_vectors, level, i):
        # Calculate mean for each batch element
        means = eigen_vectors.mean(dim=1, keepdim=True)
        # Compute binaries for each batch element
        binaries = eigen_vectors >= means 
        # Take the first 'level' rows and transpose
        binaries = binaries[:, :, int(i*level):(i+1)*level]
        # Calculate integers
        num_bits = level
        powers_of_2 = 2 ** torch.arange(start=num_bits-1, end=-1, step=-1, device=eigen_vectors.device)
        integers = torch.sum(binaries * powers_of_2[None, None, :], dim=-1, keepdim=True)
        return integers.squeeze()


    def int_to_binary(self, tensor, bit_length=4):
        # Flatten the tensor to handle it easily
        flat_tensor = tensor.flatten()
        # Convert each integer to binary format, padded to 'bit_length' bits
        binary_format = [int(np.binary_repr(num, width=bit_length)) for num in flat_tensor]
        # Reshape the list of binary strings back to the original tensor shape
        binary_tensor = np.array(binary_format).reshape(tensor.shape[0], tensor.shape[1])
        return torch.tensor(binary_tensor).cuda() 
    
    
    def unpack_binary_integers_vectorized(self, binary_tensor): 
        """ Converts an integer tensor where each integer is assumed to be a binary encoded number. """
        # Calculate the binary tensor shape and bit depth
        B, N = binary_tensor.shape
        bit_depth = 4  # Adjust this based on your expected binary length

        # Convert integers to binary strings with padding and split into separate bits
        binary_strings = np.vectorize(lambda x: format(x, '0{}b'.format(bit_depth)))(binary_tensor)
        bit_tensor = np.array([[list(map(int, list(bits))) for bits in batch] for batch in binary_strings])

        # Convert the result to a PyTorch tensor
        return torch.tensor(bit_tensor, dtype=torch.int8)  
    
    def partition_centers_by_eigenvectors(self, top_k_eigenvectors):
        B, N, _ = top_k_eigenvectors.shape  # Batch size, Number of points, Eigenvector dimensions
        partition_indices = [[] for _ in range(B)]  # List to hold partition indices for each batch

        for batch_idx in range(B):
            current_partitions = [[torch.arange(N).cuda()]]  # Start with all indices in one partition

            for dim in range(top_k_eigenvectors.shape[2]):  # Iterate through each eigenvector dimension
                new_partitions = []

                for partition in current_partitions[-1]:  # Iterate through current partitions
                    # Select the current partition's eigenvector values for the current dimension
                    eig_values = top_k_eigenvectors[batch_idx, partition, dim]
                    mean_value = eig_values.mean()  # Compute mean
                    
                    # Divide indices into two bins based on the mean
                    below_mean = partition[eig_values < mean_value]
                    above_mean = partition[eig_values >= mean_value]

                    # Add new partitions to the list
                    if len(below_mean) > 0:
                        new_partitions.append(below_mean)
                    if len(above_mean) > 0:
                        new_partitions.append(above_mean)

                current_partitions.append(new_partitions)  # Append new partitions for this dimension

            partition_indices[batch_idx] = current_partitions  # Save partitions for this batch

        return partition_indices


    def shuffle_and_reconcat_tensors(self, group_input_tokens, pos, center, partition_indices):
        B, N, D = group_input_tokens.shape  # Assuming shape is (B, 128, 384)
        #sorted_group_input_tokens = torch.empty_like(group_input_tokens)
        sorted_group_input_tokens = torch.empty(B, int(N*2), D)
        #sorted_pos = torch.empty_like(pos)
        sorted_pos = torch.empty(B, int(N*2), D)    
        sorted_center = torch.empty(B, int(N*2), 3)   

        for b in range(B):
            shuffled_tensors_group = []
            shuffled_tensors_pos = []
            shuffled_tensors_center = []

            total_indices = 0  # To verify that sum of indices equals 128 after processing all partitions

            for partition in partition_indices[b][0]:  
                # Extracting the specific indices for this partition
                indices = partition  # Ensure indices are long type for indexing

                # Selecting the tokens for the current partition from both tensors
                selected_group = group_input_tokens[b, indices]
                selected_pos = pos[b, indices]
                selected_center = center[b, indices]

                # Randomly shuffling the order
                idx = torch.randperm(indices.shape[0])
                shuffled_group = selected_group[idx]
                shuffled_pos = selected_pos[idx]
                shuffled_center = selected_center[idx]

                shuffled_group_inverse = shuffled_group.flip(0)
                shuffled_pos_inverse = shuffled_pos.flip(0)
                shuffled_center_inverse = shuffled_center.flip(0)

                shuffled_group = torch.cat((shuffled_group, shuffled_group_inverse), 0)
                shuffled_pos = torch.cat((shuffled_pos, shuffled_pos_inverse), 0)
                shuffled_center = torch.cat((shuffled_center, shuffled_center_inverse), 0)

                # Accumulating the shuffled tensors
                shuffled_tensors_group.append(shuffled_group)         
                shuffled_tensors_pos.append(shuffled_pos)
                shuffled_tensors_center.append(shuffled_center)

                total_indices += indices.shape[0]   

            # Ensuring the sum of all indices equals 128
            assert total_indices == N, "Sum of partition indices does not equal 128"

            # Concatenating the shuffled tensors for the current batch
            sorted_group_input_tokens[b] = torch.cat(shuffled_tensors_group, dim=0)
            sorted_pos[b] = torch.cat(shuffled_tensors_pos, dim=0)
            sorted_center[b] = torch.cat(shuffled_tensors_center, dim=0)

        return sorted_group_input_tokens, sorted_pos, sorted_center.cuda() 

    def get_sinusoid_encoding_table(self, position_tensor, d_hid):
        ''' Sinusoid position encoding table using an existing position tensor with batch dimension '''
        # Ensure the position tensor is a floating point tensor for proper division
        position = position_tensor.float()  # shape [B, 128, 1], where B is batch size
        
        # Compute the div term for d_hid dimensions (assuming only even indices for cos and sin)
        div_term = torch.exp(torch.arange(0, d_hid, 2).float() * -(math.log(10000.0) / d_hid)).to(position.device)
        
        # Repeat div_term for all batch elements and positions
        div_term = div_term.repeat(position.shape[0], position.shape[1], 1)
        
        # Prepare the position tensor for broadcasting
        position = position.repeat(1, 1, d_hid // 2)
        
        # Compute the sinusoidal terms
        sinusoid_table = torch.zeros(position.shape[0], position.shape[1], d_hid).to(position.device)
        sinusoid_table[:, :, 0::2] = torch.sin(position * div_term)
        sinusoid_table[:, :, 1::2] = torch.cos(position * div_term)
        
        return sinusoid_table
    

    def get_sinusoid_encoding_table_2(self, position_tensor, d_hid):
        ''' Sinusoid position encoding table for modified input shape (B, 1, 1) and output shape (B, 1, 384) '''
        # Ensure the position tensor is a floating point tensor for proper division
        position = position_tensor.float()  # shape [B, 1, 1], where B is batch size
        
        # Compute the div term for d_hid dimensions (assuming only even indices for cos and sin)
        div_term = torch.exp(torch.arange(0, d_hid, 2).float() * -(math.log(10000.0) / d_hid)).to(position.device)
        
        # Repeat div_term for all batch elements and positions
        div_term = div_term.repeat(position.shape[0], position.shape[1], 1)
        
        # Prepare the position tensor for broadcasting
        position = position.repeat(1, 1, d_hid // 2)
        
        # Compute the sinusoidal terms
        sinusoid_table = torch.zeros(position.shape[0], position.shape[1], d_hid).to(position.device)
        sinusoid_table[:, :, 0::2] = torch.sin(position * div_term)
        sinusoid_table[:, :, 1::2] = torch.cos(position * div_term)
        
        return sinusoid_table 
    

    def modify_sign_and_same_eigenvalue(self, top_k_eigenvalues, top_k_eigenvectors, mode, eps = 10e-8):

        if (mode == "first_row_eigenvector"):
            first_row_eigenvectors = top_k_eigenvectors[:, 0, :]  
            first_row_eigenvectors_sign = first_row_eigenvectors > 0
            first_row_eigenvectors_sign = (first_row_eigenvectors_sign.float() * 2 - 1)

            first_row_eigenvectors_sign_repeated = first_row_eigenvectors_sign[:, None, :].repeat(1, top_k_eigenvectors.shape[1], 1)

            top_k_eigenvectors_sign_modified = top_k_eigenvectors * first_row_eigenvectors_sign_repeated

            diffs = torch.abs(top_k_eigenvalues[:, 1:] - top_k_eigenvalues[:, :-1])

            diffs_condition = diffs < eps

            # Condition for values comparison
            value_condition = first_row_eigenvectors[:, 1:] < first_row_eigenvectors[:, :-1]

            # Combine conditions
            swap_condition = diffs_condition & value_condition

            # Identify indices for swapping
            swap_indices = swap_condition.nonzero()

            if not swap_indices.numel() == 0:
                # Expand dimensions for broadcasting in assignments
                row_indices = swap_indices[:, 0]
                col_indices = swap_indices[:, 1]

                # Columns to swap
                col1 = col_indices
                col2 = col_indices + 1

                # Perform the swap
                temp = top_k_eigenvectors_sign_modified[row_indices, :, col1].clone()
                top_k_eigenvectors_sign_modified[row_indices, :, col1] = top_k_eigenvectors_sign_modified[row_indices, :, col2]
                top_k_eigenvectors_sign_modified[row_indices, :, col2] = temp

        return top_k_eigenvectors_sign_modified        


 
    def forward(self, pts): 
        if (self.affine_geometric == False):
            neighborhood, center, neighborhood_org = self.group_divider(pts)
        else:
            neighborhood, center, neighborhood_org = self.group_divider_affine(pts, self.affine_alpha, self.affine_beta)

        if (self.method == "MAMBA" or self.method == "k_top_eigenvectors" or self.method == "local_global_traverse_k_eigenvectors_2" or self.method == "k_top_eigenvectors_eigenvectors_addd_first_lat_sequence" or self.method == "k_top_eigenvectors_wighted_traversing" or self.method == "k_top_eigenvectors_wighted_traversing_2" or self.method == "Random" or self.method == "k_top_eigenvectors_2" or self.method == "k_top_eigenvectors_Christian_Method" or self.method == "k_top_eigenvectors_weighted_adj" or self.method == "k_top_eigenvectors_weighted_adj_symmetric"):   
            group_input_tokens = self.encoder(neighborhood)  # B G N
            pos = self.pos_embed(center)          


        if (self.method == "MAMBA"):

            # reordering strategy
            center_x = center[:, :, 0].argsort(dim=-1)[:, :, None]
            center_y = center[:, :, 1].argsort(dim=-1)[:, :, None]
            center_z = center[:, :, 2].argsort(dim=-1)[:, :, None]
            group_input_tokens_x = group_input_tokens.gather(dim=1, index=torch.tile(center_x, (
                1, 1, group_input_tokens.shape[-1])))
            group_input_tokens_y = group_input_tokens.gather(dim=1, index=torch.tile(center_y, (
                1, 1, group_input_tokens.shape[-1])))
            group_input_tokens_z = group_input_tokens.gather(dim=1, index=torch.tile(center_z, (
                1, 1, group_input_tokens.shape[-1])))
            pos_x = pos.gather(dim=1, index=torch.tile(center_x, (1, 1, pos.shape[-1])))
            pos_y = pos.gather(dim=1, index=torch.tile(center_y, (1, 1, pos.shape[-1])))
            pos_z = pos.gather(dim=1, index=torch.tile(center_z, (1, 1, pos.shape[-1])))
            group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z],
                                        dim=1)
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1)

        elif (self.method == "Random"):   
             group_input_tokens = group_input_tokens
             pos = pos

        elif (self.method == "k_top_eigenvectors"):                  

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph, self.symmetric, self.self_loop)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= self.k_top_eigenvectors, smallest = self.smallest)                                       

            #sorted_values, sorted_indices = torch.sort(top_k_eigenvectors, dim=1)

            for i in range (self.k_top_eigenvectors):      
                sorted_group_input_tokens = self.sort_points_by_fiedler(group_input_tokens, top_k_eigenvectors[:, :, i]) 
                sorted_pos = self.sort_points_by_fiedler(pos, top_k_eigenvectors[:, :, i])

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)    

                sorted_group_input_tokens_t = sorted_group_input_tokens      
                sorted_pos_t = sorted_pos

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()

        elif (self.method == "k_top_eigenvectors_weighted_adj"):              

            adjacency_matrix, dist_matrix = self.create_graph_from_feature_space_gpu_weighted_adjacency(center, self.knn_graph, self.alpha, self.symmetric, self.self_loop)
            if (self.all_distance == False):
                top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= self.k_top_eigenvectors, smallest = self.smallest)                                       
            else:
                top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(dist_matrix, k= self.k_top_eigenvectors, smallest = self.smallest)                                       

            #sorted_values, sorted_indices = torch.sort(top_k_eigenvectors, dim=1)      

            for i in range (self.k_top_eigenvectors):      
                sorted_group_input_tokens = self.sort_points_by_fiedler(group_input_tokens, top_k_eigenvectors[:, :, i]) 
                sorted_pos = self.sort_points_by_fiedler(pos, top_k_eigenvectors[:, :, i])

                if (i != 0): 
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)  

                sorted_group_input_tokens_t = sorted_group_input_tokens      
                sorted_pos_t = sorted_pos

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()

        elif (self.method == "k_top_eigenvectors_weighted_adj_symmetric"):                   

            adjacency_matrix, dist_matrix = self.create_graph_from_feature_space_gpu_weighted_adjacency(center, self.knn_graph, self.alpha, self.symmetric)
            if (self.all_distance == False):
                top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors_symmetric(adjacency_matrix, k= self.k_top_eigenvectors, smallest = self.smallest)                                       
            else:
                top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors_symmetric(dist_matrix, k= self.k_top_eigenvectors, smallest = self.smallest)                                       

            #sorted_values, sorted_indices = torch.sort(top_k_eigenvectors, dim=1)      

            for i in range (self.k_top_eigenvectors):      
                sorted_group_input_tokens = self.sort_points_by_fiedler(group_input_tokens, top_k_eigenvectors[:, :, i]) 
                sorted_pos = self.sort_points_by_fiedler(pos, top_k_eigenvectors[:, :, i])

                if (i != 0): 
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)  

                sorted_group_input_tokens_t = sorted_group_input_tokens      
                sorted_pos_t = sorted_pos

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()

        elif (self.method == "k_top_eigenvectors_2"):                 

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= self.k_top_eigenvectors, smallest = self.smallest)                                       

            #sorted_values, sorted_indices = torch.sort(top_k_eigenvectors, dim=1)

            top_k_eigenvectors = torch.cat((top_k_eigenvectors[:, :, :4], top_k_eigenvectors[:, :, 6:7]), -1)     
            self.k_top_eigenvectors = top_k_eigenvectors.shape[-1]
 
            for i in range (self.k_top_eigenvectors):      
                sorted_group_input_tokens = self.sort_points_by_fiedler(group_input_tokens, top_k_eigenvectors[:, :, i])  
                sorted_pos = self.sort_points_by_fiedler(pos, top_k_eigenvectors[:, :, i])

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)  

                sorted_group_input_tokens_t = sorted_group_input_tokens      
                sorted_pos_t = sorted_pos

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()      

        elif (self.method == "k_top_eigenvectors_Christian_Method"):                  

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph) 
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= self.k_top_eigenvectors, smallest = self.smallest)                                       

            #sorted_values, sorted_indices = torch.sort(top_k_eigenvectors, dim=1)

            top_k_eigenvectors = self.modify_sign_and_same_eigenvalue(top_k_eigenvalues, top_k_eigenvectors, "first_row_eigenvector")   


            for i in range (self.k_top_eigenvectors):      
                sorted_group_input_tokens = self.sort_points_by_fiedler(group_input_tokens, top_k_eigenvectors[:, :, i]) 
                sorted_pos = self.sort_points_by_fiedler(pos, top_k_eigenvectors[:, :, i])

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)  

                sorted_group_input_tokens_t = sorted_group_input_tokens      
                sorted_pos_t = sorted_pos

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda() 
                pos = sorted_pos.cuda()          

        elif (self.method == "k_top_eigenvectors_eigenvectors_addd_first_lat_sequence"):                 

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= self.k_top_eigenvectors, smallest = self.smallest)                                       

            sorted_values, sorted_indices = torch.sort(top_k_eigenvectors, dim=1)
            emb_size = 384

            for i in range (self.k_top_eigenvectors):       

                ###sorted_pos_t = self.get_sinusoid_encoding_table(sorted_pos_t, emb_size)
                sorted_indices = self.get_sinusoid_encoding_table(sorted_indices[..., i:i+1], emb_size)  

                sorted_group_input_tokens = self.sort_points_by_fiedler(group_input_tokens, top_k_eigenvectors[:, :, i]) 
                sorted_pos = self.sort_points_by_fiedler(pos, top_k_eigenvectors[:, :, i])

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)  

                sorted_group_input_tokens_t = sorted_group_input_tokens         
                sorted_pos_t = sorted_pos

                sorted_group_input_tokens_t = torch.cat((sorted_indices, sorted_group_input_tokens_t), 1)
                sorted_group_input_tokens_t = torch.cat((sorted_group_input_tokens_t, sorted_indices), 1)

                sorted_pos_t = torch.cat((sorted_indices, sorted_pos_t), 1)
                sorted_pos_t = torch.cat((sorted_pos_t, sorted_indices), 1)  


            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1)     
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()        

        elif (self.method == "k_top_eigenvectors_and_spectral_pose"):                  

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)  
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)                                       

            for i in range (self.k_top_eigenvectors):            
                sorted_group_input_tokens = self.sort_points_by_fiedler_for_neighberhood(neighborhood, top_k_eigenvectors[:, :, i])         
                sorted_pos = self.sort_points_by_fiedler_for_center(center, top_k_eigenvectors[:, :, i])
                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)   

                sorted_group_input_tokens_t = sorted_group_input_tokens             
                sorted_pos_t = sorted_pos

            sorted_group_input_tokens_t = self.encoder(sorted_group_input_tokens_t)  # B G N 

            sorted_values, _ = torch.sort(top_k_eigenvectors, dim=1)

            center_group_0 = torch.cat((sorted_pos_t[:, :128], sorted_values[:, :, 0:1]), -1)
            center_group_1 = torch.cat((sorted_pos_t[:, 128:256], sorted_values[:, :, 1:2]), -1)
            center_group_2 = torch.cat((sorted_pos_t[:, 256:384], sorted_values[:, :, 2:3]), -1)
            center_group_3 = torch.cat((sorted_pos_t[:, 384:512], sorted_values[:, :, 3:4]), -1)  

            center = torch.cat((center_group_0, center_group_1, center_group_2, center_group_3), 1)

            sorted_pos_t = self.pos_embed(center)   

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   
                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()

        elif (self.method == "k_top_eigenvectors_and_just_spectral_pose"):                  

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)  
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)                                       

            sorted_values, sorted_index_ = torch.sort(top_k_eigenvectors, dim=1)
            position = sorted_index_[..., 0]

            for i in range (self.k_top_eigenvectors):            
                sorted_group_input_tokens = self.sort_points_by_fiedler_for_neighberhood(neighborhood, top_k_eigenvectors[:, :, i])         
                sorted_pos = self.sort_points_by_fiedler_for_center(center, top_k_eigenvectors[:, :, i])
                sorted_position = self.sort_points_by_fiedler_for_position(position, top_k_eigenvectors[:, :, i])

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)   
                    sorted_position = torch.cat((sorted_position_t, sorted_position), 1)  

                sorted_group_input_tokens_t = sorted_group_input_tokens             
                sorted_pos_t = sorted_pos 
                sorted_position_t = sorted_position 


            sorted_group_input_tokens_t = self.encoder(sorted_group_input_tokens_t)  # B G N  

            ##sorted_values, sorted_index_ = torch.sort(top_k_eigenvectors, dim=1)

            ###sorted_pos_t = torch.cat((sorted_index_[:, :, 0:1], sorted_index_[:, :, 1:2], sorted_index_[:, :, 2:3], sorted_index_[:, :, 3:4]), 1)

            emb_size = 384
            ###sorted_pos_t = self.get_sinusoid_encoding_table(sorted_pos_t, emb_size)
            sorted_pos_t = self.get_sinusoid_encoding_table(sorted_position_t[..., None], emb_size)

            #sorted_pos_t = self.pos_embed(sorted_pos_t)    

            if (self.reverse == True):           
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   
                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()    

        elif (self.method == "k_top_eigenvectors_Christian_and_just_spectral_pose"):                   

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)  
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)                                       

            top_k_eigenvectors = self.modify_sign_and_same_eigenvalue(top_k_eigenvalues, top_k_eigenvectors, "first_row_eigenvector")   

            for i in range (self.k_top_eigenvectors):            
                sorted_group_input_tokens = self.sort_points_by_fiedler_for_neighberhood(neighborhood, top_k_eigenvectors[:, :, i])         
                sorted_pos = self.sort_points_by_fiedler_for_center(center, top_k_eigenvectors[:, :, i])

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)   

                sorted_group_input_tokens_t = sorted_group_input_tokens             
                sorted_pos_t = sorted_pos 

            sorted_values, sorted_index_ = torch.sort(top_k_eigenvectors, dim=1)
            sorted_values = torch.cat((sorted_values[:, :, :3], sorted_values[:, :, :3], sorted_values[:, :, :3], sorted_values[:, :, :3]), 1)

            sorted_group_input_tokens_t = self.encoder(sorted_group_input_tokens_t)  # B G N    

            sorted_pos_t = self.pos_embed(sorted_values)    

            if (self.reverse == True):           
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   
                group_input_tokens = sorted_group_input_tokens.cuda()   
                pos = sorted_pos.cuda()         

        elif(self.method == "k_top_eigenvectors_binary_with_spectral_pose"):
            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= 128, smallest = self.smallest)

            integers = self.multilevel_travers(top_k_eigenvectors, self.level)

            integers_before_random = integers

            random_value = torch.rand(integers.shape[0], integers.shape[1]).cuda()   
            integers = integers + random_value

            integers_arg_sort = torch.argsort(integers, 1)

            integers_arg_sort_center = integers_arg_sort.unsqueeze(-1).expand(-1, -1, 3)   
            integers_arg_sort_neighbor = integers_arg_sort.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3)       
        
            # Sort points based on the sorted indices
            sorted_group_input_tokens = torch.gather(neighborhood, 1, integers_arg_sort_neighbor)       
            sorted_pos = torch.gather(center, 1, integers_arg_sort_center)

            # Expectral pose embeddings
            sorted_group_input_tokens = self.encoder(sorted_group_input_tokens)
            sorted_integers, _ = torch.sort(integers_before_random, -1)   

            sorted_pos = torch.cat((sorted_pos, sorted_integers[:, :, None]), -1)    
            sorted_pos = self.pos_embed(sorted_pos)

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens.flip(1)     
                sorted_pos_t_reverse = sorted_pos.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda()  
                pos = sorted_pos.cuda()  

        elif(self.method == "k_top_eigenvectors_binary_with_spectral_pose_different_steps"):

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= 128, smallest = self.smallest)

            integers_list = []
            integers_before_random_list = []
            sorted_group_input_tokens_list = []    
            sorted_pos_list = []   

            for j in range(self.steps):
                integers = self.multilevel_travers_2(top_k_eigenvectors, self.level, j)
                integers_list.append(integers)

            for integers in (integers_list):

                integers_before_random = integers
                random_value = torch.rand(integers.shape[0], integers.shape[1]).cuda()   
                integers = integers + random_value

                integers_arg_sort = torch.argsort(integers, 1)

                integers_arg_sort_center = integers_arg_sort.unsqueeze(-1).expand(-1, -1, 3)   
                integers_arg_sort_neighbor = integers_arg_sort.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3)       
            
                # Sort points based on the sorted indices
                sorted_group_input_tokens = torch.gather(neighborhood, 1, integers_arg_sort_neighbor)       
                sorted_pos = torch.gather(center, 1, integers_arg_sort_center)

                # Expectral pose embeddings
                sorted_group_input_tokens = self.encoder(sorted_group_input_tokens)
                sorted_integers, _ = torch.sort(integers_before_random, -1)   

                sorted_pos = torch.cat((sorted_pos, sorted_integers[:, :, None]), -1)    
                sorted_pos = self.pos_embed(sorted_pos)

                sorted_group_input_tokens_list.append(sorted_group_input_tokens)
                sorted_pos_list.append(sorted_pos)

            sorted_group_input_tokens = torch.cat(sorted_group_input_tokens_list, 1)
            sorted_pos = torch.cat(sorted_pos_list, 1)

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens.flip(1)     
                sorted_pos_t_reverse = sorted_pos.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda()        
                pos = sorted_pos.cuda() 

        elif(self.method == "k_top_eigenvectors_binary_with_spectral_pose_minus_state"):
            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= 64, smallest = self.smallest)

            integers = self.multilevel_travers(top_k_eigenvectors, self.level)
            integers_minus = self.multilevel_travers((-1)*top_k_eigenvectors, self.level)  

            integers_before_random = integers
            integers_minus_before_random = integers_minus
 

            random_value = torch.rand(integers.shape[0], integers.shape[1]).cuda()       
            integers = integers + random_value
            integers_minus = integers_minus + random_value

            integers_arg_sort = torch.argsort(integers, 1)
            integers_minus_arg_sort = torch.argsort(integers_minus, 1)

            integers_arg_sort_center = integers_arg_sort.unsqueeze(-1).expand(-1, -1, 3)   
            integers_arg_sort_neighbor = integers_arg_sort.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3)       

            integers_minus_arg_sort_center = integers_minus_arg_sort.unsqueeze(-1).expand(-1, -1, 3)   
            integers_minus_arg_sort_neighbor = integers_minus_arg_sort.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3)     
        
            # Sort points based on the sorted indices
            sorted_group_input_tokens = torch.gather(neighborhood, 1, integers_arg_sort_neighbor)       
            sorted_pos = torch.gather(center, 1, integers_arg_sort_center)

            sorted_group_input_tokens_minus = torch.gather(neighborhood, 1, integers_minus_arg_sort_neighbor)         
            sorted_pos_minus = torch.gather(center, 1, integers_minus_arg_sort_center)

            # Expectral pose embeddings
            sorted_group_input_tokens = self.encoder(sorted_group_input_tokens)
            sorted_integers, _ = torch.sort(integers_before_random, -1)   

            sorted_group_input_tokens_minus = self.encoder(sorted_group_input_tokens_minus)
            sorted_integers_minus, _ = torch.sort(integers_minus_before_random, -1)   

            sorted_pos = torch.cat((sorted_pos, sorted_integers[:, :, None]), -1)    
            sorted_pos = self.pos_embed(sorted_pos)

            sorted_pos_minus = torch.cat((sorted_pos_minus, sorted_integers_minus[:, :, None]), -1)    
            sorted_pos_minus = self.pos_embed(sorted_pos_minus)

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens.flip(1)     
                sorted_pos_t_reverse = sorted_pos.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos, sorted_pos_t_reverse), 1)   

                group_input_tokens_1 = sorted_group_input_tokens.cuda()  
                pos_1  = sorted_pos.cuda()  

                sorted_group_input_tokens_minus_reverse = sorted_group_input_tokens_minus.flip(1)     
                sorted_pos_minus_t_reverse = sorted_pos_minus.flip(1)  
                sorted_group_input_tokens_minus = torch.cat((sorted_group_input_tokens_minus, sorted_group_input_tokens_minus_reverse), 1)           
                sorted_pos_minus = torch.cat((sorted_pos_minus, sorted_pos_minus_t_reverse), 1)   

                group_input_tokens_2 = sorted_group_input_tokens_minus.cuda()  
                pos_2 = sorted_pos_minus.cuda()

                group_input_tokens = torch.cat((group_input_tokens_1, group_input_tokens_2), 1)    
                pos = torch.cat((pos_1, pos_2), 1)

        elif(self.method == "k_top_eigenvectors_binary_with_spectral_pose_2_minus_state_with_list_sign"):
            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= 128, smallest = self.smallest)

            sorted_group_input_tokens_all_list = []
            sorted_pos_all_list = []
            for sign in self.sign:

                top_k_eigenvectors = top_k_eigenvectors[:, :, :self.level]
                top_k_eigenvectors = top_k_eigenvectors * torch.tensor(sign).cuda()

                integers = self.multilevel_travers(top_k_eigenvectors, self.level)

                integers_before_random = integers

                random_value = torch.rand(integers.shape[0], integers.shape[1]).cuda()   
                integers = integers + random_value

                integers_arg_sort = torch.argsort(integers, 1)

                integers_arg_sort_center = integers_arg_sort.unsqueeze(-1).expand(-1, -1, 3)   
                integers_arg_sort_neighbor = integers_arg_sort.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3)       
            
                # Sort points based on the sorted indices
                sorted_group_input_tokens = torch.gather(neighborhood, 1, integers_arg_sort_neighbor)       
                sorted_pos = torch.gather(center, 1, integers_arg_sort_center)   

                # Expectral pose embeddings
                sorted_group_input_tokens = self.encoder(sorted_group_input_tokens)
                sorted_integers, _ = torch.sort(integers_before_random, -1)    

                #sorted_bin_integers = self.int_to_binary(sorted_integers)
                sorted_bin_integers = self.unpack_binary_integers_vectorized(sorted_integers.cpu().numpy()).cuda()        

                #sorted_pos = torch.cat((sorted_pos, sorted_bin_integers[:, :, None]), -1)  
                #######sorted_pos = torch.cat((sorted_pos, sorted_bin_integers), -1)
                sorted_bin_integers = torch.tensor(sorted_bin_integers, dtype=torch.float)
                sorted_pos = self.pos_embed(sorted_bin_integers)

                sorted_group_input_tokens_all_list.append(sorted_group_input_tokens)
                sorted_pos_all_list.append(sorted_pos)

            sorted_group_input_tokens = torch.cat(sorted_group_input_tokens_all_list, 1)
            sorted_pos = torch.cat(sorted_pos_all_list, 1)        


            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens.flip(1)     
                sorted_pos_t_reverse = sorted_pos.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda()  
                pos = sorted_pos.cuda()     

        elif (self.method == "k_top_eigenvectors_and_Reverse_AND_k_top_eigenvectors_binary_minus_state_and_Reverse"):                 

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)                                       

            ################# Method 1
            for i in range (self.k_top_eigenvectors):      
                sorted_group_input_tokens = self.sort_points_by_fiedler_for_neighberhood(neighborhood, top_k_eigenvectors[:, :, i]) 
                sorted_pos = self.sort_points_by_fiedler_for_center(center, top_k_eigenvectors[:, :, i])

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)  

                sorted_group_input_tokens_t = sorted_group_input_tokens     
                sorted_pos_t = sorted_pos

            """if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   

                group_input_tokens_method_1 = sorted_group_input_tokens.cuda()
                pos_method_1 = sorted_pos.cuda()"""

            group_input_tokens_method_1 = sorted_group_input_tokens_t.cuda()
            pos_method_1 = sorted_pos_t.cuda()
            ####group_input_tokens_method_1 = self.encoder(group_input_tokens_method_1)
            # pos_method_1 = torch.cat((pos_method_1, top_k_eigenvectors[:, :, :self.k_top_eigenvectors]), -1)   
            ####pos_method_1 = self.pos_embed(pos_method_1)

            ############## Method 2
            integers = self.multilevel_travers(top_k_eigenvectors, self.level)
            integers_minus = self.multilevel_travers((-1)*top_k_eigenvectors, self.level)  

            integers_before_random = integers
            integers_minus_before_random = integers_minus
 

            random_value = torch.rand(integers.shape[0], integers.shape[1]).cuda()       
            integers = integers + random_value
            integers_minus = integers_minus + random_value

            integers_arg_sort = torch.argsort(integers, 1)
            integers_minus_arg_sort = torch.argsort(integers_minus, 1)

            integers_arg_sort_center = integers_arg_sort.unsqueeze(-1).expand(-1, -1, 3)   
            integers_arg_sort_neighbor = integers_arg_sort.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3)       

            integers_minus_arg_sort_center = integers_minus_arg_sort.unsqueeze(-1).expand(-1, -1, 3)   
            integers_minus_arg_sort_neighbor = integers_minus_arg_sort.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3)     
        
            # Sort points based on the sorted indices
            sorted_group_input_tokens = torch.gather(neighborhood, 1, integers_arg_sort_neighbor)       
            sorted_pos = torch.gather(center, 1, integers_arg_sort_center)

            sorted_group_input_tokens_minus = torch.gather(neighborhood, 1, integers_minus_arg_sort_neighbor)         
            sorted_pos_minus = torch.gather(center, 1, integers_minus_arg_sort_center)

            # Expectral pose embeddings
            sorted_group_input_tokens = torch.cat((group_input_tokens_method_1, sorted_group_input_tokens, sorted_group_input_tokens_minus), 1) 
            sorted_group_input_tokens = self.encoder(sorted_group_input_tokens)
            ####sorted_integers, _ = torch.sort(integers_before_random, -1)   

            #############sorted_group_input_tokens_minus = self.encoder(sorted_group_input_tokens_minus)
            ####sorted_integers_minus, _ = torch.sort(integers_minus_before_random, -1)   

            ####sorted_pos = torch.cat((sorted_pos, sorted_integers[:, :, None]), -1)  
            sorted_pos = torch.cat((pos_method_1, sorted_pos, sorted_pos_minus), 1)   
            sorted_pos = self.pos_embed(sorted_pos)

            ####sorted_pos_minus = torch.cat((sorted_pos_minus, sorted_integers_minus[:, :, None]), -1)    
            ###########sorted_pos_minus = self.pos_embed(sorted_pos_minus)

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens.flip(1)     
                sorted_pos_t_reverse = sorted_pos.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos, sorted_pos_t_reverse), 1)   

                #group_input_tokens_1 = sorted_group_input_tokens.cuda()  
                #pos_1  = sorted_pos.cuda()  
                group_input_tokens = sorted_group_input_tokens.cuda()  
                pos = sorted_pos.cuda()   

                """sorted_group_input_tokens_minus_reverse = sorted_group_input_tokens_minus.flip(1)     
                sorted_pos_minus_t_reverse = sorted_pos_minus.flip(1)  
                sorted_group_input_tokens_minus = torch.cat((sorted_group_input_tokens_minus, sorted_group_input_tokens_minus_reverse), 1)           
                sorted_pos_minus = torch.cat((sorted_pos_minus, sorted_pos_minus_t_reverse), 1)   

                group_input_tokens_2 = sorted_group_input_tokens_minus.cuda()  
                pos_2 = sorted_pos_minus.cuda()

                group_input_tokens_method_2 = torch.cat((group_input_tokens_1, group_input_tokens_2), 1)    
                pos_method_2 = torch.cat((pos_1, pos_2), 1)

                group_input_tokens = torch.cat((group_input_tokens_method_1, group_input_tokens_method_2), 1)
                pos = torch.cat((pos_method_1, pos_method_2), 1)  """   

        elif (self.method == "k_top_eigenvectors_and_spectral_pose_2"):                 

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)  
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)                                       

            for i in range (self.k_top_eigenvectors):            
                sorted_group_input_tokens = self.sort_points_by_fiedler_for_neighberhood(neighborhood, top_k_eigenvectors[:, :, i])         
                sorted_pos = self.sort_points_by_fiedler_for_center(center, top_k_eigenvectors[:, :, i])
                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)   

                sorted_group_input_tokens_t = sorted_group_input_tokens              
                sorted_pos_t = sorted_pos

            sorted_group_input_tokens_t = self.encoder(sorted_group_input_tokens_t)  # B G N

            sorted_values, _ = torch.sort(top_k_eigenvectors, dim=1)

            """center_group_0 = torch.cat((sorted_pos_t[:, :128], top_k_eigenvectors[:, :, 0:1]), -1)
            center_group_1 = torch.cat((sorted_pos_t[:, 128:256], top_k_eigenvectors[:, :, 1:2]), -1)
            center_group_2 = torch.cat((sorted_pos_t[:, 256:384], top_k_eigenvectors[:, :, 2:3]), -1)
            center_group_3 = torch.cat((sorted_pos_t[:, 384:512], top_k_eigenvectors[:, :, 3:4]), -1)"""

            center = torch.cat((sorted_values[:, :, 0:1], sorted_values[:, :, 1:2], sorted_values[:, :, 2:3], sorted_values[:, :, 3:4]), 1)

            sorted_pos_t = self.pos_embed(center)  

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   
                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()

        elif (self.method == "local_global_traverse_k_eigenvectors_2"): 

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)   
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k = self.k_top_eigenvectors, smallest = self.smallest)                                       

            partition_indices = self.partition_centers_by_eigenvectors(top_k_eigenvectors)      

            partition_indices = [sublist[-1:] for sublist in partition_indices]

            sorted_group_input_tokens, sorted_pos, _ = self.shuffle_and_reconcat_tensors(group_input_tokens, pos, center, partition_indices)

            group_input_tokens = sorted_group_input_tokens.cuda()
            pos = sorted_pos.cuda()     
 
        elif (self.method == "k_top_eigenvectors_wighted_traversing"):                 

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= self.k_top_eigenvectors, smallest = self.smallest)                                       

            sorted_values, sorted_indices = torch.sort(top_k_eigenvectors, dim=1)

            for i in range (self.k_top_eigenvectors):      
                sorted_group_input_tokens = self.sort_points_by_fiedler(group_input_tokens, top_k_eigenvectors[:, :, i]) 
                sorted_pos = self.sort_points_by_fiedler(pos, top_k_eigenvectors[:, :, i])

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)  

                sorted_group_input_tokens_t = sorted_group_input_tokens      
                sorted_pos_t = sorted_pos

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()

                learnable_tokens = self.learnable_tokens.expand(group_input_tokens.shape[0], int(group_input_tokens.shape[1] / neighborhood.shape[1]), -1)

                group_input_tokens = torch.cat((group_input_tokens, learnable_tokens), 1)
                pos = torch.cat((pos, learnable_tokens), 1)


        elif (self.method == "k_top_eigenvectors_wighted_traversing_2"):                 

            learnable_tokens = self.learnable_tokens.expand(center.shape[0], int(self.k_top_eigenvectors * 2), -1)
            index_sequence = torch.tensor([0,1,2,3,4,5,6,7]).expand(center.shape[0], -1)

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k= self.k_top_eigenvectors, smallest = self.smallest)                                       

            for i in range (self.k_top_eigenvectors):      
                sorted_group_input_tokens = self.sort_points_by_fiedler(group_input_tokens, top_k_eigenvectors[:, :, i]) 
                sorted_pos = self.sort_points_by_fiedler(pos, top_k_eigenvectors[:, :, i])

                group_input_tokens = torch.cat((sorted_group_input_tokens, learnable_tokens[:, i:i+1]), 1)
                index_sequence_sin_cos = self.get_sinusoid_encoding_table_2(index_sequence[:, i:i+1, None], 384)
                sorted_pos = torch.cat((sorted_pos, learnable_tokens[:, i:i+1]), 1)

                if (i != 0):
                    sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                    sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1)  

                sorted_group_input_tokens_t = sorted_group_input_tokens      
                sorted_pos_t = sorted_pos

            if (self.reverse == True):         
                sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
                sorted_pos_t_reverse = sorted_pos_t.flip(1)  
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)   

                group_input_tokens = sorted_group_input_tokens.cuda()
                pos = sorted_pos.cuda()

                group_input_tokens_zeros = torch.zeros(group_input_tokens.shape[0], group_input_tokens.shape[1]+int(self.k_top_eigenvectors * 2), group_input_tokens.shape[2]).cuda()
                pos_zeros = torch.zeros(group_input_tokens.shape[0], group_input_tokens.shape[1]+int(self.k_top_eigenvectors * 2), group_input_tokens.shape[2]).cuda()

                for j in range(int(self.k_top_eigenvectors * 2)):

                    group_input_tokens_zeros[:, int(j*center.shape[1])+j: int((j+1)*center.shape[1])+1+j] = torch.cat((group_input_tokens[:, int(j*center.shape[1]): int((j+1)*center.shape[1])], learnable_tokens[:, i:i+1]), 1)
                    index_sequence_sin_cos = self.get_sinusoid_encoding_table_2(index_sequence[:, i:i+1, None], 384).cuda()
                    pos_zeros[:, int(j*center.shape[1])+j: int((j+1)*center.shape[1])+1+j] = torch.cat((pos[:, int(j*center.shape[1]): int((j+1)*center.shape[1])], index_sequence_sin_cos[:, 0:1]), 1)

                group_input_tokens = group_input_tokens_zeros    
                pos = pos_zeros

        if (self.method == "k_top_eigenvectors_wighted_traversing"):

            x = group_input_tokens    
            # transformer 
            x = self.drop_out(x)
            x = self.blocks(x, pos)
            learnable_tokens_softmax = torch.softmax(self.cls_head_learnable(x[:, -8:, :].reshape(-1, 384)).reshape(pos.shape[0], -1), 1)
            x = x[:, :group_input_tokens.shape[1] - learnable_tokens.shape[1], :]
            x = self.norm(x)
            x_zeros = torch.zeros_like(group_input_tokens)
            for i in range(learnable_tokens_softmax.shape[1]):
                x_zeros[:, (i*neighborhood.shape[1]): (i+1)*neighborhood.shape[1]] = x[:, (i*neighborhood.shape[1]): (i+1)*neighborhood.shape[1]] * learnable_tokens_softmax[:, i:i+1, None]

            concat_f = x_zeros[:, :].mean(1)   
            ret = self.cls_head_finetune(concat_f)    
            return ret 
        
        elif (self.method == "k_top_eigenvectors_wighted_traversing_2"):

            learnable_tokens_list = []
            x_list = []

            x = group_input_tokens    
            # transformer 
            x = self.drop_out(x)
            x = self.blocks(x, pos)

            for j in range(int(self.k_top_eigenvectors * 2)):

                learnable_tokens_list.append(x[:, int((j+1)*center.shape[1])+j: int((j+1)*center.shape[1])+1+j])
                x_list.append(x[:, int(j*center.shape[1]+j): int((j+1)*center.shape[1])+j])
     

            learnable_tokens_softmax = torch.softmax(self.cls_head_learnable(torch.cat(learnable_tokens_list, 1).reshape(-1, 384)).reshape(pos.shape[0], -1), 1)
            x = torch.cat(x_list, 1)

            x = self.norm(x)
            x_zeros = torch.zeros_like(x)
            for i in range(learnable_tokens_softmax.shape[1]):
                x_zeros[:, (i*neighborhood.shape[1]): (i+1)*neighborhood.shape[1]] = x[:, (i*neighborhood.shape[1]): (i+1)*neighborhood.shape[1]] * learnable_tokens_softmax[:, i:i+1, None]

            concat_f = x_zeros[:, :].mean(1)   
            ret = self.cls_head_finetune(concat_f)       
            return ret 

        else:    

            x = group_input_tokens    
            # transformer 
            x = self.drop_out(x)
            x = self.blocks(x, pos)
            x = self.norm(x)
            concat_f = x[:, :].mean(1)
            ret = self.cls_head_finetune(concat_f)
            return ret


class MaskMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Mamba')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm)

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


class MaskMamba_2(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.group_size = config.group_size
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Mamba')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm)

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),  
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)  

        return overall_mask.to(center.device)  # B G


    def sort_points_by_fiedler(self, points, bool_masked_pos, fiedler_vector):
        """
        Sorts points based on the Fiedler vector, optimized for GPU execution.

        Parameters:
        - points: Tensor of shape (B, N, 3) where B is the batch size, 
                N is the number of points in the point cloud, and 3 are the xyz coordinates.
        - fiedler_vector: The Fiedler vector used for sorting, of shape (B, N).

        Returns:
        - sorted_points: Tensor of sorted points.
        """
        B, N, _ = points.shape
        # Generate indices from Fiedler vector for sorting
        #fiedler_vector = fiedler_vector.real             
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)   
        
        # Expand the indices to work for x, y, z coordinates
        expanded_indices_mask = sorted_indices
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 384) 
        
        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, expanded_indices)
        sorted_bool_masked_pos = torch.gather(bool_masked_pos, 1, expanded_indices_mask)   
        
        return sorted_points, sorted_bool_masked_pos
    
    def sort_points_by_fiedler_for_neighberhood(self, points, fiedler_vector):
        """
        Sorts points based on the Fiedler vector, optimized for GPU execution.

        Parameters:
        - points: Tensor of shape (B, N, 3) where B is the batch size, 
                N is the number of points in the point cloud, and 3 are the xyz coordinates.
        - fiedler_vector: The Fiedler vector used for sorting, of shape (B, N).

        Returns:
        - sorted_points: Tensor of sorted points.
        """
        B, N, _, _ = points.shape
        # Generate indices from Fiedler vector for sorting
        fiedler_vector = fiedler_vector.real         
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)
        
        # Expand the indices to work for x, y, z coordinates
        expanded_indices = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.group_size, 3)      
        
        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, expanded_indices)
        
        return sorted_points 

    def forward(self, neighborhood, center, top_k_eigenvectors, k_top_eigenvectors, reverse, noaug=False):   
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug).cuda()  # B G   
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        pos = self.pos_embed(center)

        sorted_bool_masked_pos_list = []

        for i in range (k_top_eigenvectors):      
            sorted_group_input_tokens, sorted_bool_masked_pos = self.sort_points_by_fiedler(group_input_tokens, bool_masked_pos, top_k_eigenvectors[:, :, i]) 
            sorted_neighborhood = self.sort_points_by_fiedler_for_neighberhood(neighborhood, top_k_eigenvectors[:, :, i]) 

            sorted_pos, _ = self.sort_points_by_fiedler(pos, bool_masked_pos, top_k_eigenvectors[:, :, i])

            sorted_pos_full = sorted_pos
            sorted_group_input_tokens = sorted_group_input_tokens[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
            sorted_pos = sorted_pos[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
            sorted_pos_mask = sorted_pos_full[sorted_bool_masked_pos].reshape(batch_size, -1, C)

            if (i != 0):
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1) 
                sorted_pos_mask = torch.cat((sorted_pos_mask_t, sorted_pos_mask), 1)  
                sorted_pos_full = torch.cat((sorted_pos_full_t, sorted_pos_full), 1) 
                sorted_neighborhood = torch.cat((sorted_neighborhood_t, sorted_neighborhood), 1)  


            sorted_group_input_tokens_t = sorted_group_input_tokens     
            sorted_pos_t = sorted_pos
            sorted_pos_mask_t = sorted_pos_mask
            sorted_pos_full_t = sorted_pos_full
            sorted_neighborhood_t = sorted_neighborhood

            sorted_bool_masked_pos_list.append(sorted_bool_masked_pos)

        if (reverse == True):         
            sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
            sorted_pos_t_reverse = sorted_pos_t.flip(1)  
            sorted_pos_mask_t_reverse = sorted_pos_mask_t.flip(1)  
            sorted_pos_full_t_reverse = sorted_pos_full_t.flip(1)  
            sorted_neighborhood_t_reverse = sorted_neighborhood_t.flip(1)  
            sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
            sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)
            sorted_pos_mask = torch.cat((sorted_pos_mask_t, sorted_pos_mask_t_reverse), 1) 
            sorted_pos_full = torch.cat((sorted_pos_full_t, sorted_pos_full_t_reverse), 1) 
            sorted_neighborhood = torch.cat((sorted_neighborhood_t, sorted_neighborhood_t_reverse), 1) 

            x_vis = sorted_group_input_tokens.cuda()  
            pos = sorted_pos.cuda()

            sorted_bool_masked_pos_tensor = torch.cat(sorted_bool_masked_pos_list, -1)
            sorted_bool_masked_pos_tensor = sorted_bool_masked_pos_tensor.flip(-1)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis) 

        return x_vis, sorted_bool_masked_pos_list, sorted_pos_mask.cuda(), sorted_pos_full.cuda(), sorted_bool_masked_pos_tensor.cuda(), sorted_neighborhood.cuda(), self.mask_ratio


class MaskMamba_3(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Mamba')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.config.rms_norm)

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),  
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)  

        return overall_mask.to(center.device)  # B G


    def sort_points_by_fiedler(self, points, bool_masked_pos, fiedler_vector, noaug= False):
        """
        Sorts points based on the Fiedler vector, optimized for GPU execution. 

        Parameters:
        - points: Tensor of shape (B, N, 3) where B is the batch size, 
                N is the number of points in the point cloud, and 3 are the xyz coordinates.
        - fiedler_vector: The Fiedler vector used for sorting, of shape (B, N).

        Returns:
        - sorted_points: Tensor of sorted points.
        """
        B, N, _ = points.shape
        # Generate indices from Fiedler vector for sorting
        #fiedler_vector = fiedler_vector.real             
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)   
        
        # Expand the indices to work for x, y, z coordinates
        expanded_indices_mask = sorted_indices
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 384) 

        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, expanded_indices)
        sorted_bool_masked_pos = torch.gather(bool_masked_pos, 1, expanded_indices_mask)   

        ############# find index for learnable
        if (noaug == False):
            sorted_indices_learnable_tokens = sorted_indices[sorted_bool_masked_pos].reshape(B, 38) 
        else:
            sorted_indices_learnable_tokens = torch.zeros((B, 38)).cuda()       
        
        return sorted_points, sorted_bool_masked_pos, sorted_indices_learnable_tokens, sorted_indices
    
    def sort_points_by_fiedler_for_neighberhood(self, points, fiedler_vector):
        """
        Sorts points based on the Fiedler vector, optimized for GPU execution.

        Parameters:
        - points: Tensor of shape (B, N, 3) where B is the batch size, 
                N is the number of points in the point cloud, and 3 are the xyz coordinates.
        - fiedler_vector: The Fiedler vector used for sorting, of shape (B, N).

        Returns:
        - sorted_points: Tensor of sorted points.
        """
        B, N, _, _ = points.shape
        # Generate indices from Fiedler vector for sorting
        fiedler_vector = fiedler_vector.real         
        _, sorted_indices = torch.sort(fiedler_vector, dim=1)
        
        # Expand the indices to work for x, y, z coordinates
        expanded_indices = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 3)      
        
        # Sort points based on the sorted indices
        sorted_points = torch.gather(points, 1, expanded_indices)
        
        return sorted_points 
    
    def find_indices_vectorized(self, a, sorted_indices_learnable_tokens_2):
        B, N = a.shape
        T = sorted_indices_learnable_tokens_2.size(1)

        # Expand 'a' to compare with each element in 'sorted_indices_learnable_tokens_2'
        a_expanded = a.unsqueeze(2).expand(B, N, T)

        # Expand 'sorted_indices_learnable_tokens_2' for comparison
        sorted_indices_expanded = sorted_indices_learnable_tokens_2.unsqueeze(1).expand(B, N, T)

        # Compare and find indices where values match
        matches = (a_expanded == sorted_indices_expanded).nonzero(as_tuple=True)

        # Extract the third component of 'matches', which are the indices of matches
        # Since 'matches[2]' gives us the flat index positions across the tensor,
        # We need to reshape it to match the shape of 'a'
        output = matches[2].view(B, N)

        return output.cuda()

    def forward(self, neighborhood, center, top_k_eigenvectors, k_top_eigenvectors, reverse, noaug=False):   
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug).cuda()  # B G   
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        pos = self.pos_embed(center)

        sorted_bool_masked_pos_list = []
        sorted_found_indices_list = []
        sorted_found_indices_list_reverse = []

        for i in range (k_top_eigenvectors): 
            if (i == 0):      
                sorted_group_input_tokens, sorted_bool_masked_pos, sorted_indices_learnable_tokens_, sorted_indices = self.sort_points_by_fiedler(group_input_tokens, bool_masked_pos, top_k_eigenvectors[:, :, i], noaug) 
                sorted_indices_learnable_tokens = sorted_indices_learnable_tokens_
            else:
                sorted_group_input_tokens, sorted_bool_masked_pos, _, sorted_indices = self.sort_points_by_fiedler(group_input_tokens, bool_masked_pos, top_k_eigenvectors[:, :, i], noaug) 
            
            sorted_found_indices_list.append(self.find_indices_vectorized(sorted_indices_learnable_tokens, sorted_indices)[..., None])
            sorted_found_indices_list_reverse.append(self.find_indices_vectorized(sorted_indices_learnable_tokens, sorted_indices.flip(-1))[..., None])
 
            sorted_neighborhood = self.sort_points_by_fiedler_for_neighberhood(neighborhood, top_k_eigenvectors[:, :, i]) 

            sorted_pos, _, _, _ = self.sort_points_by_fiedler(pos, bool_masked_pos, top_k_eigenvectors[:, :, i], noaug)

            sorted_pos_full = sorted_pos 
            sorted_group_input_tokens = sorted_group_input_tokens[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
            sorted_pos = sorted_pos[~sorted_bool_masked_pos].reshape(batch_size, -1, C)
            sorted_pos_mask = sorted_pos_full[sorted_bool_masked_pos].reshape(batch_size, -1, C)

            if (i != 0):
                sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens), 1)         
                sorted_pos = torch.cat((sorted_pos_t, sorted_pos), 1) 
                sorted_pos_mask = torch.cat((sorted_pos_mask_t, sorted_pos_mask), 1)  
                sorted_pos_full = torch.cat((sorted_pos_full_t, sorted_pos_full), 1) 
                sorted_neighborhood = torch.cat((sorted_neighborhood_t, sorted_neighborhood), 1)  

            sorted_group_input_tokens_t = sorted_group_input_tokens     
            sorted_pos_t = sorted_pos
            sorted_pos_mask_t = sorted_pos_mask
            sorted_pos_full_t = sorted_pos_full
            sorted_neighborhood_t = sorted_neighborhood

            sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
            sorted_pos_t_reverse = sorted_pos_t.flip(1)  
            sorted_pos_mask_t_reverse = sorted_pos_mask_t.flip(1)  
            sorted_pos_full_t_reverse = sorted_pos_full_t.flip(1)  
            sorted_neighborhood_t_reverse = sorted_neighborhood_t.flip(1) 

            sorted_bool_masked_pos_list.append(sorted_bool_masked_pos)  

        sorted_found_indices_tensor = torch.cat(sorted_found_indices_list, -1)
        sorted_found_indices_list_reverse.reverse()
        sorted_found_indices_tensor_reverse = torch.cat(sorted_found_indices_list_reverse, -1)

        sorted_found_indices_tensor_final = torch.cat((sorted_found_indices_tensor, sorted_found_indices_tensor_reverse), -1)

        if (reverse == True):         
            sorted_group_input_tokens_t_reverse = sorted_group_input_tokens_t.flip(1) 
            sorted_pos_t_reverse = sorted_pos_t.flip(1)  
            sorted_pos_mask_t_reverse = sorted_pos_mask_t.flip(1)  
            sorted_pos_full_t_reverse = sorted_pos_full_t.flip(1)  
            sorted_neighborhood_t_reverse = sorted_neighborhood_t.flip(1)  
            sorted_group_input_tokens = torch.cat((sorted_group_input_tokens_t, sorted_group_input_tokens_t_reverse), 1)           
            sorted_pos = torch.cat((sorted_pos_t, sorted_pos_t_reverse), 1)
            sorted_pos_mask = torch.cat((sorted_pos_mask_t, sorted_pos_mask_t_reverse), 1) 
            sorted_pos_full = torch.cat((sorted_pos_full_t, sorted_pos_full_t_reverse), 1) 
            sorted_neighborhood = torch.cat((sorted_neighborhood_t, sorted_neighborhood_t_reverse), 1) 

            x_vis = sorted_group_input_tokens.cuda()  
            pos = sorted_pos.cuda()

            sorted_bool_masked_pos_tensor = torch.cat(sorted_bool_masked_pos_list, -1)            
            sorted_bool_masked_pos_tensor = sorted_bool_masked_pos_tensor.flip(-1)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis) 

        return x_vis, sorted_bool_masked_pos_list, sorted_pos_mask.cuda(), sorted_pos_full.cuda(), sorted_bool_masked_pos_tensor.cuda(), sorted_neighborhood.cuda(), sorted_found_indices_tensor_final.cuda()






class MambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, config=None):
        super().__init__()
        if hasattr(config, "use_external_dwconv_at_last"):
            self.use_external_dwconv_at_last = config.use_external_dwconv_at_last
        else:
            self.use_external_dwconv_at_last = False
        self.blocks = MixerModel(d_model=embed_dim,
                                 n_layer=depth,
                                 rms_norm=config.rms_norm,
                                 drop_path=config.drop_path)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        x = self.blocks(x, pos)

        #x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        x = self.head(self.norm(x)) 
        return x


@MODELS.register_module()
class Point_MAE_Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if (config.transformer_config.method == "MAMBA"):
            self.MAE_encoder = MaskMamba(config)
        elif (config.transformer_config.method == "k_top_eigenvectors_seperate_learnable_tokens" or config.transformer_config.method == "k_top_eigenvectors_seperate_learnable_tokens_weighted_adjacency"):
            self.MAE_encoder = MaskMamba_2(config)
        elif (config.transformer_config.method == "k_top_eigenvectors_shared_learnable_tokens"):
            self.MAE_encoder = MaskMamba_3(config)    
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim)) 
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.MAE_decoder = MambaDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            config=config,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        self.method = config.transformer_config.method
        self.reverse = config.transformer_config.reverse
        self.k_top_eigenvectors = config.transformer_config.k_top_eigenvectors
        self.smallest = config.transformer_config.smallest 
        self.knn_graph = config.transformer_config.knn_graph 
        self.alpha = config.transformer_config.alpha
        self.symmetric = config.transformer_config.symmetric
        self.self_loop = config.transformer_config.self_loop


    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            #self.loss_func = ChamferDistanceL1().cuda()
            self.loss_func = chamfer_distance

        elif loss_type == 'cdl2':
            #self.loss_func = ChamferDistanceL2().cuda()  
            self.loss_func = chamfer_distance
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def create_graph_from_feature_space_gpu(self, points, k=5, self_loop = False):
        """
        Create a graph from point cloud data in a feature space using k-nearest neighbors, optimized for GPU.
        
        Parameters:
        - points: Tensor of shape (B, N, F) where B is the batch size,
                N is the number of points in the point cloud, and F is the feature dimensions.
                This tensor should already be on the GPU.
        - k: The number of nearest neighbors to consider for graph edges.
        
        Returns:
        - adjacency_matrix: Tensor of shape (B, N, N) representing the adjacency matrix of the graph.
        """
        # Ensure the input tensor is on the GPU
        points = points.to('cuda')
        
        B, N, _ = points.shape
        
        # Compute pairwise distances, shape (B, N, N), on GPU    
        dist_matrix = torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1)
        
        # Find the k-nearest neighbors for each point, excluding itself
        _, indices = torch.topk(-dist_matrix, k=k+1, largest=True, dim=-1)
        #indices = indices[:, :, 1:]  # Remove self-loops
        if (self_loop):
            indices = indices[:, :, :]  # Remove self-loops     
        else:
            indices = indices[:, :, 1:]  # Remove self-loops
        
        # Create adjacency matrix on GPU
        adjacency_matrix = torch.zeros(B, N, N, device='cuda')
        b_idx = torch.arange(B, device='cuda')[:, None, None]
        adjacency_matrix[b_idx, torch.arange(N, device='cuda')[:, None], indices] = 1
        adjacency_matrix[b_idx, indices, torch.arange(N, device='cuda')[:, None]] = 1  # Ensure symmetry
        
        return adjacency_matrix

    def calc_top_k_eigenvalues_eigenvectors(self, adj_matrices, k, smallest):   
        """
        Calculate the top k eigenvalues and eigenvectors of a batch of adjacency matrices 
        using Random-walk normalization.

        Parameters:
        - adj_matrices: A tensor of shape (B, 128, 128) representing a batch of adjacency matrices.
        - k: The number of top eigenvalues (and corresponding eigenvectors) to return.

        Returns:
        - top_k_eigenvalues: A tensor of shape (B, k) containing the top k eigenvalues for each matrix.
        - top_k_eigenvectors: A tensor of shape (B, 128, k) containing the eigenvectors corresponding to the top k eigenvalues for each matrix.
        """
        B, N, _ = adj_matrices.shape
        top_k_eigenvalues = torch.zeros((B, k+1)).cuda()
        top_k_eigenvectors = torch.zeros((B, N, k+1)).cuda()
        eigenvalues_l = torch.zeros((B, N)).cuda()
        eigenvectors_l = torch.zeros((B, N, N)).cuda()

        for i in range(B):
            # Extract the i-th adjacency matrix
            A = adj_matrices[i]

            # Ensure A is symmetric
            #A = (A + A.t()) / 2

            # Compute the degree matrix D
            D = torch.diag(torch.sum(A, dim=1))  

            # Compute D^-1
            D_inv = torch.diag(1.0 / torch.diag(D))

            # Perform Random-walk normalization: D^-1 * A
            ####normalized_A = torch.matmul(D_inv, A)
            I = torch.eye(N).cuda()    
            normalized_A = I - torch.matmul(D_inv, A)      
 
            #eigenvalues, eigenvectors = torch.linalg.eigh(normalized_A)
            eigenvalues, eigenvectors = torch.linalg.eig(normalized_A)   
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real

            # Select the top k eigenvalues and corresponding eigenvectors
            if (smallest == False):
                top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=True, sorted=True)   
                top_vecs = eigenvectors[:, top_indices]
            else:
                top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=False, sorted=True)    
                top_vecs = eigenvectors[:, top_indices]   

            # Store the results
            top_k_eigenvalues[i] = top_vals
            top_k_eigenvectors[i, :, :] = top_vecs               

            eigenvalues_l[i] = eigenvalues
            eigenvectors_l[i, :, :] = eigenvectors    

        return top_k_eigenvalues[:, 1:], top_k_eigenvectors[:, :, 1:], eigenvalues_l, eigenvectors_l
    

    def calc_top_k_eigenvalues_eigenvectors_old(self, adj_matrices, k, smallest):  
        """
        Calculate the top k eigenvalues and eigenvectors of a batch of adjacency matrices 
        using Random-walk normalization.

        Parameters:
        - adj_matrices: A tensor of shape (B, 128, 128) representing a batch of adjacency matrices.
        - k: The number of top eigenvalues (and corresponding eigenvectors) to return.

        Returns:
        - top_k_eigenvalues: A tensor of shape (B, k) containing the top k eigenvalues for each matrix.
        - top_k_eigenvectors: A tensor of shape (B, 128, k) containing the eigenvectors corresponding to the top k eigenvalues for each matrix.
        """
        B, N, _ = adj_matrices.shape

        # Ensure A is symmetric
        adj_matrices = (adj_matrices + adj_matrices.transpose(1, 2)) / 2

        # Compute the degree matrices D and D^-1
        degrees = torch.sum(adj_matrices, dim=2)
        D_inv = torch.zeros_like(adj_matrices)
        for i in range(B):
            D_inv[i] = torch.diag(1.0 / degrees[i])

        # Perform Random-walk normalization: D^-1 * A
        I = torch.eye(N, device=adj_matrices.device)
        normalized_A = I.unsqueeze(0).repeat(B, 1, 1) - torch.matmul(D_inv, adj_matrices)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eig(normalized_A)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        # Select the top k eigenvalues and corresponding eigenvectors
        if smallest:
            top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=False, sorted=True)
        else:
            top_vals, top_indices = torch.topk(eigenvalues, k+1, largest=True, sorted=True)
        
        # Gather the top eigenvectors
        top_vecs = torch.gather(eigenvectors, 2, top_indices.unsqueeze(1).expand(-1, N, -1))

        # Slice off the smallest eigenvalue and corresponding eigenvector
        return top_vals[:, 1:], top_vecs[:, :, 1:], eigenvalues, eigenvectors


    def create_graph_from_feature_space_gpu_weighted_adjacency(self, points, k=5, alpha=1, symmetric = False, self_loop = False):
        """
        Create a graph from point cloud data in a feature space using k-nearest neighbors with Euclidean distances as weights, optimized for GPU.
        
        Parameters:
        - points: Tensor of shape (B, N, F) where B is the batch size,
                N is the number of points in the point cloud, and F is the feature dimensions.
                This tensor should already be on the GPU.
        - k: The number of nearest neighbors to consider for graph edges.
        
        Returns:
        - adjacency_matrix: Tensor of shape (B, N, N) representing the weighted adjacency matrix of the graph,
                            where weights are the Euclidean distances between points.
        """
        points = points.to('cuda')
        B, N, _ = points.shape
        
        # Compute pairwise Euclidean distances, shape (B, N, N), on GPU
        dist_matrix = torch.sqrt(torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1)) 
        
        # Find the k-nearest neighbors for each point, including itself
        distances, indices = torch.topk(-dist_matrix, k=k+1, largest=True, dim=-1)
        #distances, indices = -distances[:, :, 1:], indices[:, :, 1:]  # Remove self-loops
        distances, indices = -distances[:, :, :], indices[:, :, :]  

        if (self_loop):
            indices = indices[:, :, :]  # Remove self-loops    
            distances = distances[:, :, :]  # Remove self-loops    
        else:
            indices = indices[:, :, 1:]  # Remove self-loops    
            distances = distances[:, :, 1:]  # Remove self-loops     
        
        # Create a weighted adjacency matrix on GPU
        adjacency_matrix = torch.zeros(B, N, N, device='cuda') 
        b_idx = torch.arange(B, device='cuda')[:, None, None]
        n_idx = torch.arange(N, device='cuda')[:, None]  
        
        # Use gathered distances as weights
        distances_new = torch.exp((-1) * alpha * (distances)**2)
        adjacency_matrix[b_idx, n_idx, indices] = distances_new
        if (symmetric):
            adjacency_matrix[b_idx, indices, n_idx] = distances_new  # Ensure symmetry    

        dist_matrix_new = torch.exp((-1) * alpha * (dist_matrix)**2)  
        
        return adjacency_matrix, dist_matrix_new     


    def forward(self, pts, noaug = False, vis=False, **kwargs):
        neighborhood, center, neighborhood_org = self.group_divider(pts)


        if (self.method == "MAMBA"):

            x_vis, mask = self.MAE_encoder(neighborhood, center)
            B, _, C = x_vis.shape  # B VIS C

            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points)

            if vis:  # visualization
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                # full_points = torch.cat([rebuild_points,vis_points], dim=0)
                full_center = torch.cat([center[mask], center[~mask]], dim=0)
                # full = full_points + full_center.unsqueeze(1)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return ret1, ret2, full_center
            else:
                return loss1   
            

        elif(self.method == "k_top_eigenvectors_seperate_learnable_tokens"):     

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph, self.self_loop)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)
            
            x_vis, sorted_bool_masked_pos_list, sorted_pos_mask, sorted_pos_full, sorted_bool_masked_pos_tensor, sorted_neighborhood, mask_ratio = self.MAE_encoder(neighborhood, center, top_k_eigenvectors, self.k_top_eigenvectors, self.reverse, noaug)
            B, _, C = x_vis.shape  # B VIS C
            n_masked_tokens_after_masking = int(mask_ratio * neighborhood_org.shape[1])
            n_visible_tokens_after_masking = neighborhood_org.shape[1] - n_masked_tokens_after_masking

 
            if (noaug == True): 
                return x_vis

            _, N, _ = sorted_pos_mask.shape 
            mask_token = self.mask_token.expand(B, N, -1)

            x_full_list = []

            for cnt, i in enumerate(sorted_bool_masked_pos_list):

                x_full = torch.zeros((x_vis.shape[0], center.shape[1], x_vis.shape[-1])).cuda()

                B, T, D = x_full.shape

                mask_token_part = mask_token[:, (cnt*n_masked_tokens_after_masking):(cnt+1)*n_masked_tokens_after_masking, :]
                x_vis_part = x_vis[:, (cnt*n_visible_tokens_after_masking):(cnt+1)*n_visible_tokens_after_masking, :]

                # Generate indices where i == 1 and i == 0
                mask_indices = torch.where(i == 1)
                vis_indices = torch.where(i == 0)

                # Update x_full where i is 1
                x_full[mask_indices] = mask_token_part.view(-1, D)[0:len(mask_indices[0])]

                # Update x_full where i is 0
                x_full[vis_indices] = x_vis_part.reshape(-1, D)[0:len(vis_indices[0])]

                x_full_list.append(x_full)

            cnt = 4
            x_full_tensor_1 = torch.cat(x_full_list, 1)
            x_full_tensor_2 = torch.zeros((x_full_tensor_1.shape[0], x_full_tensor_1.shape[1], x_full_tensor_1.shape[-1])).cuda()
            mask_token_part = mask_token[:, (cnt*n_masked_tokens_after_masking): , :]
            x_vis_part = x_vis[:, (cnt*n_visible_tokens_after_masking): , :]

            # Generate indices where i == 1 and i == 0
            mask_indices = torch.where(sorted_bool_masked_pos_tensor == 1)
            vis_indices = torch.where(sorted_bool_masked_pos_tensor == 0)

            # Update x_full where i is 1
            x_full_tensor_2[mask_indices] = mask_token_part.view(-1, D)[0:len(mask_indices[0])]

            # Update x_full where i is 0
            x_full_tensor_2[vis_indices] = x_vis_part.reshape(-1, D)[0:len(vis_indices[0])]

            x_full_tensor = torch.cat((x_full_tensor_1, x_full_tensor_2), 1)  

            x_rec = self.MAE_decoder(x_full_tensor, sorted_pos_full, N)

            sorted_bool_masked_pos_tensor_1 = torch.cat(sorted_bool_masked_pos_list, 1)
            sorted_bool_masked_pos_tensor_final = torch.cat((sorted_bool_masked_pos_tensor_1, sorted_bool_masked_pos_tensor), 1)

            x_rec = x_rec[sorted_bool_masked_pos_tensor_final].reshape(B, -1, C)

            B, M, C = x_rec.shape  
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = sorted_neighborhood[sorted_bool_masked_pos_tensor_final].reshape(B * M, -1, 3) 
            loss1 = self.loss_func(rebuild_points, gt_points) 

            return loss1[0]       
            #return loss1       
        
        elif(self.method == "k_top_eigenvectors_seperate_learnable_tokens_weighted_adjacency"):       

            adjacency_matrix, dist_matrix = self.create_graph_from_feature_space_gpu_weighted_adjacency(center, self.knn_graph, self.alpha, self.symmetric, self.self_loop)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)
            
            x_vis, sorted_bool_masked_pos_list, sorted_pos_mask, sorted_pos_full, sorted_bool_masked_pos_tensor, sorted_neighborhood, mask_ratio = self.MAE_encoder(neighborhood, center, top_k_eigenvectors, self.k_top_eigenvectors, self.reverse, noaug)
            B, _, C = x_vis.shape  # B VIS C
            n_masked_tokens_after_masking = int(mask_ratio * neighborhood_org.shape[1])
            n_visible_tokens_after_masking = neighborhood_org.shape[1] - n_masked_tokens_after_masking

 
            if (noaug == True): 
                return x_vis

            _, N, _ = sorted_pos_mask.shape 
            mask_token = self.mask_token.expand(B, N, -1)

            x_full_list = []

            for cnt, i in enumerate(sorted_bool_masked_pos_list):

                x_full = torch.zeros((x_vis.shape[0], center.shape[1], x_vis.shape[-1])).cuda()

                B, T, D = x_full.shape

                mask_token_part = mask_token[:, (cnt*n_masked_tokens_after_masking):(cnt+1)*n_masked_tokens_after_masking, :]
                x_vis_part = x_vis[:, (cnt*n_visible_tokens_after_masking):(cnt+1)*n_visible_tokens_after_masking, :]

                # Generate indices where i == 1 and i == 0
                mask_indices = torch.where(i == 1)
                vis_indices = torch.where(i == 0)

                # Update x_full where i is 1
                x_full[mask_indices] = mask_token_part.view(-1, D)[0:len(mask_indices[0])]

                # Update x_full where i is 0
                x_full[vis_indices] = x_vis_part.reshape(-1, D)[0:len(vis_indices[0])]

                x_full_list.append(x_full)

            cnt = 4
            x_full_tensor_1 = torch.cat(x_full_list, 1)
            x_full_tensor_2 = torch.zeros((x_full_tensor_1.shape[0], x_full_tensor_1.shape[1], x_full_tensor_1.shape[-1])).cuda()
            mask_token_part = mask_token[:, (cnt*n_masked_tokens_after_masking): , :]
            x_vis_part = x_vis[:, (cnt*n_visible_tokens_after_masking): , :]

            # Generate indices where i == 1 and i == 0
            mask_indices = torch.where(sorted_bool_masked_pos_tensor == 1)
            vis_indices = torch.where(sorted_bool_masked_pos_tensor == 0)

            # Update x_full where i is 1
            x_full_tensor_2[mask_indices] = mask_token_part.view(-1, D)[0:len(mask_indices[0])]

            # Update x_full where i is 0
            x_full_tensor_2[vis_indices] = x_vis_part.reshape(-1, D)[0:len(vis_indices[0])]

            x_full_tensor = torch.cat((x_full_tensor_1, x_full_tensor_2), 1)  

            x_rec = self.MAE_decoder(x_full_tensor, sorted_pos_full, N)

            sorted_bool_masked_pos_tensor_1 = torch.cat(sorted_bool_masked_pos_list, 1)
            sorted_bool_masked_pos_tensor_final = torch.cat((sorted_bool_masked_pos_tensor_1, sorted_bool_masked_pos_tensor), 1)

            x_rec = x_rec[sorted_bool_masked_pos_tensor_final].reshape(B, -1, C)

            B, M, C = x_rec.shape  
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = sorted_neighborhood[sorted_bool_masked_pos_tensor_final].reshape(B * M, -1, 3) 
            loss1 = self.loss_func(rebuild_points, gt_points) 

            return loss1 


        elif(self.method == "k_top_eigenvectors_shared_learnable_tokens"):      

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)
            
            x_vis, sorted_bool_masked_pos_list, sorted_pos_mask, sorted_pos_full, sorted_bool_masked_pos_tensor, sorted_neighborhood, sorted_found_indices_tensor_final = self.MAE_encoder(neighborhood, center, top_k_eigenvectors, self.k_top_eigenvectors, self.reverse, noaug)
            B, _, C = x_vis.shape  # B VIS C
 
            if (noaug == True):   
                return x_vis

            _, N, _ = sorted_pos_mask.shape 
            mask_token = self.mask_token.expand(B, 38, -1) 

            x_full_list = []

            for cnt, i in enumerate(sorted_bool_masked_pos_list):

                x_full = torch.zeros((x_vis.shape[0], center.shape[1], x_vis.shape[-1])).cuda()

                B, T, D = x_full.shape

                ##mask_token_part = mask_token[:, (cnt*38):(cnt+1)*38, :]
                mask_token_part = mask_token
                x_vis_part = x_vis[:, (cnt*26):(cnt+1)*26, :]

                # Generate indices where i == 1 and i == 0
                ##mask_indices = torch.where(i == 1)
                batch_indices = torch.arange(B).unsqueeze(1).expand(B, 38)

                mask_indices = sorted_found_indices_tensor_final[:, :, cnt]
                vis_indices = torch.where(i == 0)

                # Update x_full where i is 1
                ####x_full[mask_indices] = mask_token_part.view(-1, D)[0:len(mask_indices[0])]
                x_full[batch_indices, mask_indices] = mask_token_part

                # Update x_full where i is 0
                x_full[vis_indices] = x_vis_part.reshape(-1, D)[0:len(vis_indices[0])]

                x_full_list.append(x_full)  

            cnt = 4
            x_full_tensor_1 = torch.cat(x_full_list, 1)  
            x_full_tensor_2 = torch.zeros((x_full_tensor_1.shape[0], x_full_tensor_1.shape[1], x_full_tensor_1.shape[-1])).cuda()

            mask_indices = sorted_found_indices_tensor_final[:, :, cnt:]  
            mask_indices_1 = mask_indices[..., 0]
            mask_indices_2 = mask_indices[..., 1] + 64
            mask_indices_3 = mask_indices[..., 2] + 128
            mask_indices_4 = mask_indices[..., 3] + 192

            mask_indices = torch.cat((mask_indices_1, mask_indices_2, mask_indices_3, mask_indices_4), -1)  

            ###mask_token_part = mask_token[:, (cnt*38): , :]
            mask_token_part = torch.cat((mask_token, mask_token, mask_token, mask_token), 1)
            # mask_token_part_list = []
            # for i in range(4):
            #     i += 4
            #     mask_token_part = mask_token[sorted_found_indices_tensor_final[:, :, i]].reshape(B, -1, 384)
            #     mask_token_part_list.append(mask_token_part)
            # mask_token_part = torch.cat(mask_token_part_list, 1)    
            x_vis_part = x_vis[:, (cnt*26): , :]

            # Generate indices where i == 1 and i == 0
            ###mask_indices = torch.where(sorted_bool_masked_pos_tensor == 1)

            vis_indices = torch.where(sorted_bool_masked_pos_tensor == 0)

            # Update x_full where i is 1
            ######x_full_tensor_2[mask_indices] = mask_token_part.view(-1, D)[0:len(mask_indices[0])]
            batch_indices = torch.arange(B).unsqueeze(1).expand(B, int(4*38))
            x_full_tensor_2[batch_indices, mask_indices] = mask_token_part

            # Update x_full where i is 0
            x_full_tensor_2[vis_indices] = x_vis_part.reshape(-1, D)[0:len(vis_indices[0])] 

            x_full_tensor = torch.cat((x_full_tensor_1, x_full_tensor_2), 1)  

            x_rec = self.MAE_decoder(x_full_tensor, sorted_pos_full, N)

            sorted_bool_masked_pos_tensor_1 = torch.cat(sorted_bool_masked_pos_list, 1)
            sorted_bool_masked_pos_tensor_final = torch.cat((sorted_bool_masked_pos_tensor_1, sorted_bool_masked_pos_tensor), 1)

            x_rec = x_rec[sorted_bool_masked_pos_tensor_final].reshape(B, -1, C)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = sorted_neighborhood[sorted_bool_masked_pos_tensor_final].reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points) 

            return loss1         
                

        """elif(self.method == "k_top_eigenvectors_seperate_learnable_tokens"):       

            adjacency_matrix = self.create_graph_from_feature_space_gpu(center, self.knn_graph)
            top_k_eigenvalues, top_k_eigenvectors, eigenvalues, eigenvectors = self.calc_top_k_eigenvalues_eigenvectors(adjacency_matrix, k=self.k_top_eigenvectors, smallest = self.smallest)
            
            x_vis, sorted_bool_masked_pos_list, sorted_pos_mask, sorted_pos_full, sorted_bool_masked_pos_tensor, sorted_neighborhood = self.MAE_encoder(neighborhood, center, top_k_eigenvectors, self.k_top_eigenvectors, self.reverse, noaug)
            B, _, C = x_vis.shape  # B VIS C
 
            if (noaug == True): 
                return x_vis

            _, N, _ = sorted_pos_mask.shape 
            mask_token = self.mask_token.expand(B, N, -1)

            x_full_list = []

            for cnt, i in enumerate(sorted_bool_masked_pos_list):

                x_full = torch.zeros((x_vis.shape[0], center.shape[1], x_vis.shape[-1])).cuda()

                B, T, D = x_full.shape

                mask_token_part = mask_token[:, (cnt*38):(cnt+1)*38, :]
                x_vis_part = x_vis[:, (cnt*26):(cnt+1)*26, :]

                # Generate indices where i == 1 and i == 0
                mask_indices = torch.where(i == 1)
                vis_indices = torch.where(i == 0)

                # Update x_full where i is 1
                x_full[mask_indices] = mask_token_part.view(-1, D)[0:len(mask_indices[0])]

                # Update x_full where i is 0
                x_full[vis_indices] = x_vis_part.reshape(-1, D)[0:len(vis_indices[0])]

                x_full_list.append(x_full)

            cnt = 4
            x_full_tensor_1 = torch.cat(x_full_list, 1)
            x_full_tensor_2 = torch.zeros((x_full_tensor_1.shape[0], x_full_tensor_1.shape[1], x_full_tensor_1.shape[-1])).cuda()
            mask_token_part = mask_token[:, (cnt*38): , :]
            x_vis_part = x_vis[:, (cnt*26): , :]

            # Generate indices where i == 1 and i == 0
            mask_indices = torch.where(sorted_bool_masked_pos_tensor == 1)
            vis_indices = torch.where(sorted_bool_masked_pos_tensor == 0)

            # Update x_full where i is 1
            x_full_tensor_2[mask_indices] = mask_token_part.view(-1, D)[0:len(mask_indices[0])]

            # Update x_full where i is 0
            x_full_tensor_2[vis_indices] = x_vis_part.reshape(-1, D)[0:len(vis_indices[0])]

            x_full_tensor = torch.cat((x_full_tensor_1, x_full_tensor_2), 1)  

            x_rec = self.MAE_decoder(x_full_tensor, sorted_pos_full, N)

            sorted_bool_masked_pos_tensor_1 = torch.cat(sorted_bool_masked_pos_list, 1)
            sorted_bool_masked_pos_tensor_final = torch.cat((sorted_bool_masked_pos_tensor_1, sorted_bool_masked_pos_tensor), 1)

            x_rec = x_rec[sorted_bool_masked_pos_tensor_final].reshape(B, -1, C)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = sorted_neighborhood[sorted_bool_masked_pos_tensor_final].reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points) 

            return loss1"""



############## plot     
import open3d as o3d
import numpy as np


def apply_color_gradient_and_save(neighborhood_org, integers_minus, filename="sorted_neighborhood_org.ply"):
    N, M, C = 128, 32, 3
    colors = np.linspace([0, 0, 1], [1, 0, 0], N)  # Gradient from blue to red

    # Sorting neighborhood_org according to integers_minus
    sorted_neighborhood_org = neighborhood_org[integers_minus].cpu().numpy()  # Shape: (128, 32, 3)
    flat_points = sorted_neighborhood_org.reshape(-1, 3)  # Flatten to (128*32, 3)

    # Replicate colors for each token's neighbors
    token_colors = np.repeat(colors, M, axis=0)  # Shape: (128*32, 3)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flat_points)
    pcd.colors = o3d.utility.Vector3dVector(token_colors)

    # Save to PLY file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to '{filename}'")

#apply_color_gradient_and_save_for_eigenvectors(neighborhood_org[0], top_k_eigenvectors[0:1, :, 0], filename="sorted_neighborhood_org.ply")

def apply_color_gradient_and_save_for_eigenvectors(neighborhood_org, top_k_eigenvectors, filename="sorted_neighborhood_org.ply"):
    N, M, C = 128, 32, 3
    colors = np.linspace([0, 0, 1], [1, 0, 0], N)  # Gradient from blue to red

    # Sorting neighborhood_org according to integers_minus
    sorted_values, sorted_indices = torch.sort(top_k_eigenvectors, dim=1)
    sorted_neighborhood_org = neighborhood_org[sorted_indices].cpu().numpy()  # Shape: (128, 32, 3)
    flat_points = sorted_neighborhood_org.reshape(-1, 3)  # Flatten to (128*32, 3)

    # Replicate colors for each token's neighbors
    token_colors = np.repeat(colors, M, axis=0)  # Shape: (128*32, 3)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flat_points)
    pcd.colors = o3d.utility.Vector3dVector(token_colors)

    # Save to PLY file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to '{filename}'")

############### plot eigenvetors
import numpy as np
import matplotlib.pyplot as plt

# Simulate a 128x128 tensor with random data as an example
# Replace this with your actual data, i.e., EIGENVECTORS
"""EIGENVECTORS = np.random.randn(128, 128)

# Set up the figure and axes for the plots
fig, axs = plt.subplots(16, 8, figsize=(24, 32))  # Create a 16x8 grid of subplots

# Indexing for the axes
for i in range(16):
    for j in range(8):
        ax = axs[i, j]
        idx = i * 8 + j  # Calculate the column index from the grid position
        ax.plot(EIGENVECTORS[:, idx])  # Plot the eigenvector
        ax.set_title(f'Eigenvector {idx}')
        ax.set_xlim([0, 127])  # Set x-axis limits
        ax.grid(True)  # Optional: add a grid for easier visualization

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure to a file
plt.savefig('/mnt/data/eigenvectors_grid.png')
"""



###################### plot edges with point cloudimport numpy as np

def save_ply_with_edges(filename, points, adjacency_matrix):
    # Calculate the number of edges (assuming undirected graph, count each edge once)
    num_edges = np.sum(np.triu(adjacency_matrix, 1))  # Only count upper triangle excluding diagonal

    with open(filename, 'w') as ply_file:
        # PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(len(points)))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("element edge {}\n".format(num_edges))
        ply_file.write("property int vertex1\n")
        ply_file.write("property int vertex2\n")
        ply_file.write("end_header\n")

        # Vertex elements
        for point in points:
            # Ensure that the formatting of numbers is consistent; use formatted string for safety
            ply_file.write("{:.6f} {:.6f} {:.6f}\n".format(point[0], point[1], point[2]))

        # Edge elements - only write an edge once
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i + 1, adjacency_matrix.shape[1]):  # Only consider upper triangle
                if adjacency_matrix[i, j] == 1:
                    ply_file.write("{} {}\n".format(i, j))

# Example usage
"""points = np.random.rand(128, 3)  # Generate some random points
# Create an adjacency matrix for a random graph
adjacency_matrix = (np.random.rand(128, 128) > 0.95).astype(int)
# Ensure the matrix is symmetric
adjacency_matrix = np.triu(adjacency_matrix) + np.triu(adjacency_matrix, 1).T

save_ply_with_edges('/mnt/data/graph.ply', points, adjacency_matrix)"""


