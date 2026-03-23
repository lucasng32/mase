import torch
import triton
import triton.language as tl

@triton.jit
def bsr_fwd_kernel(
    Q, K, V, Out,
    indptr, indices,
    valid_mask, group_ids,
    stride_q_m, stride_q_h, stride_q_d,
    stride_k_n, stride_k_h, stride_k_d,
    stride_v_n, stride_v_h, stride_v_d,
    stride_o_m, stride_o_h, stride_o_d,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    # Grid: (Total_Block_Rows, Heads)
    row_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    start_block_idx = tl.load(indptr + row_idx)
    end_block_idx = tl.load(indptr + row_idx + 1)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE, HEAD_DIM], dtype=tl.float32)
    
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    
    q_offs = (row_idx * BLOCK_SIZE + offs_m[:, None]) * stride_q_m + head_idx * stride_q_h + offs_d[None, :] * stride_q_d
    q_mask = (offs_m[:, None] < BLOCK_SIZE) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(Q + q_offs, mask=q_mask, other=0.0)
    
    # Load row-level masking data
    row_valid = tl.load(valid_mask + row_idx * BLOCK_SIZE + offs_m, mask=offs_m < BLOCK_SIZE, other=False)
    row_group = tl.load(group_ids + row_idx * BLOCK_SIZE + offs_m, mask=offs_m < BLOCK_SIZE, other=-1)
    
    k_mask = (offs_d[:, None] < HEAD_DIM) & (offs_m[None, :] < BLOCK_SIZE)
    
    for block_idx in range(start_block_idx, end_block_idx):
        col_idx = tl.load(indices + block_idx)
        
        k_offs = (col_idx * BLOCK_SIZE + offs_m[None, :]) * stride_k_n + head_idx * stride_k_h + offs_d[:, None] * stride_k_d
        v_offs = (col_idx * BLOCK_SIZE + offs_m[:, None]) * stride_v_n + head_idx * stride_v_h + offs_d[None, :] * stride_v_d
        
        k = tl.load(K + k_offs, mask=k_mask, other=0.0)
        v = tl.load(V + v_offs, mask=q_mask, other=0.0)
        
        qk = tl.dot(q, k)
        
        # Intra-block Group Masking
        col_valid = tl.load(valid_mask + col_idx * BLOCK_SIZE + offs_m, mask=offs_m < BLOCK_SIZE, other=False)
        col_group = tl.load(group_ids + col_idx * BLOCK_SIZE + offs_m, mask=offs_m < BLOCK_SIZE, other=-2)
        
        # Match only if (valid AND same group)
        group_match = (row_group[:, None] == col_group[None, :]) & col_valid[None, :]
        qk = tl.where(group_match, qk, -float('inf'))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
    out = acc / l_i[:, None]
    # Zero out rows that are entirely padding
    out = tl.where(row_valid[:, None], out, 0.0)
    
    out_offs = (row_idx * BLOCK_SIZE + offs_m[:, None]) * stride_o_m + head_idx * stride_o_h + offs_d[None, :] * stride_o_d
    tl.store(Out + out_offs, out.to(Out.dtype.element_ty), mask=q_mask)

def run_miniflash_bsr(q, k, v, indptr, indices, valid_mask, group_ids):
    """
    Minimal Block-Sparse Row (BSR) Attention
    q, k, v: [M_flat, Heads, Head_Dim]
    """
    # Self-healing contiguous check
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    M_flat, n_heads, head_dim = q.shape
    
    BLOCK_SIZE = 16
    total_block_rows = M_flat // BLOCK_SIZE
    
    out = torch.empty_like(q)
    
    grid = (total_block_rows, n_heads)
    
    # Needs to be power of 2 for Triton
    triton_head_dim = triton.next_power_of_2(head_dim)
    
    bsr_fwd_kernel[grid](
        q, k, v, out,
        indptr, indices,
        valid_mask, group_ids,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=triton_head_dim
    )
    
    return out