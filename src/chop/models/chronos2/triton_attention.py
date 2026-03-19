import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    q_offset = off_hz // H
    h_offset = off_hz % H

    # base pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset * stride_qz + h_offset * stride_qh,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + q_offset * stride_kz + h_offset * stride_kh,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + q_offset * stride_vz + h_offset * stride_vh,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_block_ptr)
    
    # loop over k, v and update accumulator
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        
        # compute qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # compute new m
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # compute new l
        p = tl.math.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc
        alpha = tl.math.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        # update acc
        acc += tl.dot(p.to(v.dtype), v)

        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back result
    acc = acc / l_i[:, None]
    
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset * stride_oz + h_offset * stride_oh,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))

def triton_flash_attention(q, k, v, sm_scale):
    # q, k, v shapes: [batch, seq_len, num_heads, head_dim] (or transposed if strides changed)
    # However, triton make_block_ptr requires correct strides.
    # The kernel expects the dimensions to be Z (batch), H (head), N_CTX (seq_len), DMODEL.
    # Since we can pass strides explicitly, the physical layout can be anything!
    
    # Check that shapes match
    Z, N_CTX, H, D_HEAD = q.shape
    
    assert k.shape == q.shape
    assert v.shape == q.shape
    
    Out = torch.empty_like(q)
    
    BLOCK_M = 128
    BLOCK_N = 64 if D_HEAD > 64 else 128
    
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H, 1)
    
    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale,
        Out,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        Out.stride(0), Out.stride(2), Out.stride(1), Out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D_HEAD,
        num_warps=4,
        num_stages=2,
    )
    
    return Out
