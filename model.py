import math
from collections import OrderedDict
from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
from cached_convnet import CachedConvNet

#입력 길이를 chunk_size 배수로 맞추고, 필요한 패딩을 적용하는 함수
#x: 패딩을 적용할 입력 텐서
#chunk_size: 입력 길이를 이 값의 배수로 맞춰야 하는 기준
#pad: F.pad()에 전달할 추가적인 패딩 튜플
def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    # chunk_size의 배수로 만들기 위해 추가해야 할 길이 초기화
    mod = 0
    # x의 마지막 차원 길이가 chunk_size의 배수가 아니면
    if (x.shape[-1] % chunk_size) != 0:
        # 부족한 만큼 mod에 저장 (chunk_size - 나머지)
        mod = chunk_size - (x.shape[-1] % chunk_size)
    #참고로 pad(x,(a,b))에서 a는 왼쪽, b는 오른쪽
    # 오른쪽 끝에 mod만큼 0으로 패딩 (chunk_size 배수로 맞춤) 이떄, F.pad()는 기본적으로 padding 값을 0으로 채움
    x = F.pad(x, (0, mod))
    # 지정된 pad(입력값으로 주어지는)만큼 앞뒤로 추가 패딩 적용
    x = F.pad(x, pad)

    # 패딩된 텐서와 추가된 mod 값을 반환
    return x, mod

#[B, C, T] 형식의 데이터를 LayerNorm 처리하기 위해 순서를 바꿨다가 다시 되돌리는 LayerNorm 래퍼
class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        # nn.LayerNorm 초기화 (args에는 normalized_shape 등 파라미터가 들어감)
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T] 형식의 입력 텐서 -> 참고로 B:배치, C: 채널 수, T: 시간 길이
        """
        x = x.permute(0, 2, 1)  # [B, C, T] → [B, T, C]로 순서 변경 (LayerNorm이 마지막 차원에 적용되도록)
        x = super().forward(x)  # nn.LayerNorm의 forward 호출하여 정규화 수행
        x = x.permute(0, 2, 1)  # [B, T, C] → [B, C, T]로 다시 원래 순서로 되돌림
        return x  # 정규화된 텐서 반환

#Depthwise + Pointwise Conv 레이어 -> 경량화된 CNN 구조로, 채널 수를 줄이거나 늘릴 때 사용됨
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()  # nn.Module 초기화

        self.layers = nn.Sequential(  # 여러 층을 순차적으로 적용
            # Depthwise convolution: 채널별로 따로 커널을 적용 (groups=in_channels)
            nn.Conv1d(in_channels, in_channels, kernel_size, stride,
                      padding, groups=in_channels, dilation=dilation),
            # LayerNorm 적용을 위한 [B, C, T] → [B, T, C] 변형 및 정규화
            LayerNormPermuted(in_channels),
            # 비선형 활성화 함수
            nn.ReLU(),
            # Pointwise convolution: 1x1 convolution으로 채널 수 변경
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1,
                      padding=0),
            # 다시 LayerNorm + ReLU 적용
            LayerNormPermuted(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # 정의한 순차적 레이어들에 입력 x를 통과시킴
        return self.layers(x)

#인코더
#딜레이를 고려한 인코더로 딜레이를 점차 늘리며 과거 정보를 점점 더 많이 반영
class DilatedCausalConvEncoder(nn.Module):
    """
    A dilated causal convolution based encoder for encoding
    time domain audio input into latent space.
    """

    def __init__(self, channels, num_layers, kernel_size=3):
        super(DilatedCausalConvEncoder, self).__init__()  # nn.Module 초기화
        self.channels = channels  # 채널 수 저장
        self.num_layers = num_layers  # 레이어 개수 저장
        self.kernel_size = kernel_size  # 커널 크기 저장

        # 각 레이어별 버퍼 길이 계산: (커널 크기 - 1) * dilation(팽창)
        # dilation은 2^i로 증가 → 과거 정보를 점점 더 넓게 보는 구조
        self.buf_lengths = [(kernel_size - 1) * 2**i
                            for i in range(num_layers)]

        # 각 레이어에서 사용하는 입력 버퍼의 시작 인덱스 계산
        # 이전 레이어까지의 누적 길이를 기준으로 함
        self.buf_indices = [0]
        for i in range(num_layers - 1):
            self.buf_indices.append(
                self.buf_indices[-1] + self.buf_lengths[i])

        # Dilated causal conv 레이어들을 정의(딕셔너리로 사용)
        # dilation을 늘려가며 점점 더 넓은 과거 정보를 반영하도록 설계
        _dcc_layers = OrderedDict()
        for i in range(num_layers):
            # Depthwise + Pointwise로 구성된 경량 CNN 사용
            dcc_layer = DepthwiseSeparableConv(
                channels, channels, kernel_size=3, stride=1,
                padding=0, dilation=2**i)  # dilation은 1, 2, 4, 8, ... 식으로 증가
            # 'dcc_0', 'dcc_1', ... 이름으로 layer 등록 {'dcc_0': dcc_layer} 이런식으로 하나씩 저장
            _dcc_layers.update({'dcc_%d' % i: dcc_layer})
        
        # Sequential로 묶어서 하나의 모듈로 저장
        self.dcc_layers = nn.Sequential(_dcc_layers)

    # 각 레이어별 context buffer 초기화(이전에 본 정보들)
    def init_ctx_buf(self, batch_size, device):
        """
        Returns an initialized context buffer for a given batch size.
        """
        return torch.zeros(
            (batch_size, self.channels,
             (self.kernel_size - 1) * (2**self.num_layers - 1)),
            device=device)  # 총 context buffer 길이 = (커널크기-1) * (2^num_layers - 1)

    
    # 입력 x와 이전 context를 바탕으로 인코딩 결과와 새로운 context buffer 반환
    def forward(self, x, ctx_buf):
        """
        Encodes input audio `x` into latent space, and aggregates
        contextual information in `ctx_buf`. Also generates new context
        buffer with updated context.
        """
        T = x.shape[-1]  # 입력 시퀀스 길이 [B, C, T] 중 T

        for i in range(self.num_layers):  # 각 DCC 레이어에 대해 반복
            buf_start_idx = self.buf_indices[i]  # 현재 레이어의 버퍼 시작 위치
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]  # 끝 위치

            # 현재 레이어 입력 = (해당 레이어의 context buffer 일부 + 현재 입력)
            dcc_in = torch.cat(
                (ctx_buf[..., buf_start_idx:buf_end_idx], x), dim=-1)

            # context buffer 업데이트: 최근 입력(dcc_in)으로 해당 구간(ctx_buf)을 덮어씀(참고로 \이거 걍 줄바꿈임)
            ctx_buf[..., buf_start_idx:buf_end_idx] = \
                dcc_in[..., -self.buf_lengths[i]:]

            # 잔차 연결 (residual connection) 적용
            x = x + self.dcc_layers[i](dcc_in)

        return x, ctx_buf  # 인코딩된 출력과 업데이트된 context buffer 반환


# Transformer 디코더 레이어 정의 (causal attention 구조, 마지막 토큰만 사용)
class CausalTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    """
    Adapted from:
    https://github.com/alexmt-scale/causal-transformer-decoder
    """

    # Self-attention → Cross-attention → FeedForward 순서로 수행
    def forward(
        self,
        tgt: Tensor,                    # 디코더 입력: [B, T, C]
        memory: Optional[Tensor] = None,  # 인코더 출력 (optional): [B, S, C]
        chunk_size: int = 1            # 마지막 몇 개의 토큰만 계산할지 (기본은 1개)
    ) -> Tensor:
        # 마지막 토큰 (또는 여러 개, chunk_size 개)만 attention의 query로 사용
        tgt_last_tok = tgt[:, -chunk_size:, :]  # [B, chunk_size, C]

        # ---- Self-Attention ----
        # query: 마지막 토큰만, key/value: 전체 tgt
        tmp_tgt, sa_map = self.self_attn(
            tgt_last_tok,  # query
            tgt,           # key
            tgt,           # value
            attn_mask=None,  # mask 없이도 causal 효과 있음 (마지막 토큰만 계산하므로)
            key_padding_mask=None,
        )
        # residual connection + dropout + layer norm
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # ---- Cross-Attention (encoder-decoder attention) ----
        if memory is not None:
            tmp_tgt, ca_map = self.multihead_attn(
                tgt_last_tok,  # query: 마지막 토큰
                memory,        # key
                memory,        # value
                attn_mask=None,  # 전체 memory에 attention 허용
                key_padding_mask=None,
            )
            # residual connection + dropout + layer norm
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        # ---- Feed-Forward Network ----
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))  # FFN
        )
        # residual connection + dropout + layer norm (이걸 총 3번 하는듯)
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)

        # 마지막 토큰의 출력 + attention map 반환
        return tgt_last_tok, sa_map, ca_map

#위 layer들을 쌓은 디코더
#지정된 길이만큼의 과거 정보 (ctx_len)를 고려하여 chunk 단위로 decoding
# Transformer 기반 Causal 디코더: 과거 정보만 보고 chunk 단위로 디코딩
class CausalTransformerDecoder(nn.Module):
    """
    A causal transformer decoder which decodes input vectors using
    precisely `ctx_len` past vectors in the sequence, and using no future
    vectors at all.
    """

    def __init__(self, model_dim, ctx_len, chunk_size, num_layers,
                 nhead, use_pos_enc, ff_dim, dropout):
        super(CausalTransformerDecoder, self).__init__()
        self.num_layers = num_layers              # Transformer layer 개수
        self.model_dim = model_dim                # 모델 차원
        self.ctx_len = ctx_len                    # 과거 context 길이
        self.chunk_size = chunk_size              # chunk 단위 처리 크기
        self.nhead = nhead                        # multi-head attention 개수
        self.use_pos_enc = use_pos_enc            # positional encoding 사용 여부

        # unfold 연산을 위한 설정: 과거 + 현재 chunk 길이 만큼 sliding window
        self.unfold = nn.Unfold(kernel_size=(ctx_len + chunk_size, 1), stride=chunk_size)

        # Positional encoding: 위치 정보 제공 (모델에게 순서를 알려주는 장치 -> "입력의 순서")
        self.pos_enc = PositionalEncoding(model_dim, max_len=200)

        # Transformer 디코더 레이어들 쌓기
        self.tf_dec_layers = nn.ModuleList([
            CausalTransformerDecoderLayer(
                d_model=model_dim, nhead=nhead,
                dim_feedforward=ff_dim,
                batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])

        
    # 디코더용 context buffer 초기화: [B, num_layers+1, ctx_len, model_dim]
    def init_ctx_buf(self, batch_size, device):
        return torch.zeros(
            (batch_size, self.num_layers + 1, self.ctx_len, self.model_dim),
            device=device)

    
    # 입력을 chunk 단위로 나누고, 그 앞에 ctx_len 만큼의 과거를 붙여주는 함수
    def _causal_unfold(self, x):
        """
        x: [B, ctx_len + L, C]
        ctx_len: int
        Returns: [B * L, ctx_len + chunk_size, C]
        """
        B, T, C = x.shape
        x = x.permute(0, 2, 1)                      # [B, C, T]
        x = self.unfold(x.unsqueeze(-1))            # [B, C*(ctx_len+chunk), L]
        x = x.permute(0, 2, 1)                      # [B, L, C*(ctx_len+chunk)]
        x = x.reshape(B, -1, C, self.ctx_len + self.chunk_size) #unfold로 만든 데이터를 다시 [batch, chunk 수, 채널, chunk 길이]로 정리
        x = x.reshape(-1, C, self.ctx_len + self.chunk_size)  # [B*L, C, ctx+chunk]
        x = x.permute(0, 2, 1)                      # [B*L, ctx+chunk, C]
        return x

    
    # 전체 디코딩 흐름
    def forward(self, tgt, mem, ctx_buf, probe=False):
        """
        tgt: [B, model_dim, T] - 디코더 입력
        mem: [B, model_dim, T] - 인코더 출력
        ctx_buf: [B, num_layers+1, ctx_len, model_dim] - context buffer
        """

        # 입력 길이를 chunk_size 배수로 맞추고 패딩
        mem, _ = mod_pad(mem, self.chunk_size, (0, 0))
        tgt, mod = mod_pad(tgt, self.chunk_size, (0, 0))

        B, C, T = tgt.shape

        tgt = tgt.permute(0, 2, 1)  # [B, T, C]
        mem = mem.permute(0, 2, 1)  # [B, T, C]

        # 인코더 context를 현재 mem 앞에 붙이기
        mem = torch.cat((ctx_buf[:, 0, :, :], mem), dim=1)
        # context buffer 갱신 (다음 step을 위한 최신 context 저장)
        ctx_buf[:, 0, :, :] = mem[:, -self.ctx_len:, :]

        # unfold로 chunk화 + context 앞붙이기
        mem_ctx = self._causal_unfold(mem)
        if self.use_pos_enc:
            mem_ctx = mem_ctx + self.pos_enc(mem_ctx)

        K = 1000  # attention을 나눠서 처리해 OOM 방지

        for i, tf_dec_layer in enumerate(self.tf_dec_layers):
            # 현재 레이어의 context를 tgt 앞에 붙이기
            tgt = torch.cat((ctx_buf[:, i + 1, :, :], tgt), dim=1)
            # context buffer 갱신
            ctx_buf[:, i + 1, :, :] = tgt[:, -self.ctx_len:, :]

            # unfold로 chunk + context 붙인 입력 생성
            tgt_ctx = self._causal_unfold(tgt)
            if self.use_pos_enc and i == 0:
                tgt_ctx = tgt_ctx + self.pos_enc(tgt_ctx)

            # output 초기화: 마지막 chunk만 쓸 거라 zeros로 생성
            tgt = torch.zeros_like(tgt_ctx)[:, -self.chunk_size:, :]

            # K 단위로 나눠서 attention 수행 (OOM 방지)
            for i in range(int(math.ceil(tgt.shape[0] / K))):
                tgt[i*K:(i+1)*K], _sa_map, _ca_map = tf_dec_layer(
                    tgt_ctx[i*K:(i+1)*K], mem_ctx[i*K:(i+1)*K], self.chunk_size)

            # 모든 chunk 처리 후 원래 모양으로 복원
            tgt = tgt.reshape(B, T, C)

        tgt = tgt.permute(0, 2, 1)  # [B, C, T]로 다시 복원

        if mod != 0:
            tgt = tgt[..., :-mod]  # 패딩 제거

        return tgt, ctx_buf  # 디코딩 결과와 context buffer 반환


#인코더-디코더 구조 기반 마스크 생성기
#입력 오디오 + 라벨을 인코딩하고, Transformer 디코더로 마스크를 생성
class MaskNet(nn.Module):
    def __init__(self, enc_dim, num_enc_layers, dec_dim, dec_buf_len,
                 dec_chunk_size, num_dec_layers, use_pos_enc, skip_connection, proj, decoder_dropout):
        super(MaskNet, self).__init__()
        self.skip_connection = skip_connection  # skip 연결 여부
        self.proj = proj  # encoder-decoder 사이 차원 변환 여부

        # Dilated causal convolution 기반 인코더 초기화
        self.encoder = DilatedCausalConvEncoder(
            channels=enc_dim,
            num_layers=num_enc_layers)

        # encoder → decoder 차원 변환: 인코더 출력 e에 적용
        self.proj_e2d_e = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())

        # encoder → decoder 차원 변환: 라벨 통합값 l에 적용
        self.proj_e2d_l = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())

        # decoder → encoder 차원 복원: 디코더 출력 m에 적용
        self.proj_d2e = nn.Sequential(
            nn.Conv1d(dec_dim, enc_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())

        # Transformer 기반 causal decoder 정의
        self.decoder = CausalTransformerDecoder(
            model_dim=dec_dim,
            ctx_len=dec_buf_len,
            chunk_size=dec_chunk_size,
            num_layers=num_dec_layers,
            nhead=8,
            use_pos_enc=use_pos_enc,
            ff_dim=2 * dec_dim,
            dropout=decoder_dropout)

        
    # 인코딩 → 라벨 통합 → 디코딩 → 마스크 생성 (전체 forward 흐름)
    def forward(self, x, l, enc_buf, dec_buf):
        """
        x: [B, C, T] - 입력 오디오
        l: [B, C] - 라벨 임베딩
        enc_buf: 인코더 context 버퍼
        dec_buf: 디코더 context 버퍼
        """

        # 1. Dilated causal convolution 인코더를 통해 입력 인코딩
        e, enc_buf = self.encoder(x, enc_buf)  # e: 인코딩 결과

        # 2. 라벨 정보와 인코딩 결과를 곱하여 label-aware 특성 생성
        l = l.unsqueeze(2) * e  # [B, C, 1] * [B, C, T] → [B, C, T]

        # 3. decoder 차원으로 맞춰 변환이 필요한 경우
        if self.proj:
            e = self.proj_e2d_e(e)  # 인코딩된 입력 e 차원 변환
            m = self.proj_e2d_l(l)  # 라벨 통합 특성 l 차원 변환
            m, dec_buf = self.decoder(m, e, dec_buf)  # 디코더 통과
        else:
            m, dec_buf = self.decoder(l, e, dec_buf)  # 변환 없이 바로 디코더 입력

        # 4. 디코더 출력 m을 encoder 차원으로 복원
        if self.proj:
            m = self.proj_d2e(m)

        # 5. skip connection 적용: label-aware 입력 l + 디코더 출력 m
        if self.skip_connection:
            m = l + m

        # 최종 마스크 m과 context buffer 반환
        return m, enc_buf, dec_buf


#전체 모델을 구성하는 main 클래스
#입력 오디오 → 전처리 Conv → 라벨 인코딩 → 마스크 생성 → 마스크 적용 → ConvTranspose로 최종 음성 복원
#labe_num 필요가 없어지고 num_speakers로 바꿈
class Net(nn.Module):
    def __init__(self, num_speakers, L=8,
                 enc_dim=512, num_enc_layers=10,
                 dec_dim=256, dec_buf_len=100, num_dec_layers=2,
                 dec_chunk_size=72, out_buf_len=2,
                 use_pos_enc=True, skip_connection=True, proj=True, lookahead=True, decoder_dropout=0.0, convnet_config=None):
        super(Net, self).__init__()  # 부모 클래스(nn.Module) 초기화

        self.L = L  # 오디오를 몇 샘플 단위로 나눌지 결정하는 stride 값
        self.dec_chunk_size = dec_chunk_size  # 디코더가 한 번에 처리하는 chunk 크기
        self.out_buf_len = out_buf_len  # 디코더 출력 버퍼 길이
        self.enc_dim = enc_dim  # 인코더의 출력 차원 수
        self.lookahead = lookahead  # 미래 정보를 사용할지 여부

        self.convnet_config = convnet_config  # convnet 관련 설정 저장

        # convnet 프리넷이 활성화되어 있다면, CachedConvNet 초기화
        if convnet_config['convnet_prenet']:
            self.convnet_pre = CachedConvNet(
                1,  # 입력 채널 수 (1채널 오디오)
                convnet_config['kernel_sizes'],  # 각 레이어의 커널 크기 리스트
                convnet_config['dilations'],  # 각 레이어의 dilation 리스트
                convnet_config['dropout'],  # 드롭아웃 비율
                convnet_config['combine_residuals'],  # 잔차 연결 방식
                convnet_config['use_residual_blocks'],  # 잔차 블록 사용 여부
                convnet_config['out_channels'],  # 출력 채널 수
                use_2d=False)  # 1D convolution 사용 (오디오이므로)

        # 입력 오디오를 latent 특성으로 변환하는 conv layer
        kernel_size = 3 * L if lookahead else L  # lookahead 여부에 따라 커널 크기 결정
        self.in_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,  # 입력: 1채널 오디오
                out_channels=enc_dim,  # 출력 차원: enc_dim
                kernel_size=kernel_size,  # 위에서 설정한 커널 크기
                stride=L,  # stride = L
                padding=0,  # padding 없음
                bias=False),  # bias 사용 안 함
            nn.ReLU())  # ReLU 활성화 함수 적용

        # 라벨 임베딩 층: 1차원 라벨을 enc_dim 차원으로 변환 -> 기존에 이 부분이 라벨 1개만 입력
        # label_len = 1  # 라벨 길이는 1로 고정
        # self.label_embedding = nn.Sequential(
        #     nn.Linear(label_len, 512),  # 1 → 512 차원
        #     nn.LayerNorm(512),  # 정규화
        #     nn.ReLU(),
        #     nn.Linear(512, enc_dim),  # 512 → enc_dim
        #     nn.LayerNorm(enc_dim),  # 정규화
        #     nn.ReLU())

        self.label_embedding = nn.Sequential(
            nn.Linear(num_speakers, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.ReLU())

        # 마스크 생성기 초기화 (Dilated Encoder + Transformer Decoder 포함)
        self.mask_gen = MaskNet(
            enc_dim=enc_dim,  # 인코더 출력 차원
            num_enc_layers=num_enc_layers,  # 인코더 레이어 수
            dec_dim=dec_dim,  # 디코더 모델 차원
            dec_buf_len=dec_buf_len,  # 디코더 context buffer 길이
            dec_chunk_size=dec_chunk_size,  # 디코더 chunk 크기
            num_dec_layers=num_dec_layers,  # 디코더 레이어 수
            use_pos_enc=use_pos_enc,  # positional encoding 사용 여부
            skip_connection=skip_connection,  # skip connection 여부
            proj=proj,  # encoder-decoder 간 차원 변환 여부
            decoder_dropout=decoder_dropout)  # 디코더 dropout 비율

        # 디코더 출력 → 오디오 파형 복원 conv transpose layer
        self.out_conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=enc_dim,  # 입력 채널: 인코더 차원
                out_channels=1,  # 출력 채널: 1채널 오디오 복원
                kernel_size=(out_buf_len + 1) * L,  # 커널 크기: stride를 고려하여 결정
                stride=L,  # stride
                padding=out_buf_len * L,  # padding 크기
                bias=False),  # bias 없음
            nn.Tanh())  # 출력 값을 -1 ~ 1로 제한 (오디오 신호처럼)

        
    #context buffer 초기화
    def init_buffers(self, batch_size, device):
        # 인코더 context buffer 초기화
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size, device)
        # 디코더 context buffer 초기화
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size, device)
        # 출력 버퍼 초기화 (마지막 몇 프레임을 이어 붙이기 위해 사용)
        out_buf = torch.zeros(batch_size, self.enc_dim, self.out_buf_len, device=device)
        return enc_buf, dec_buf, out_buf  # 초기화된 세 가지 버퍼 반환

    
    # 전체 모델의 흐름 처리. 추론 시 또는 실시간 스트리밍에도 대응 가능
    def forward(self, x, target_index, init_enc_buf=None, init_dec_buf=None,
                init_out_buf=None, convnet_pre_ctx=None, pad=True):
        """
        Extracts the audio corresponding to the `label` in the given
        `mixture`. Generates `chunk_size` samples per iteration.
        """

        # label = torch.zeros(x.shape[0], 1, device=x.device)  # 라벨은 현재 모두 0으로 설정된 one-hot 벡터 생성
        label = F.one_hot(target_index, num_classes=3).float().to(x.device)  # [1, 3]
        mod = 0  # 패딩 여부와 길이를 추적하기 위한 변수

        if pad:
            pad_size = (self.L, self.L) if self.lookahead else (0, 0)  # lookahead를 사용할 경우 앞뒤로 L만큼 패딩
            x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)  # 입력 x를 chunk_size의 배수로 맞추고 필요한 경우 패딩 적용

        if hasattr(self, 'convnet_pre'):  # 사전 정의된 convnet 프리넷이 있을 경우
            if convnet_pre_ctx is None:
                convnet_pre_ctx = self.convnet_pre.init_ctx_buf(x.shape[0], x.device)  # convnet 프리넷용 context buffer 초기화

            convnet_out, convnet_pre_ctx = self.convnet_pre(x, convnet_pre_ctx)  # CachedConvNet을 통해 x 전처리 및 context 갱신

            if self.convnet_config['skip_connection'] == 'add':
                x = x + convnet_out  # convnet 출력과 입력을 더함 (skip 연결 방식)
            elif self.convnet_config['skip_connection'] == 'multiply':
                x = x * convnet_out  # convnet 출력과 입력을 곱함
            else:
                x = convnet_out  # skip 연결 없이 convnet 출력만 사용

        if init_enc_buf is None or init_dec_buf is None or init_out_buf is None:  # context buffer가 주어지지 않은 경우
            assert init_enc_buf is None and \
                init_dec_buf is None and \
                init_out_buf is None, \
                "Both buffers have to initialized, or both of them have to be None."  # 버퍼는 모두 주어지거나 모두 None이어야 함

            enc_buf, dec_buf, out_buf = self.init_buffers(x.shape[0], x.device)  # 버퍼 새로 초기화
        else:
            enc_buf, dec_buf, out_buf = init_enc_buf, init_dec_buf, init_out_buf  # 주어진 버퍼 사용

        x = self.in_conv(x)  # Conv1d를 통해 입력 오디오를 latent representation으로 변환

        l = self.label_embedding(label)  # 라벨 벡터를 임베딩하여 인코더 차원에 맞춤

        m, enc_buf, dec_buf = self.mask_gen(x, l, enc_buf, dec_buf)  # 인코딩된 입력과 라벨 임베딩으로 마스크 생성 (context buffer도 갱신)

        x = x * m  # 마스크를 인코딩된 입력에 곱해 원하는 소리만 추출

        x = torch.cat((out_buf, x), dim=-1)  # 이전 출력 버퍼와 현재 출력 이어붙임 (컨텍스트 유지 목적)

        out_buf = x[..., -self.out_buf_len:]  # 출력 버퍼 갱신 (가장 최근 out_buf_len만 유지)

        x = self.out_conv(x)  # ConvTranspose1d를 통해 latent representation을 오디오 파형으로 복원

        if mod != 0:
            x = x[:, :, :-mod]  # 처음 입력에 패딩을 추가했었다면, 그만큼 제거

        if init_enc_buf is None:
            return x  # 추론 시엔 최종 오디오만 반환
        else:
            return x, enc_buf, dec_buf, out_buf, convnet_pre_ctx  # 실시간 스트리밍용: context buffer들도 함께 반환

