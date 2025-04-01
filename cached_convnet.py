# based on https://github.com/YangangCao/Causal-U-Net/blob/main/cunet.py
import torch # PyTorch 라이브러리
import torch.nn as nn # PyTorch의 신경망 구축에 관련된 여러 클래스와 함수들을 포함하는 모듈
import torch.nn.functional as F # 신경망에서 주로 사용하는 함수들(예: 활성화 함수, 손실 함수, 컨볼루션 연산 등)을 포함

# 잔차 블록(입력 특징을 유지하면서도 학습을 원활하게 만드는 블록,잔차 연결을 이용해 정보 손실을 줄여줌)
class ResidualBlock(nn.Module):
    """
    Based on https://github.com/f90/Seq-U-Net/blob/master/sequnet_res.py
    """
    # ResidualBlock 클래스의 생성자 함수
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout,
                 use_2d):
        super().__init__() # 부모 클래스인 nn.Module의 생성자를 호출하여 PyTorch의 모듈 기능을 사용가능하게 함
        self.use_2d = use_2d # 1D or 2D 컨볼루션 선택
        # 2D 컨볼루션 선택 (영상처리) -> nn.Conv2d, nn.Conv2d 사용
        if use_2d:
            # 컨볼루션 레이어 정의, 입력 데이터 처리 필터 역할
            self.filter = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, dilation=dilation)
            # 필터된 결과에 게이트를 곱하여 제어
            self.gate = nn.Conv2d(in_channels, out_channels,
                                  kernel_size, dilation=dilation)
            # 드롭아웃 적용 -> 과적합 방지
            self.dropout = nn.Dropout2d(dropout)
        # 1D 컨볼루션 선택 (음성처리) -> nn.Conv1d, nn.Conv1d 사용
        else:
            # 컨볼루션 레이어 정의, 입력 데이터 처리 필터 역할
            self.filter = nn.Conv1d(in_channels, out_channels,
                                    kernel_size, dilation=dilation)
            # 필터된 결과에 게이트를 곱하여 제어
            self.gate = nn.Conv1d(in_channels, out_channels,
                                  kernel_size, dilation=dilation)
            # 드롭아웃 적용 -> 과적합 방지
            self.dropout = nn.Dropout1d(dropout)
        # 패딩 크기 계산: 커널 크기와 확장 계수(dilation)로 결정됨
        self.output_crop = dilation * (kernel_size - 1)

    # 입력 데이터 x에 대해 잔차 블록을 실행하는 forward 함수
    def forward(self, x):
        # 필터 처리된 결과에 tanh 활성화 함수 적용
        filtered = torch.tanh(self.filter(x))
        # 게이트 처리된 결과에 sigmoid 활성화 함수 적용
        gated = torch.sigmoid(self.gate(x))
        # 필터와 게이트된 결과 곱해 잔차(residual) 값 생성
        residual = filtered * gated

        
        # pad dim 1 of x to match residual
        if self.use_2d:
            # 입력 x의 크기를 잔차와 맞추기 위해 패딩 적용
            x = F.pad(x, (0, 0, 0, 0, 0, residual.shape[1] - x.shape[1]))
            # 잔차와 입력 데이터의 크기를 맞추고 합성 (Skip connection)
            output = x[..., self.output_crop:, self.output_crop:] + residual 
        else:
            # 입력 x의 크기를 잔차와 맞추기 위해 패딩 적용
            x = F.pad(x, (0, 0, 0, residual.shape[1] - x.shape[1]))
            # 잔차와 입력 데이터의 크기를 맞추고 합성 (Skip connection)
            output = x[..., self.output_crop:] + residual # Skip connection
        
        # 드롭아웃 적용 (과적합 방지)
        output = self.dropout(output)
        return output

# 과거 데이터만 사용하도록 하는 컨볼루션 블록(인과적 컨볼루션 블록)
class CausalConvBlock(nn.Module):
    # CausalConvBlock 클래스의 생성자 함수
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout,
                 use_2d):
        super().__init__()
        # 1D 또는 2D 컨볼루션 선택
        if use_2d:
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
            dropout_layer = nn.Dropout2d
        else:
            conv_layer = nn.Conv1d
            batchnorm_layer = nn.BatchNorm1d
            dropout_layer = nn.Dropout1d
        # 인과적 컨볼루션 블록 정의
        # 인과적 컨볼루션을 위한 레이어들을 순차적으로 정의
        self.conv = nn.Sequential(
            conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation),
            batchnorm_layer(num_features=out_channels), # 배치 정규화 적용
            dropout_layer(dropout), # 드롭아웃 적용 (과적합 방지)
            nn.LeakyReLU(inplace=True), # LeakyReLU 활성화 함수 적용
        )
    # 입력 x에 대해 순방향 전파를 실행하는 함수
    def forward(self, x):
        """
        1D Causal convolution.
        """
        # 정의된 conv 레이어를 사용하여 입력 x에 대해 처리한 결과를 반환
        return self.conv(x)

# U-Net 구조 모델, 각 레이어에 대해 컨볼루션을 처리하며, 캐시된 컨텍스트 버퍼를 사용하는 구조
class CachedConvNet(nn.Module):
    # 모델의 초기화 함수
    def __init__(self, num_channels, kernel_sizes, dilations,
                 dropout, combine_residuals, use_residual_blocks,
                 out_channels, use_2d, use_pool=False, pool_kernel=2):
        super().__init__()
        # 커널 크기와 dilation(확장 계수) 리스트 길이가 같다는 조건 검증
        assert (len(kernel_sizes) == len(dilations)
                ), "kernel_sizes and dilations must be the same length"
        assert (len(kernel_sizes) == len(out_channels)), \
            "kernel_sizes and out_channels must be the same length"
        self.num_layers = len(kernel_sizes) # 전체 레이어 개수
        self.ctx_height = max(out_channels) # 최대 채널 개수 저장
        self.down_convs = nn.ModuleList() # 다운샘플링 컨볼루션 레이어 리스트
        self.num_channels = num_channels # 입력 채널 개수
        self.kernel_sizes = kernel_sizes # 커널 크기 리스트
        self.combine_residuals = combine_residuals # 잔차 연결 방식
        self.use_2d = use_2d # 2D 여부
        self.use_pool = use_pool # 풀링 여부

        # 각 레이어의 컨텍스트 버퍼 크기 계산
        # compute buffer lengths for each layer
        self.buf_lengths = [
            (k - 1) * d for k, d in zip(kernel_sizes, dilations)]

        # 컨텍스트 버퍼의 시작 인덱스 계산
        # Compute buffer start indices for each layer
        self.buf_indices = [0]
        for i in range(len(kernel_sizes) - 1):
            self.buf_indices.append(
                self.buf_indices[-1] + self.buf_lengths[i])

        # 잔차 블록을 사용할지 인과적 블록을 사용할지 선택
        if use_residual_blocks: # 잔차 블록
            block = ResidualBlock
        else: # 인과적 블록
            block = CausalConvBlock

        if self.use_pool:
            # self.use_pool이 True일 경우, 이 코드에서 평균 풀링 계층(nn.AvgPool1d)을 초기화
            self.pool = nn.AvgPool1d(kernel_size=pool_kernel)

        # 각 레이어에 대해 컨볼루션 블록 추가
        for i in range(self.num_layers):
            in_channel = num_channels if i == 0 else out_channels[i - 1]
            self.down_convs.append(
                block(
                    in_channels=in_channel,
                    out_channels=out_channels[i],
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    dropout=dropout,
                    use_2d=use_2d))
    # 각 레이어의 컨텍스트 버퍼 초기화 함수, 배치 크기와 장치에 맞춰 컨텍스트 버퍼를 생성
    def init_ctx_buf(self, batch_size, device, height=None):
        """
        Initialize context buffer for each layer.
        """
        # 컨텍스트 버퍼는 모든 값이 0인 텐서로 초기화
        if height is not None:
            up_ctx = torch.zeros(
                (batch_size, self.ctx_height, height, sum(self.buf_lengths))).to(device)
        else:
            up_ctx = torch.zeros(
                (batch_size, self.ctx_height, sum(self.buf_lengths))).to(device)
        return up_ctx
    
    # 순전파 함수
    def forward(self, x, ctx):
        # x: 입력 데이터(배치, 채널, 시간)
        # ctx: 컨텍스트 버퍼
        """
         Args:
             x: [B, in_channels, T]
                 Input
             ctx: {[B, channels, self.buf_length[0]], ...}
                 A list of tensors holding context for each unet layer. (len(ctx) == self.num_layers)
         Returns:
             x: [B, out_channels, T]
             ctx: {[B, channels, self.buf_length[0]], ...}
                 Updated context buffer with output as the
                 last element.
         """
        # 풀링을 적용하는 옵션을 확인
        if self.use_pool:
            x = self.pool(x) # 평균 풀링 적용
        
        # 각 레이어에서 컨볼루션을 처리하고, 컨텍스트 버퍼를 업데이트
        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            # concatenate context buffer with input
            # 입력(x)과 컨텍스트(ctx)를 결합 -> conv_in
            if self.use_2d:
                conv_in = torch.cat(
                    (ctx[..., :x.shape[1], :x.shape[-2], buf_start_idx:buf_end_idx], x), dim=-1)
            else:
                conv_in = torch.cat(
                    (ctx[..., :x.shape[-2], buf_start_idx:buf_end_idx], x), dim=-1)

            # Push current output to the context buffer
            # conv_in 컨텍스트 버퍼에 업데이트
            if self.use_2d:
                ctx[..., :x.shape[1], :x.shape[-2],
                    buf_start_idx:buf_end_idx] = conv_in[..., -self.buf_lengths[i]:]
            else:
                ctx[..., :x.shape[1], buf_start_idx:buf_end_idx] = conv_in[..., -
                                                                           self.buf_lengths[i]:]

            # pad second-to-last index of input with self.buf_lengths[i] // 2 zeros
            # on each side to ensure that height of output is the same as input
            # 2D 컨볼루션을 사용할 경우, conv_in의 높이와 너비 차원에 대해 (self.buf_lengths[i] // 2)만큼 패딩을 추가하여 출력 크기가 입력과 같도록 한다.
            if self.use_2d:
                conv_in = F.pad(
                    conv_in, (0, 0, self.buf_lengths[i] // 2, self.buf_lengths[i] // 2))
            # 잔차 결합 방식에 따라 연산 수행
            if self.combine_residuals == 'add':
                x = x + self.down_convs[i](conv_in) # 잔차 결합 방식에 따라 잔차 값 더함
            elif self.combine_residuals == 'multiply':
                x = x * self.down_convs[i](conv_in) # 잔차 결합 방식에 따라 잔차 값 곱합
            else:
                x = self.down_convs[i](conv_in) # 그 외의 경우, 컨볼루션 결과를 x에 바로 할당

        # 풀링이 사용되는 경우(self.use_pool=True), 이 코드에서 x를 self.pool.kernel_size[0]에 맞춰 업샘플링
        if self.use_pool:
            x = F.interpolate(x, scale_factor=self.pool.kernel_size[0])

        return x, ctx # x: 변환된 출력 데이터, ctx: 업데이트된 컨텍스트 버퍼
