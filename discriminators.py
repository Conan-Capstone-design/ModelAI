#실제 vs. 생성 음성 구분하기 위한 여러 종류의 판별기(1D/2D, 다양한 주기)와 손실 함수를 정의함
#DiscriminatorS(일반 1D Conv), DiscriminatorP(주기 기반 2D Conv), MultiPeriodDiscriminator(여러 주기)
#음성 퀄리티와 자연스러움을 높이는 데 핵심적인 역할

import torch    # 텐서 연산, 자동 미분 등을 제공하여 딥러닝 모델을 구현
from torch import nn    # nn : 신경망 계층과 관련된 클래스
from torch.nn import Conv1d, Conv2d     #1차원, 2차원 합성곱 계층
from torch.nn import functional as F    #다양한 함수형 API(활성화 함수, 손실 함수 등) 제공
from torch.nn.utils import spectral_norm, weight_norm   #모델 파라미터에 norm 정규화를 적용 - 학습 안정성이나 일반화 성능을 향상

LRELU_SLOPE = 0.1   #활성화 함수에서 음수 입력 구간의 기울기


#주어진 모듈 m이 합성곱 계층(“Conv”)이라면, 가중치를 정규분포(평균 mean, 표준편차 std)로 초기화
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

#padding : 합성곱 연산이나 주기 단위 변환을 할 때 필요한 공간을 확보하기 위해 입력의 앞뒤를 채워 넣는 작업
#합성곱 연산 시 ‘패딩’ 크기를 자동으로 계산 : (커널 크기 × 확장 - 확장) / 2
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


#단순 1D Conv(1차원 합성곱 계층들을 여러 개 쌓아 만든 판별기) 형태
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm   
        #use_spectral_norm가 True면 spectral_norm, False면 weight_norm을 사용해 각 합성곱 계층에 정규화를 적용
        self.convs = nn.ModuleList(     #1차원 Conv 계층들을 차례대로 넣음
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]       #groups: 학습 파라미터 수나 특성 추출 방식을 조절
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))   #마지막 출력을 1채널로 줄이는 후처리용 Conv 계층

#실제 순전파 로직을 정의
    def forward(self, x):
        fmap = []       #각 합성곱 계층의 출력을 저장(특징맵을 나중에 feature loss 등에 활용)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)          #모든 conv 통과 후 마지막 conv_post를 거쳐 1채널의 결과를 얻음
        x = torch.flatten(x, 1, -1)     #배치 차원/채널 차원을 제외하고 펼침

        return x, fmap      #(최종 출력, 중간 feature map들의 리스트) 형태로 반환


#2차원 합성곱 형태, 주기성 판별 전용
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period        #오디오 시퀀스를 나눌 길이
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))     #1채널로 출력

    def forward(self, x):
        fmap = []

        #(배치, 채널, 시간길이) 형태의 1D 텐서 받아옴
        # 1d to 2d
        b, c, t = x.shape
        #시간길이가 period로 나누어떨어지지 않으면), reflect 패딩을 수행해 길이를 맞춤
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)     #(배치, 채널, 시간길이/period, period) 형태로 재배열 -> 2D 형태

#self.convs에 등록된 2D Convolution 레이어를 순차적으로 거쳐, Leaky ReLU 활성화를 적용
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)      # #각 단계 출력을 fmap에 저장
        x = self.conv_post(x)       #conv_post로 채널 수를 1로 줄임임
        fmap.append(x)
        x = torch.flatten(x, 1, -1)     #최종 판별결과를 펼침침

        return x, fmap      #(결과, 모든 특성맵 목록)을 반환


#여러 개의 판별기를 한꺼번에 모은 클래스
#하나의 오디오 입력에 대해 다양한 주기를 가진 판별기가 각각 진짜인지 판별
class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, periods=[2, 3, 5, 7, 11, 17, 23, 37]):
        super(MultiPeriodDiscriminator, self).__init__()

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]       #1D 방식
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

#실제값(진짜 오디오) y와 모델이 생성한 오디오 y_hat을 각각 모든 내부 판별기에 통과
    def forward(self, y, y_hat):
        y_d_rs = []     #출력
        y_d_gs = []     #출력
        fmap_rs = []        #특징맵
        fmap_gs = []        #특징맵
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            # for j in range(len(fmap_r)):
            #     print(i,j,y.shape,y_hat.shape,fmap_r[j].shape,fmap_g[j].shape)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    #y_d_rs / y_d_gs : 모든 판별기의 최종 판별 결과(실제 vs. 생성)
    #fmap_rs / fmap_gs : 각 판별기의 중간 feature map들


#periods에 23과 37을 추가, 내부 로직은 MultiPeriodDiscriminator와 동일
class MultiPeriodDiscriminatorV2(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminatorV2, self).__init__()
        # periods = [2, 3, 5, 7, 11, 17]
        periods = [2, 3, 5, 7, 11, 17, 23, 37]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []  #
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            # for j in range(len(fmap_r)):
            #     print(i,j,y.shape,y_hat.shape,fmap_r[j].shape,fmap_g[j].shape)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


#판별기 손실 계산 함수
#disc_real_outputs : 실제 오디오에 대한 판별 결과들
#disc_generated_outputs : 생성 오디오에 대한 판별 결과들
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
#진짜는 1에 가깝게(즉 (1 - dr)²가 작아지도록), 가짜는 0에 가깝게(즉 dg²가 작아지도록) 만드는 MSE를 사용

    return loss, r_losses, g_losses


#특성맵 손실을 계산 - 생성된 신호가 실제 신호의 내부적 구조와 비슷해지도록 학습을 유도
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))
#판별기의 중간 레이어 출력(실제: rl, 생성: gl)을 비교해 평균냄

    return loss


#생성기 손실 함수
def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)       #판별기의 출력 dg를 1에 가깝게
        gen_losses.append(l)
        loss += l

    return loss, gen_losses     #여러 판별기의 출력합 최종 loss, 개별 값 목록 반환


#VAE 모델에서 “정규분포” 가정 하에 잠재 변수를 학습할 때 사용
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    #하위 4개 : 잠재 변수의 평균·분산(로그 형태) 정보
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()     #유효한 위치(패딩이 아닌 영역 등)를 가려내기 위한 mask : 1로 표시된 부분에 대해서만 KL 손실을 합산

    kl = logs_p - logs_q - 0.5      #KL divergence 공식
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
