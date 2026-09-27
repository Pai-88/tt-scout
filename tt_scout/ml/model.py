"""BallNet: three stacked frames in, a heatmap of where the ball is in the last frame out. Plus its loss and peak decoding.

Design, in the TrackNet family (Huang et al. 2019; TrackNetV2/V3): a small U-Net. The encoder halves the resolution four
times so a layer can see ~150 px of context (enough to tell a ball in flight from a shoe or a white logo); the skip connections
carry the full-resolution detail back, because a ball is only ~3 px across at 640x352. Motion is not computed by hand: with
three frames stacked as 9 channels the first layers learn their own frame differences.

Loss: the penalty-reduced focal loss of CenterNet (Zhou et al. 2019). Only the peak pixel is a positive; pixels near it are
negatives whose penalty shrinks with (1 - target)^4, so a prediction one pixel off costs little. Nearly every pixel is empty
background, and the focal term (1 - p)^2 / p^2 stops those easy pixels from swamping the few that matter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class BallNet(nn.Module):
    def __init__(self, frames=3, width=(16, 32, 64, 128, 192)):
        super().__init__()
        c = width
        self.config = dict(frames=frames, width=list(width))
        self.e1, self.e2, self.e3, self.e4 = block(3 * frames, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])
        self.mid = block(c[3], c[4])
        self.d4, self.d3, self.d2, self.d1 = block(c[4] + c[3], c[3]), block(c[3] + c[2], c[2]), block(c[2] + c[1], c[1]), block(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)
        nn.init.constant_(self.head.bias, -4.6)               # start by predicting "no ball" everywhere (p = 0.01)

    @staticmethod
    def up(x, like):
        return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        m = self.mid(F.max_pool2d(e4, 2))
        d = self.d4(torch.cat([self.up(m, e4), e4], 1))
        d = self.d3(torch.cat([self.up(d, e3), e3], 1))
        d = self.d2(torch.cat([self.up(d, e2), e2], 1))
        d = self.d1(torch.cat([self.up(d, e1), e1], 1))
        return self.head(d)                                   # logits, (B, 1, H, W)


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet focal loss. target: Gaussian heatmaps with the peak pixel exactly 1 (all zeros where the ball is hidden)."""
    logp, log1mp = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = torch.sigmoid(logits)
    pos = target.eq(1.0).float()
    pos_loss = -(logp * (1 - p) ** alpha * pos).sum()
    neg_loss = -(log1mp * p ** alpha * (1 - target) ** beta * (1 - pos)).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1.0)


def peaks(logits, k=3, thresh=0.3, window=7):
    """Up to k local maxima per image above `thresh`: list over the batch of [(x, y, score)] in heatmap pixels. Sub-pixel
    position per axis from a parabola through log p at the peak and its two neighbours: for a Gaussian blob log p is exactly a
    parabola, so this recovers the centre (a 3x3 probability centroid would pull it most of the way back to the integer pixel)."""
    p = torch.sigmoid(logits)
    keep = (p >= F.max_pool2d(p, window, stride=1, padding=window // 2)) & (p >= thresh)
    B, _, H, W = p.shape
    scores, idx = (p * keep).view(B, -1).topk(min(k, H * W), dim=1)
    lp = torch.log(p.clamp(min=1e-6)).cpu()
    scores, idx = scores.cpu().tolist(), idx.cpu().tolist()

    def vertex(a, b, c):
        den = a - 2 * b + c
        return 0.0 if den >= 0 else float(max(-0.5, min(0.5, 0.5 * (a - c) / den)))

    out = []
    for bi in range(B):
        found = []
        for sc, i in zip(scores[bi], idx[bi]):
            if sc < thresh:
                break
            y, x = divmod(i, W)
            row, col = lp[bi, 0, y], lp[bi, 0, :, x]
            dx = vertex(float(row[x - 1]), float(row[x]), float(row[x + 1])) if 0 < x < W - 1 else 0.0
            dy = vertex(float(col[y - 1]), float(col[y]), float(col[y + 1])) if 0 < y < H - 1 else 0.0
            found.append((x + dx, y + dy, sc))
        out.append(found)
    return out
