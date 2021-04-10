import torch

def SIoU_Plus_Plus_3D(gz, gy, gx, gr, pz, py, px, pr, eps=1e-4):
    dist = torch.sqrt((gz-pz)**2 + (gy-py)**2 + (gx-px)**2+eps)
    if gr + pr >= dist:
        if gr + dist < pr:
            pr = torch.clamp(pr, min=1e-8)
            siou = (gr / pr) ** 3
        elif pr + dist < gr:
            gr = torch.clamp(gr, min=1e-8)
            siou = (pr / gr) ** 3
        else:
            cos1 = (gr ** 2 + dist ** 2 - pr ** 2) / (2 * gr * dist + eps)
            h1 = gr * (1 - cos1)
            v1 = 3.1415926 * gr * h1 ** 2 - 3.1415926 * h1 ** 3 / 3
            cos2 = (pr ** 2 + dist ** 2 - gr ** 2) / (2 * pr * dist + eps)
            h2 = pr * (1 - cos2)
            v2 = 3.1415926 * pr * h2 ** 2 - 3.1415926 * h2 ** 3 / 3
            ua = (3.1415926 * 4 * (pr ** 3 + gr ** 3) / 3) - (v1 + v2)
            ua = torch.clamp(ua, min=1e-8)
            eta = torch.acos((gr ** 2 - dist ** 2 + pr ** 2) / (2 * gr * pr + eps)) / 3.1415926
            siou = (v1 + v2) / ua - eta
    else:
        siou = torch.tensor(0).float()
        siou.requires_grad=True
    dist_ratio = dist / (dist + pr + gr)
    sdiou = siou - dist_ratio
    sdiou_loss = 1.0 - sdiou
    return sdiou_loss
