import torch
import torch.nn.functional as F

def _kl_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)

    logsoftmax_1 = torch.log_softmax(prob1.clamp(min=1e-8))
    logsoftmax_2 = torch.log_softmax(prob2.clamp(min=1e-8))
    kl = F.kl_div(logsoftmax_1, logsoftmax_2, reduction='batchmean')
    return kl

def _jensen_shannon_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
    return jsd * 0.5

