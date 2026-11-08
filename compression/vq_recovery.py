"""CPU training primitives from the two independent recovery runs."""
import numpy as np
import torch
import torch.nn.functional as F

class VQBookEntry:
    """Trainable codebook (with FP16 STE) + frozen codes and frozen g128 scales."""

    def __init__(self, name, book, codes, scales, d):
        self.name = name
        self.d = d
        self.rows, self.groups = scales.shape
        self.nin = self.groups * 128
        b_init = torch.from_numpy(book.astype(np.float16).astype(np.float32).copy())
        self.book0 = b_init.clone()
        self.book = torch.nn.Parameter(b_init.clone())
        self.codes_flat = torch.from_numpy(codes.astype(np.int64).reshape(-1))
        self.scales_3d = torch.from_numpy(scales.astype(np.float32).copy()) \
            .reshape(self.rows, self.groups, 1)

    def make_leaf_weight(self):
        with torch.no_grad():
            b_ste = self.book.detach().half().float()
            w = (b_ste[self.codes_flat].reshape(self.rows, self.groups, 128)
                 * self.scales_3d).reshape(self.rows, self.nin)
        return w.requires_grad_(True)

    def accumulate_book_grad(self, w_leaf, l2_anchor=0.0):
        g = w_leaf.grad
        grad_rec = (g.reshape(self.rows, self.groups, 128)
                    * self.scales_3d).reshape(-1, self.d)
        bg = torch.zeros_like(self.book)
        bg.index_add_(0, self.codes_flat, grad_rec)
        if l2_anchor > 0.0:
            bg.add_(self.book.detach() - self.book0, alpha=l2_anchor)
        self.book.grad = bg
        w_leaf.grad = None

    def detach_weight(self):
        with torch.no_grad():
            b_f16 = self.book.detach().half().float()
            w = (b_f16[self.codes_flat].reshape(self.rows, self.groups, 128)
                 * self.scales_3d).reshape(self.rows, self.nin)
            return w.half().float()
class VQBilinearEntry:
    """FP32 master codebook + clamped g128 scales with fast leaf-gradient extraction."""

    def __init__(self, name, book, codes, scales, d, clamp_rel=0.10):
        self.name = name
        self.d = d
        self.rows, self.groups = scales.shape
        self.nin = self.groups * 128
        b_init = torch.from_numpy(book.astype(np.float32).copy())
        self.book = torch.nn.Parameter(b_init)
        s_init = torch.from_numpy(scales.astype(np.float32).copy())
        self.scales = torch.nn.Parameter(s_init.clone())
        self.s_min = s_init * (1.0 - clamp_rel)
        self.s_max = s_init * (1.0 + clamp_rel)
        self.codes_flat = torch.from_numpy(codes.astype(np.int64).reshape(-1))

    def make_leaf_weight(self):
        with torch.no_grad():
            b = self.book.detach()
            s_ste = self.scales.detach().half().float().unsqueeze(2)
            w = (b[self.codes_flat].reshape(self.rows, self.groups, 128)
                 * s_ste).reshape(self.rows, self.nin)
        return w.requires_grad_(True)

    def accumulate_grads(self, w_leaf):
        with torch.no_grad():
            g3 = w_leaf.grad.reshape(self.rows, self.groups, 128)
            b = self.book.detach()
            rec3 = b[self.codes_flat].reshape(self.rows, self.groups, 128)
            self.scales.grad = (g3 * rec3).sum(dim=2)
            del rec3
            s_ste = self.scales.detach().half().float().unsqueeze(2)
            grad_rec = (g3 * s_ste).reshape(-1, self.d)
            bg = torch.zeros_like(self.book)
            bg.index_add_(0, self.codes_flat, grad_rec)
            self.book.grad = bg
            w_leaf.grad = None

    def clamp_scales(self):
        with torch.no_grad():
            self.scales.data.clamp_(self.s_min, self.s_max)

    def detach_weight(self):
        with torch.no_grad():
            b_f16 = self.book.detach().half().float()
            s_f16 = self.scales.detach().half().float().unsqueeze(2)
            w = (b_f16[self.codes_flat].reshape(self.rows, self.groups, 128)
                 * s_f16).reshape(self.rows, self.nin)
            return w.half().float()

def recovery_loss(model, row, teacher_cache, training):
    ids = torch.tensor([row['ids']], dtype=torch.long)
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    places = torch.tensor(row['positions'], dtype=torch.long)
    logits = F.linear(hidden[places], model.lm_head.weight)
    targets = torch.tensor(row['targets'], dtype=torch.long)
    ce = F.cross_entropy(logits, targets)
    loss = ce
    is_terminal_agent = (row.get('kind') == 'agent' and row['targets'][0] != 151657)
    is_short_en = (training.get('short_english_terminal_weight', False)
                   and row.get('kind') == 'language'
                   and 'teacher_key' not in row
                   and len(row['targets']) <= 5)
    if is_terminal_agent or is_short_en:
        tw = float(training.get('terminal_turn_weight', 1.0))
        fg = float(training.get('terminal_first_token_gamma', 1.0))
        per_tok = F.cross_entropy(logits, targets, reduction='none')
        weights = torch.ones_like(per_tok)
        weights[0] = fg
        weights = weights / weights.sum()
        loss = tw * (per_tok * weights).sum()
    kl = None
    if 'teacher_key' in row:
        pos = torch.tensor(row['teacher_positions'], dtype=torch.long)
        actual = F.linear(hidden[pos], model.lm_head.weight).log_softmax(-1)
        target = torch.from_numpy(
            teacher_cache[row['teacher_key']]).log_softmax(-1)
        kl = (target.exp() * (target - actual)).sum(-1).mean()
        loss = loss + training['language_KL_weight'] * kl
    return loss, ce, kl
