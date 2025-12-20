import torch
import torch.nn as nn
import torch.nn.functional as F

class BellLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_p = torch.pow((y - p), 2)
        y_p_div = -1.0 * torch.div(y_p, 162.0)
        exp_y_p = torch.exp(y_p_div)
        loss = 300 * (1.0 - exp_y_p)
        return torch.mean(loss)


class LogCosh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(p - y))
        return torch.mean(loss)


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(p, y))


class GL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.sigma = sigma

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gl = self.eps / (self.lam ** 2) * (1 - torch.exp(-1 * ((y - p) ** 2) / (self.sigma ** 2)))
        return gl.mean()


class RMBell(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y)


class RMLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.logcosh(p, y)


class RMGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.rmse = RMSE()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.gl(p, y)


class RMBellLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y) + self.logcosh(p, y)


class RMBellGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y) + self.gl(p, y)


class BellLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.bell = BellLoss()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.logcosh(p, y)


class BellGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.bell = BellLoss()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.gl(p, y)


class BellLCoshGL(nn.Module):
    def __init__(self):
        super().__init__()
        self.bell = BellLoss()
        self.logcosh = LogCosh()
        self.gl = GL()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.logcosh(p, y) + self.gl(p, y)


class LogCoshGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.logcosh = LogCosh()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.logcosh(p, y) + self.gl(p, y)

class MAELoss(nn.Module):
    """Mean Absolute Error loss"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x - y))

class MSELoss(nn.Module):
    """Mean Squared Error loss"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.pow(x - y, 2))

class CCCLoss(nn.Module):
    """Lin's Concordance Correlation Coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Measures the agreement between two variables

    It is a product of
    - precision (pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)

    Interpretation
    - rho =  1: perfect agreement
    - rho =  0: no agreement
    - rho = -1: perfect disagreement

    Args:
        eps (float, optional): Avoiding division by zero. Defaults to 1e-8.

    original: https://github.com/DresvyanskiyDenis/ABAW_2024/blob/main/src/audio/loss/loss.py
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes CCC loss

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: 1 - CCC loss value
        """
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        weight_emotion: float = 1.0,
        weight_personality: float = 1.0,
        weight_ah: float = 1.0,
        emo_weights=None,
        ah_weights=None,
        personality_loss_type: str = "ccc",     # см. список ниже
        eps: float = 1e-8,
        lam_gl: float = 1.0,
        eps_gl: float = 600,
        sigma_gl: float = 8,
    ):
        super().__init__()
        self.weight_emotion = weight_emotion
        self.weight_personality = weight_personality
        self.weight_ah = weight_ah

        # Эмоции — всегда CrossEntropy (в этой версии)
        self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)

        # AH — бинарная single-label классификация (2 класса) через CE
        self.ah_loss = nn.CrossEntropyLoss(weight=ah_weights)

        # Персональные качества — выбираем по имени
        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(),
            "mse": MSELoss(),
            "bell": BellLoss(),
            "logcosh": LogCosh(),
            "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(),
            "rmse_bell": RMBell(),
            "rmse_logcosh": RMLCosh(),
            "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(),
            "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(),
            "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(),
            "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }
        if personality_loss_type not in loss_types:
            raise ValueError(
                f"Unknown personality_loss_type: {personality_loss_type}. Available: {list(loss_types.keys())}"
            )
        self.personality_loss = loss_types[personality_loss_type]
        self.personality_loss_type = personality_loss_type

    def forward(self, outputs, labels):
        loss = 0.0

        # Эмоции (классификация)
        if 'emotion_logits' in outputs and 'emotion' in labels:
            true_emotion = labels['emotion']              # (B,)
            pred_emotion = outputs['emotion_logits']      # (B, C_emo)
            loss += self.weight_emotion * self.emotion_loss(pred_emotion, true_emotion)

        # Персональные качества (регрессия по 5 признакам)
        if 'personality_scores' in outputs and 'personality' in labels:
            true_personality = labels['personality']          # (B, 5)
            pred_personality = outputs['personality_scores']  # (B, 5)

            if self.personality_loss_type == "ccc":
                loss_per = 0.0
                for i in range(5):  # по каждому из 5 признаков
                    loss_per += self.personality_loss(true_personality[:, i], pred_personality[:, i])
                loss += (loss_per) * self.weight_personality
            else:
                loss += self.weight_personality * self.personality_loss(true_personality, pred_personality)

        # Третья задача: AH (absence_full / presence_full)
        if 'ah_logits' in outputs and 'ah' in labels:
            true_ah = labels['ah']           # (B,) int64 с классами 0/1
            pred_ah = outputs['ah_logits']   # (B, 2)
            loss += self.weight_ah * self.ah_loss(pred_ah, true_ah)

        return loss

def binarize_with_nan(x, threshold=0.5):
    # Создаем маску NaN
    nan_mask = torch.isnan(x)

    # Бинаризуем (не затрагивая NaN)
    binary = torch.zeros_like(x)
    binary[x > threshold] = 1.0

    # Восстанавливаем NaN там, где они были
    binary[nan_mask] = float('nan')

    return binary


class MultiTaskLossWithNaN(nn.Module):
    def __init__(
        self,
        weight_emotion: float = 1.0,
        weight_personality: float = 1.0,
        weight_ah: float = 1.0,                 # NEW
        emo_weights=None,
        ah_weights=None,                        # NEW
        personality_loss_type: str = "ccc",
        emotion_loss_type: str = 'BCE',         # как у тебя: CE/BCE
        eps: float = 1e-8,
        lam_gl: float = 1.0,
        eps_gl: float = 600,
        sigma_gl: float = 8,
    ):
        super().__init__()
        self.weight_emotion = weight_emotion
        self.weight_personality = weight_personality
        self.weight_ah = weight_ah

        # Эмоции — CE или BCE (оставляю твою логику)
        if emotion_loss_type == 'CE':
            self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)
            self.emotion_loss_type = emotion_loss_type
        elif emotion_loss_type == 'BCE':
            self.emotion_loss = nn.BCEWithLogitsLoss(weight=emo_weights)
            self.emotion_loss_type = emotion_loss_type
        else:
            raise ValueError(f"Unknown emotion_loss_type: {emotion_loss_type}")

        # AH — всегда CE (single-label 2-класса)
        self.ah_loss = nn.CrossEntropyLoss(weight=ah_weights)  # NEW

        # Персональные качества — выбираем по имени
        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(),
            "mse": MSELoss(),
            "bell": BellLoss(),
            "logcosh": LogCosh(),
            "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(),
            "rmse_bell": RMBell(),
            "rmse_logcosh": RMLCosh(),
            "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(),
            "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(),
            "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(),
            "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }
        if personality_loss_type not in loss_types:
            raise ValueError(
                f"Unknown personality_loss_type: {personality_loss_type}. Available: {list(loss_types.keys())}"
            )
        self.personality_loss = loss_types[personality_loss_type]
        self.personality_loss_type = personality_loss_type

    def forward(self, outputs, labels):
        loss = 0.0

        # EMOTION branch (с маской)
        emo_mask = labels.get('valid_emo', None)
        pred_emotion_all = outputs.get('emotion_logits')
        if pred_emotion_all is not None:
            if emo_mask is None:
                true_emotion = labels['emotion']
                pred_emotion = pred_emotion_all
                valid_any = True
            else:
                valid_any = emo_mask.any()
                if valid_any:
                    true_emotion = labels['emotion'][emo_mask]
                    pred_emotion = pred_emotion_all[emo_mask]

            if pred_emotion_all is not None and (emo_mask is None or valid_any):
                if self.emotion_loss_type == 'BCE':
                    true_emotion = binarize_with_nan(true_emotion, threshold=0)
                loss += self.weight_emotion * self.emotion_loss(pred_emotion, true_emotion)

        # PERSONALITY branch (с маской/NaN по каждому признаку)
        per_mask = labels.get('valid_per', None)
        pred_personality_all = outputs.get('personality_scores')
        if pred_personality_all is not None:
            if per_mask is None:
                true_personality = labels['personality']
                pred_personality = pred_personality_all
                per_valid_any = True
            else:
                per_valid_any = per_mask.any()
                if per_valid_any:
                    true_personality = labels['personality'][per_mask]
                    pred_personality = pred_personality_all[per_mask]

            if pred_personality_all is not None and (per_mask is None or per_valid_any):
                if self.personality_loss_type == "ccc":
                    loss_per = 0.0
                    valid_traits = 0
                    for i in range(5):
                        trait_mask = ~torch.isnan(true_personality[:, i])
                        if trait_mask.any():
                            loss_per += self.personality_loss(
                                true_personality[trait_mask, i],
                                pred_personality[trait_mask, i]
                            )
                            valid_traits += 1
                    if valid_traits > 0:
                        loss += (loss_per / valid_traits) * self.weight_personality
                else:
                    loss += self.weight_personality * self.personality_loss(
                        true_personality, pred_personality
                    )

        # AH branch (с поддержкой маски valid_ah, если она есть)
        ah_logits_all = outputs.get('ah_logits')           # (B, 2)
        if ah_logits_all is not None:
            ah_mask = labels.get('valid_ah', None)
            if ah_mask is None:
                # ⚠️ Если сюда зайдём без маски, а в y есть NaN — CE упадёт.
                # В идеале mask всегда передавать из train; здесь подстрахуемся:
                true_ah = labels['ah']
                if true_ah.dtype != torch.long:
                    # заменим NaN на -1, заигнорим их маской
                    nan_mask = torch.isnan(true_ah) if true_ah.dtype.is_floating_point else torch.zeros_like(true_ah, dtype=torch.bool)
                    if nan_mask.any():
                        ah_mask = ~nan_mask
                        true_ah = true_ah[ah_mask]
                        pred_ah = ah_logits_all[ah_mask]
                    else:
                        pred_ah = ah_logits_all
                    true_ah = true_ah.long()
                    ah_valid_any = true_ah.numel() > 0
                else:
                    pred_ah = ah_logits_all
                    ah_valid_any = True
            else:
                ah_valid_any = ah_mask.any()
                if ah_valid_any:
                    true_ah = labels['ah'][ah_mask]
                    if true_ah.dtype != torch.long:
                        true_ah = true_ah.long()
                    pred_ah = ah_logits_all[ah_mask]

            if ah_valid_any:
                loss += self.weight_ah * self.ah_loss(pred_ah, true_ah)

        if not isinstance(loss, torch.Tensor):
            device = (
                (outputs.get("emotion_logits") or outputs.get("personality_scores") or outputs.get("ah_logits")).device
                if (outputs.get("emotion_logits") is not None
                    or outputs.get("personality_scores") is not None
                    or outputs.get("ah_logits") is not None)
                else torch.device("cpu")
            )
            loss = torch.tensor(0.0, requires_grad=True, device=device)

        return loss

def prototype_contrastive_loss(embeddings, labels, prototypes, num_classes, temperature=0.1):
    """
    embeddings: [B, D]
    labels: [B]
    prototypes: [P, D] = [num_classes * n_proto, D]
    """
    device = embeddings.device
    B, D = embeddings.shape
    P = prototypes.shape[0]
    n_proto = P // num_classes

    # Нормализуем
    emb_norm = F.normalize(embeddings, dim=1)           # [B, D]
    proto_norm = F.normalize(prototypes, dim=1)         # [P, D]

    # Cosine similarity matrix: [B, P]
    sim = torch.matmul(emb_norm, proto_norm.t()) / temperature  # [B, P]

    # Маска положительных пар: [B, P]
    # Для каждого батч-элемента — прототипы его класса
    labels_exp = labels.unsqueeze(1).expand(-1, n_proto)  # [B, n_proto]
    proto_indices = torch.arange(n_proto, device=device).unsqueeze(0)  # [1, n_proto]
    pos_proto_ids = labels_exp * n_proto + proto_indices  # [B, n_proto] — индексы прототипов класса y_i
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)    # [B, P]
    pos_mask.scatter_(1, pos_proto_ids, True)             # отмечаем позитивы

    # Числитель: сумма exp(sim) по позитивным прототипам
    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask).sum(dim=1)             # [B]

    # Знаменатель: сумма exp(sim) по всем прототипам
    all_sum = exp_sim.sum(dim=1)                          # [B]

    # Loss per sample
    loss = -torch.log(pos_sum / all_sum + 1e-8)           # [B]
    return loss.mean()

def prototype_contrastive_loss_2(embeddings, labels, prototypes, num_classes, temperature=0.1):
    device = embeddings.device
    B, D = embeddings.shape
    P = prototypes.shape[0]
    n_proto = P // num_classes

    emb_norm = F.normalize(embeddings, dim=1)         # [B,D]
    proto_norm = F.normalize(prototypes, dim=1)       # [P,D]
    sim = torch.matmul(emb_norm, proto_norm.t()) / temperature  # [B,P]

    # ids прототипов своего класса: [B, n_proto]
    proto_indices = torch.arange(n_proto, device=device).unsqueeze(0)  # [1,n_proto]
    class_proto_ids = labels.unsqueeze(1) * n_proto + proto_indices    # [B,n_proto]

    # выбираем лучший прототип своего класса: [B]
    sim_in_class = sim.gather(1, class_proto_ids)          # [B,n_proto]
    best_in_class = sim_in_class.argmax(dim=1)             # [B]
    pos_ids = class_proto_ids[torch.arange(B, device=device), best_in_class]  # [B]

    # pos_mask: только один прототип-позитив
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    pos_mask.scatter_(1, pos_ids.unsqueeze(1), True)

    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask).sum(dim=1)      # [B]
    all_sum = exp_sim.sum(dim=1)                   # [B]
    loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
    return loss.mean()
