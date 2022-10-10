import torch


class FGM:
    def __init__(self, model):
        """
        Usage:
         fgm = FGM(model)
         for batch_input, batch_label in data:
               loss = model(batch_input, batch_label)
               loss.backward()

               # adversarial training
               fgm.attack()
               loss_adv = model(batch_input, batch_label)
               loss_adv.backward()
               fgm.restore()

               optimizer.step()
               model.zero_grad()
        :param model:
        """
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}


class EMA:

    def __init__(self, model, decay):
        """
        Usage:
        ema = EMA(model, 0.999)
        ema.register()

        # update shadow weights
        def train():
            optimizer.step()
            ema.update()

        # apply shadow weights；eval
        def evaluate():
            ema.apply_shadow()
            # evaluate
            ema.restore()
        :param model:
        :param decay:
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AWP:
    def __init__(
            self,
            model,
            cal_loss,
            adv_param="weight",
            adv_lr=1,
            adv_eps=0.2,
            start_epoch=0,
            adv_step=1,
            device="cpu",
    ):
        self.model = model
        self.cal_loss = cal_loss
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.device = device

    def attack_backward(self, batch, epoch=0):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        loss = 0
        self.save()
        for i in range(self.adv_step):
            self.attack_step()
            loss += self.cal_loss(model=self.model, batch=batch, device=self.device)
            self.model.zero_grad()

        self.restore()

        return loss

    def attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self, ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
