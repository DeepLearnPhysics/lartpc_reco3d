import torch
import torch.nn as nn
import torch.nn.functional as F

from mlreco.models.layers.cluster_cnn.losses.lovasz import lovasz_softmax_flat

class ClassificationLoss(nn.Module):
    """
    Abstract base class for all classification loss functions.
    """

    def __init__(self, regularization_weight=1.0, ignore_index=-1, **kwargs):
        super(ClassificationLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss(reduction='mean')
        self.w = regularization_weight
        if self.w < 0 or self.w > 1:
            raise ValueError('Regularization weight must be in [0, 1]')
        self.ignore_index = ignore_index
        self.eps = kwargs.get('eps', 1e-6) # Numerical stability constant
        
    def ignore_classes(self, logits, labels):
        """
        Ignore certain classes in the loss computation.
        """
        mask = labels != self.ignore_index
        x = logits[mask]
        y = labels[mask]
        return x, y
        
    def regularization(self, logits, labels):
        """
        Regularization function for the classification loss.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError('Regularization not implemented for ClassificationLoss')
    
    def forward(self, logits, labels):
        """
        Forward pass of the focal loss.
        
        Inputs
        ------
            logits: torch.Tensor
                Raw logits from the network.
            labels: torch.Tensor
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor
                Focal loss for the input logits and labels.
        """
        x, y = self.ignore_classes(logits, labels)
        reg_loss = self.regularization(x, y)
        ce_loss = self.xentropy(x, y)
        
        loss = self.w * reg_loss + (1 - self.w) * ce_loss
        
        output = {
            'loss': loss,
            'reg_loss': self.w * float(reg_loss),
            'ce_loss': (1 - self.w) * float(ce_loss)
        }
        
        return output
    
    
class CrossEntropyLoss(ClassificationLoss):
    """
    Cross entropy loss for classification tasks.
    """
    
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)
        self.w = 1.0
        
    def regularization(self, logits, labels):
        """
        Regularization function for the cross entropy loss.
        
        Inputs
        ------
            logits: torch.Tensor (N, C)
                Raw logits from the network.
            labels: torch.Tensor (N, )
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor (float)
                Cross entropy loss for the input logits and labels.
        """
        return 0.0
    
    
class FocalLoss(ClassificationLoss):
    """
    Focal loss for classification tasks.
    """
    
    def __init__(self, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = kwargs['gamma']
        # self.alpha = kwargs['alpha']
        self.ignore_index = kwargs.get('ignore_index', -1)
        
        self.nll_loss = nn.NLLLoss(reduction='none',
                                #    weight=self.alpha, 
                                   ignore_index=self.ignore_index)
    
    def regularization(self, logits, labels):
        """
        Forward pass of the focal loss.
        
        Inputs
        ------
            logits: torch.Tensor
                Raw logits from the network.
            labels: torch.Tensor
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor
                Focal loss for the input logits and labels.
        """
        log_p = F.log_softmax(logits, dim=1)
        ce = self.nll_loss(log_p, labels)
        
        all_rows = torch.arange(len(logits))
        log_pt = log_p[all_rows, labels]
        
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma
        
        loss = focal_term * ce
        
        return loss.mean()
    
    # def forward(self, logits, labels):
    #     """
    #     Forward pass of the focal loss.
        
    #     Inputs
    #     ------
    #         logits: torch.Tensor
    #             Raw logits from the network.
    #         labels: torch.Tensor
    #             True labels for the input logits.
                
    #     Returns
    #     -------
    #         loss: torch.Tensor
    #             Focal loss for the input logits and labels.
    #     """
    #     x, y = self.ignore_classes(logits, labels)
    #     loss = self.regularization(x, y)
        
    #     return loss
    
    
class DiceLoss(ClassificationLoss):
    """
    Dice loss for classification tasks.
    """
    
    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        
    def regularization(self, logits, labels):
        """
        Regularization function for the dice loss.
        
        Inputs
        ------
            logits: torch.Tensor (N, C)
                Raw logits from the network.
            labels: torch.Tensor (N, )
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor (float)
                Dice loss for the input logits and labels.
        """
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        
        y = F.one_hot(labels, num_classes=logits.shape[1]).float()
        dice = (2 * (p * y).sum(dim=0) + self.eps) / (p.sum(dim=0) + y.sum(dim=0) + self.eps)
        dice = dice.mean()
        loss = 1.0 - dice
        
        return loss
    
    
class LogCoshDiceLoss(ClassificationLoss):
    """
    LogCoshDice loss for classification tasks.
    """
    
    def __init__(self, **kwargs):
        super(LogCoshDiceLoss, self).__init__(**kwargs)
        
    def regularization(self, logits, labels):
        """
        Regularization function for the logcosh dice loss.
        
        Inputs
        ------
            logits: torch.Tensor (N, C)
                Raw logits from the network.
            labels: torch.Tensor (N, )
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor (float)
                Logcosh dice loss for the input logits and labels.
        """
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        
        y = F.one_hot(labels, num_classes=logits.shape[1]).float()
        dice = (2 * (p * y).sum(dim=0) + self.eps) / (p.sum(dim=0) + y.sum(dim=0) + self.eps)
        dice = dice.mean()
        loss = torch.log(torch.cosh(1.0 - dice))
        
        return loss
    
    
class JaccardLoss(ClassificationLoss):
    """
    Jaccard loss for classification tasks.
    """
    
    def __init__(self, **kwargs):
        super(JaccardLoss, self).__init__(**kwargs)
        
    def regularization(self, logits, labels):
        """
        Regularization function for the jaccard loss.
        
        Inputs
        ------
            logits: torch.Tensor (N, C)
                Raw logits from the network.
            labels: torch.Tensor (N, )
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor (float)
                Jaccard loss for the input logits and labels.
        """
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        
        y = F.one_hot(labels, num_classes=logits.shape[1]).float()
        intersection = (p * y).sum(dim=0)
        union = p.sum(dim=0) + y.sum(dim=0) - intersection
        jaccard = (intersection + self.eps) / (union + self.eps)
        jaccard = jaccard.mean()
        loss = 1.0 - jaccard
        
        return loss
    
    
class LovaszSoftmaxLoss(ClassificationLoss):
    """
    Lovasz softmax loss for classification tasks.
    """
    
    def __init__(self, **kwargs):
        super(LovaszSoftmaxLoss, self).__init__(**kwargs)
        
    def regularization(self, logits, labels):
        """
        Regularization function for the lovasz softmax loss.
        
        Inputs
        ------
            logits: torch.Tensor (N, C)
                Raw logits from the network.
            labels: torch.Tensor (N, )
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor (float)
                Lovasz softmax loss for the input logits and labels.
        """
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        loss = lovasz_softmax_flat(p, labels)
        
        return loss
    
    
class TverskyLoss(ClassificationLoss):
    """
    Tversky loss for classification tasks.
    """
    
    def __init__(self, **kwargs):
        super(TverskyLoss, self).__init__(**kwargs)
        self.a = kwargs['fp_weight']
        self.b = kwargs['fn_weight']
        
    def regularization(self, logits, labels):
        """
        Regularization function for the tversky loss.
        
        Inputs
        ------
            logits: torch.Tensor (N, C)
                Raw logits from the network.
            labels: torch.Tensor (N, )
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor (float)
                Tversky loss for the input logits and labels.
        """
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        
        y = F.one_hot(labels, num_classes=logits.shape[1]).float()
        tp = (p * y).sum(dim=0)
        fp = (p * (1 - y)).sum(dim=0)
        fn = ((1 - p) * y).sum(dim=0)
        tversky = (tp + self.eps) / (tp + self.a * fp + self.b * fn + self.eps)
        tversky = tversky.mean()
        loss = 1.0 - tversky
        
        return loss
    
    
class FocalTverskyLoss(ClassificationLoss):
    """
    Focal tversky loss for classification tasks.
    """
    
    def __init__(self, **kwargs):
        super(FocalTverskyLoss, self).__init__(**kwargs)
        self.gamma = kwargs['gamma']
        self.a = kwargs['fp_weight']
        self.b = kwargs['fn_weight']
        
    def regularization(self, logits, labels):
        """
        Regularization function for the focal tversky loss.
        
        Inputs
        ------
            logits: torch.Tensor (N, C)
                Raw logits from the network.
            labels: torch.Tensor (N, )
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor (float)
                Focal tversky loss for the input logits and labels.
        """
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        
        y = F.one_hot(labels, num_classes=logits.shape[1]).float()
        tp = (p * y).sum(dim=0)
        fp = (p * (1 - y)).sum(dim=0)
        fn = ((1 - p) * y).sum(dim=0)
        tversky = (tp + self.eps) / (tp + self.a * fp + self.b * fn + self.eps)
        focal_tversky = (1 - tversky)**self.gamma
        loss = focal_tversky.mean()
        
        return loss
    
    
class LogDiceLoss(ClassificationLoss):
    """
    Log dice loss for classification tasks.
    """
    
    def __init__(self, **kwargs):
        super(LogDiceLoss, self).__init__(**kwargs)
        
        self.gamma_ce = kwargs['gamma_ce']
        self.gamma_dice = kwargs['gamma_dice']
    def regularization(self, logits, labels):
        """
        Regularization function for the log dice loss.
        
        Inputs
        ------
            logits: torch.Tensor (N, C)
                Raw logits from the network.
            labels: torch.Tensor (N, )
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor (float)
                Log dice loss for the input logits and labels.
        """
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        
        y = F.one_hot(labels, num_classes=logits.shape[1]).float()
        dice = (2 * (p * y).sum(dim=0) + self.eps) / (p.sum(dim=0) + y.sum(dim=0) + self.eps)
        dice = dice.mean()
        loss = -torch.log(torch.clamp(dice, min=self.eps))
        
        return loss
    
    def forward(self, logits, labels):
        """
        Forward pass of the focal loss.
        
        Inputs
        ------
            logits: torch.Tensor
                Raw logits from the network.
            labels: torch.Tensor
                True labels for the input logits.
                
        Returns
        -------
            loss: torch.Tensor
                Focal loss for the input logits and labels.
        """
        x, y = self.ignore_classes(logits, labels)
        reg_loss = self.regularization(x, y)
        ce_loss = self.xentropy(x, y)
        
        l1 = self.w * (reg_loss) ** self.gamma_dice
        l2 = (1 - self.w) * (ce_loss) ** self.gamma_ce
        
        loss = l1 + l2
        
        output = {
            'loss': loss,
            'reg_loss': float(l1),
            'ce_loss': float(l2)
        }
        
        return output