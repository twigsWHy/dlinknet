import mindspore
import mindspore.nn as nn


class dice_bce_loss(nn.LossBase):
    def __init__(self, batch=True, reduction="mean"):
        super(dice_bce_loss, self).__init__(reduction)
        self.batch = batch
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.sum = mindspore.ops.ReduceSum(keep_dims=False)

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = self.sum(y_true)
            j = self.sum(y_pred)
            intersection = self.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def construct(self, predict, target):
        a = self.bce_loss(predict, target)
        b = self.soft_dice_loss(target, predict)
        return a + b


class iou_bce_loss(nn.LossBase):
    def __init__(self, batch=True, reduction="mean"):
        super(iou_bce_loss, self).__init__(reduction)
        self.batch = batch
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.sum = mindspore.ops.ReduceSum(keep_dims=False)

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = self.sum(y_true)
            j = self.sum(y_pred)
            intersection = self.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (intersection + smooth) / (i + j - intersection + smooth)  # iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def construct(self, predict, target):
        a = self.bce_loss(predict, target)
        b = self.soft_dice_loss(target, predict)
        return a + b
