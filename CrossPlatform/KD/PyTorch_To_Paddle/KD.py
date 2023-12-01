import paddle.nn as nn
import paddle.nn.functional as F

#  Distilling the Knowledge in a Neural Network
class KD(nn.Layer):
    def __init__(self, T, alpha, beta, epoches):
        '''
        成员属性: 
            1. T: 温度
            2. alpha: 超参数1
            3. beta: 超参数2
            4. epoches: 训练周期
            5. soft_loss: 软损失函数 (默认KL散度)
            6. hard_loss: 硬损失函数 (默认交叉熵损失函数)
        '''
        super(KD, self).__init__()
        
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.epoches = epoches
        
        self.soft_loss = nn.KLDivLoss(reduction="batchmean")
        self.hard_loss = nn.CrossEntropyLoss()  # hard loss

    #  JS散度
    def JSDivLoss(self, y_s, y_t):
        p_s = F.softmax(y_s / self.T, axis=1)
        p_t = F.softmax(y_t / self.T, axis=1)
        log_mean_output = ((p_s + p_t ) / 2).log()
        return (self.soft_loss(log_mean_output, p_s) + self.soft_loss(log_mean_output, p_t)) / 2

    def forward(self, y_s, y_t, label, epoch, LOSS='KL'):
        """
        参数解释： 
            1. y_s: Student Model Output
            2. y_t: Teacher Model Output
            3. label: 标签真实值
            4. epoch: 当前训练周期
            5. LOSS: soft loss函数的选取
        """
        # soft loss
        # 1. KL散度
        if LOSS == 'KL':
            soft_loss = self.soft_loss(F.softmax(y_s / self.T, axis=1), F.softmax(y_t / self.T, axis=1))
        # 2. JS散度
        elif LOSS == 'JS':
            soft_loss = self.JSDivLoss(y_s, y_t)
        # 3. KL + JS
        else:
            if epoch > self.epoches / 2:
                soft_loss = self.soft_loss(F.softmax(y_s / self.T, axis=1), F.softmax(y_t / self.T, axis=1))
            else:
                soft_loss = self.JSDivLoss(y_s, y_t)
        
        # hard_loss
        hard_loss = self.hard_loss(y_s, label)
        
        # loss = alpha * soft_loss + beta * hard_loss
        loss = self.alpha * soft_loss + self.beta * hard_loss
        return loss
