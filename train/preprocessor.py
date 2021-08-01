class Preprocessor(nn.Module):
    def __init__(self,max_val=4,alpha=0.05):
        super().__init__()
        self.max_val = max_val
        self.alpha = alpha

    def forward(self,x,invert=False):
        if invert:
            x = inverse_logit(x)
            x = (x-self.alpha)/(1-2*self.alpha)
            log_det = 
            return x, log_det
        else:
            x += uniform_dist(0,1,x.shape)  # Dequantization

            # Logit Trick
            x /= self.max_val
            x = (1-2*self.alpha)*x + self.alpha
            new_x = torch.log(x) - torch.log(1-x)
            log_det = 

            return x, log_det