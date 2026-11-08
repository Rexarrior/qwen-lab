"""Rank-8 correction with straight-through INT4 codes and FP16 FiLM/decoder weights."""
import torch
import torch.nn.functional as F
from factor_decoder import condition

class FullLayerFilmDecoder(torch.nn.Module):
    """Full trainable MLP synthesis function + per-layer FiLM with FP16 STE."""
    def __init__(self,params):
        super().__init__()
        for i in range(3):
            for kind in ('weight','bias'):
                setattr(self,kind+str(i),torch.nn.Parameter(torch.from_numpy(params[str(i)+'.'+kind].astype('<f4'))))
        self.film_gain=torch.nn.Parameter(torch.ones(28,2,64))
        self.film_bias=torch.nn.Parameter(torch.zeros(28,2,64))
    def forward(self,codes,cond):
        layer=int(round(float(cond[0,9])*27))
        assert 0<=layer<28 and torch.all(cond[:,9]==cond[0,9])
        gain=self.film_gain+(self.film_gain.half().float()-self.film_gain).detach()
        bias=self.film_bias+(self.film_bias.half().float()-self.film_bias).detach()
        x=torch.cat((codes/4,cond),dim=1)
        for i in range(3):
            w=getattr(self,'weight'+str(i));b=getattr(self,'bias'+str(i))
            w_ste=w+(w.half().float()-w).detach()
            b_ste=b+(b.half().float()-b).detach()
            x=F.linear(x,w_ste,b_ste)
            if i!=2:
                x=x*gain[layer,i]+bias[layer,i]
                x=x*torch.sigmoid(x)
        return x+codes/2
    def base_parameters_half(self):
        return {f'{i}.{k}':getattr(self,f'{k}{i}').detach().cpu().numpy().astype('<f2') for i in range(3) for k in ('weight','bias')}

class QATLinear(torch.nn.Module):
    def __init__(self,linear,decoder,records,pairs):
        super().__init__();assert linear.bias is None
        self.register_buffer('weight',linear.weight.detach());self.base_energy=float(self.weight.square().sum())
        self.decoder=decoder
        self.A=torch.nn.Parameter(torch.from_numpy(pairs[0].astype('<f4')))
        self.B=torch.nn.Parameter(torch.from_numpy(pairs[1].astype('<f4')))
        self.rows=records
        for i,row in enumerate(records):
            self.register_buffer('condition'+str(i),torch.from_numpy(condition(row['family'],row['role'],row['layer'],row['vectors'])))
        self.exported_factors=None
    def factors(self):
        if self.exported_factors is not None:return self.exported_factors
        values=[]
        for i,master in enumerate((self.A,self.B)):
            q=master+(master.round().clamp(-8,7)-master).detach()
            y=self.decoder(q,getattr(self,'condition'+str(i)))*self.rows[i]['FP16_RMS']
            y=y+(y.half().float()-y).detach()
            assert torch.isfinite(y).all()
            values.append(y.T if i==0 else y)
        return tuple(values)
    def forward(self,x):
        a,b=self.factors();return F.linear(x,self.weight)+F.linear(F.linear(x,a),b)
    def delta_energy(self):
        a,b=self.factors();return ((b.T@b)*(a@a.T)).sum()
def hidden(model,row):
    ids=torch.tensor([row['ids']],dtype=torch.long,device=model.model.embed_tokens.weight.device)
    return model.model(input_ids=ids,use_cache=False).last_hidden_state[0]

def loss_for(model,row,teacher_cache,training):
    h=hidden(model,row);places=torch.tensor(row['positions'],device=h.device)
    logits=F.linear(h[places],model.lm_head.weight)
    targets=torch.tensor(row['targets'],device=h.device)
    ce=F.cross_entropy(logits,targets);kl=None;loss=ce
    if row.get('kind')=='agent' and training.get('terminal_turn_weight',1.0)!=1.0 and row['targets'][0]!=151657:
        tw=float(training['terminal_turn_weight'])
        fg=float(training.get('terminal_first_token_gamma',1.0))
        if fg!=1.0 and len(row['targets'])>1:
            per_tok=F.cross_entropy(logits,targets,reduction='none')
            w=torch.ones_like(per_tok);w[0]=fg;w=w/w.sum()
            loss=tw*float(len(row['targets']))*(per_tok*w).sum() if training.get('terminal_scale_by_len',False) else tw*(per_tok*w).sum()
        else:
            loss=tw*ce
    if 'teacher_key' in row:
        positions=torch.tensor(row['teacher_positions'],device=h.device)
        actual=F.linear(h[positions],model.lm_head.weight).log_softmax(-1)
        target=torch.from_numpy(teacher_cache[row['teacher_key']]).to(h.device).log_softmax(-1)
        assert actual.shape==target.shape
        kl=(target.exp()*(target-actual)).sum(-1).mean();loss=loss+training['language_KL_weight']*kl
    return loss,ce,kl
