"""Resumable MAE training for the isolated native transfer study."""
from copy import deepcopy
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .training import seed_everything


def rng_state(generator):
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        sampler=generator.get_state(),python=random.getstate(),
        numpy=(n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])))


def restore_rng(state,generator):
    torch.set_rng_state(state['torch']);generator.set_state(state['sampler']);random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(payload,path):
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)


def fit(model,train,inner,output,*,seed,epochs=150,patience=20,device='cpu',metadata=None,resume=False,deadline=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    latest=output/'latest.pt';best_file=output/'best.pt'
    if latest.exists() and not resume:raise ValueError('Existing training state; explicit resume required')
    if resume and not latest.exists():raise ValueError('No checkpoint to resume')
    if not 1<=epochs<=150:raise ValueError('Epoch ceiling outside frozen protocol')
    seed_everything(seed);model.to(device)
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train,batch_size=32,shuffle=True,generator=generator)
    inner_loader=DataLoader(inner,batch_size=32)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-7)
    metadata=metadata or {}
    history=[];best=float('inf');best_epoch=0;stale=0;best_state=None;start_epoch=0
    if resume:
        state=torch.load(latest,map_location='cpu',weights_only=True)
        if state['metadata']!=metadata or state['seed']!=seed or state['patience']!=patience:
            raise ValueError('Resume metadata changed')
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer_state_dict'])
        history=state['history'];best=state['best_loss'];best_epoch=state['best_epoch'];stale=state['stale_epochs']
        best_state=state['best_state_dict'];start_epoch=state['epoch'];restore_rng(state['rng'],generator)
    def check_deadline():
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Soft deadline; previous complete epoch retained')
    for epoch in range(start_epoch,epochs):
        if stale>=patience:break
        check_deadline();model.train();total=0.;count=0
        for x,y in train_loader:
            check_deadline();optimizer.zero_grad(set_to_none=True)
            loss=(model(x.to(device))-y.to(device)).abs().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training MAE')
            loss.backward();optimizer.step();total+=loss.item()*len(x);count+=len(x)
        model.eval();validation=0.;items=0
        with torch.inference_mode():
            for x,y in inner_loader:
                check_deadline();loss=(model(x.to(device))-y.to(device)).abs().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite inner MAE')
                validation+=loss.item()*len(x);items+=len(x)
        value=validation/items
        history.append(dict(epoch=epoch+1,train_mae=total/count,inner_mae=value))
        if value<best:
            best=value;best_epoch=epoch+1;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        payload=dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch+1,seed=seed,patience=patience,best_loss=best,best_epoch=best_epoch,stale_epochs=stale,
            best_state_dict=best_state,history=history,rng=rng_state(generator),metadata=metadata)
        atomic_save(payload,latest)
    if best_state is None:raise RuntimeError('No selected checkpoint')
    model.load_state_dict(best_state)
    atomic_save(dict(state_dict=best_state,best_epoch=best_epoch,best_inner_mae=best,metadata=metadata),best_file)
    return dict(epochs_ran=len(history),best_epoch=best_epoch,best_inner_mae=best,
        stale_epochs=stale,patience_exhausted=stale>=patience,ceiling_reached=len(history)==epochs,
        history=history)
