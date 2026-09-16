"""Configurable, resumable training engine isolated from frozen pilot code."""
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .training import seed_everything
from .transfer_training import rng_state,restore_rng,atomic_save
from .hyperdense_transfer import ieee_precision


def finite(value):
    if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):raise RuntimeError('Nonfinite tensor')
    if isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)


def run(model,train,inner,output,*,seed,epochs,learning_rate=.001,schedule='constant',calibration=False,device='cpu',metadata=None,resume=False,deadline=None):
    if learning_rate not in (.0001,.0003,.001) or schedule not in ('constant','exponential') or not 1<=epochs<=150:raise ValueError('Unsupported settings')
    if calibration and not isinstance(inner,torch.Tensor):raise ValueError('Calibration accepts inner inputs only')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);latest=output/'latest.pt'
    if latest.exists()!=resume:raise ValueError('Resume flag does not match existing state')
    config=dict(seed=seed,learning_rate=learning_rate,schedule=schedule,calibration=calibration,patience=20,batch_size=32)
    seed_everything(seed);model.to(device);generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train,batch_size=32,shuffle=True,generator=generator)
    inner_loader=DataLoader(inner,batch_size=32)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,betas=(.9,.999),eps=1e-7)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.98 if schedule=='exponential' else 1.)
    history=[];best=float('inf');best_epoch=0;stale=0;best_state=None;start=0;timings=[]
    if resume:
        state=torch.load(latest,map_location='cpu',weights_only=True)
        if state['metadata']!=(metadata or {}) or state['config']!=config:raise ValueError('Resume metadata changed')
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer_state_dict']);scheduler.load_state_dict(state['scheduler_state_dict'])
        history=state['history'];best=state['best_loss'];best_epoch=state['best_epoch'];stale=state['stale_epochs'];best_state=state['best_state_dict'];start=state['epoch'];timings=state['timings']
        restore_rng(state['rng'],generator)
    def check():
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Soft deadline; latest complete epoch retained')
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    with ieee_precision():
        for epoch in range(start,epochs):
            if not calibration and stale>=20:break
            check();sync();lap=time.perf_counter();model.train();total=0.;count=0;lr=optimizer.param_groups[0]['lr']
            for x,y in train_loader:
                check();optimizer.zero_grad(set_to_none=True);loss=(model(x.to(device))-y.to(device)).abs().mean();finite(loss)
                loss.backward();optimizer.step();total+=loss.item()*len(x);count+=len(x)
            sync();train_seconds=time.perf_counter()-lap;lap=time.perf_counter();model.eval();val=0.;items=0
            with torch.inference_mode():
                for batch in inner_loader:
                    check()
                    if calibration:finite(model(batch.to(device)))
                    else:
                        x,y=batch;loss=(model(x.to(device))-y.to(device)).abs().mean();finite(loss);val+=loss.item()*len(x);items+=len(x)
            sync();inner_seconds=time.perf_counter()-lap;lap=time.perf_counter()
            entry=dict(epoch=epoch+1,lr=lr,train_mae=total/count)
            if calibration:
                # Same-size selected-state payload for I/O timing; never a validation-selected checkpoint.
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best=None;best_epoch=epoch+1
            else:
                value=val/items;entry['inner_mae']=value
                if value<best:
                    best=value;best_epoch=epoch+1;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:stale+=1
            history.append(entry);scheduler.step();finite(model.state_dict());finite(optimizer.state_dict())
            payload=dict(protocol='hyperdense-transfer-training-v2',state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
                epoch=epoch+1,config=config,metadata=metadata or {},best_loss=best,best_epoch=best_epoch,stale_epochs=stale,best_state_dict=best_state,
                rng=rng_state(generator),history=history,timings=timings+[dict(epoch=epoch+1,train_seconds=train_seconds,inner_seconds=inner_seconds,checkpoint_seconds=None,total_seconds=None)],timing_note='Current checkpoint write duration is available only in the returned result; earlier completed timings are retained.')
            atomic_save(payload,latest);sync();checkpoint_seconds=time.perf_counter()-lap
            timings.append(dict(epoch=epoch+1,train_seconds=train_seconds,inner_seconds=inner_seconds,checkpoint_seconds=checkpoint_seconds,total_seconds=train_seconds+inner_seconds+checkpoint_seconds))
        if best_state is None:raise RuntimeError('No complete epoch')
        model.load_state_dict(best_state)
        if not calibration:atomic_save(dict(state_dict=best_state,metadata=metadata or {},best_epoch=best_epoch,best_inner_mae=best),output/'best.pt')
    return dict(history=history,timings=timings,epochs_ran=len(history),best_epoch=None if calibration else best_epoch,best_inner_mae=best,
        stale_epochs=stale,patience_exhausted=stale>=20,calibration=calibration,validation_scored=not calibration,test_scored=False)
