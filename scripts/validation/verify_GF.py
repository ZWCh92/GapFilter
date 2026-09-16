import ast, importlib, json, pathlib, sys, traceback
import numpy as np
import torch
root=pathlib.Path('/data1/wangzeyu/st_ot/GapFilter')
results=[]
def check(name, fn):
    try:
        detail=fn()
        results.append({'test':name,'status':'PASS','detail':str(detail)})
        print('PASS',name,detail,flush=True)
    except Exception as e:
        results.append({'test':name,'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()})
        print('FAIL',name,repr(e),flush=True)
mods='numpy scipy pandas anndata scanpy squidpy sklearn skimage h5py matplotlib seaborn numba zarr tifffile PIL hydra omegaconf yaml pytorch_lightning torchmetrics tensorboard timm transformers peft accelerate huggingface_hub safetensors loguru typer dotenv tqdm ipykernel torchvision'.split()
for name in mods: check('dependency:'+name,lambda name=name: importlib.import_module(name).__name__)
for v in range(1,5):
    for component in ['dataset','model','train','predict']:
        name=f'gapfilter.gapfilter_{v}.{component}'
        check('module:'+name,lambda name=name: importlib.import_module(name).__name__)
for name in ['gapfilter.features','gapfilter.plots','gapfilter.baseline.KNN','seal','seal.models.load_model','seal.models.encoder_factory','seal.utils.eval_utils','seal.utils.hest_download']:
    check('module:'+name,lambda name=name: importlib.import_module(name).__name__)
def roundtrip():
    import anndata as ad,scipy.sparse as sp
    a=ad.AnnData(sp.csr_matrix(np.arange(12,dtype=np.float32).reshape(4,3)))
    p=root/'reports/GF_environment/smoke.h5ad';a.write_h5ad(p);b=ad.read_h5ad(p)
    assert (b.X != a.X).nnz == 0
    return b.shape
check('h5ad_sparse_roundtrip',roundtrip)
def modeltest(v,device):
    cfg=dict(feature_dim=8,gene_dim=3,hidden_dims=[16,8],d_model=8,n_heads=2,dropout=0.0,residual_rank=2)
    cls=importlib.import_module(f'gapfilter.gapfilter_{v}.model').GapFilterModel
    model=cls(cfg).to(device)
    q=torch.randn(4,8,device=device)
    if v<=2:
        out=model(q,torch.randn_like(q))
    else:
        if v==4:
            basis=torch.linalg.qr(torch.randn(3,2,device=device)).Q
            model.attn.set_residual_basis(basis,torch.ones(2,device=device))
        args=[q,torch.randn(4,2,8,device=device),torch.randn(4,2,3,device=device),torch.ones(4,2,dtype=torch.bool,device=device)]
        if v==4: args.append(torch.rand(4,2,device=device))
        out=model(*args)
    assert out.shape==(4,3) and torch.isfinite(out).all()
    loss=out.square().mean();loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    optimizer=model.configure_optimizers()['optimizer'];optimizer.step()
    if device=='cuda': torch.cuda.synchronize()
    return {'shape':list(out.shape),'loss':float(loss.detach().cpu())}
for device in ['cpu','cuda']:
    for v in range(1,5):check(f'GF{v}_forward_backward_optimizer_{device}',lambda v=v,device=device:modeltest(v,device))
check('GPU_inventory',lambda:{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'available':torch.cuda.is_available(),'count':torch.cuda.device_count(),'name':torch.cuda.get_device_name(0)})
(root/'reports/GF_environment/verification.json').write_text(json.dumps(results,indent=2))
sys.exit(int(any(r['status']=='FAIL' for r in results)))
