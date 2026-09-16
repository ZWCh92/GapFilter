import importlib,json,traceback,pathlib,sys
root=pathlib.Path('/data1/wangzeyu/st_ot/GapFilter');results=[]
def check(name,fn):
 try:
  detail=fn();results.append({'test':name,'status':'PASS','detail':str(detail)});print('PASS',name,detail,flush=True)
 except Exception as e:
  results.append({'test':name,'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()});print('FAIL',name,repr(e),flush=True)
for name in ['config','preprocessing','preprocessing.archives','preprocessing.base','preprocessing.config','preprocessing.registry','preprocessing.utils','preprocessing.visium','preprocessing.visiumhd','preprocessing.xenium','data']:
 check('module:'+name,lambda name=name:importlib.import_module(name).__name__)
def spatial():
 import numpy as np,scanpy as sc,squidpy as sq,anndata as ad
 rng=np.random.default_rng(5)
 a=ad.AnnData(rng.poisson(4,size=(100,80)).astype('float32'));a.obsm['spatial']=rng.random((100,2))
 sc.pp.highly_variable_genes(a,flavor='seurat_v3',n_top_genes=20)
 assert a.var.highly_variable.sum()==20
 sq.gr.spatial_neighbors(a,coord_type='generic',n_neighs=4)
 sq.gr.spatial_autocorr(a,mode='moran',genes=list(a.var_names[:5]),n_perms=2,n_jobs=1,show_progress_bar=False)
 assert len(a.uns['moranI'])==5
 return {'hvg':20,'moran_genes':5}
check('Seurat_v3_and_spatial_Moran',spatial)
def parquet():
 import pandas as pd
 p=root/'reports/GF_environment/smoke.parquet';a=pd.DataFrame({'spot':[1,2],'value':[3.,4.]});a.to_parquet(p);assert pd.read_parquet(p).equals(a);return 'roundtrip'
check('parquet',parquet)
(root/'reports/GF_environment/preprocessing-verification.json').write_text(json.dumps(results,indent=2))
sys.exit(int(any(x['status']=='FAIL' for x in results)))
