
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class ZihaoDataset(BaseSegDataset):

    METAINFO = {
        'classes':['background', 'pelagic_sediments', 'basalt', 'basalt_breccia', 'polymetallic_sulfide'],
        'palette':[[127,127,127], [200,0,0], [0,200,0], [144,238,144], [30,30,30]]
    }
    

    def __init__(self,
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)