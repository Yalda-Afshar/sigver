from .gpds import GPDSDataset
from .mcyt import MCYTDataset
from .cedar import CedarDataset
from .persian import PersianDataset
from .brazilian import BrazilianDataset, BrazilianDatasetWithoutSimpleForgeries

available_datasets = {'gpds': GPDSDataset,
                      'mcyt': MCYTDataset,
                      'cedar': CedarDataset,
                      'brazilian': BrazilianDataset,
                      'brazilian-nosimple': BrazilianDatasetWithoutSimpleForgeries,
                      'persian': PersianDataset}
