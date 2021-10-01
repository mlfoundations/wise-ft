import os
import torch
import wilds

from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from wilds.common.data_loaders import get_train_loader, get_eval_loader

class FMOW:
    test_subset = None

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 subset='test',
                 classnames=None,
                 **kwargs):

        self.dataset = wilds.get_dataset(dataset='fmow', root_dir=location)

        self.train_dataset = self.dataset.get_subset('train', transform=preprocess)
        self.train_loader = get_train_loader("standard", self.train_dataset, num_workers=num_workers, batch_size=batch_size)

        self.test_dataset = self.dataset.get_subset(self.test_subset, transform=preprocess)
        self.test_loader = get_eval_loader("standard", self.test_dataset, num_workers=num_workers, batch_size=batch_size)

        self.classnames = [
            "airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture",
            "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership",
            "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution",
            "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
            "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital",
            "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility",
            "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
            "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track",
            "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall",
            "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank",
            "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal",
            "water_treatment_facility", "wind_farm", "zoo"
        ]

    def post_loop_metrics(self, labels, preds, metadata, args):
        metadata = torch.stack(metadata)
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]

class FMOWID(FMOW):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'id_test'
        super().__init__(*args, **kwargs)

class FMOWOOD(FMOW):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'test'
        super().__init__(*args, **kwargs)