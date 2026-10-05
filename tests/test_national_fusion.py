import tempfile
import unittest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from unittest.mock import patch

from backend.models.geo_intelligence.fusion_inputs import prepare, read
from backend.models.geo_intelligence.national_fusion import NationalFusionCNN, augment


class NationalInputTests(unittest.TestCase):
    def bands(self, size=64):
        return np.broadcast_to(np.array([400, 800, 1000, 3000, 2000, 1500, 0, 0, 0], dtype='float32')[:, None, None],
                               (9, size, size)).copy()

    def test_indices_and_units(self):
        out = prepare(self.bands())
        self.assertAlmostEqual(float(out[0, 0, 0]), .04)
        self.assertAlmostEqual(float(out[6, 0, 0]), .5, places=6)
        self.assertAlmostEqual(float(out[8, 0, 0]), .2, places=6)

    def test_no_stretching_missing_grid(self):
        a = self.bands()
        a[:, :20] = np.nan
        with self.assertRaises(ValueError):
            prepare(a)
        a = self.bands()
        a[:, :2] = np.nan
        self.assertTrue(np.isfinite(prepare(a)).all())

    def test_matching_physical_footprint(self):
        with tempfile.TemporaryDirectory() as folder:
            results = []
            for size, resolution in [(128, 10), (64, 20)]:
                p = Path(folder)/f'{resolution}.tif'
                with rasterio.open(p, 'w', driver='GTiff', height=size, width=size, count=9,
                                   dtype='float32', crs='EPSG:32644', transform=from_origin(400000, 2400000, resolution, resolution)) as dst:
                    dst.write(self.bands(size))
                results.append(read(p))
            np.testing.assert_allclose(*results, rtol=1e-5)

    def test_augmentation_keeps_index_physics(self):
        x = torch.tensor(prepare(self.bands())[None].repeat(2, axis=0))
        out = augment(x, np.random.default_rng(13))
        torch.testing.assert_close(out[:, 6], (out[:, 3]-out[:, 2])/(out[:, 3]+out[:, 2]))
        self.assertFalse(torch.equal(out[:, :6], x[:, :6]))

    def test_normalization_not_updated_by_inference(self):
        model = NationalFusionCNN()
        x = torch.tensor(prepare(self.bands())[None].repeat(2, axis=0))
        model.fit_normalization(x, torch.tensor([[20.], [50.]]))
        before = model.band_mean.clone()
        model.eval()
        with torch.inference_mode():
            result = model(x*2, torch.tensor([[90.], [100.]]))
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(model.band_mean, before)

    def test_legacy_live_request_matches_640m_training(self):
        from backend.models.geo_intelligence import world_fusion
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'patch.tif'
            with rasterio.open(p, 'w', driver='GTiff', height=64, width=64, count=9, dtype='float32',
                               crs='EPSG:32644', transform=from_origin(400000, 2400000, 10, 10)) as dst:
                dst.write(self.bands())
            with patch.object(world_fusion, 'model_bundle', return_value=(None, {})), \
                 patch.object(world_fusion, 'predict', return_value={'score': .5}), \
                 patch('backend.cloud.fetch_patch', return_value=(p, {'valid_fraction': 1})) as fetch:
                result = world_fusion.inspect_point(80, 21)
            fetch.assert_called_once_with(80, 21, size=64, min_valid=.85, resolution_m=10)
            self.assertEqual(len(result['input_band_order']), 9)

    def test_unapproved_national_model_is_not_loaded(self):
        import json
        from backend.models.geo_intelligence import world_fusion
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'national_fusion_report.json').write_text(json.dumps({'eligible_for_deployment': False}))
            torch.save(world_fusion.WorldFusionCNN().state_dict(), root/'world_fusion.pt')
            (root/'world_fusion_metadata.json').write_text(json.dumps({'version': 'baseline', 'gravity_mean': 0, 'gravity_variance': 1}))
            world_fusion.model_bundle.cache_clear()
            try:
                with patch.object(world_fusion, 'ARTIFACTS', root):
                    self.assertEqual(world_fusion.model_bundle()[1]['version'], 'baseline')
            finally:
                world_fusion.model_bundle.cache_clear()


if __name__ == '__main__':
    unittest.main()
