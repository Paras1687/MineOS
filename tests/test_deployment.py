import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from backend.imagery_window import read_window
from backend.models.geo_intelligence.world_fusion import file_sha256


class DeploymentTests(unittest.TestCase):
    def test_jobs_reuse_only_matching_point_and_mode(self):
        from backend import server
        point=server.Point(latitude=22,longitude=80)
        with patch.object(server,'jobs',{}), patch.object(server.pool,'submit') as submit:
            first=server.point_prediction(point)
            self.assertEqual(first,server.point_prediction(point))
            self.assertNotEqual(first,server.nearby_prediction(point))
            self.assertNotEqual(first,server.point_prediction(server.Point(latitude=23,longitude=80)))
            self.assertEqual(submit.call_count,3)
            server.jobs[first['job_id']]['status']='failed'
            self.assertNotEqual(first,server.point_prediction(point))

    def test_window_read_matches_selected_pixels(self):
        data = np.arange(1, 257*257+1, dtype='float32').reshape(257, 257)
        transform = from_origin(400000, 2500000, 10, 10)
        with MemoryFile() as memory:
            with memory.open(driver='GTiff', width=257, height=257, count=1,
                             dtype='float32', crs='EPSG:32644', transform=transform) as src:
                src.write(data, 1)
                target = from_origin(400640, 2499360, 10, 10)
                result = read_window(src, target, src.crs, 64, categorical=True)
                np.testing.assert_array_equal(result, data[64:128, 64:128])

    def test_gravity_checksum_streams(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'grid'
            content=b'test-data'*300000
            path.write_bytes(content)
            with patch.object(Path,'read_bytes',side_effect=AssertionError('Full read forbidden')):
                self.assertEqual(file_sha256(path),hashlib.sha256(content).hexdigest())


if __name__ == '__main__':
    unittest.main()
