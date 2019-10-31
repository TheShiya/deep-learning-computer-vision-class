from .grader import Grader, Case, MultiCase

import numpy as np
import torchvision.transforms.functional as TF
from os import path
import torch
import pystk

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15

pystk_config = pystk.GraphicsConfig.hd()
pystk_config.screen_width = 128
pystk_config.screen_height = 96

pystk.init(pystk_config)


def quat_rot(r, v, inverse=False):
    r, v, inv = torch.as_tensor(r, dtype=torch.float32), torch.as_tensor(v, dtype=torch.float32), 1 - 2 * float(inverse)
    return torch.as_tensor([(1 - 2 * (r[(i + 1) % 3] ** 2 + r[(i + 2) % 3] ** 2)) * v[i] +
                            2 * (r[i] * r[(i + 1) % 3] - r[(i + 2) % 3] * r[3] * inv) * v[(i + 1) % 3] +
                            2 * (r[i] * r[(i + 2) % 3] + r[(i + 1) % 3] * r[3] * inv) * v[(i + 2) % 3] for i in
                            range(3)])


class PySTKGrader(Grader):
    use_planner = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.C = self.module.control
        self.P = None
        if self.use_planner:
            self.P = self.module.load_model().eval()
    
    @staticmethod
    def _to_world(aim_point_image, proj, view, height=0):
        x, y, W, H = *aim_point_image, pystk_config.screen_width, pystk_config.screen_height
        pv_inv = np.linalg.pinv(proj @ view)
        xy, d = pv_inv.dot([float(x) / (W / 2) - 1, 1 - float(y) / (H / 2), 0, 1]), pv_inv[:, 2]
        x0, x1 = xy[:-1] / xy[-1], (xy+d)[:-1] / (xy+d)[-1]
        t = (height-x0[1]) / (x1[1] - x0[1])
        if t < 1e-3 or t > 10:
            # Project the point forward by a certain distance, if it would end up behind
            t = 10
        return t * x1 + (1-t) * x0

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """
        node_idx = np.searchsorted(track.path_distance[..., 1], distance) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_kart(x, kart):
        return quat_rot(kart.rotation, x - kart.location, True)
    
    def _test(self, track, max_frames):
        config = pystk.RaceConfig(num_kart=1, laps=1)
        config.track = track
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        config.render = self.P is not None

        k = pystk.Race(config)
        try:
            state = pystk.WorldState()
            track = pystk.Track()

            k.start()
            k.step()

            last_rescue = 0
            for t in range(max_frames):
                state.update()
                track.update()

                kart = state.players[0].kart

                if kart.race_result:
                    break
                
                if self.P is None:
                    aim_point_world = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
                else:
                    image = np.array(k.render_data[0].image)
                    aim_point_image = self.P(TF.to_tensor(image)[None]).squeeze(0)
                    
                    proj = np.array(state.players[0].camera.projection).T
                    view = np.array(state.players[0].camera.view).T
                    aim_point_world = self._to_world(aim_point_image, proj, view, kart.location[1])
                
                aim_point_car = self._to_kart(aim_point_world, kart)
                current_vel = np.linalg.norm(kart.velocity)
                action = self.C(aim_point_car, current_vel)

                if current_vel <= 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                    action.rescue = True
                    last_rescue = t

                k.step(action)
        finally:
            k.stop()
            del k
        if kart.race_result:
            return 1, '%0.1f s' % kart.finish_time
        return kart.overall_distance / track.length, '%0.1f%% done' % (100 * kart.overall_distance / track.length)


class ControllerGrader(PySTKGrader, Grader):
    """Controller"""
    use_planner = False
    
    @Case(score=5)
    def test_lighthouse(self):
        """lighthouse"""
        return self._test('lighthouse', 550)

    @Case(score=5)
    def test_hacienda(self):
        """hacienda"""
        return self._test('hacienda', 700)

    @Case(score=5)
    def test_snowtuxpeak(self):
        """snowtuxpeak"""
        return self._test('snowtuxpeak', 700)

    @Case(score=5)
    def test_zengarden(self):
        """zengarden"""
        return self._test('zengarden', 600)

    @Case(score=5)
    def test_cornfield_crossing(self):
        """cornfield_crossing"""
        return self._test('cornfield_crossing', 750)

    @Case(score=5)
    def test_scotland(self):
        """scotland"""
        return self._test('scotland', 700)


class PlannerGrader(PySTKGrader, Grader):
    """Planner"""
    use_planner = True

    @Case(score=10)
    def test_lighthouse(self, it=0):
        """lighthouse"""
        return self._test('lighthouse', 650)

    @Case(score=10)
    def test_hacienda(self, it=0):
        """hacienda"""
        return self._test('hacienda', 700)

    @Case(score=10)
    def test_snowtuxpeak(self, it=0):
        """snowtuxpeak"""
        return self._test('snowtuxpeak', 700)

    @Case(score=10)
    def test_zengarden(self, it=0):
        """zengarden"""
        return self._test('zengarden', 700)

    @Case(score=10)
    def test_cornfield_crossing(self, it=0):
        """cornfield_crossing"""
        return self._test('cornfield_crossing', 950)

    @Case(score=10)
    def test_scotland(self, it=0):
        """scotland"""
        return self._test('scotland', 850)


class NewLevelrGrader(PySTKGrader, Grader):
    """Test level"""
    use_planner = True

    @Case(score=10)
    def test_cocoa_temple(self, it=0):
        """cocoa temple"""
        return self._test('cocoa_temple', 800)
