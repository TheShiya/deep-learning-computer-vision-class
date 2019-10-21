"""
EDIT THIS FILE AT YOUR OWN RISK!
It will not ship with your code, editing it will only change the test cases locally, and might make you fail our
remote tests.
"""
import torch
from .grader import Grader, Case, MultiCase

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def point_in_box(p, x0, y0, x1, y1):
    return x0 <= p[0] < x1 and y0 <= p[1] < y1


def point_close(p, x0, y0, x1, y1, d=5):
    return ((x0 + x1 - 1) / 2 - p[0]) ** 2 + ((y0 + y1 - 1) / 2 - p[1]) ** 2 < d ** 2


def box_iou(p, x0, y0, x1, y1, t=0.5):
    iou = abs(min(p[0] + p[2], x1) - max(p[0] - p[2], x0)) * abs(min(p[1] + p[3], y1) - max(p[1] - p[3], y0)) / \
          abs(max(p[0] + p[2], x1) - min(p[0] - p[2], x0)) * abs(max(p[1] + p[3], y1) - min(p[1] - p[3], y0))
    return iou > t


class PR:
    def __init__(self, min_size=20, is_close=point_in_box):
        self.min_size = min_size
        self.total_det = 0
        self.det = []
        self.is_close = is_close

    def add(self, d, lbl):
        small_lbl = [b for b in lbl if abs(b[2] - b[0]) * abs(b[3] - b[1]) < self.min_size]
        large_lbl = [b for b in lbl if abs(b[2] - b[0]) * abs(b[3] - b[1]) >= self.min_size]
        used = [False] * len(large_lbl)
        for s, *p in d:
            match = False
            for i, box in enumerate(large_lbl):
                if not used[i] and self.is_close(p, *box):
                    match = True
                    used[i] = True
                    break
            if match:
                self.det.append((s, 1))
            else:
                match_small = False
                for i, box in enumerate(small_lbl):
                    if self.is_close(p, *box):
                        match_small = True
                        break
                if not match_small:
                    self.det.append((s, 0))
        self.total_det += len(large_lbl)

    @property
    def curve(self):
        true_pos, false_pos = 0, 0
        r = []
        for t, m in sorted(self.det, reverse=True):
            if m:
                true_pos += 1
            else:
                false_pos += 1
            prec = true_pos / (true_pos + false_pos)
            recall = true_pos / self.total_det
            r.append((prec, recall))
        return r

    @property
    def average_prec(self, n_samples=11):
        max_prec = 0
        cur_rec = 1
        precs = []
        for prec, recall in self.curve[::-1]:
            max_prec = max(max_prec, prec)
            while cur_rec > recall:
                precs.append(max_prec)
                cur_rec -= 1.0 / n_samples
        return sum(precs) / len(precs)


class ExtractPeakGrader(Grader):
    """extract_peak"""

    def test_det(self, p, hm, min_score=0):
        centers = [(cx, cy) for s, cx, cy in p]
        assert len(centers) == len(set(centers)), "Duplicate detection"
        assert all([0 <= cx < hm.size(1) and 0 <= cy < hm.size(0) for cx, cy in centers]), "Peak out of bounds"
        assert all([s > min_score for s, cx, cy in p]), "Returned a peak below min_score"
        assert all([s == hm[cy, cx] for s, cx, cy in p]), "Score does not match heatmap"

    @Case(score=5)
    def test_format(self, min_score=0):
        """return value"""
        ep = self.module.extract_peak
        for i in range(50, 200, 10):
            img = torch.randn(3 * i, 2 * i)
            p = ep(img, max_pool_ks=3, min_score=min_score, max_det=i)
            assert len(p) <= i, "Expected at most %d peaks, got %d" % (i, len(p))
            self.test_det(p, img, min_score=min_score)

    @Case(score=5)
    def test_radius1(self, min_score=0):
        """radius=1"""
        img = torch.randn(54, 123)
        p = self.module.extract_peak(img, max_pool_ks=1, min_score=min_score, max_det=100000)
        assert len(p) == (img > 0).sum(), 'Expected exactly %d detections, got %d' % (len(p), (img > 0).sum())
        self.test_det(p, img, min_score=min_score)

    @Case(score=5)
    def test_manyl(self, min_score=0, max_pool_ks=5):
        """peak extraction"""
        from functools import partial
        ep = partial(self.module.extract_peak, max_pool_ks=max_pool_ks, min_score=min_score, max_det=100)
        assert len(ep(torch.zeros((10, 10)))) == 0, "No peak expected"
        assert len(ep(torch.arange(100).view(10, 10).float())) == 1, "Single peak expected"
        assert len(ep(torch.ones((10, 10)))) == 100, "100 peaks expected"
        assert len(ep((torch.arange(100).view(10, 10) == 55).float())) == 1, "Single peak expected"
        assert len(ep((torch.arange(100).view(10, 10) == 55).float() - 1)) == 0, "No peak expected"

    @Case(score=5)
    def test_random(self, min_score=0, max_pool_ks=5):
        """randomized test"""
        from functools import partial
        ep = partial(self.module.extract_peak, max_pool_ks=max_pool_ks, min_score=min_score, max_det=100)
        img = torch.zeros((100, 100))
        c = torch.randint(0, 100, (100, 2))
        pts = set()
        for i, p in enumerate(c):
            if i == 0 or (c[:i] - p[None]).abs().sum(dim=1).min() > max_pool_ks:
                pts.add((float(p[0]), float(p[1])))
                img[p[1], p[0]] = 1
                if len(pts) >= 10:
                    break
        p_img = 1 * img
        for k in range(1, max_pool_ks+1, 2):
            p_img += torch.nn.functional.avg_pool2d(img[None, None], k, padding=k//2, stride=1)[0, 0]
            p = ep(p_img)
            self.test_det(p, p_img, min_score)
            ret_pts = {(float(cx), float(cy)) for s, cx, cy in p}
            assert ret_pts == pts, "Returned the wrong peaks for randomized test"


class DetectorGrader(Grader):
    """Detector"""

    @Case(score=5)
    def test_format(self):
        """return value"""
        det = self.module.load_model().eval()
        for i, (img, *gts) in enumerate(self.module.utils.DetectionSuperTuxDataset('dense_data/valid', min_size=0)):
            d = det.detect(img)
            assert len(d) <= 100, 'Returned more than 100 detections'
            assert all(len(i) == 4 for i in d), 'Each detection should be a tuple (class, score, cx, cy)'
            if i > 10:
                break


class DetectionGrader(Grader):
    """Detection model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        det = self.module.load_model().eval()

        # Compute detections
        self.pr_box = [PR() for _ in range(3)]
        self.pr_dist = [PR(is_close=point_close) for _ in range(3)]
        for img, *gts in self.module.utils.DetectionSuperTuxDataset('dense_data/valid', min_size=0):
            d = det.detect(img)
            for i, gt in enumerate(gts):
                self.pr_box[i].add([j[1:] for j in d if j[0] == i], gt)
                self.pr_dist[i].add([j[1:] for j in d if j[0] == i], gt)

    @Case(score=10)
    def test_box_ap0(self, min_val=0.5, max_val=0.75):
        """Average precision (inside box c=0)"""
        ap = self.pr_box[0].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    @Case(score=10)
    def test_box_ap1(self, min_val=0.25, max_val=0.45):
        """Average precision (inside box c=1)"""
        ap = self.pr_box[1].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    @Case(score=10)
    def test_box_ap2(self, min_val=0.6, max_val=0.85):
        """Average precision (inside box c=2)"""
        ap = self.pr_box[2].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    @Case(score=15)
    def test_dist_ap0(self, min_val=0.5, max_val=0.72):
        """Average precision (distance c=0)"""
        ap = self.pr_dist[0].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    @Case(score=15)
    def test_dist_ap1(self, min_val=0.25, max_val=0.45):
        """Average precision (distance c=1)"""
        ap = self.pr_dist[1].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    @Case(score=15)
    def test_dist_ap2(self, min_val=0.6, max_val=0.85):
        """Average precision (distance c=2)"""
        ap = self.pr_dist[2].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap


class DetectionSizeGrader(Grader):
    """Detection and size model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        det = self.module.load_model().eval()

        # Compute detections
        self.pr = [PR(is_close=box_iou) for _ in range(3)]
        for img, *gts in self.module.utils.DetectionSuperTuxDataset('dense_data/valid', min_size=0):
            d = det.detect_with_size(img)
            for i, gt in enumerate(gts):
                self.pr[i].add([j[1:] for j in d if j[0] == i], gt)

    @Case(score=3, extra_credit=True)
    def test_box_ap0(self, min_val=0.7):
        """Average precision (iou > 0.5  c=0)"""
        ap = self.pr[0].average_prec
        return ap >= min_val, 'AP = %0.3f' % ap

    @Case(score=3, extra_credit=True)
    def test_box_ap1(self, min_val=0.2):
        """Average precision (iou > 0.5  c=1)"""
        ap = self.pr[1].average_prec
        return ap >= min_val, 'AP = %0.3f' % ap

    @Case(score=3, extra_credit=True)
    def test_box_ap2(self, min_val=0.5):
        """Average precision (iou > 0.5  c=2)"""
        ap = self.pr[2].average_prec
        return ap >= min_val, 'AP = %0.3f' % ap
