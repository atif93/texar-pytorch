"""
Unit tests for average recoder.
"""

import unittest

from texar.torch.utils.average_recorder import _SingleAverageRecorder, AverageRecorder


class AverageRecorderTest(unittest.TestCase):
    r"""Tests average recoder.
    """

    def test_single_average_recoder(self):
        r"""Tests :class:`~texar.torch.utils._SingleAverageRecorder`
        """
        recoder = _SingleAverageRecorder(5)
        for i in range(100):
            self.assertEqual(recoder.add(1), 1.)
            self.assertEqual(recoder.avg(), 1.)

        recoder = _SingleAverageRecorder()
        for i in range(100):
            self.assertEqual(recoder.add(1), 1.)
            self.assertEqual(recoder.avg(), 1.)

        def _cal_ground_truth(n):
            r"""Calculates ((n-4)^2 + ... + n^5) / (n-4 + ... + n)
            """
            lb = max(n - 4, 0)
            _sum = 0
            _w = 0
            for i in range(lb, n + 1):
                _sum += i * i
                _w += i
            if _w == 0:
                return 0
            return _sum / _w

        recoder = _SingleAverageRecorder(5)
        for i in range(100):
            self.assertEqual(recoder.add(i, i), _cal_ground_truth(i))
            self.assertEqual(recoder.avg(), _cal_ground_truth(i))

    def test_average_recorder(self):
        r"""Tests :class:`~texar.torch.utils.AverageRecorder`
        """
        recorder = AverageRecorder(5)
        for _ in range(100):
            self.assertEqual(recorder.add([1., 2.]), [1., 2.])
            self.assertEqual(recorder.add([1.]), [1., 2.])
            self.assertEqual(recorder.avg(), [1., 2.])
            self.assertEqual(recorder.avg(0), 1.)
            self.assertEqual(recorder.avg(1), 2.)
            self.assertEqual(recorder.avg([0, 1]), [1., 2.])

        recorder = AverageRecorder()
        for _ in range(100):
            self.assertEqual(recorder.add({'1': 1, '2': 2}), {'1': 1., '2': 2.})
            self.assertEqual(recorder.add({'1': 1}), {'1': 1., '2': 2.})
            self.assertEqual(recorder.avg(), {'1': 1., '2': 2.})
            self.assertEqual(recorder.avg('1'), 1.)
            self.assertEqual(recorder.avg('2'), 2.)
            self.assertEqual(recorder.avg(['1', '2']), {'1': 1., '2': 2.})


if __name__ == "__main__":
    unittest.main()
