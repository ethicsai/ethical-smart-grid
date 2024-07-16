"""
Test that our custom environment works according to the PettingZoo Parallel API.
"""


import unittest
from pettingzoo.test import parallel_api_test

from smartgrid import make_basic_smartgrid


class TestPettingZoo(unittest.TestCase):

    def test_parallel_api(self):
        env = make_basic_smartgrid(max_step=10_000)
        parallel_api_test(env, num_cycles=1_000)


if __name__ == '__main__':
    unittest.main()
