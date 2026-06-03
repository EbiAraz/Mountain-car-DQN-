import importlib.util
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_NUMPY and HAS_TORCH:
    import numpy as np
    import torch
    from mountain_car_dqn import DQN, select_action


@unittest.skipUnless(HAS_NUMPY and HAS_TORCH, "numpy and torch are required for DQN tests")
class TestMountainCarDQN(unittest.TestCase):
    def test_dqn_output_shape(self):
        model = DQN(state_size=2, action_size=3)
        batch = torch.randn(4, 2)
        output = model(batch)
        self.assertEqual(output.shape, (4, 3))

    def test_select_action_returns_valid_action(self):
        model = DQN(state_size=2, action_size=3)
        state = np.array([0.0, 0.0], dtype=np.float32)
        action = select_action(model, state, epsilon=0.0, action_size=3, device=torch.device("cpu"))
        self.assertIn(action, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
