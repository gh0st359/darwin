import unittest

from darwin.causal import CausalModel
from darwin.types import Transition


class CausalModelTests(unittest.TestCase):
    def test_learns_conditional_toggle_effect(self) -> None:
        model = CausalModel(min_samples=2)
        model.learn(
            Transition(
                before={"switch_on": False},
                action="toggle_switch",
                after={"switch_on": True},
                reward=1.0,
            )
        )
        model.learn(
            Transition(
                before={"switch_on": True},
                action="toggle_switch",
                after={"switch_on": False},
                reward=0.0,
            )
        )

        off_prediction = model.predict({"switch_on": False}, "toggle_switch")
        on_prediction = model.predict({"switch_on": True}, "toggle_switch")

        self.assertEqual(off_prediction.state["switch_on"], True)
        self.assertEqual(on_prediction.state["switch_on"], False)

    def test_expected_reward_uses_observed_payoff(self) -> None:
        model = CausalModel(min_samples=2)
        model.learn(Transition(before={}, action="open_curtains", after={}, reward=0.9))
        model.learn(Transition(before={}, action="open_curtains", after={}, reward=1.1))

        estimate = model.expected_reward({}, "open_curtains")

        self.assertAlmostEqual(estimate.mean, 1.0)
        self.assertEqual(estimate.samples, 2)


if __name__ == "__main__":
    unittest.main()

