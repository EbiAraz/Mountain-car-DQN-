# Mountain-car-DQN-

Minimal DQN implementation for `MountainCar-v0` using **gymnasium** and **torch**.

## Install

```bash
pip install gymnasium torch numpy
```

## Run training

```bash
python mountain_car_dqn.py --episodes 500 --batch-size 64 --max-steps 200
```

## Run tests

```bash
python -m unittest discover -s tests -q
```
