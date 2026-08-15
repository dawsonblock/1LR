from pathlib import Path
from lgae_v3.config import load_config

def test_default_yaml_loads():
    p=Path(__file__).parents[1]/"configs/default.yaml"
    c=load_config(p)
    assert c.fiber.d_base < c.fiber.d_max
