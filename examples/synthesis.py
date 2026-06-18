from pathlib import Path

from sonoforge.experiments import generate_run, sweep, write_run

Path("outputs").mkdir(exist_ok=True)
settings = sweep({"duration": 0.05}, [0], [220, 440, 880])
print(write_run("outputs/tones", generate_run(settings)))
