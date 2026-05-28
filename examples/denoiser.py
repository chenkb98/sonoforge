import json
import numpy as np
from sonoforge.diffusion import (
    initialize_model,
    linear_schedule,
    training_pairs,
    train_denoiser,
    training_loss,
    sample,
)

clean = np.stack([np.sin(2 * np.pi * np.arange(8) / 8 + phase) for phase in np.linspace(0, 6, 24)])
noisy, epsilon, times = training_pairs(clean, linear_schedule(10), seed=7)
initial = initialize_model(8, seed=7)
trained = train_denoiser(initial, noisy, times, epsilon, steps=100)
print(
    json.dumps(
        {
            "synthetic": True,
            "before": training_loss(initial, noisy, times, epsilon),
            "after": training_loss(trained, noisy, times, epsilon),
            "generated_frames": len(sample(trained, linear_schedule(10), 3)),
        }
    )
)
