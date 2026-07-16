## High learning rate can lead to plateaue.

Q to ChatGPT: high learning rate leads to loss being high but not fluctuating. Is that common?

Yes, it is possible and fairly common for a high learning rate to produce a loss that is high but not obviously fluctuating. Many people expect "too high LR → loss oscillates wildly", but that is only one failure mode.

There are a few scenarios:

1. Learning rate is too high, but the optimizer keeps moving in a bad direction

Example:

LR = 1e-5  (good)
loss:
10 → 8 → 6 → 4 → 3

LR = 1e-2  (too high)
loss:
10 → 9.8 → 9.7 → 9.6 → 9.5

The second case may look stable, but it is actually failing because updates are too large and the optimizer cannot settle into a good basin. The loss decreases very slowly or plateaus at a bad value.

This is common with:

Adam/AdamW
large batch training
models with normalization layers
transformers

