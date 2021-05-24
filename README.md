TODO:
- loss(model=model) ?
- gpu support ?
- reset `module.previous_input`. Either automatically if we detect shape inconsistencies, or manually through a method, or both. Maybe detection + call method in optimizer.zero_grad() ? (because the batch size is static currently)
- `__str__` and `__repr__`
- why `gradwrtinput = (self.last_input[0]-self.last_input[1])*2/30` to match pytorch ?
- fix `eval()`, `train()` and `last_input`

DESIGN
- do not use `gradfn` field but use `grad` field in tensors so that we can access them directly with the optimizer.
- modules update the `grad` field of their tensors parameters directly in the backward pass.
- optimizers have a list of pointers to tensors and simply read `grad` field.