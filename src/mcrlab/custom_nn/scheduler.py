# -----------
# > Imports <
# -----------
from torch.optim.lr_scheduler import StepLR, LambdaLR



# ----------
# > Getter <
# ----------
def get_scheduler(name:str, optimizer, 
                  warmup_epochs=0, check_valid=True,
                  step_size=7, gamma=0.1,
                  **kwargs):
    # FIXME -> add warm up, maybe make warm up wrapper!
    name = name.lower()

    scheduler = None

    if name == "step":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs)
    # ...

    # Wrap it in warmup scheduler
    if warmup_epochs > 0 and scheduler is not None:
        warmup = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, scheduler],
            milestones=[warmup_epochs]
        )

    if check_valid:
        if scheduler is None:
            raise ValueError(f"Could not create Scheduler with name '{name}'.")

    return scheduler



# -------------------
# > Warm-Up Wrapper <
# -------------------

# class WarmUpScheduleWrapper:
#     def __init__(self, optimizer, scheduler, warmup_epochs=1):
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.runned_warmup_epochs = 0
#         self.warmup_epochs = warmup_epochs
#         self.warmup_active = True
#         def warmup_scheduler_fn(epoch):
#             return min(1.0, (epoch + 1) / self.warmup_epochs)

#         self.warmup_scheduler = LambdaLR(optimizer, warmup_scheduler_fn)

#     def step(self):
#         if self.warmup_active:
#             if self.runned_warmup_epochs >= self.warmup_epochs:
#                 self.warmup_active = False
#             else:
#                 self.warmup_scheduler.step()
#                 self.runned_warmup_epochs += 1
#         else:
#             self.scheduler.step()













