# Lottery Ticket Hypothesis

forked from [Rahul-Vigneswaran K's repo](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch).

---
Here's an example argument set.
```
--lr=2e-4
--batch_size=60
--end_iter=25
--resume
--gpu=1
--dataset=cifar10
--arch_type=conv2
--prune_percent=10
--prune_iterations=27
--split_conv_and_fc
--fc_prune_percent=20
```
For now there's a weird bug that makes every `#param-testacc` graph identical when trying experiments multiple time.
I think it's due to saving/loading mechanism from original repo, i should fix asap.
