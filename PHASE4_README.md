# Phase 4: Training Options

You have two options for training the DQN agent:

## Option 1: Quick Test (~30 minutes) ⚡

Perfect for testing that everything works before committing to a long training run:

```bash
python run_phase4_quick_test.py
```

**Parameters:**

- Steps: 100,000 (vs 1,000,000 full)
- Time: ~30 minutes on CPU
- Expected performance: Basic learning, may not surpass baseline

## Option 2: Full Training (8-12 hours) 🎯

For best results and portfolio-quality training:

```bash
python src/phase4_train_dqn.py
```

**Parameters:**

- Steps: 1,000,000
- Time: 8-12 hours on CPU, 2-4 hours on GPU
- Expected performance: 2-3x better than baseline (300-500 reward)

## What Gets Created

Both options create:

```
models/phase4/
  ├── dqn_final.pt              # Final trained model
  └── dqn_checkpoint_*.pt       # Checkpoints during training

data/phase4/
  └── training_metrics.json     # Complete training statistics

outputs/phase4/
  └── training_results.png      # Training curves visualization
```

## After Training

Open the analysis notebook:

```bash
jupyter notebook notebooks/phase4_dqn_training.ipynb
```

This notebook will:

- ✅ Load and visualize training results
- ✅ Compare DQN vs random baseline (146.95 ± 93.14)
- ✅ Test the trained agent
- ✅ Analyze action distribution
- ✅ Generate gameplay visualizations

## Monitoring Progress

While training runs, you can monitor:

1. **Progress bar**: Shows current step, epsilon, buffer size
2. **Evaluation rewards**: Printed every 50k steps (full) or 10k steps (quick)
3. **Checkpoints**: Saved every 100k steps (full) or 25k steps (quick)

## Tips

- **Use GPU if available**: Training is 3-4x faster
- **Start with quick test**: Verify everything works before full training
- **Training can be interrupted**: Resume by loading latest checkpoint
- **Monitor system resources**: Training uses ~2GB RAM for replay buffer

## Troubleshooting

### Out of Memory

Edit training script and reduce:

```python
buffer_size=50_000   # (default: 100_000)
batch_size=16        # (default: 32)
```

### Too Slow

Options:

1. Use quick test version (100k steps)
2. Enable GPU if available
3. Reduce evaluation frequency

### Not Learning

- Ensure preprocessing module works (Phase 3)
- Check replay buffer is filling up
- Try running for more steps
- Verify training loss is decreasing

## Next Steps

After training completes and you analyze results:

1. **Commit to GitHub**:

   ```bash
   git add -A
   git commit -m "Phase 4 complete: DQN agent training"
   git push
   ```

2. **Move to Phase 5**: DQN improvements (Double DQN, Dueling Networks, Prioritized Replay)

---

**Questions?** Check `docs/PHASE4_GUIDE.md` for detailed documentation.
